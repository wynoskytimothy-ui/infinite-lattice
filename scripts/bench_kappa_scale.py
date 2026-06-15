#!/usr/bin/env python3
"""
Kappa-routed O(1) candidate generation — latency + accuracy + pool size vs N.

Compares two query paths on a distractor-flooded corpus (gold always kept):

  FULL   : brain.search       — full lexical scan + kappa fusion + routed teach.
           Exact, but scoring touches postings that grow with N.
  SCALE  : brain.scale_search — kappa buckets + rare-term exact recall give a
           BOUNDED candidate pool, scored restricted. Query work ~ independent
           of N. The formula-driven path.

We report: candidate pool size (should stay flat as N grows), latency for both,
and nDCG@10 / Recall@10 for both (does the bounded pool keep the gold?).

Run:  python scripts/bench_kappa_scale.py [scifact|nfcorpus] [--sizes 5000,20000,50000]
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import words
from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def distractors(corpus, n, seed=7):
    freq: Counter = Counter()
    lengths = []
    for text in corpus.values():
        ws = words(text)
        lengths.append(min(len(ws), 200))
        freq.update(ws)
    vocab = np.array(list(freq.keys()), dtype=object)
    w = np.array([freq[v] for v in vocab], dtype=np.float64)
    w /= w.sum()
    mean_len = max(20, int(np.mean(lengths)) // 2)
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        ln = max(15, int(rng.poisson(mean_len)))
        out[f"DISTRACT_{i}"] = " ".join(vocab[rng.choice(len(vocab), size=ln, p=w)])
    return out


def latency(fn, qsample, reps=2):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for q in qsample:
            fn(q)
        best = min(best, (time.perf_counter() - t0) / len(qsample) * 1000)
    return best


def accuracy(search_fn, queries, test_ids, test_q):
    nd = rc = pool = 0.0
    for qid in test_ids:
        res = search_fn(queries[qid])
        nd += ndcg10(res.local_ids, test_q[qid])
        rc += recall10(res.local_ids, test_q[qid])
        pool += res.kappa_candidates
    n = len(test_ids)
    return nd / n, rc / n, pool / n


def run(name, sizes):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    base_n = len(corpus)
    qsample = [queries[q] for q in test_ids[:60]]

    print(f"\n{'='*86}")
    print(f"  {name.upper()} — kappa-routed scale path vs full scan (gold kept, flooded)")
    print(f"{'='*86}")
    print(f"{'N docs':>9} {'pool':>6} {'full ms':>8} {'scale ms':>9} {'spdup':>6} "
          f"{'nDCG full':>10} {'nDCG scale':>11} {'R@10 full':>10} {'R@10 scale':>11}")

    for target in sizes:
        c = dict(corpus)
        if target > base_n:
            c.update(distractors(corpus, target - base_n))

        brain = MultiCorpusBrain()
        brain.stack_corpus(name, c, queries=queries, train_qrels=train_q)

        full = lambda q: brain.search(q, corpus=name, k=10)
        scale = lambda q: brain.scale_search(q, corpus=name, k=10)

        full_ms = latency(full, qsample)
        scale_ms = latency(scale, qsample)
        nd_f, rc_f, _ = accuracy(full, queries, test_ids, test_q)
        nd_s, rc_s, pool = accuracy(scale, queries, test_ids, test_q)

        N = brain._corpora[name].n_docs
        spd = full_ms / scale_ms if scale_ms else 0.0
        print(f"{N:>9,} {pool:>6.0f} {full_ms:>7.2f} {scale_ms:>8.2f} {spd:>5.1f}x "
              f"{nd_f:>10.4f} {nd_s:>11.4f} {rc_f:>10.4f} {rc_s:>11.4f}")
        del brain, c

    print(f"\n  pool = mean candidate set scored by scale_search (flat = N-independent).")
    print(f"  full scan grows with N; scale path stays bounded. nDCG shows gold retention.")


def main():
    name = "scifact"
    sizes = [None, 20000, 50000]
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--sizes":
            sizes = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif not args[i].startswith("-"):
            name = args[i]
            i += 1
        else:
            i += 1

    corpus, _, _, _ = load(name)
    base_n = len(corpus)
    sizes = sorted({base_n if s is None else max(s, base_n) for s in sizes})
    print("KAPPA-ROUTED O(1) CANDIDATE GENERATION — scale benchmark")
    run(name, sizes)


if __name__ == "__main__":
    main()
