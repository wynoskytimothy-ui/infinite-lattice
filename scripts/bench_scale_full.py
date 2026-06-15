#!/usr/bin/env python3
"""
Scaling law with ACCURACY: fast + accurate + small footprint vs corpus size N.

Unlike bench_scaling.py (which replicates docs and only measures speed), this
floods the corpus with realistic DISTRACTOR documents while keeping every gold
doc, then asks the real question at scale:

  - SPEED    : does query latency stay flat as N grows? (champion vs full scan)
  - ACCURACY : does gold stay on top as 10x-50x distractors pile in?
  - FOOTPRINT: postings / bytes-per-doc, full vs champion (top-M) lists.

Distractors are frequency-weighted random word docs (in-distribution unigram
statistics, zero real relevance) — a fair, hard flood test. Gold docs are always
present so nDCG@10 / Recall@10 are measured on real held-out queries at each N.

Run:  python scripts/bench_scale_full.py [scifact|nfcorpus] [--sizes 5000,20000,50000] [-M 400]
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def build_distractor_sampler(corpus, seed=1234):
    """Frequency-weighted word pool + length distribution from the real corpus."""
    freq: Counter = Counter()
    lengths = []
    for text in corpus.values():
        ws = words(text)
        lengths.append(min(len(ws), 200))
        freq.update(ws)
    vocab = list(freq.keys())
    weights = np.array([freq[w] for w in vocab], dtype=np.float64)
    weights /= weights.sum()
    mean_len = max(20, int(np.mean(lengths)) // 2)  # shorter distractors save memory
    rng = np.random.default_rng(seed)
    return vocab, weights, mean_len, rng


def make_distractors(n, vocab, weights, mean_len, rng):
    """n synthetic docs of frequency-weighted random words (irrelevant by design)."""
    out = {}
    vocab_arr = np.array(vocab, dtype=object)
    for i in range(n):
        ln = max(15, int(rng.poisson(mean_len)))
        idx = rng.choice(len(vocab), size=ln, p=weights)
        out[f"DISTRACT_{i}"] = " ".join(vocab_arr[idx])
    return out


def build_champion(idx, M):
    denom = idx._d_denom
    cd, ct = {}, {}
    for p, di in idx._d_pdoc.items():
        ptf = idx._d_ptf[p]
        if di.size <= M:
            cd[p], ct[p] = di, ptf
        else:
            tf = ptf.astype(np.float64)
            top = np.argpartition(tf / (tf + denom[di]), -M)[-M:]
            cd[p], ct[p] = di[top], ptf[top]
    return cd, ct


def champ_score(idx, cd, ct, query):
    N = max(1, len(idx.alive))
    k1p1 = idx.k1 + 1
    cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
    denom, df = idx._d_denom, idx.df
    scores = np.zeros(len(idx._d_docs), dtype=np.float64)
    for tok, qwt in idx._multiview(query).items():
        p = idx.token_prime.get(tok)
        if p is None:
            continue
        dfp = df[p]
        if dfp == 0:
            continue
        if cap is not None and tok[0] == "3" and dfp > cap:
            continue
        di = cd.get(p)
        if di is None:
            continue
        tfa = ct[p].astype(np.float64)
        scores[di] += (qwt * np.log(1 + (N - dfp + 0.5) / (dfp + 0.5)) * k1p1) \
            * tfa / (tfa + denom[di])
    return scores


def topk_from_scores(idx, scores, k=10):
    kk = min(k, len(idx._d_docs))
    part = np.argpartition(scores, -kk)[-kk:]
    part = part[np.argsort(scores[part])[::-1]]
    docs = idx._d_docs
    return [docs[i] for i in part if scores[i] > 0.0]


def accuracy(idx, score_fn, queries, test_ids, test_q):
    nd = rc = 0.0
    for qid in test_ids:
        ranked = topk_from_scores(idx, score_fn(queries[qid]), 10)
        nd += ndcg10(ranked, test_q[qid])
        rc += recall10(ranked, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n


def latency(score_fn, qsample, reps=2):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for q in qsample:
            score_fn(q)
        best = min(best, (time.perf_counter() - t0) / len(qsample) * 1000)
    return best


def run(name, sizes, M):
    corpus, queries, _, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    base_n = len(corpus)
    qsample = [queries[q] for q in test_ids[:60]]
    vocab, weights, mean_len, rng = build_distractor_sampler(corpus)

    print(f"\n{'='*78}")
    print(f"  {name.upper()} scaling — base {base_n:,} real docs (all gold kept), "
          f"champion M={M}")
    print(f"{'='*78}")
    print(f"{'N docs':>9} {'idtype':>7} {'B/doc':>7} {'champMB':>8} {'fullMB':>7} "
          f"{'full ms':>8} {'champ ms':>9} {'nDCG full':>10} {'nDCG champ':>11} {'R@10':>7}")

    for target in sizes:
        c = dict(corpus)
        n_extra = max(0, target - base_n)
        if n_extra:
            c.update(make_distractors(n_extra, vocab, weights, mean_len, rng))

        idx = AppendOnlyLatticeIndex()
        for d, t in c.items():
            idx.add(d, t)
        idx.finalize()
        N = len(idx.alive)
        idt = "uint16" if N < 65536 else "uint32"

        full_post = sum(a.size for a in idx._d_pdoc.values())
        bpp = (2 if N < 65536 else 4) + 2  # docid + f16 tf
        full_mb = (full_post * bpp + idx._d_denom.nbytes) / 1e6

        cd, ct = build_champion(idx, M)
        champ_post = sum(a.size for a in cd.values())
        champ_mb = (champ_post * bpp + idx._d_denom.nbytes) / 1e6
        bdoc = champ_mb * 1e6 / N

        full_ms = latency(idx._dense_score_array, qsample)
        champ_ms = latency(lambda q: champ_score(idx, cd, ct, q), qsample)

        nd_full, _ = accuracy(idx, idx._dense_score_array, queries, test_ids, test_q)
        nd_champ, rc_champ = accuracy(
            idx, lambda q: champ_score(idx, cd, ct, q), queries, test_ids, test_q,
        )

        print(f"{N:>9,} {idt:>7} {bdoc:>6.0f} {champ_mb:>7.1f} {full_mb:>6.1f} "
              f"{full_ms:>7.2f} {champ_ms:>8.2f} {nd_full:>10.4f} {nd_champ:>11.4f} "
              f"{rc_champ:>7.4f}")
        del idx, cd, ct, c

    print(f"\n  full scan: latency grows ~linearly with N (lossless).")
    print(f"  champion (top-M={M}): latency ~flat = bounded work at ANY N.")
    print(f"  nDCG columns show gold retention as distractors flood the corpus.")


def main():
    name = "scifact"
    sizes = [None, 20000, 50000]
    M = 400
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--sizes":
            sizes = [int(x) for x in args[i + 1].split(",")]
            i += 2
        elif a == "-M":
            M = int(args[i + 1])
            i += 2
        elif not a.startswith("-"):
            name = a
            i += 1
        else:
            i += 1

    corpus, _, _, _ = load(name)
    base_n = len(corpus)
    sizes = [base_n if s is None else s for s in sizes]
    sizes = sorted({max(s, base_n) for s in sizes})

    print("SCALING: fast + accurate + small footprint vs corpus size N")
    run(name, sizes, M)


if __name__ == "__main__":
    main()
