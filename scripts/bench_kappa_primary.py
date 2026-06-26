#!/usr/bin/env python3
"""
κ-primary ingest mode — footprint / speed / accuracy vs full multi-view.

Compares stack_corpus(index_mode=...) on SciFact held-out test:
  full          — word + trigram + prefix postings, search() full scan
  kappa_primary — word postings only, search() -> scale_search bounded pool

Run:  python scripts/bench_kappa_primary.py [scifact|nfcorpus]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def compact_bytes(idx) -> tuple[int, int, int]:
    postings = sum(len(p) for p in idx.postings.values())
    compact_b = postings * 5
    n = max(1, len(idx.alive))
    return postings, compact_b, compact_b // n


def run(name: str) -> None:
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    qsample = [queries[q] for q in test_ids[:80]]
    mp = 1 if name == "scifact" else 2

    print(f"\n{'='*78}")
    print(f"  {name.upper()} — kappa-primary vs full multi-view")
    print(f"{'='*78}")
    print(f"{'mode':<16} {'postings':>10} {'B/doc':>8} {'build s':>8} {'q ms':>7} "
          f"{'nDCG@10':>9} {'R@10':>7} {'pool':>6}")

    for mode in ("full", "kappa_primary"):
        t0 = time.perf_counter()
        brain = MultiCorpusBrain()
        brain.stack_corpus(
            name, corpus,
            queries=queries, train_qrels=train_q,
            index_mode=mode,
        )
        build_s = time.perf_counter() - t0
        branch = brain._corpora[name]
        postings, compact_b, bdoc = compact_bytes(branch.idx)

        t1 = time.perf_counter()
        for q in qsample:
            brain.search(q, corpus=name, k=10)
        q_ms = (time.perf_counter() - t1) / len(qsample) * 1000

        nd = rc = pool = 0.0
        for qid in test_ids:
            res = brain.search(queries[qid], corpus=name, k=10)
            nd += ndcg10(res.local_ids, test_q[qid])
            rc += recall10(res.local_ids, test_q[qid])
            pool += res.kappa_candidates
        n = len(test_ids)
        pool_avg = pool / n

        print(
            f"{mode:<16} {postings:>10,} {bdoc:>8,} {build_s:>7.1f} {q_ms:>6.1f} "
            f"{nd/n:>9.4f} {rc/n:>7.4f} {pool_avg:>6.0f}"
        )
        del brain


def main():
    name = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "scifact"
    print("KAPPA-PRIMARY INGEST — footprint / speed / accuracy")
    run(name)


if __name__ == "__main__":
    main()
