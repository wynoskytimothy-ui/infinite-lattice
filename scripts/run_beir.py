#!/usr/bin/env python3
"""
run_beir.py - full-stack BEIR evaluation for the lattice retrieval engine.

    python scripts/run_beir.py <dataset> [min_pairs]

Reports nDCG@10 / Recall@10 / MRR@10 for the lexical engine and (if the dataset
has a train split) the supervised-bridge full stack, on the held-out test split.
Test-only corpora (no train qrels) report lexical only - bridges need training
data. This is the canonical benchmark for the package.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges
from scripts.bench_active_learning import best_search


def mrr10(ranked, rels):
    for i, d in enumerate(ranked[:10]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(fn, queries, test_q, test_ids):
    nd = rc = mr = 0.0
    for qid in test_ids:
        r = fn(queries[qid])
        nd += ndcg10(r, test_q[qid])
        rc += recall10(r, test_q[qid])
        mr += mrr10(r, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n, mr / n


def run(name, min_pairs=2):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*64}\n{name}: {len(corpus):,} docs | "
          f"train {len(train_q):,} q | test {len(test_ids):,} q")

    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    build_s = time.perf_counter() - t0
    print(f"  ingest: {build_s:.1f}s ({len(idx.alive)/build_s:,.0f} docs/s)")

    nd0, rc0, mr0 = evaluate(lambda q: idx.search(q, 10), queries, test_q, test_ids)
    print(f"  lexical engine:        nDCG@10 {nd0:.4f}  Recall@10 {rc0:.4f}  MRR@10 {mr0:.4f}")

    if train_q:
        br = RelevanceBridges(idx, len(idx.alive), min_pairs=min_pairs).learn(
            queries, train_q, corpus)
        nd1, rc1, mr1 = evaluate(lambda q: best_search(idx, br, q),
                                 queries, test_q, test_ids)
        print(f"  + supervised bridges:  nDCG@10 {nd1:.4f}  Recall@10 {rc1:.4f}  "
              f"MRR@10 {mr1:.4f}  ({nd1-nd0:+.4f} nDCG)")
        return nd0, nd1
    else:
        print("  (test-only corpus: no train qrels -> bridges not trainable; lexical only)")
        return nd0, None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run(name, mp)


if __name__ == "__main__":
    main()
