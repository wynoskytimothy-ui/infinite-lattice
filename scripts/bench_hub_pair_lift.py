#!/usr/bin/env python3
"""Measure hub-penalty + rare-pair kappa lift on SciFact held-out test."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def eval_brain(brain, name, queries, test_q, test_ids):
    nd = rc = 0.0
    t0 = time.perf_counter()
    for qid in test_ids:
        res = brain.search(queries[qid], corpus=name, k=10)
        nd += ndcg10(res.local_ids, test_q[qid])
        rc += recall10(res.local_ids, test_q[qid])
    n = len(test_ids)
    ms = (time.perf_counter() - t0) / n * 1000
    return nd / n, rc / n, ms


def stack(name, corpus, queries, train_q, *, hub: float, pair: bool):
    brain = MultiCorpusBrain()
    brain.HUB_IDF_GATE = hub
    brain.ENABLE_PAIR_KEYS = pair
    brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)
    return brain


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    print(f"Hub-penalty + rare-pair kappa lift — {name} ({len(test_ids)} test q)")
    print(f"{'config':<28} {'nDCG@10':>9} {'R@10':>8} {'ms/q':>8}")
    print(f"{'prior kappa-primary':<28} {'0.7428':>9} {'0.8413':>8} {'~8.0':>8}")

    configs = [
        ("no hub, no pair", 99.0, False),
        ("hub penalty only", 2.0, False),
        ("pair keys only", 99.0, True),
        ("hub + pair (default)", 2.0, True),
    ]
    for label, hub, pair in configs:
        brain = stack(name, corpus, queries, train_q, hub=hub, pair=pair)
        nd, rc, ms = eval_brain(brain, name, queries, test_q, test_ids)
        print(f"{label:<28} {nd:>9.4f} {rc:>8.4f} {ms:>7.1f}")
        del brain


if __name__ == "__main__":
    main()
