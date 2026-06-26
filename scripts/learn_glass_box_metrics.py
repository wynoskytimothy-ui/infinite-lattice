#!/usr/bin/env python3
"""
Learn glass-box metrics from any corpus and apply cross-corpus.

Stack corpora on one brain, audit each (train or test qrels), merge hub
diluters / bucket priors / bridge patterns into GlassBoxMemory, then re-eval.

  python scripts/learn_glass_box_metrics.py
  python scripts/learn_glass_box_metrics.py scifact nfcorpus

Metrics learned on SciFact automatically apply when searching NFCorpus
(same shared prime vocab — no reindex, append-only teach-style memory).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10

MEMORY_PATH = Path("logs/glass_box_learned.json")


def eval_corpus(brain, name, queries, qrels, qids):
    nd = rc = 0.0
    t0 = time.perf_counter()
    for qid in qids:
        res = brain.search(queries[qid], corpus=name, k=10)
        nd += ndcg10(res.local_ids, qrels[qid])
        rc += recall10(res.local_ids, qrels[qid])
    n = len(qids)
    ms = (time.perf_counter() - t0) / n * 1000
    return nd / n, rc / n, ms


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not names:
        names = ["scifact", "nfcorpus"]

    print("CROSS-CORPUS GLASS-BOX METRIC LEARNING")
    print(f"Corpora: {', '.join(names)}\n")

    brain = MultiCorpusBrain()
    corpora_data = {}

    for name in names:
        corpus, queries, train_q, test_q = load(name)
        test_ids = [q for q in test_q if q in queries]
        corpora_data[name] = (corpus, queries, train_q, test_q, test_ids)
        print(f"Stacking {name} ...", flush=True)
        brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)

    print("\n--- Before learning (no cross-corpus metrics) ---")
    brain.glass_box.enabled = False
    for name in names:
        _, queries, _, test_q, test_ids = corpora_data[name]
        nd, rc, ms = eval_corpus(brain, name, queries, test_q, test_ids)
        print(f"  {name:<12} nDCG@10={nd:.4f}  R@10={rc:.4f}  {ms:.1f} ms/q")

    print("\n--- Learning metrics from each corpus (test qrels audit) ---")
    brain.glass_box.enabled = True
    for name in names:
        _, queries, _, test_q, test_ids = corpora_data[name]
        summary = brain.learn_glass_box_metrics(
            name, queries, test_q, qids=test_ids,
        )
        hubs = summary.get("learned_hubs", [])[:8]
        print(
            f"  {name}: pool={summary['queries_gold_in_pool_pct']}% "
            f"top10={summary['queries_gold_in_top10_pct']}% "
            f"hubs={hubs}"
        )

    brain.glass_box.save(MEMORY_PATH)
    print(f"\nSaved merged memory -> {MEMORY_PATH}")
    print(f"  hub diluters: {len(brain.glass_box.hub_diluters)} terms")
    print(f"  bucket mix: {dict(brain.glass_box.bucket_counts)}")

    print("\n--- After learning (metrics applied to all corpora) ---")
    for name in names:
        _, queries, _, test_q, test_ids = corpora_data[name]
        nd, rc, ms = eval_corpus(brain, name, queries, test_q, test_ids)
        print(f"  {name:<12} nDCG@10={nd:.4f}  R@10={rc:.4f}  {ms:.1f} ms/q")


if __name__ == "__main__":
    main()
