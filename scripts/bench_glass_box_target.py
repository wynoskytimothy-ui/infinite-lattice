#!/usr/bin/env python3
"""Benchmark glass_box_search configs toward SciFact nDCG@10 ~0.8."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from aethos_bridges import RelevanceBridges, bridge_search
from aethos_glass_box_search import glass_box_search, GlassBoxSearchConfig
from aethos_encyclopedia_teacher import load_glossary
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from scripts.knowledge_bridges import inject


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


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"{name}: {len(corpus)} docs, {len(test_ids)} test queries")

    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    print(f"  ingest {time.perf_counter() - t0:.1f}s")

    gloss = load_glossary(name)
    br = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)
    br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=1)

    br_kb = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)
    br_kb.learn_rarest_corridors(queries, train_q, corpus, min_pairs=1)
    inject(br_kb, gloss, idx, N, per_term=10, gate=2.0)

    configs = {
        "lexical_bm25": None,
        "bridge_search": "bridge",
        "glass_default": "default",
        "glass_target": "target",
        "glass_lattice": "lattice",
        "glass_target_kb": "target_kb",
    }

    results = []
    for label, mode in configs.items():
        if mode is None:
            fn = lambda q, _m=mode: idx.search(q, 10)
        elif mode == "bridge":
            fn = lambda q, _m=mode: bridge_search(idx, br, q)[:10]
        elif mode == "default":
            fn = lambda q, _m=mode: glass_box_search(
                idx, br, q, 10, glossary=gloss, corpus=corpus,
            )
        elif mode == "target":
            cfg = GlassBoxSearchConfig.scifact_target()
            fn = lambda q, _m=mode, _cfg=cfg: glass_box_search(
                idx, br, q, 10, glossary=gloss, config=_cfg, corpus=corpus,
            )
        elif mode == "lattice":
            from aethos_lattice_lexical import lattice_lexical_scorer
            cfg = GlassBoxSearchConfig.scifact_lattice()
            scorer = lattice_lexical_scorer(idx, mode="lattice_pure", pair_w=0.0)
            fn = lambda q, _m=mode, _cfg=cfg, _s=scorer: glass_box_search(
                idx, br, q, 10, glossary=gloss, config=_cfg, scorer=_s, corpus=corpus,
            )
        else:
            cfg = GlassBoxSearchConfig.scifact_target()
            fn = lambda q, _m=mode, _cfg=cfg: glass_box_search(
                idx, br_kb, q, 10, glossary=gloss, config=_cfg, corpus=corpus,
            )
        nd, rc, mr = evaluate(fn, queries, test_q, test_ids)
        results.append((label, nd, rc, mr))
        print(f"  {label:<22} nDCG@10 {nd:.4f}  R@10 {rc:.4f}  MRR@10 {mr:.4f}")

    best = max(results, key=lambda x: x[1])
    print(f"\n  best: {best[0]} nDCG@10={best[1]:.4f}  (gap to 0.8: {0.8 - best[1]:+.4f})")


if __name__ == "__main__":
    main()
