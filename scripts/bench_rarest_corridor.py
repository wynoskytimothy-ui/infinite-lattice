#!/usr/bin/env python3
"""Bench miss-r1 recovery stack on SciFact held-out."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_append_index import words
from aethos_glass_box_metrics import _prefix_hits, _rarest_terms
from aethos_multi_corpus import IdfCache, MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def eval_brain(brain, test_ids, queries, test_q):
    branch = brain._corpora["scifact"]
    idf = IdfCache(branch.idx, branch.n_docs)
    nd = rc = top1 = 0.0
    miss_r1 = miss_r1_top1 = 0
    for qid in test_ids:
        query = queries[qid]
        r = brain.search(query, corpus="scifact", k=10)
        rel = test_q[qid]
        local = r.local_ids
        nd += ndcg10(local, rel)
        rc += recall10(local, rel)
        if local and local[0] in rel:
            top1 += 1
        rarest = _rarest_terms(words(query), idf)
        for local_id, score in rel.items():
            if score <= 0:
                continue
            toks = set(words(branch.texts.get(local_id, "")))
            h1, _ = _prefix_hits(toks, rarest, 1)
            if h1:
                continue
            miss_r1 += 1
            if local and local[0] == local_id:
                miss_r1_top1 += 1
    n = len(test_ids)
    return {
        "ndcg10": nd / n,
        "recall10": rc / n,
        "top1_pct": 100 * top1 / n,
        "miss_r1_top1_pct": 100 * miss_r1_top1 / max(miss_r1, 1),
        "miss_r1_n": miss_r1,
    }


def main() -> None:
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]

    configs = [
        ("baseline (no miss-r1)", False, False),
        ("corridors only", True, False),
        ("corridors + teach", True, True),
    ]
    for label, corridors, teach in configs:
        brain = MultiCorpusBrain()
        brain.ENABLE_RARE_CORRIDORS = corridors
        brain.ENABLE_MISS_R1_TEACH = teach
        brain.stack_corpus("scifact", corpus, queries=queries, train_qrels=train_q)
        br = brain._corpora["scifact"]
        n_cor = len(br.pair_bridges.corridor_bridge) if br.pair_bridges else 0
        n_def = len(br.teach.definitions)
        m = eval_brain(brain, test_ids, queries, test_q)
        print(
            f"{label:<22} cor={n_cor:<4} defs={n_def:<4} "
            f"nDCG={m['ndcg10']:.4f}  R@10={m['recall10']:.4f}  "
            f"top1={m['top1_pct']:.1f}%  miss-r1-top1={m['miss_r1_top1_pct']:.1f}% "
            f"({m['miss_r1_n']} inst)"
        )


if __name__ == "__main__":
    main()
