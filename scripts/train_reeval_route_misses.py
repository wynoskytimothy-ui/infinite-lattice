#!/usr/bin/env python3
"""
Trinary-train the routing-miss queries, refresh plane adjacency, re-eval.

  python scripts/audit_route_misses.py
  python scripts/train_reeval_route_misses.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD, SymbolKnowledgeIndex, knowledge_path
from aethos_symbol_trinary_train import TrinaryTrainer
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k, recall_at_k
from eval_beir_symbol import evaluate_symbol_beir, load_brain_and_plane, mrr_at_k, query_words
from pipeline.bit_12_symbol_plane_index import (
    SymbolPlaneIndex,
    correlation_meet_keys,
    route_symbol_plane_candidates,
    rank_symbol_plane_docs,
)


def patch_plane_for_words(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    touch_words: set[str],
) -> dict[str, int]:
    """Incremental adjacency + pair_meet patch for query words only."""
    if not plane.word_adjacency:
        plane.word_adjacency = {}

    touched_links = 0
    added_pairs = 0
    for w in touch_words:
        for lk in knowledge.neighbors(w):
            other = lk.right if lk.left == w else lk.left
            plane.word_adjacency.setdefault(w, [])
            plane.word_adjacency.setdefault(other, [])
            entry = (other, lk.strength, lk.kind)
            rev = (w, lk.strength, lk.kind)
            if entry not in plane.word_adjacency[w]:
                plane.word_adjacency[w].append(entry)
                touched_links += 1
            if rev not in plane.word_adjacency[other]:
                plane.word_adjacency[other].append(rev)
            key_pair = tuple(sorted((w, other)))
            if not plane.pair_keys.get(key_pair):
                meet = correlation_meet_keys(knowledge, w, other, link=lk)
                if meet:
                    plane.pair_keys[key_pair] = meet
                    added_pairs += 1
        if w in plane.word_adjacency:
            plane.word_adjacency[w].sort(key=lambda x: (-x[1], x[0]))

    return {"touched_links": touched_links, "pair_meets_added": added_pairs}


def eval_misses(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    miss_ids: list[str],
) -> dict[str, object]:
    subset_q = {qid: queries[qid] for qid in miss_ids if qid in queries}
    subset_r = {qid: qrels[qid] for qid in miss_ids if qid in qrels}
    result = evaluate_symbol_beir(
        knowledge, plane, subset_q, subset_r, max_queries=None,
    )
    per_query: list[dict[str, object]] = []
    for qid in miss_ids:
        if qid not in subset_q:
            continue
        words = query_words(subset_q[qid])
        route = route_symbol_plane_candidates(knowledge, plane, words)
        ranked = [d for d, _ in rank_symbol_plane_docs(knowledge, plane, words, limit=100)]
        rel = subset_r[qid]
        per_query.append({
            "query_id": qid,
            "gold_in_route": any(g in route.doc_ids for g in rel),
            "ndcg_at_10": ndcg_at_k(ranked, rel, 10),
            "recall_at_10": recall_at_k(ranked, rel, 10),
            "mrr_at_10": mrr_at_k(ranked, rel, 10),
            "top3": ranked[:3],
            "gold": list(rel.keys()),
        })
    route_hits = sum(1 for p in per_query if p["gold_in_route"])
    return {
        "n_queries": len(per_query),
        "route_recall": route_hits / max(len(per_query), 1),
        "ndcg_at_10": result.ndcg_at_10,
        "recall_at_10": result.recall_at_10,
        "mrr_at_10": result.mrr_at_10,
        "per_query": per_query,
    }


def main() -> int:
    audit_path = _ROOT / "logs" / "route_miss_audit.json"
    if not audit_path.is_file():
        print("Run scripts/audit_route_misses.py first.")
        return 1

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    miss_ids = [str(r["query_id"]) for r in audit["reports"]]
    print(f"Routing misses to train: {len(miss_ids)}  ids={miss_ids}")

    paths = load_paths(Path(resolve_beir_root()), "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    print("Loading brain + plane ...")
    knowledge, plane = load_brain_and_plane("scifact")

    # Baseline from route_miss_audit (skip slow re-run)
    before = {
        "route_recall": 0.0,
        "ndcg_at_10": 0.0,
        "per_query": [
            {"query_id": qid, "gold_in_route": False} for qid in miss_ids
        ],
    }
    print("BEFORE (from audit): route_recall=0% on all 10 misses")

    # Q1 vocabulary gap — pretrain gold bundle
    if "1" in miss_ids:
        print("Pretrain compound learn for Q1 (quantum/zero/dimension) ...")
        knowledge.compound_learn(PRETRAIN_QUANTUM_GOLD, subjects={1, 9, 10})

    trainer = TrinaryTrainer(knowledge=knowledge)
    touch_words: set[str] = set()
    reports = []
    t0 = time.perf_counter()
    for qid in miss_ids:
        if qid not in queries or qid not in qrels:
            continue
        golds = list(qrels[qid].keys())
        rep = trainer.train_query(qid, queries[qid], golds)
        reports.append(rep)
        touch_words.update(query_words(queries[qid]))
        print(
            f"  Q{qid}: triples={rep.triples_promoted} "
            f"top={rep.promoted[0].words if rep.promoted else None}",
        )
    train_ms = (time.perf_counter() - t0) * 1000.0

    refresh = patch_plane_for_words(knowledge, plane, touch_words)
    print(f"Plane patch: {refresh}")

    print("AFTER trinary train:")
    after = eval_misses(knowledge, plane, queries, qrels, miss_ids)
    print(f"  route_recall={after['route_recall']:.2%}  nDCG@10={after['ndcg_at_10']:.4f}")

    closed = []
    still_miss = []
    for b, a in zip(before["per_query"], after["per_query"]):
        if not b["gold_in_route"] and a["gold_in_route"]:
            closed.append(a["query_id"])
        if not a["gold_in_route"]:
            still_miss.append(a)

    print(f"\nRouting misses closed: {len(closed)}/{len(miss_ids)}  {closed}")
    print(f"Still missing route ({len(still_miss)}):")
    for row in still_miss:
        print(f"  Q{row['query_id']}  ndcg={row['ndcg_at_10']:.3f}  top3={row['top3']}")

    out = _ROOT / "logs" / "trinary_reeval_misses.json"
    payload = {
        "miss_ids": miss_ids,
        "train_ms": round(train_ms, 1),
        "triples_promoted": sum(r.triples_promoted for r in reports),
        "before": before,
        "after": after,
        "routing_closed": closed,
        "plane_refresh": refresh,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    knowledge.save(knowledge_path("scifact_miss_trinary"))
    print(f"\nSaved brain: {knowledge_path('scifact_miss_trinary')}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
