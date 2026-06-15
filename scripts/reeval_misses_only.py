#!/usr/bin/env python3
"""Re-eval 10 route misses after trinary train."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import knowledge_path
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k, recall_at_k
from eval_beir_symbol import load_brain_and_plane, mrr_at_k, query_words
from pipeline.bit_12_symbol_plane_index import (
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)
from scripts.train_reeval_route_misses import patch_plane_for_words

MISS_IDS = ["1", "3", "13", "36", "48", "54", "94", "99", "127", "132"]


def load_brain_and_trained_plane(
    brain_name: str,
    *,
    plane_dataset: str = "scifact",
) -> tuple:
    kpath = knowledge_path(brain_name)
    if not kpath.is_file():
        raise FileNotFoundError(f"brain not found: {kpath}")
    plane_path = knowledge_path(plane_dataset).parent / f"{plane_dataset}_plane.pkl"
    return load_brain_and_plane(
        brain_name,
        brain_path=kpath,
        plane_path=plane_path,
    )


def main() -> int:
    paths = load_paths(Path(resolve_beir_root()), "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    brain_name = "scifact_miss_trinary"
    if not knowledge_path(brain_name).is_file():
        brain_name = "scifact"
    print(f"Loading {brain_name} brain + scifact plane ...", flush=True)
    knowledge, plane = load_brain_and_trained_plane(brain_name)

    touch = set()
    for qid in MISS_IDS:
        touch.update(query_words(queries[qid]))
    patch = patch_plane_for_words(knowledge, plane, touch)
    print(f"Plane patch: {patch}", flush=True)

    results = []
    for qid in MISS_IDS:
        words = query_words(queries[qid])
        t0 = time.perf_counter()
        route = route_symbol_plane_candidates(knowledge, plane, words)
        ranked = [d for d, _ in rank_symbol_plane_docs(knowledge, plane, words, limit=100)]
        ms = (time.perf_counter() - t0) * 1000
        rel = qrels[qid]
        in_route = any(g in route.doc_ids for g in rel)
        results.append({
            "query_id": qid,
            "gold_in_route": in_route,
            "ndcg_at_10": ndcg_at_k(ranked, rel, 10),
            "recall_at_10": recall_at_k(ranked, rel, 10),
            "mrr_at_10": mrr_at_k(ranked, rel, 10),
            "top5": ranked[:5],
            "gold": list(rel.keys()),
            "ms": round(ms, 1),
        })
        print(
            f"  Q{qid}: route={in_route}  ndcg={results[-1]['ndcg_at_10']:.3f}  "
            f"recall={results[-1]['recall_at_10']:.3f}  {ms:.0f}ms",
            flush=True,
        )

    closed = [r["query_id"] for r in results if r["gold_in_route"]]
    ndcg_mean = sum(r["ndcg_at_10"] for r in results) / len(results)
    recall_mean = sum(r["recall_at_10"] for r in results) / len(results)
    out = _ROOT / "logs" / "trinary_reeval_misses.json"
    payload = {
        "brain": brain_name,
        "before_route_recall": 0.0,
        "before_ndcg_at_10": 0.0,
        "after_route_recall": len(closed) / len(MISS_IDS),
        "after_ndcg_at_10": round(ndcg_mean, 6),
        "after_recall_at_10": round(recall_mean, 6),
        "routing_closed": closed,
        "per_query": results,
        "plane_patch": patch,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nRouting closed: {len(closed)}/10  {closed}", flush=True)
    print(f"nDCG@10: 0.000 -> {ndcg_mean:.4f}  Recall@10: {recall_mean:.4f}", flush=True)
    print(f"Report: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
