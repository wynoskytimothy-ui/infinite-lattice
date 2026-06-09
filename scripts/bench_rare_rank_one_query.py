#!/usr/bin/env python3
"""
Benchmark rare-weighted correlation ranking (no trinary training).

  python scripts/bench_rare_rank_one_query.py
  python scripts/bench_rare_rank_one_query.py --query-id 54
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_rare_rank import gold_rank, rank_docs_rare_weighted
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k, recall_at_k
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    query_symbol_plane_keys,
    route_symbol_plane_candidates,
)

ROUTE_MISS_AUDIT = _ROOT / "logs" / "route_miss_audit.json"
DEFAULT_OUT = _ROOT / "logs" / "rare_rank_benchmark.json"


def bench_query(
    knowledge,
    plane,
    qid: str,
    query_text: str,
    rel: dict[str, int],
    *,
    route_max: int = 600,
) -> dict[str, object]:
    words = query_words(query_text)
    gold_ids = list(rel.keys())

    t0 = time.perf_counter()
    route = route_symbol_plane_candidates(
        knowledge, plane, words, max_candidates=route_max,
    )
    route_ms = (time.perf_counter() - t0) * 1000.0

    gold_in_route = any(g in route.doc_ids for g in rel)

    t0 = time.perf_counter()
    keys = query_symbol_plane_keys(knowledge, plane, words)
    kappa_scored = sorted(
        (
            (did, plane.score_overlap(keys, did))
            for did in route.doc_ids
        ),
        key=lambda x: -x[1],
    )
    kappa_ranked = [did for did, s in kappa_scored if s > 0]
    seen = set(kappa_ranked)
    for did in route.doc_ids:
        if did not in seen:
            kappa_ranked.append(did)
    kappa_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    rare_ranked = [
        doc_id for doc_id, _ in rank_docs_rare_weighted(
            knowledge,
            plane,
            words,
            route.doc_ids,
            knowledge.corpus,
        )
    ]
    seen_r = set(rare_ranked)
    for did in route.doc_ids:
        if did not in seen_r:
            rare_ranked.append(did)
    rare_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "query_id": qid,
        "query": query_text[:160],
        "gold": gold_ids,
        "gold_in_route": gold_in_route,
        "route_pool": len(route.doc_ids),
        "route_ms": round(route_ms, 2),
        "kappa_rank_ms": round(kappa_ms, 2),
        "rare_rank_ms": round(rare_ms, 2),
        "before": {
            "top5": kappa_ranked[:5],
            "gold_rank": gold_rank(kappa_ranked, gold_ids),
            "ndcg_at_10": round(ndcg_at_k(kappa_ranked, rel, 10), 6),
            "recall_at_10": round(recall_at_k(kappa_ranked, rel, 10), 6),
        },
        "after": {
            "top5": rare_ranked[:5],
            "gold_rank": gold_rank(rare_ranked, gold_ids),
            "ndcg_at_10": round(ndcg_at_k(rare_ranked, rel, 10), 6),
            "recall_at_10": round(recall_at_k(rare_ranked, rel, 10), 6),
        },
        "gold_moved_into_top10": (
            gold_rank(kappa_ranked, gold_ids) is None
            or gold_rank(kappa_ranked, gold_ids) > 10
        ) and (
            gold_rank(rare_ranked, gold_ids) is not None
            and gold_rank(rare_ranked, gold_ids) <= 10
        ),
    }


def aggregate(reports: list[dict[str, object]]) -> dict[str, object]:
    n = max(len(reports), 1)
    return {
        "n_queries": len(reports),
        "gold_in_route": sum(1 for r in reports if r["gold_in_route"]),
        "mean_ndcg_before": round(
            sum(r["before"]["ndcg_at_10"] for r in reports) / n, 6,
        ),
        "mean_ndcg_after": round(
            sum(r["after"]["ndcg_at_10"] for r in reports) / n, 6,
        ),
        "mean_recall_before": round(
            sum(r["before"]["recall_at_10"] for r in reports) / n, 6,
        ),
        "mean_recall_after": round(
            sum(r["after"]["recall_at_10"] for r in reports) / n, 6,
        ),
        "gold_moved_into_top10": sum(
            1 for r in reports if r.get("gold_moved_into_top10")
        ),
        "gold_rank_improved": sum(
            1 for r in reports
            if (r["after"]["gold_rank"] or 999)
            < (r["before"]["gold_rank"] or 999)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Rare-weighted rank benchmark")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--brain", default=None)
    ap.add_argument("--query-id", default=None, help="Single query (default: Q54 + 10 misses)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--route-max", type=int, default=600)
    args = ap.parse_args()

    brain = args.brain or args.dataset
    paths = load_paths(Path(resolve_beir_root()), args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    knowledge, plane = load_brain_and_plane(brain)

    if args.query_id:
        qids = [str(args.query_id)]
    else:
        qids = ["54"]
        if ROUTE_MISS_AUDIT.is_file():
            audit = json.loads(ROUTE_MISS_AUDIT.read_text(encoding="utf-8"))
            for rep in audit.get("reports", []):
                qid = str(rep["query_id"])
                if qid not in qids:
                    qids.append(qid)

    reports: list[dict[str, object]] = []
    for qid in qids:
        if qid not in queries or qid not in qrels:
            print(f"  skip {qid}: not in test split", flush=True)
            continue
        print(f"  bench query {qid}...", flush=True)
        reports.append(
            bench_query(
                knowledge,
                plane,
                qid,
                queries[qid],
                qrels[qid],
                route_max=args.route_max,
            )
        )
        partial = {
            "dataset": args.dataset,
            "brain": brain,
            "no_trinary_training": True,
            "reports": reports,
            "aggregate_all": aggregate(reports),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    result = {
        "dataset": args.dataset,
        "brain": brain,
        "no_trinary_training": True,
        "q54": next((r for r in reports if r["query_id"] == "54"), None),
        "route_misses": [r for r in reports if r["query_id"] != "54"],
        "aggregate_all": aggregate(reports),
        "aggregate_misses": aggregate(
            [r for r in reports if r["query_id"] != "54"],
        ),
        "reports": reports,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}", flush=True)
    agg = result["aggregate_all"]
    print(
        f"  {agg['n_queries']} queries  "
        f"ndcg {agg['mean_ndcg_before']:.4f} -> {agg['mean_ndcg_after']:.4f}  "
        f"recall {agg['mean_recall_before']:.4f} -> {agg['mean_recall_after']:.4f}  "
        f"gold->top10: {agg['gold_moved_into_top10']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
