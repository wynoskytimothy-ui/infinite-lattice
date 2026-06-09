#!/usr/bin/env python3
"""Sweep max_keys / neighbors — accuracy vs latency curve (kappa mode, zero-shot)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval_beir_symbol import evaluate_symbol_beir, load_brain_and_plane
from eval_beir import load_paths, load_qrels, load_queries
from beir_data_root import resolve_beir_root


def sweep(
    knowledge,
    plane,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    qids: list[str],
    *,
    max_keys_values: list[int],
    max_corr_values: list[int] | None = None,
    route_max: int = 600,
    rank_limit: int = 100,
    expand_correlations: bool = True,
) -> list[dict[str, object]]:
    max_corr_values = max_corr_values or [8]
    rows: list[dict[str, object]] = []

    for max_keys in max_keys_values:
        for max_corr in max_corr_values:
            t0 = time.perf_counter()
            result = evaluate_symbol_beir(
                knowledge,
                plane,
                {q: queries[q] for q in qids},
                {q: qrels[q] for q in qids},
                max_queries=None,
                rank_limit=rank_limit,
                route_max=route_max,
                max_keys=max_keys,
                max_corr_neighbors=max_corr,
                expand_correlations=expand_correlations,
                save_failures=0,
                mode="kappa",
            )
            wall_ms = (time.perf_counter() - t0) * 1000.0

            rows.append({
                "max_keys": max_keys,
                "max_corr_neighbors": max_corr,
                "expand_correlations": expand_correlations,
                "route_max": route_max,
                "ndcg_at_10": round(result.ndcg_at_10, 6),
                "recall_at_10": round(result.recall_at_10, 6),
                "mrr_at_10": round(result.mrr_at_10, 6),
                "route_recall": round(result.route_recall, 6),
                "mean_query_ms": round(result.mean_query_ms, 3),
                "p95_query_ms": round(result.p95_query_ms, 3),
                "eval_wall_ms": round(wall_ms, 1),
            })
            print(
                f"keys={max_keys:4d}  corr={max_corr}  "
                f"ndcg={result.ndcg_at_10:.4f}  recall={result.recall_at_10:.4f}  "
                f"route={result.route_recall:.4f}  "
                f"ms={result.mean_query_ms:.2f}",
                flush=True,
            )

    return rows


def pick_best(rows: list[dict[str, object]]) -> dict[str, object]:
    """Best accuracy, then fastest among within 1% route recall of peak."""
    if not rows:
        return {}
    peak_route = max(r["route_recall"] for r in rows)
    peak_ndcg = max(r["ndcg_at_10"] for r in rows)
    threshold = peak_route - 0.01
    ndcg_threshold = peak_ndcg - 0.005
    eligible = [
        r for r in rows
        if r["route_recall"] >= threshold and r["ndcg_at_10"] >= ndcg_threshold
    ]
    if not eligible:
        eligible = rows
    fastest = min(eligible, key=lambda r: (r["mean_query_ms"], -r["ndcg_at_10"]))
    best_acc = max(rows, key=lambda r: (r["ndcg_at_10"], r["recall_at_10"], r["route_recall"]))
    return {
        "fastest_good": fastest,
        "best_accuracy": best_acc,
        "peak_route_recall": peak_route,
        "peak_ndcg_at_10": peak_ndcg,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="kappa max_keys accuracy/latency sweep")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument(
        "--max-keys",
        default="32,64,128,192,256,384,512,768,1024",
        help="comma-separated max_keys values",
    )
    p.add_argument("--max-corr", default="4,8,12", help="comma-separated neighbor caps")
    p.add_argument("--out", default="logs/kappa_keys_curve.json")
    args = p.parse_args()

    max_keys_values = [int(x.strip()) for x in args.max_keys.split(",") if x.strip()]
    max_corr_values = [int(x.strip()) for x in args.max_corr.split(",") if x.strip()]

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    qids = [q for q in qrels if q in queries][: args.max_queries]

    print(f"Loading brain ({args.dataset}) ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)
    print(f"Sweeping {len(qids)} queries, {len(max_keys_values)} key caps ...", flush=True)

    rows = sweep(
        knowledge,
        plane,
        queries,
        qrels,
        qids,
        max_keys_values=max_keys_values,
        max_corr_values=max_corr_values,
    )
    summary = pick_best(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "n_queries": len(qids),
        "sweep": rows,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fg = summary.get("fastest_good", {})
    ba = summary.get("best_accuracy", {})
    print("\n--- summary ---")
    print(
        f"fastest (within 1% route / 0.5% ndcg of peak): "
        f"max_keys={fg.get('max_keys')} corr={fg.get('max_corr_neighbors')} "
        f"ndcg={fg.get('ndcg_at_10')} route={fg.get('route_recall')} "
        f"ms={fg.get('mean_query_ms')}",
    )
    print(
        f"best accuracy: max_keys={ba.get('max_keys')} corr={ba.get('max_corr_neighbors')} "
        f"ndcg={ba.get('ndcg_at_10')} recall={ba.get('recall_at_10')} "
        f"route={ba.get('route_recall')} ms={ba.get('mean_query_ms')}",
    )
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
