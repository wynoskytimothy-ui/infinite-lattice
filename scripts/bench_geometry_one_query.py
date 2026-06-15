#!/usr/bin/env python3
"""Benchmark geometry-native fast train on one SciFact query."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_geometry_ingest import bench_one_query, save_benchmark


def main() -> int:
    p = argparse.ArgumentParser(description="Geometry speed benchmark — one query")
    p.add_argument("--query-id", default="54", help="SciFact test query id (default: 54)")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--brain", default=None, help="brain name (default: --dataset)")
    p.add_argument("--pretrain-q1", action="store_true", help="compound-learn Q1 gold bundle")
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args()

    result = bench_one_query(
        args.query_id,
        dataset=args.dataset,
        brain_name=args.brain,
        pretrain_q1=args.pretrain_q1,
    )
    out = save_benchmark(result, Path(args.out) if args.out else None)

    print(f"Q{result['query_id']}: {result['query']!r}")
    print(f"  load_ms         : {result['load_ms']:.1f}")
    print(f"  route_before_ms : {result['route_before_ms']:.1f}")
    print(f"  train_ms        : {result['train_ms']:.1f}")
    print(f"  route_after_ms  : {result['route_after_ms']:.1f}")
    print(f"  rank_after_ms   : {result['rank_after_ms']:.1f}")
    print(f"  gold_in_route   : {result['gold_in_route']}  (before: {result['gold_in_route_before']})")
    print(f"  nDCG@10         : {result['ndcg_at_10']:.4f}")
    print(f"  top5            : {result['top5']}")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
