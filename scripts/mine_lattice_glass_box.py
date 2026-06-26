#!/usr/bin/env python3
"""
Mine lattice_retriever_v1 glass-box traces on SciFact — 25+ diagnostic lenses.

Answers: rarest/2nd-rarest/compound/cross-doc-bridge patterns for gold vs non-gold.

  python scripts/mine_lattice_glass_box.py --max-queries 50
  python scripts/mine_lattice_glass_box.py --max-queries 300 --out logs/lattice_glass_mine.json
  python scripts/mine_lattice_glass_box.py --query-id 123  # single query deep dive
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.glass_box_mine import (
    LENS_CATALOG,
    aggregate_lenses,
    headroom_summary,
    mine_query,
)
from scripts.bench_lattice_retriever_v1 import build_retriever, load_scifact


def _print_lens_table(lenses: list) -> None:
    print(f"\n{'ID':<34} {'rate':>7} {'hits':>5}  lens")
    print("-" * 72)
    for lr in sorted(lenses, key=lambda x: -x.rate):
        print(f"{lr.id:<34} {lr.rate:>7.1%} {lr.gold_hits:>5}  {lr.name}")


def _print_query_dive(rec) -> None:
    print(f"\n=== Q{rec.qid} ===")
    print(f"query: {rec.query[:120]}")
    print(f"gold: {rec.gold_ids} | pool={rec.pool_size} mode={rec.route_mode}")
    print(f"ranks: {rec.gold_ranks} | top10: {rec.top10_ids[:5]}...")
    print(f"rarity: {rec.terms_by_rarity[:6]}")
    print(f"routing_pins: {rec.routing_pins}")
    print(f"compounds: {[(a, b, df) for a, b, _, df in rec.compound_pairs[:4]]}")
    if rec.bridge_terms:
        print(f"bridge rare terms (top1↔gold): {rec.bridge_terms}")
    if rec.cage_bridge_terms:
        print(f"cage correlation bridge: {rec.cage_bridge_terms[:6]}")
    if rec.subword_bridge_terms:
        print(f"subword surface bridge: {rec.subword_bridge_terms[:6]}")
    active = [k for k, v in rec.buckets.items() if v]
    print(f"active lenses ({len(active)}): {', '.join(sorted(active)[:12])}...")


def main() -> int:
    p = argparse.ArgumentParser(description="Glass-box mine — lattice v1 SciFact diagnostics")
    p.add_argument("--max-queries", type=int, default=50)
    p.add_argument("--query-id", type=str, default=None, help="Deep dive one qid")
    p.add_argument("--fast", action="store_true", help="fast_ingest index (floor only)")
    p.add_argument("--out", default=None)
    p.add_argument("--top-bridges", type=int, default=10, help="Print top bridge-query samples")
    args = p.parse_args()

    corpus, queries, qrels = load_scifact()
    qids = [q for q in qrels if q in queries]
    if args.query_id:
        qids = [args.query_id] if args.query_id in qrels else [args.query_id]
    elif args.max_queries:
        qids = qids[: args.max_queries]

    print(f"Glass-box mine: {len(qids)} queries | {len(corpus)} docs", flush=True)
    t0 = time.time()
    r = build_retriever(corpus, fast_ingest=args.fast)
    print(f"index {time.time() - t0:.1f}s | L2={len(r.semantic.registry.promotions)}", flush=True)

    records = []
    t1 = time.time()
    for i, qid in enumerate(qids):
        if qid not in queries:
            continue
        gold = {d for d, s in qrels[qid].items() if s > 0}
        if not gold:
            continue
        rec = mine_query(r, qid, queries[qid], gold)
        records.append(rec)
        if args.query_id:
            _print_query_dive(rec)
        if (i + 1) % 25 == 0:
            print(f"  mined {i + 1}/{len(qids)}", flush=True)
    print(f"mined {len(records)} queries in {time.time() - t1:.1f}s", flush=True)

    lenses = aggregate_lenses(records)
    headroom = headroom_summary(records)

    _print_lens_table(lenses)
    print(f"\n--- headroom (fix class -> extra gold recoverable) ---")
    for k, v in headroom.items():
        print(f"  {k}: {v}")

    bridges = [rec for rec in records if rec.buckets.get("L23_top10_bridge_to_gold")]
    bridges.sort(key=lambda x: x.pool_size)
    print(f"\n--- bridge samples (top doc shares rare w/ gold, n={len(bridges)}) ---")
    for rec in bridges[: args.top_bridges]:
        print(f"  Q{rec.qid}: pool={rec.pool_size} bridge={rec.bridge_terms[:4]} | {rec.query[:70]}")

    out = Path(args.out or Path(__file__).resolve().parents[1] / "logs" / "lattice_glass_mine.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": len(records),
        "headroom": headroom,
        "lenses": [lr.explain() for lr in lenses],
        "lens_catalog": [{"id": a, "name": b, "description": c} for a, b, c in LENS_CATALOG],
        "per_query": [
            {
                "qid": rec.qid,
                "query": rec.query,
                "gold_ids": list(rec.gold_ids),
                "pool_size": rec.pool_size,
                "route_mode": rec.route_mode,
                "gold_in_pool": rec.gold_in_pool,
                "gold_ranks": rec.gold_ranks,
                "terms_by_rarity": list(rec.terms_by_rarity),
                "routing_pins": list(rec.routing_pins),
                "compound_pairs": [
                    {"w1": a, "w2": b, "pin": p, "pin_df": df} for a, b, p, df in rec.compound_pairs
                ],
                "bridge_terms": list(rec.bridge_terms),
                "cage_bridge_terms": list(rec.cage_bridge_terms),
                "subword_bridge_terms": list(rec.subword_bridge_terms),
                "top10": list(rec.top10_ids),
                "buckets": rec.buckets,
            }
            for rec in records
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
