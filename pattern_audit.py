#!/usr/bin/env python3
"""
Pattern placement audit — all queries, gold vs false, all signals at once.

Places every scoring signal and pipeline bit on gold documents vs false top-1
documents across the full eval set, then prints an aggregate tuner report.

Usage:
  python pattern_audit.py
  python pattern_audit.py --checkpoint brains/scifact_quality.eval.pkl
  python pattern_audit.py --max-queries 100 --json reports/pattern.json
  python pattern_audit.py --failures-only   # detail lines for non-PERFECT queries
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from eval_checkpoint import checkpoint_path, load_checkpoint
from pipeline.pattern_placement import (
    SIGNAL_NAMES,
    audit_query_patterns,
    format_pattern_report,
)


def _record_to_dict(rec) -> dict:
    out = {
        "qid": rec.qid,
        "query": rec.query_text,
        "pattern": rec.pattern,
        "ndcg10": rec.ndcg10,
        "recall10": rec.recall10,
        "route_tier": rec.route_tier,
        "n_candidates": rec.n_candidates,
        "n_kappa_keys": rec.n_kappa_keys,
        "z_obs_q": rec.z_obs_q,
        "gold_ids": rec.gold_ids,
        "gold_in_candidates": rec.gold_in_candidates,
        "gold_best_rank": rec.gold_best_rank,
        "top1_id": rec.top1_id,
        "top1_is_gold": rec.top1_is_gold,
        "score_gap": rec.score_gap,
        "signal_delta": rec.signal_delta,
        "fix_hint": rec.fix_hint,
    }
    if rec.gold:
        out["gold_signals"] = rec.gold.signal_dict()
        out["gold_total"] = rec.gold.total
        out["gold_kappa_jaccard"] = rec.gold.kappa_jaccard
    if rec.false_top1:
        out["false_signals"] = rec.false_top1.signal_dict()
        out["false_total"] = rec.false_top1.total
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHOS pattern placement audit")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--mode", default="quality", choices=("quality", "scale"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-kappa", action="store_true", help="Disable signal 8a scoring")
    parser.add_argument("--json", type=Path, default=None, metavar="PATH")
    parser.add_argument(
        "--failures-only",
        action="store_true",
        help="Print per-query detail for non-PERFECT patterns",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint or checkpoint_path(args.dataset, args.mode)
    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}")
        print("Run:  python run_ab.py --stage build --skip-training")
        return 1

    bundle = load_checkpoint(ckpt)
    qids = bundle.qids
    if args.max_queries is not None:
        qids = qids[: args.max_queries]

    print(f"Pattern audit: {ckpt.name}  queries={len(qids)}  docs={bundle.n_docs}", flush=True)
    t0 = time.perf_counter()
    report = audit_query_patterns(
        bundle,
        qids=qids,
        enable_kappa_scoring=not args.no_kappa,
        progress_every=args.progress_every,
    )
    elapsed = time.perf_counter() - t0
    print(format_pattern_report(report))
    print(f"  Wall time: {elapsed:.1f}s  ({elapsed / max(len(qids), 1) * 1000:.0f} ms/query)", flush=True)

    if args.failures_only:
        print("\nPer-query detail (non-PERFECT):")
        for rec in report.records:
            if rec.pattern == "PERFECT":
                continue
            print(f"\n  q={rec.qid}  {rec.pattern}  NDCG={rec.ndcg10:.3f}  tier={rec.route_tier}  |C|={rec.n_candidates}")
            if rec.gold:
                gs = " ".join(f"{k}={v:.2f}" for k, v in rec.gold.signal_dict().items() if v > 0)
                print(f"    gold {rec.gold.doc_id} rank={rec.gold.rank}  {gs or '(no signals)'}")
            if rec.false_top1:
                fs = " ".join(f"{k}={v:.2f}" for k, v in rec.false_top1.signal_dict().items() if v > 0)
                print(f"    false {rec.false_top1.doc_id}  {fs or '(no signals)'}")
            if rec.signal_delta:
                ds = " ".join(f"{k}={v:+.2f}" for k, v in rec.signal_delta.items() if abs(v) > 0.01)
                if ds:
                    print(f"    delta gold-false: {ds}")
            if rec.fix_hint:
                print(f"    hint: {rec.fix_hint}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["queries"] = [_record_to_dict(r) for r in report.records]
        payload["signal_names"] = list(SIGNAL_NAMES)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  Wrote {args.json}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
