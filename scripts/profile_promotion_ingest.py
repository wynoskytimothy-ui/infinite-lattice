#!/usr/bin/env python3
"""
Profile promotion-on ingest — per-doc timing to localize eager O(corpus) work.

Usage:
  python scripts/profile_promotion_ingest.py              # 50-doc fixture slice
  python scripts/profile_promotion_ingest.py --scifact 50 # first N SciFact docs
  python scripts/profile_promotion_ingest.py --fast       # compare fast_ingest baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lattice_retriever_v1.promotion_ingest_profile import (
    analyze_flatness,
    fixture_corpus_slice,
    profile_ingest,
)


def load_scifact_slice(n: int) -> dict[str, str]:
    from scripts.bench_lattice_retriever_v1 import load_scifact

    corpus, _, _ = load_scifact()
    items = list(corpus.items())[:n]
    return dict(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-doc promotion ingest profiler")
    parser.add_argument("--n", type=int, default=50, help="Doc count for fixture slice")
    parser.add_argument("--scifact", type=int, metavar="N", help="Use first N SciFact docs instead of fixtures")
    parser.add_argument("--fast", action="store_true", help="Run with fast_ingest=True (engine half-off)")
    parser.add_argument("--max-ratio", type=float, default=4.0, help="Flatness threshold (last/first quartile median)")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args()

    if args.scifact:
        corpus = load_scifact_slice(args.scifact)
        label = f"scifact_{args.scifact}"
    else:
        corpus = fixture_corpus_slice(args.n)
        label = f"fixture_{args.n}"

    mode = "fast_ingest" if args.fast else "promotion_on"
    print(f"profile_ingest: {label} | {len(corpus)} docs | {mode}", flush=True)

    profile = profile_ingest(corpus, fast_ingest=args.fast)
    report = profile.explain()
    flat = analyze_flatness(profile.totals_ms, max_ratio=args.max_ratio)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"\n{'doc':>8} {'total':>8} {'promo':>8} {'flush':>8} {'cand':>6} {'words':>6}")
    for t in profile.timings:
        print(
            f"{t.doc_id:>8} {t.total_ms:8.2f} {t.promotion_ms:8.2f} "
            f"{t.flush_ms:8.2f} {t.flush_candidates:6d} {t.n_words:6d}"
        )

    pm = report["phase_medians_ms"]
    print(f"\n--- flatness ({label}) ---")
    print(f"  median total/doc     {pm['total']:.2f} ms")
    print(f"  median promotion     {pm['promotion_observe_text']:.2f} ms")
    print(f"  median l2_flush      {pm['l2_flush']:.2f} ms")
    print(f"  median semantic      {pm['semantic_observe']:.2f} ms")
    print(f"  first-quartile med   {flat['median_first_quartile_ms']:.2f} ms")
    print(f"  last-quartile med    {flat['median_last_quartile_ms']:.2f} ms")
    print(f"  quartile ratio       {flat['quartile_ratio']:.2f}  (flat if <= {args.max_ratio})")
    print(f"  L2 promotions final  {report['final_l2_promotions']}")

    if flat["flat"]:
        print("\n  VERDICT: per-doc cost looks flat (lazy ingest OK on this slice)")
    else:
        print("\n  VERDICT: per-doc cost RISING — likely eager corpus-scaling work in promotion path")
        print("  Check flush_ms growth and flush_candidates; profile _should_promote_l2 parent loops.")


if __name__ == "__main__":
    main()
