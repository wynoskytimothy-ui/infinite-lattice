#!/usr/bin/env python3
"""Run token level audits — one level at a time or full stack."""

from __future__ import annotations

import argparse
import sys

from aethos_pipeline import AethosPipeline
from aethos_token_levels import TokenLevel, format_audit_report, run_level_audits
from diagnose_corpus import SMALL_CORPUS


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHOS token level audit")
    parser.add_argument(
        "--level",
        choices=[lv.value for lv in TokenLevel],
        help="Run a single level (default: all)",
    )
    parser.add_argument("--rebuild-every", type=int, default=2)
    args = parser.parse_args()

    pipe = AethosPipeline(rebuild_every=args.rebuild_every)
    pipe.ingest(*SMALL_CORPUS)

    levels = None
    if args.level:
        levels = [TokenLevel(args.level)]

    results = run_level_audits(pipe, SMALL_CORPUS, levels=levels)
    print(format_audit_report(results))

    if args.level is None:
        print("Next tiers (not audited here): CODE/URL species, LM head / Hilbert training.")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
