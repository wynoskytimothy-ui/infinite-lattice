#!/usr/bin/env python3
"""
Unified BEIR benchmark for the clean pipeline.

Full-corpus only. Writes JSON to logs/clean_bench_<dataset>.json.

Usage:
  python -m aethos_clean.bench_beir --dataset scifact
  python -m aethos_clean.bench_beir --dataset scifact --max-queries 20 --rebuild
  python -m aethos_clean.bench_beir --dataset scifact --from-checkpoint brains/scifact_quality.eval.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from aethos_clean.pipeline import CleanPipeline
from beir_data_root import resolve_beir_root
from eval_checkpoint import checkpoint_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AETHOS clean pipeline BEIR benchmark")
    p.add_argument("--dataset", default="scifact", help="BEIR dataset name")
    p.add_argument("--preset", default="lean", help="gates preset (lean, accuracy_max)")
    p.add_argument("--beir-root", default=None, help="override BEIR data root")
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--rebuild", action="store_true", help="ignore saved checkpoint")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--from-checkpoint", default=None, help="skip build, load eval pkl")
    p.add_argument(
        "--retrain-composites",
        action="store_true",
        help="composite-only train on checkpoint (no full rebuild)",
    )
    p.add_argument("--no-gates", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default=None, help="results JSON path")
    return p.parse_args(argv)


def run_benchmark(args: argparse.Namespace) -> int:
    beir_root = args.beir_root or resolve_beir_root()
    print(f"BEIR root: {beir_root}", flush=True)

    if args.from_checkpoint or args.retrain_composites:
        ckpt = args.from_checkpoint or str(checkpoint_path(args.dataset, "quality"))
        pipe = CleanPipeline.from_checkpoint(ckpt, preset=args.preset)
        print(f"loaded checkpoint: {ckpt}", flush=True)
        if args.retrain_composites:
            print("composite-only retrain...", flush=True)
            pipe.retrain_composites(verbose=True)
    else:
        pipe = CleanPipeline.from_beir(
            args.dataset,
            preset=args.preset,
            beir_root=beir_root,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
        print(f"indexing {args.dataset}  preset={args.preset} ...", flush=True)
        storage = pipe.index(
            rebuild=args.rebuild,
            skip_training=args.skip_training,
            verbose=args.verbose,
        )
        print(storage.summary(), flush=True)

    result = pipe.evaluate(check_gates=not args.no_gates, verbose=args.verbose)
    print(result.summary(), flush=True)
    if result.gate_report:
        print(result.gate_report, flush=True)

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent
        / "logs"
        / f"clean_bench_{args.dataset}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "beir_root": str(beir_root),
        "result": result.to_dict(),
        "gate_report": result.gate_report,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved: {out}", flush=True)

    return 0 if (result.gate_passed is not False) else 1


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
