#!/usr/bin/env python3
"""
Run the full glass-box audit suite on one or more BEIR corpora.

Includes:
  - 32 append-index correlation probes (+ glass-box target/lattice)
  - 20 pollution distinguishability tests (T01–T20)
  - MultiCorpusBrain gold-bridge audit (κ-primary scale_search path)

Auto-discovers datasets under BEIR_DATA_DIR / beir_datasets that have
corpus.jsonl + test qrels. Skips corpora missing on disk; warns when train
or glossary are absent.

Examples:
  python scripts/run_glass_box_suite.py scifact nfcorpus
  python scripts/run_glass_box_suite.py --all-trained
  python scripts/run_glass_box_suite.py nfcorpus --lattice --index-mode kappa_primary
  python scripts/run_glass_box_suite.py scifact --max-queries 50 --skip-brain
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from beir_data_root import resolve_beir_root
from scripts.bench_supervised_bridges import find_ds


DEFAULT_MIN_PAIRS = {"scifact": 1, "nfcorpus": 2, "fiqa": 2}
LOG_DIR = _ROOT / "logs"


def discover_trained_datasets() -> list[str]:
    root = Path(resolve_beir_root())
    if not root.is_dir():
        return []
    found: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "corpus.jsonl").exists():
            continue
        if not (child / "qrels" / "test.tsv").exists():
            continue
        if (child / "qrels" / "train.tsv").exists():
            found.append(child.name)
    return found


def corpus_available(name: str) -> bool:
    try:
        find_ds(name)
        return True
    except SystemExit:
        return False


def run_step(label: str, cmd: list[str]) -> int:
    print(f"\n{'='*72}\n  {label}\n  {' '.join(cmd)}\n", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=str(_ROOT))
    print(f"  -> {label}: exit {rc} ({time.perf_counter()-t0:.1f}s)", flush=True)
    return rc


def run_dataset(
    name: str,
    *,
    max_queries: int,
    index_mode: str,
    lattice: bool,
    skip_append: bool,
    skip_pollution: bool,
    skip_brain: bool,
    min_pairs: int,
) -> dict[str, int]:
    if not corpus_available(name):
        print(f"\nSKIP {name}: not found under {resolve_beir_root()}", flush=True)
        return {"skipped": 1}

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    mp = min_pairs or DEFAULT_MIN_PAIRS.get(name, 2)
    mq = f"--max-queries {max_queries}" if max_queries else ""
    results: dict[str, int] = {}

    if not skip_append:
        cmd = [
            sys.executable, "scripts/audit_append_index_glass_box.py",
            name,
            "--index-mode", index_mode,
            "--min-pairs", str(mp),
            "--out", str(LOG_DIR / f"append_glass_box_{name}.json"),
            "--rules-md", str(LOG_DIR / f"append_glass_box_rules_{name}.md"),
        ]
        if max_queries:
            cmd.extend(["--max-queries", str(max_queries)])
        results["append"] = run_step(f"{name} append probes (32)", cmd)

    if not skip_pollution:
        cmd = [
            sys.executable, "scripts/audit_glass_box_pollution.py",
            name,
            "--index-mode", index_mode,
            "--min-pairs", str(mp),
            "--out", str(LOG_DIR / f"glass_box_pollution_{name}.json"),
            "--rules-out", str(LOG_DIR / f"glass_box_pollution_rules_{name}.md"),
        ]
        if lattice:
            cmd.append("--lattice")
        if max_queries:
            cmd.extend(["--max-queries", str(max_queries)])
        results["pollution"] = run_step(f"{name} pollution audit (20 tests)", cmd)

    if not skip_brain:
        brain_mode = "kappa_primary" if index_mode == "kappa_primary" or lattice else index_mode
        cmd = [
            sys.executable, "scripts/audit_glass_box_gold_bridge.py",
            name,
            "--index-mode", brain_mode,
            "--out", str(LOG_DIR / f"glass_box_gold_audit_{name}.json"),
        ]
        if max_queries:
            cmd.extend(["--max-queries", str(max_queries)])
        results["brain"] = run_step(f"{name} trained brain audit", cmd)

    return results


def main() -> int:
    p = argparse.ArgumentParser(description="Run glass-box audit suite on BEIR corpora")
    p.add_argument(
        "datasets", nargs="*",
        help="corpus names (default: scifact nfcorpus if present)",
    )
    p.add_argument("--all-trained", action="store_true", help="Every corpus with train.tsv")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--index-mode", default="full", choices=("full", "kappa_primary"))
    p.add_argument("--lattice", action="store_true", help="Pollution audit uses lattice κ-pool")
    p.add_argument("--min-pairs", type=int, default=0)
    p.add_argument("--skip-append", action="store_true")
    p.add_argument("--skip-pollution", action="store_true")
    p.add_argument("--skip-brain", action="store_true")
    args = p.parse_args()

    if args.all_trained:
        names = discover_trained_datasets()
        if not names:
            print(f"No trained datasets under {resolve_beir_root()}", flush=True)
            return 1
    elif args.datasets:
        names = list(args.datasets)
    else:
        names = [n for n in ("scifact", "nfcorpus") if corpus_available(n)]

    print(f"Glass-box suite: {', '.join(names)}", flush=True)
    print(f"  BEIR root: {resolve_beir_root()}", flush=True)
    print(f"  index_mode={args.index_mode}  lattice={args.lattice}", flush=True)

    rc = 0
    for name in names:
        res = run_dataset(
            name,
            max_queries=args.max_queries,
            index_mode=args.index_mode,
            lattice=args.lattice,
            skip_append=args.skip_append,
            skip_pollution=args.skip_pollution,
            skip_brain=args.skip_brain,
            min_pairs=args.min_pairs,
        )
        if res.get("skipped"):
            continue
        for step, code in res.items():
            if code != 0:
                rc = code

    print(f"\n{'='*72}\n  Suite complete. Logs in {LOG_DIR}/", flush=True)
    for pattern in (
        "append_glass_box_*.json",
        "append_glass_box_rules_*.md",
        "glass_box_pollution_*.json",
        "glass_box_pollution_rules_*.md",
        "glass_box_gold_audit_*.json",
    ):
        for path in sorted(LOG_DIR.glob(pattern)):
            print(f"    {path.name}", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
