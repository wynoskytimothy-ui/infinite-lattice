#!/usr/bin/env python3
"""
Run lattice_retriever_v1 brain-loop gate tests.

Usage:
    python scripts/run_brain_gates.py           # smoke subset (default)
    python scripts/run_brain_gates.py --smoke   # same as default
    python scripts/run_brain_gates.py --full    # all lattice_retriever_v1/tests/

Exit 0 only when every selected test module passes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SMOKE_TESTS = [
    "lattice_retriever_v1/tests/test_deny_imports.py",
    "lattice_retriever_v1/tests/test_stage01.py",
    "lattice_retriever_v1/tests/test_stage08.py",
    "lattice_retriever_v1/tests/test_k_meet.py",
    "lattice_retriever_v1/tests/test_brain_loop.py",
    "lattice_retriever_v1/tests/test_corpus_prime.py",
    "lattice_retriever_v1/tests/test_neuron_room.py",
    "lattice_retriever_v1/tests/test_k_meet_index.py",
]


def _run_pytest(paths: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", *paths, "-q"]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AETHOS brain loop gate runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--smoke",
        action="store_true",
        help="run brain-loop smoke gates (default)",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="run all lattice_retriever_v1/tests/",
    )
    args = parser.parse_args(argv)

    if args.full:
        paths = ["lattice_retriever_v1/tests/"]
    else:
        paths = SMOKE_TESTS

    rc = _run_pytest(paths)
    if rc == 0:
        print(f"PASS — {len(paths)} gate module(s)")
    else:
        print(f"FAIL — exit {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
