#!/usr/bin/env python3
"""Run AETHOS tests, stress, demos, and pipeline smoke."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(label: str, script: str) -> int:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n")
    proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
    return proc.returncode


def main() -> int:
    steps = [
        ("Test suite (unittest)", "test_aethos.py"),
        ("Lattice core demo (no tokens)", "aethos_core.py"),
        ("Golden coordinate verify", "aethos_golden_coords.py"),
        ("Depth-5 stress", "stress_depth5.py"),
        ("Intersection codec demo", "aethos_codec.py"),
        ("Intersection policy demo", "demo_intersection_policy.py"),
        ("Token savings demo", "aethos_token_savings.py"),
        ("Unified pipeline demo", "aethos_pipeline.py"),
        ("Hilbert space tests", "test_hilbert_space.py"),
        ("Token level audits", "run_token_levels.py"),
        ("Token level tests", "test_token_levels.py"),
    ]
    for label, script in steps:
        code = _run(label, script)
        if code != 0:
            return code

    print("\n" + "=" * 60)
    print("SYSTEM CAPABILITIES (see ARCHITECTURE.md)")
    print("=" * 60)
    print("""
  [x] 32 independent lattices (4 branches x 8 vectors)
  [x] Transgressor n with velocity change at each anchor
  [x] Recursive k-prime / k-chain formulas (PDF k=2 exact)
  [x] Odd-prime anchors only (2 skipped in PRIMES chain)
  [x] Swap meet on all 32 wings (e.g. 3@11 = 11@3)
  [x] Triple promotion (3,5)+(3,7) -> (3,5,7) all 4 branches
  [x] Countable anchor species (primes, evens, 2^n, sqrt-scaled, ...)
  [x] Origin tree: 3 dimensions per origin, 32 wings per room
  [x] Depth-5 stress: 364 origins, 11,648 wing-rooms computed
  [x] 100 active nodes -> unbounded deterministic positions
  [x] Intersection codec: encode/decode data via formula dots
  [x] L1-L9 promotion + natural clusters + intersection policy
  [x] Lattice core separate from token processor (aethos_core)
  [x] Token processor L1-L9 on top of formula_coord
  [x] AethosPipeline: core + tokens + codec/word dots
  [x] Hilbert space from lattice (inner product, spring, correlations)
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
