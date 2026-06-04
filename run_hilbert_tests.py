#!/usr/bin/env python3
"""Build and verify lattice-derived Hilbert space."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    steps = [
        ("Hilbert space dedicated tests", "test_hilbert_space.py"),
        ("Hilbert tests in main suite", "test_aethos.py"),
        ("Hilbert-from-lattice demo", "aethos_hilbert_lattice.py"),
        ("Spring complex plane demo", "aethos_complex_spring.py"),
        ("Hilbert tower demo", "aethos_hilbert.py"),
    ]
    for label, script in steps:
        print(f"\n{'=' * 60}\n{label}\n{'=' * 60}\n")
        proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=str(ROOT))
        if proc.returncode != 0:
            return proc.returncode

    print("\n" + "=" * 60)
    print("HILBERT SPACE: ALL TESTS PASSED")
    print("=" * 60)
    print("""
  [x] 32-wing orthonormal basis (label inner product)
  [x] Gram matrix identity on wing sample
  [x] Norm + normalize superposition
  [x] Projection onto basis direction
  [x] Spring complex z=X+iY at triggers
  [x] VA1/VA2 mirror pairs |z|^2 (Born proxy)
  [x] 4 branch fan at trigger
  [x] Meet boost on coordinate collision
  [x] Robust inner product + L4-L6 correlations
  [x] Hilbert tower scaling + core facade
  [x] End-to-end: build -> state -> measure
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
