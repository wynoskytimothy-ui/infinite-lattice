#!/usr/bin/env python3
"""Concrete demo: lattice formula -> 3D complex plane Psi = (z, zeta)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_lattice import BranchKind
from aethos_sequences import SequenceKind, make_chain, segment_index_chain, sum_chain, canon_on_chain
from aethos_complex_plane import wing_transform


def main() -> None:
    kind = SequenceKind.PRIMES
    k = 5
    n = 10
    A = make_chain(kind, k)
    s = segment_index_chain(A, n)
    print("=== 3D COMPLEX PLANE — FORMULA PIPELINE ===\n")
    print(f"1. Chain:  A = {A}  (kind={kind.value}, k={k})")
    print(f"2. Segment: n={n}  =>  s={s}  (0..{k})")
    print(f"3. Depth:   sum(A)={sum_chain(A)}  =>  zeta={canon_on_chain(BranchKind.VA1, A, n)[2]}")
    print("4–6. Branch x wing -> Psi:\n")
    print(f"   {'Branch':<6} {'(X,Y,Z)':<22} {'z':<12} {'zeta':<6} |z|^2")
    print("   " + "-" * 58)
    for b in BranchKind:
        xyz = canon_on_chain(b, A, n)
        psi = wing_transform(b, A, n, wing=1)
        print(
            f"   {b.name:<6} {str(xyz):<22} {str(psi.z):<12} {psi.zeta:<6.0f} {psi.modulus_squared:.0f}"
        )
    print("\nNative address: alpha = (A, b, w, n)")
    print("Object: trajectory space + readout (z, zeta) — see ONTOLOGY.md")


if __name__ == "__main__":
    main()
