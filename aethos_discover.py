#!/usr/bin/env python3
"""Probe latent lattice formula capabilities (no token layer)."""

from __future__ import annotations

from aethos_lattice import BranchKind, LatticeBank32, LatticeId
from aethos_recursive import (
    LatticeBank32K,
    canon_recursive,
    extension_witness,
    find_cross_meets,
    segment_index,
    try_compose_triple,
    verify_matches_spec_k2,
)
from aethos_sequences import IntersectionType, SequenceKind, canon_on_chain, cross_type_meet, make_chain


def run_audit() -> None:
    print("=" * 70)
    print("AETHOS LATTICE FORMULA AUDIT (core only)")
    print("=" * 70)

    print("\n[1] Four formulas, PDF k=2:", verify_matches_spec_k2())

    chain4 = (3, 5, 7, 11)
    plateau = [
        n
        for n in range(1, max(chain4) + 2)
        if canon_recursive(BranchKind.VA1, chain4, n)[2] == sum(chain4)
    ]
    print(f"[2] Z interior plateau P={chain4} at n={plateau}")

    b3 = LatticeBank32.single_prime(3)
    b11 = LatticeBank32.single_prime(11)
    swap = sum(1 for lid in LatticeId if b3[lid].at(11) == b11[lid].at(3))
    print(f"[3] Solo swap meet 3@11 = 11@3 on wings: {swap}/32")

    b3541 = LatticeBank32.prime_pair(3, 541)
    b35 = LatticeBank32.prime_pair(3, 5)
    print(
        "[4] Pair endpoint swap (3,541)@5 = (3,5)@541:",
        b3541[LatticeId.L01].at(5) == b35[LatticeId.L01].at(541),
    )

    solo_ext = sum(
        1 for lid in LatticeId if b3[lid].at(5) == LatticeBank32K((3, 5))[lid].at(3)
    )
    print(f"[5] Solo 3@5 = pair (3,5)@3 (NOT expected): {solo_ext}/32")

    r = try_compose_triple(3, 5, 7)
    print(f"[6] Triple compose confirmations: {len(r['triple_confirmations'])}")

    coll = LatticeBank32.single_prime(5).find_same_n_collisions(7)
    print(f"[7] Same-n wing collision groups (p=5, n=7): {len(coll)}")

    left = IntersectionType.build("primes", SequenceKind.PRIMES, 4)
    right = IntersectionType.build("evens", SequenceKind.EVENS, 4)
    xspec = sum(
        1 for nl in range(1, 50) for nr in range(1, 50) if cross_type_meet(left, right, nl, nr)
    )
    print(f"[8] Cross-species meets (primes vs evens, n<50): {xspec}")

    print("[9] Species-local meets:")
    print("    evens 2@4 = 4@2:", canon_on_chain(BranchKind.VA1, (2,), 4) == canon_on_chain(BranchKind.VA1, (4,), 2))
    print(
        "    (2,4)@8 = (2,8)@4:",
        canon_on_chain(BranchKind.VA1, (2, 4), 8) == canon_on_chain(BranchKind.VA1, (2, 8), 4),
    )

    w = extension_witness(chain4, LatticeId.L01, 600)
    print(f"[10] Extension witnesses (3,5,7,11): {len(w)} (swap_like={sum(1 for x in w if x['swap_like'])})")

    meets = find_cross_meets(b3, b11, LatticeId.L01, 300)
    print(f"[11] Cross-meets 3 vs 11 on L01 (n<300): {len(meets)}", meets[:3])

    print("\n[12] Branch phases at (3,5,7), n=5:")
    for b in BranchKind:
        print(f"    {b.name}: {canon_recursive(b, (3, 5, 7), 5)}")

    print("\n[13] k -> velocity boundaries (L01):")
    for k in (2, 4, 6):
        ch = make_chain(SequenceKind.PRIMES, k)
        b = LatticeBank32K(ch)[LatticeId.L01].velocity_boundaries()
        print(f"    k={k}: {b} == {list(ch)} -> {b == list(ch)}")

    print("\n" + "=" * 70)
    print("See derivations/lattice_formula_capabilities.md for interpretation.")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
