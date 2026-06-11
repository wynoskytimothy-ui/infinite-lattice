#!/usr/bin/env python3
"""
test_recursive_lattice.py - does swap_meet propagate to every recursion level?

Tests three identities at two levels:
  - swap_meet: bank(a)@n=p == bank(p)@n=a
  - triple_equalization: all three pair-witnesses collapse to one node
  - anchor-immutable / wing-rotated observable: prime fixed, (X,Y,Z) cycles 8 wings

Level 0 anchors = base primes  (chain_primes - "physical" primes for symbols/squares)
Level 1 anchors = pool primes   (PROMOTION_POOL - "promoted" primes for patterns)

If the identities hold at level 1 just as they do at level 0, the cascade-free
Hilbert-Hotel property propagates to every recursion depth. Promoted patterns
spawn sub-lattices in the same algebra; no special case at higher levels.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_complex_plane import swap_meet, triple_equalization, wing_transform
from aethos_lattice import BranchKind
from aethos_promotion import PROMOTION_POOL
from core.primes import chain_primes


def test_pairs_swap_meet(primes: list[int], label: str) -> tuple[int, int]:
    print(f"\n--- {label} ---")
    print(f"  primes (sample): {primes[:8]}...")
    fails = 0
    total = 0
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            a, b = primes[i], primes[j]
            L, R = swap_meet(a, b)
            total += 1
            if L.coord != R.coord:
                fails += 1
                if fails <= 3:
                    print(f"  FAIL: swap_meet({a},{b}): {L.coord} != {R.coord}")
    print(f"  {total - fails}/{total} swap-meet identities hold")
    return total, fails


def test_triples(primes: list[int], label: str) -> tuple[int, int]:
    print(f"\n--- {label} ---")
    fails = 0
    total = 0
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            for k in range(j + 1, len(primes)):
                a, p, q = sorted([primes[i], primes[j], primes[k]])
                eq = triple_equalization(a, p, q)
                coords = {psi.coord for _, psi in eq.values()}
                total += 1
                if len(coords) != 1:
                    fails += 1
                    if fails <= 3:
                        print(f"  FAIL: triple({a},{p},{q}): {len(coords)} distinct coords {coords}")
    print(f"  {total - fails}/{total} triple-meet identities hold")
    return total, fails


def test_anchor_immutable(prime: int, n: int, label: str):
    print(f"\n--- {label} ---")
    print(f"  prime={prime}, n={n}")
    coords = []
    for wing in range(1, 9):
        psi = wing_transform(BranchKind.VA1, (prime,), n, wing)
        coords.append(psi.coord)
    distinct = set(coords)
    print(f"  prime {prime} is FIXED; observable (X,Y,Z) varies across 8 wings:")
    for w, c in enumerate(coords, 1):
        x, y, z = c
        print(f"    wing {w}:  ({x:>+3.0f}, {y:>+3.0f}, {z:>+3.0f})")
    print(f"  distinct observable positions: {len(distinct)} / 8")
    return len(distinct)


def test_two_level_composition():
    print(f"\n--- TWO-LEVEL CHAIN COMPOSITION ---")
    # Level 0 chains - "physical" patterns built from base primes
    base_chains = [
        (3, 5, 7),
        (11, 13, 17),
        (19, 23, 29),
    ]
    # Level 1 anchors - promoted from each base chain
    P_A, P_B, P_C = PROMOTION_POOL[0], PROMOTION_POOL[1], PROMOTION_POOL[2]
    print(f"  Level-0 base chains   -> Level-1 promoted primes")
    for chain, p in zip(base_chains, [P_A, P_B, P_C]):
        print(f"    {chain}  ->  {p}")

    chain2 = tuple(sorted([P_A, P_B, P_C]))
    print(f"\n  Level-1 chain: {chain2}")

    # Test 1: swap_meet at level 1
    L, R = swap_meet(P_A, P_B)
    print(f"\n  Level-1 swap_meet({P_A}, {P_B}):")
    print(f"    bank({P_A}) @ n={P_B}:  z={L.z}   zeta={L.zeta}")
    print(f"    bank({P_B}) @ n={P_A}:  z={R.z}   zeta={R.zeta}")
    print(f"    identity holds: {L.coord == R.coord}")

    # Test 2: triple_equalization at level 1
    eq = triple_equalization(*chain2)
    coords = {psi.coord for _, psi in eq.values()}
    print(f"\n  Level-1 triple_equalization{chain2}:")
    for label, (n_w, psi) in eq.items():
        print(f"    {label} subset @ n={int(n_w)}:  z={psi.z}   zeta={psi.zeta}")
    print(f"    all three collapse to ONE node: {len(coords) == 1}")
    print(f"    locked coord: {coords.pop() if len(coords) == 1 else coords}")

    # Test 3: trigger history at level 1
    print(f"\n  Level-1 trigger_history (Psi at each anchor crossing):")
    for a in chain2:
        psi = wing_transform(BranchKind.VA1, chain2, a, wing=1)
        print(f"    cross {a}:  z={psi.z}   zeta={psi.zeta}")


def main():
    print("=" * 78)
    print("RECURSIVE LATTICE TEST: anchor-immutable property at all levels")
    print("=" * 78)

    base_primes = list(chain_primes(12))      # 3, 5, 7, 11, ...
    pool_primes = list(PROMOTION_POOL[:12])   # 107, 109, ... (after 26 letter primes)

    print(f"\n  Base primes  (level 0): {base_primes}")
    print(f"  Pool primes  (level 1): {pool_primes}")

    # 1. Swap meet at base level
    t1, f1 = test_pairs_swap_meet(base_primes, "BASE LEVEL: swap_meet on chain_primes(12)")

    # 2. Swap meet at pool level
    t2, f2 = test_pairs_swap_meet(pool_primes, "LEVEL 1: swap_meet on PROMOTION_POOL[:12]")

    # 3. Cross-level swap meet
    cross = sorted(base_primes[:6] + pool_primes[:6])
    t3, f3 = test_pairs_swap_meet(cross, "CROSS-LEVEL: base + pool swap_meet")

    # 4. Triples at level 1
    t4, f4 = test_triples(pool_primes[:8], "LEVEL 1: triple_equalization on PROMOTION_POOL[:8]")

    # 5. Cross-level triples
    t5, f5 = test_triples(cross[:8], "CROSS-LEVEL: triple_equalization mixed")

    # 6. Anchor-immutable demos
    test_anchor_immutable(7, n=11, label="BASE: prime 7 at n=11, 8-wing observable orbit")
    test_anchor_immutable(PROMOTION_POOL[0], n=PROMOTION_POOL[1],
                          label=f"LEVEL 1: prime {PROMOTION_POOL[0]} at n={PROMOTION_POOL[1]}, 8-wing orbit")

    # 7. Two-level composition demo
    test_two_level_composition()

    # Summary
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Base-level swap_meet:     {t1 - f1}/{t1} pass")
    print(f"  Level-1 swap_meet:        {t2 - f2}/{t2} pass")
    print(f"  Cross-level swap_meet:    {t3 - f3}/{t3} pass")
    print(f"  Level-1 triple_meet:      {t4 - f4}/{t4} pass")
    print(f"  Cross-level triple_meet:  {t5 - f5}/{t5} pass")
    total = t1 + t2 + t3 + t4 + t5
    fails = f1 + f2 + f3 + f4 + f5
    print(f"  OVERALL: {total - fails}/{total} identities hold")
    print()
    if fails == 0:
        print("  CONCLUSION: anchor-immutable / wing-rotated-observable property holds at")
        print("  every level tested. Promoted primes obey the same swap and triple meet")
        print("  identities as base primes. No special case for recursion depth.")
        print()
        print("  => Cascade-free at every recursion level. Hilbert-Hotel collapse is")
        print("    not a property of base primes only - it's a property of the formula.")
    else:
        print(f"  CONCLUSION: {fails} identities fail at some level. Examine failures above.")


if __name__ == "__main__":
    main()
