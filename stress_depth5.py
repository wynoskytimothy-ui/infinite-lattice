#!/usr/bin/env python3
"""
Depth-5 stress run for AETHOS origin tree + lattice banks.
Run: python stress_depth5.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter

from aethos_lattice import BranchKind, LatticeId
from aethos_origins import OriginTree
from aethos_recursive import LatticeBank32K, canon_recursive, segment_index
from aethos_sequences import SequenceKind, make_chain


def geometric_origin_count(max_depth: int) -> int:
    """Nodes in full 3-ary origin tree through depth max_depth."""
    return sum(3**i for i in range(max_depth + 1))


def main() -> int:
    max_depth = 5
    expected_origins = geometric_origin_count(max_depth)
    expected_wing_rooms = expected_origins * 32

    print("=" * 60)
    print(f"AETHOS DEPTH-{max_depth} STRESS")
    print("=" * 60)
    print(f"  Expected origins:     {expected_origins:,}")
    print(f"  Expected wing-rooms:  {expected_wing_rooms:,}\n")

    # --- Build origin tree ---
    t0 = time.perf_counter()
    tree = OriginTree.bootstrap(max_depth=max_depth)
    t_build = time.perf_counter() - t0

    n_origins = tree.root.count_descendant_origins()
    wing_estimate = tree.lattice_count_estimate()
    origins = list(tree.walk())

    print(f"  Tree build:           {t_build:.3f}s")
    print(f"  Origins counted:      {n_origins:,}")
    print(f"  Wing-room estimate:   {wing_estimate:,}")
    assert n_origins == expected_origins, f"origin count {n_origins} != {expected_origins}"
    assert wing_estimate == expected_wing_rooms

    # --- Every origin has exactly 3 children (except depth=max_depth leaves) ---
    leaves = [o for o in origins if o.depth == max_depth]
    internal = [o for o in origins if o.depth < max_depth]
    assert len(leaves) == 3**max_depth
    assert all(len(o.children) == 3 for o in internal)
    assert all(len(o.children) == 0 for o in leaves)
    print(f"  Internal nodes:       {len(internal):,}")
    print(f"  Leaf nodes (depth 5): {len(leaves):,}")

    # --- Compute all 32 wings at n=10 for every origin ---
    n_transgress = 10
    t1 = time.perf_counter()
    wing_coords: list[tuple[str, str, tuple]] = []
    for o in origins:
        wings = o.wings_at(n_transgress)
        assert len(wings) == 32
        for lid_name, coord in wings.items():
            wing_coords.append((o.id, lid_name, coord))
    t_wings = time.perf_counter() - t1
    unique_global = len({c for _, _, c in wing_coords})
    print(f"\n  All wings @ n={n_transgress}:")
    print(f"    Total coordinates:  {len(wing_coords):,}")
    print(f"    Unique global:      {unique_global:,}")
    print(f"    Time:               {t_wings:.3f}s")
    print(f"    Rate:               {len(wing_coords)/t_wings:,.0f} coords/s")

    # --- k-chain stress (recursive, no tree) ---
    chain = make_chain(SequenceKind.PRIMES, 12)
    bank = LatticeBank32K(chain)
    t2 = time.perf_counter()
    hits = 0
    for n in range(0, 500):
        all_c = bank.at_all(n)
        assert len(all_c) == 32
        hits += 1
    t_bank = time.perf_counter() - t2
    seg_at_500 = segment_index(chain, 500)
    print(f"\n  k=12 prime chain {chain[:4]}...{chain[-1]}:")
    print(f"    32 wings x 500 n:   {t_bank:.3f}s")
    print(f"    segment(500):       {seg_at_500}")
    print(f"    VA1@500:            {canon_recursive(BranchKind.VA1, chain, 500)}")

    # --- Depth sample: chain length grows with origin depth ---
    depths = Counter(len(o.anchor_chain) for o in origins)
    print(f"\n  Anchor chain lengths by origin count:")
    for k in sorted(depths):
        print(f"    len={k}: {depths[k]} origins")

    print("\n" + "=" * 60)
    print("DEPTH-5 STRESS: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
