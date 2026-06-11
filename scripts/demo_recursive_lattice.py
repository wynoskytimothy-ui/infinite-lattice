#!/usr/bin/env python3
"""
demo_recursive_lattice.py - build a 3-level lattice; verify identities at each.

Demonstrates the cascade-free recursive hierarchy in action:
  Level 0: base primes (alphabet-style)
  Level 1: promoted "word" primes (each labeling a chain of 3 letters)
  Level 2: promoted "phrase" primes (each labeling a chain of words)

Verifies:
  - swap_meet identity holds at level 2 (phrase pairs)
  - triple_equalization holds at level 1 (word triples)
  - walk_down recovers original base chain at any depth
  - walk_up finds containing higher-level primes
  - chambers returns 32 distinct (X,Y,Z) addresses at every level
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_recursive_lattice import RecursiveLattice
from core.primes import chain_primes


def main():
    print("=" * 78)
    print("RECURSIVE LATTICE DEMO - 3-level hierarchy")
    print("=" * 78)

    lat = RecursiveLattice()

    # =================================================================
    # Level 0: base primes as 'letters'
    # =================================================================
    base = chain_primes(12)
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"]
    for p, l in zip(base, letters):
        lat.register_base(p, label=l)

    print(f"\nLevel 0 base primes (letter -> prime):")
    for l, p in zip(letters, base):
        print(f"  '{l}' = {p}", end="   " if (letters.index(l) + 1) % 4 else "\n")
    print()

    # =================================================================
    # Level 1: promote letter triples to 'word' primes
    # =================================================================
    print("\n--- Level 1: promote letter triples to word primes ---")
    word_abc = lat.promote((base[0], base[1], base[2]), label="abc")
    word_def = lat.promote((base[3], base[4], base[5]), label="def")
    word_ghi = lat.promote((base[6], base[7], base[8]), label="ghi")
    word_jkl = lat.promote((base[9], base[10], base[11]), label="jkl")

    print(f"  'abc' chain={(base[0], base[1], base[2])} -> prime {word_abc}")
    print(f"  'def' chain={(base[3], base[4], base[5])} -> prime {word_def}")
    print(f"  'ghi' chain={(base[6], base[7], base[8])} -> prime {word_ghi}")
    print(f"  'jkl' chain={(base[9], base[10], base[11])} -> prime {word_jkl}")

    # =================================================================
    # Level 2: promote word triples to phrase primes
    # =================================================================
    print("\n--- Level 2: promote word chains to phrase primes ---")
    phrase_ABC = lat.promote((word_abc, word_def, word_ghi), label="abc.def.ghi")
    phrase_ALL = lat.promote((word_abc, word_def, word_ghi, word_jkl), label="abc.def.ghi.jkl")
    print(f"  'abc.def.ghi'     chain=({word_abc}, {word_def}, {word_ghi})     -> prime {phrase_ABC}")
    print(f"  'abc.def.ghi.jkl' chain=({word_abc}, {word_def}, {word_ghi}, {word_jkl}) -> prime {phrase_ALL}")

    # =================================================================
    # Identity tests at higher levels
    # =================================================================
    print("\n" + "=" * 78)
    print("IDENTITY TESTS AT HIGHER LEVELS")
    print("=" * 78)

    print(f"\n--- swap_meet at LEVEL 2 ({phrase_ABC}, {phrase_ALL}) ---")
    L, R = lat.swap_meet(phrase_ABC, phrase_ALL)
    print(f"  bank({phrase_ABC})@n={phrase_ALL}:  z={L.z}  zeta={L.zeta}")
    print(f"  bank({phrase_ALL})@n={phrase_ABC}:  z={R.z}  zeta={R.zeta}")
    print(f"  identity holds: {L.coord == R.coord}")

    print(f"\n--- triple_equalization at LEVEL 1 ({word_abc}, {word_def}, {word_ghi}) ---")
    eq = lat.triple_meet(word_abc, word_def, word_ghi)
    l1_triple_coords = {psi.coord for _, psi in eq.values()}
    for label, (n_w, psi) in eq.items():
        print(f"  {label} subset @ n={int(n_w)}:  z={psi.z}  zeta={psi.zeta}")
    print(f"  all collapse: {len(l1_triple_coords) == 1}   locked: {next(iter(l1_triple_coords))}")

    print(f"\n--- triple_equalization at LEVEL 2 ({word_abc}, {word_def}, {phrase_ABC}) ---")
    eq2 = lat.triple_meet(word_abc, word_def, phrase_ABC)
    l2_triple_coords = {psi.coord for _, psi in eq2.values()}
    for label, (n_w, psi) in eq2.items():
        print(f"  {label} subset @ n={int(n_w)}:  z={psi.z}  zeta={psi.zeta}")
    print(f"  all collapse: {len(l2_triple_coords) == 1}   locked: {next(iter(l2_triple_coords))}")

    # =================================================================
    # Walks
    # =================================================================
    print("\n" + "=" * 78)
    print("WALKS THROUGH THE HIERARCHY")
    print("=" * 78)

    print(f"\n--- walk_down({phrase_ALL}) [phrase 'abc.def.ghi.jkl'] ---")
    base_chain = lat.walk_down(phrase_ALL)
    base_labels = [lat.resolve(p).label for p in base_chain]
    print(f"  base primes: {base_chain}")
    print(f"  as letters:  {base_labels}")

    print(f"\n--- walk_up({base[0]}) [letter 'a'] ---")
    print(f"  letter '{lat.resolve(base[0]).label}' (prime {base[0]}) is contained in:")
    for p in lat.walk_up(base[0]):
        node = lat.resolve(p)
        print(f"    {p} (L{node.level}) '{node.label}'")

    # =================================================================
    # Tree render
    # =================================================================
    print("\n" + "=" * 78)
    print(f"TREE rooted at {phrase_ALL} ('abc.def.ghi.jkl')")
    print("=" * 78)
    print(lat.render_tree(phrase_ALL))

    # =================================================================
    # Chambers at each level
    # =================================================================
    print("\n" + "=" * 78)
    print("CHAMBERS AT EACH LEVEL (32 addresses per prime)")
    print("=" * 78)

    for prime, descr in [
        (base[0], "L0 base 'a'"),
        (word_abc, "L1 word 'abc'"),
        (phrase_ABC, "L2 phrase 'abc.def.ghi'"),
    ]:
        chs = lat.chambers(prime)
        coords = {c[2] for c in chs}
        node = lat.resolve(prime)
        print(f"\n  prime {prime} ({descr}, level {node.level}):")
        print(f"    sample chambers (first 4 of 32):")
        for branch, wing, coord in chs[:4]:
            print(f"      {branch.name}/wing{wing}: {coord}")
        print(f"    distinct (X,Y,Z) addresses: {len(coords)} / 32")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 78)
    print("HIERARCHY STATS")
    print("=" * 78)
    s = lat.stats()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 78)
    print("VERIFICATION SUMMARY")
    print("=" * 78)
    checks = [
        ("Level 2 swap_meet identity", L.coord == R.coord),
        ("Level 1 triple equalization (3 pair-witnesses collapse)", len(l1_triple_coords) == 1),
        ("Level 2 triple equalization (cross-level mixed)", len(l2_triple_coords) == 1),
        ("walk_down recovers correct base chain length", len(base_chain) == 12),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")


if __name__ == "__main__":
    main()
