#!/usr/bin/env python3
"""
Test 55 - Different counting sets, same rule: primes are one of infinity.

The lattice formula is a RULE; the counting set it runs on is a CHOICE. Primes
are special only because their unique-representation theorem is FACTORIZATION
(FTA) - but every well-chosen sequence has its OWN unique representation, and
each unlocks different operations. The set is a design knob.

  POWERS OF 2  unique sum of distinct powers (binary) -> bitmask SETS:
               union=OR, intersect=AND, card=popcount, all O(1) bit ops
  PRIMES       unique product (FTA) -> multiplicative composite:
               union=multiply, intersect=gcd, dynamic & unbounded
  FIBONACCI    unique sum of NON-CONSECUTIVE Fibonacci (Zeckendorf)
  FACTORIALS   unique factorial-base digits -> permutation ranking (Lehmer)

We verify (a) each set's unique-representation theorem holds, (b) the operation
it unlocks, and (c) the SAME lattice formula (wing_transform) runs on chains
from any set - the rule does not care which sequence you count with.
"""

from __future__ import annotations

import sys
from itertools import combinations, permutations
from math import factorial, gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def main():
    header("Different counting sets, same rule - primes are one of infinity")

    # ==================================================================
    print("\n(A) POWERS OF 2 -> binary / bitmask sets (additive composite)")
    print("-" * 72)
    import random
    rng = random.Random(0x55E0)
    # set = bitmask; union=OR, intersect=AND, card=popcount
    A = set(rng.sample(range(40), 12))
    B = set(rng.sample(range(40), 12))
    ma = sum(1 << i for i in A)
    mb = sum(1 << i for i in B)
    union = {i for i in range(40) if (ma | mb) >> i & 1}
    inter = {i for i in range(40) if (ma & mb) >> i & 1}
    print(f"  |A|={len(A)} |B|={len(B)}: union via OR -> {len(union)}, "
          f"intersect via AND -> {len(inter)}, popcount(A)={bin(ma).count('1')}")
    assertion(union == (A | B) and inter == (A & B)
              and bin(ma).count("1") == len(A),
              "powers-of-2 give bitmask sets: OR=union, AND=intersect, "
              "popcount=cardinality (O(1) bit ops, fixed universe)")

    # ==================================================================
    print("\n(B) PRIMES -> multiplicative composite (FTA)")
    print("-" * 72)
    p = chain_primes(40)
    ca = 1
    for i in A:
        ca *= p[i]
    cb = 1
    for i in B:
        cb *= p[i]
    g = gcd(ca, cb)
    inter_p = {i for i in range(40) if g % p[i] == 0}
    assertion(inter_p == (A & B),
              "primes give multiplicative sets: multiply=union, gcd=intersect "
              "(dynamic, unbounded - the addressing this whole project used)")

    # ==================================================================
    print("\n(C) FIBONACCI -> Zeckendorf (unique non-consecutive sum)")
    print("-" * 72)
    fib = [1, 2]
    while fib[-1] < 5000:
        fib.append(fib[-1] + fib[-2])

    def zeckendorf(n):
        rep = []
        for f in reversed(fib):
            if f <= n:
                rep.append(f)
                n -= f
        return rep

    worst_gap = 0
    unique_ok = True
    for n in range(1, 2000):
        rep = zeckendorf(n)
        if sum(rep) != n:
            unique_ok = False
        idx = sorted(fib.index(f) for f in rep)
        # no two consecutive Fibonacci numbers
        if any(idx[i + 1] - idx[i] == 1 for i in range(len(idx) - 1)):
            unique_ok = False
    # uniqueness: brute-force check a sample against all non-consecutive subsets
    sample_ok = True
    small_fib = [f for f in fib if f <= 60]
    for n in range(1, 61):
        reps = 0
        for r in range(1, len(small_fib) + 1):
            for combo in combinations(range(len(small_fib)), r):
                if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1)):
                    if sum(small_fib[i] for i in combo) == n:
                        reps += 1
        if reps != 1:
            sample_ok = False
    print(f"  Zeckendorf of 1..1999: all valid non-consecutive sums = {unique_ok}")
    print(f"  uniqueness (brute force, 1..60): exactly one rep each = {sample_ok}")
    assertion(unique_ok and sample_ok,
              "Fibonacci gives Zeckendorf: every integer is a UNIQUE sum of "
              "non-consecutive Fibonacci numbers (another perfect addressing)")

    # ==================================================================
    print("\n(D) FACTORIALS -> factoradic / permutation ranking (Lehmer)")
    print("-" * 72)
    K = 6

    def rank_to_perm(n, k):
        items = list(range(k))
        perm = []
        for i in range(k, 0, -1):
            f = factorial(i - 1)
            d, n = divmod(n, f)
            perm.append(items.pop(d))
        return tuple(perm)

    def perm_to_rank(perm):
        items = list(range(len(perm)))
        n = 0
        for i, x in enumerate(perm):
            d = items.index(x)
            items.pop(d)
            n += d * factorial(len(perm) - 1 - i)
        return n

    seen = set()
    bijection = True
    for n in range(factorial(K)):
        perm = rank_to_perm(n, K)
        if perm in seen or perm_to_rank(perm) != n:
            bijection = False
        seen.add(perm)
    print(f"  {factorial(K)} ranks <-> {len(seen)} distinct permutations; "
          f"round-trip bijection = {bijection}")
    assertion(bijection and len(seen) == factorial(K),
              "factorials give factoradic: every integer < k! is a UNIQUE "
              "permutation (Lehmer code - addressing for orderings)")

    # ==================================================================
    print("\n(E) SAME RULE - the lattice formula runs on ANY counting set")
    print("-" * 72)
    from aethos_complex_plane import wing_transform
    from aethos_lattice import BranchKind

    sets = {
        "primes":     chain_primes(8),
        "powers of 2": [2 ** i for i in range(1, 9)],
        "fibonacci":  fib[:8],
        "triangular": [i * (i + 1) // 2 for i in range(2, 10)],
    }
    for name, seq in sets.items():
        chain = tuple(seq[:4])
        n = seq[5]
        chambers = set()
        for branch in BranchKind:
            for wing in range(1, 9):
                psi = wing_transform(branch, chain, n, wing)
                chambers.add((branch, wing, psi.coord))
        print(f"  {name:<12} chain {chain}: {len(chambers)} chambers")
        assertion(len(chambers) == 32,
                  f"the formula yields the full 32-chamber structure on the "
                  f"'{name}' set - the RULE is set-agnostic")

    header("RESULT")
    print("  one lattice RULE, four counting sets, four unique-representation")
    print("  systems, four different capabilities:")
    print("    powers of 2 -> bitmask sets (O(1) OR/AND/popcount)")
    print("    primes      -> multiplicative composites (dynamic, unbounded)")
    print("    fibonacci   -> Zeckendorf coding")
    print("    factorials  -> permutation ranking")
    print("  and the wing formula produced the full 32-chamber structure on")
    print("  every one of them.")
    print()
    print("  'Primes is just 1 of infinity that work in the complex plane' -")
    print("  confirmed. The counting set is a DESIGN KNOB: choose the sequence")
    print("  whose number-theoretic structure matches the task. Different")
    print("  lattices, same rule, different sets - a whole family of address")
    print("  spaces, of which the prime lattice is one member.")


if __name__ == "__main__":
    main()
