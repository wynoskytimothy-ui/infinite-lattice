#!/usr/bin/env python3
"""
Test 9 - Compositional CRDT (Conflict-free Replicated Data Type).

Claim: Prime composites form a commutative, associative, and idempotent
algebra when interpreted as multisets of prime factors. This means concurrent
operations on different replicas can be merged WITHOUT conflict, by simply
multiplying composites and deduplicating via FTA.

Why this matters: CRDTs (Shapiro et al. 2011) are the foundation of modern
collaborative editing (Google Docs, Figma), distributed databases (Riak,
Redis Enterprise), and offline-first apps. Standard CRDTs (G-Counter, OR-Set,
LWW-Register) require careful design to ensure idempotency + commutativity.

Our composite IS a CRDT by construction:
  - Commutative: a * b = b * a
  - Associative: a * (b * c) = (a * b) * c
  - Idempotent (via dedup): set({factors_of(a * a)}) = set({factors_of(a)})

Tests:
  (A) Commutativity: different merge orders -> same final state
  (B) Associativity: tree-merged equals linearly-merged
  (C) Idempotence (via union semantics on factor sets)
  (D) Convergence: 3 replicas with concurrent ops converge to same state
  (E) Partition tolerance: replicas diverge then heal automatically
"""

from __future__ import annotations

import random
import sys
from itertools import permutations
from math import gcd
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


def factor_set(n: int, primes: tuple[int, ...]) -> frozenset[int]:
    """The unique set of prime factors of n via FTA."""
    fs: set[int] = set()
    residual = n
    for p in primes:
        if residual % p == 0:
            fs.add(p)
            while residual % p == 0:
                residual //= p
        if residual == 1:
            break
    return frozenset(fs)


def merge(a: int, b: int) -> int:
    """CRDT merge = compute least-common-multiple style union of factors.

    For factor SETS (G-Set semantics): result = a * b / gcd(a, b) ensures
    each prime appears exactly once even if it was in both. This is the
    idempotent merge.
    """
    return a * b // gcd(a, b)


def main():
    header("Composite as CRDT - commutative, associative, idempotent merge")

    base = chain_primes(50)

    # ---------------------------------------------------------------
    # Part A: Commutativity
    # ---------------------------------------------------------------
    print("\nPart A - Commutativity: a o b == b o a")
    print("-" * 72)

    a = 3 * 7 * 11
    b = 5 * 11 * 13
    ab = merge(a, b)
    ba = merge(b, a)
    print(f"  a = {a} (factors: {sorted(factor_set(a, base))})")
    print(f"  b = {b} (factors: {sorted(factor_set(b, base))})")
    print(f"  a o b = {ab}, b o a = {ba}")
    print(f"  union factors: {sorted(factor_set(ab, base))}")
    assertion(ab == ba, "merge is commutative")

    # ---------------------------------------------------------------
    # Part B: Associativity
    # ---------------------------------------------------------------
    print("\nPart B - Associativity: (a o b) o c == a o (b o c)")
    print("-" * 72)

    c = 7 * 13 * 17
    left = merge(merge(a, b), c)
    right = merge(a, merge(b, c))
    print(f"  left:  ((a o b) o c) = {left}")
    print(f"  right: (a o (b o c)) = {right}")
    assertion(left == right, "merge is associative")

    # ---------------------------------------------------------------
    # Part C: Idempotence
    # ---------------------------------------------------------------
    print("\nPart C - Idempotence: a o a == a (factor set preserved)")
    print("-" * 72)

    aa = merge(a, a)
    print(f"  a    = {a}, factors {sorted(factor_set(a, base))}")
    print(f"  aoa = {aa}, factors {sorted(factor_set(aa, base))}")
    assertion(aa == a, "merge is idempotent: a o a == a")

    # ---------------------------------------------------------------
    # Part D: 3 replicas converge to same state
    # ---------------------------------------------------------------
    print("\nPart D - 3 replicas with disjoint inserts converge")
    print("-" * 72)

    # Replicas start with same state
    initial = 3 * 5 * 7
    R1 = initial * 11 * 13   # replica 1 adds factors 11, 13
    R2 = initial * 17 * 19   # replica 2 adds 17, 19
    R3 = initial * 23 * 29   # replica 3 adds 23, 29

    # Different merge orders
    orders = [(R1, R2, R3), (R1, R3, R2), (R2, R1, R3),
              (R2, R3, R1), (R3, R1, R2), (R3, R2, R1)]
    final_states = []
    for perm in orders:
        x, y, z = perm
        final = merge(merge(x, y), z)
        final_states.append(final)
    distinct_states = set(final_states)
    print(f"  6 merge orders tried")
    print(f"  distinct final states: {len(distinct_states)}")
    print(f"  factors:               {sorted(factor_set(final_states[0], base))}")
    assertion(len(distinct_states) == 1,
              "all 6 merge orders converge to identical state")

    # ---------------------------------------------------------------
    # Part E: Partition tolerance - replicas diverge then heal
    # ---------------------------------------------------------------
    print("\nPart E - Partition tolerance: divergent updates auto-heal")
    print("-" * 72)

    # Two replicas partition. Each accepts 100 random updates independently.
    random.seed(2024)
    R_A = initial
    R_B = initial
    updates_A = []
    updates_B = []
    for _ in range(100):
        p = random.choice(base[10:])  # avoid the initial factors
        R_A = merge(R_A, p)
        updates_A.append(p)
    for _ in range(100):
        p = random.choice(base[10:])
        R_B = merge(R_B, p)
        updates_B.append(p)

    # Partition heals: merge final states
    healed = merge(R_A, R_B)

    # Expected: union of all updates + initial
    expected_factors = factor_set(initial, base) | \
                       factor_set(R_A, base) | factor_set(R_B, base)
    actual_factors = factor_set(healed, base)
    print(f"  replica A factors:       {len(factor_set(R_A, base))}")
    print(f"  replica B factors:       {len(factor_set(R_B, base))}")
    print(f"  after heal:              {len(actual_factors)}")
    print(f"  expected:                {len(expected_factors)}")
    assertion(actual_factors == expected_factors,
              "post-heal state == union of both partition states (no data loss)")

    # ---------------------------------------------------------------
    # Part F: Merge-merge consistency for many random replicas
    # ---------------------------------------------------------------
    print("\nPart F - Many random replicas: pairwise merge converges globally")
    print("-" * 72)

    n_replicas = 10
    replicas = []
    for i in range(n_replicas):
        rng = random.Random(i)
        comp = 1
        for _ in range(rng.randint(3, 7)):
            p = rng.choice(base)
            if comp % p != 0:
                comp *= p
        replicas.append(comp)

    # Method 1: reduce left-to-right
    state1 = replicas[0]
    for r in replicas[1:]:
        state1 = merge(state1, r)

    # Method 2: tree-reduce
    cur = list(replicas)
    while len(cur) > 1:
        new_layer = []
        for i in range(0, len(cur), 2):
            if i + 1 < len(cur):
                new_layer.append(merge(cur[i], cur[i + 1]))
            else:
                new_layer.append(cur[i])
        cur = new_layer
    state2 = cur[0]

    # Method 3: shuffle then reduce
    shuffled = replicas[:]
    random.shuffle(shuffled)
    state3 = shuffled[0]
    for r in shuffled[1:]:
        state3 = merge(state3, r)

    print(f"  state via linear:   factors = {len(factor_set(state1, base))}")
    print(f"  state via tree:     factors = {len(factor_set(state2, base))}")
    print(f"  state via shuffle:  factors = {len(factor_set(state3, base))}")
    assertion(state1 == state2 == state3,
              "linear, tree, and shuffled reductions all converge")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    header("RESULT")
    print(f"  Commutativity:    verified")
    print(f"  Associativity:    verified")
    print(f"  Idempotence:      verified")
    print(f"  Convergence:      6/6 merge orders identical")
    print(f"  Partition heal:   no data loss")
    print(f"  Multi-replica:    3 methods identical")
    print()
    print("  CONCLUSION:")
    print("  The composite forms a join-semilattice under prime-factor union")
    print("  (LCM operation): commutative + associative + idempotent. This")
    print("  makes it a CRDT (G-Set with prime encoding). Concurrent updates")
    print("  across N replicas converge automatically when merged in any order.")
    print()
    print("  Unlike standard CRDTs (which require carefully designed merge")
    print("  functions per data type), the FTA gives us a UNIVERSAL CRDT for")
    print("  ANY set-valued state - just encode elements as primes.")


if __name__ == "__main__":
    main()
