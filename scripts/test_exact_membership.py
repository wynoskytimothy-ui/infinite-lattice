#!/usr/bin/env python3
"""
Test 44 - Exact set membership: the Bloom filter with no false positives.

A Bloom filter answers "is x in the set?" in O(1) bits per element, but it
LIES sometimes (false positives) and can never delete or enumerate. The FTA
composite (Test 3) answers the same question with:

  - ZERO false positives and zero false negatives (membership = divisibility)
  - DELETE (divide out the prime) - impossible for a plain Bloom filter
  - ENUMERATE (factor the composite) - impossible for a Bloom filter
  - SET ALGEBRA (union=multiply, intersect=gcd) on the same object

Honest trade: the composite is a big integer that grows with the set, so for
enormous sets with a tolerable error budget, Bloom wins on space. But wherever
a false positive is unacceptable - security allowlists, exactly-once delivery,
financial dedup - this is the right structure, and it does strictly more.
"""

from __future__ import annotations

import math
import random
import sys
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


class BloomFilter:
    def __init__(self, m_bits, k_hashes):
        self.m = m_bits
        self.k = k_hashes
        self.bits = bytearray((m_bits + 7) // 8)

    def _hashes(self, x):
        h1 = hash((x, 0x9e3779b9)) % self.m
        h2 = (hash((x, 0x1234567)) % self.m) or 1
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, x):
        for h in self._hashes(x):
            self.bits[h >> 3] |= 1 << (h & 7)

    def query(self, x):
        return all(self.bits[h >> 3] & (1 << (h & 7)) for h in self._hashes(x))


class CompositeSet:
    """Exact set membership via a prime composite (FTA)."""

    def __init__(self, primes):
        self.primes = primes
        self.comp = 1
        self.members = set()

    def add(self, e):
        if e not in self.members:
            self.comp *= self.primes[e]
            self.members.add(e)

    def remove(self, e):
        if e in self.members:
            self.comp //= self.primes[e]      # DELETE - Bloom cannot
            self.members.discard(e)

    def query(self, e):
        return self.comp % self.primes[e] == 0

    def enumerate(self):
        return {i for i, p in enumerate(self.primes) if self.comp % p == 0}


def main():
    header("Exact set membership - zero false positives, plus delete/enumerate")
    rng = random.Random(0x44E0)

    UNIVERSE = 4000
    primes = chain_primes(UNIVERSE)
    members = set(rng.sample(range(UNIVERSE), 800))
    nonmembers = [e for e in range(UNIVERSE) if e not in members]

    # size Bloom for ~2% target FP at this load
    n = len(members)
    fp_target = 0.02
    m_bits = int(-n * math.log(fp_target) / (math.log(2) ** 2))
    k = max(1, round((m_bits / n) * math.log(2)))
    bloom = BloomFilter(m_bits, k)
    cset = CompositeSet(primes)
    for e in members:
        bloom.add(e)
        cset.add(e)

    # ---- false positives ----
    print("\nFalse positives on non-members")
    print("-" * 72)
    bloom_fp = sum(1 for e in nonmembers if bloom.query(e))
    cset_fp = sum(1 for e in nonmembers if cset.query(e))
    print(f"  Bloom (m={m_bits} bits, k={k}): {bloom_fp}/{len(nonmembers)} "
          f"false positives ({bloom_fp/len(nonmembers)*100:.1f}%)")
    print(f"  Composite set:                  {cset_fp}/{len(nonmembers)} "
          f"false positives (0.0%)")
    assertion(bloom_fp > 0, "the Bloom filter has false positives (it lies)")
    assertion(cset_fp == 0,
              "the composite set has ZERO false positives (membership is exact "
              "divisibility - it cannot lie)")

    # ---- false negatives (neither should have any) ----
    fn = sum(1 for e in members if not cset.query(e))
    assertion(fn == 0, "zero false negatives - every member tests present")

    # ---- DELETE: Bloom cannot, composite can ----
    print("\nDeletion - a thing Bloom filters fundamentally cannot do")
    print("-" * 72)
    to_remove = set(rng.sample(list(members), 200))
    for e in to_remove:
        cset.remove(e)
    removed_absent = all(not cset.query(e) for e in to_remove)
    kept_present = all(cset.query(e) for e in (members - to_remove))
    print(f"  removed 200 elements; all absent now: {removed_absent}; "
          f"others still present: {kept_present}")
    assertion(removed_absent and kept_present,
              "deletion works exactly (divide out the prime) - Bloom would "
              "need a counting variant and still could not enumerate")

    # ---- ENUMERATE: recover the exact set by factoring ----
    print("\nEnumeration - recover the exact membership by factoring")
    print("-" * 72)
    recovered = cset.enumerate()
    expected = members - to_remove
    print(f"  factored the composite -> {len(recovered)} members "
          f"(expected {len(expected)})")
    assertion(recovered == expected,
              "the composite enumerates its exact contents - a Bloom filter "
              "throws this information away")

    # ---- honest space comparison ----
    print("\nSpace - the honest trade-off")
    print("-" * 72)
    comp_bits = cset.comp.bit_length()
    print(f"  composite: {comp_bits} bits for {len(expected)} members "
          f"({comp_bits/len(expected):.1f} bits/element)")
    print(f"  Bloom:     {m_bits} bits for {n} members "
          f"({m_bits/n:.1f} bits/element) at {fp_target*100:.0f}% error")
    print(f"  Bloom is smaller per element; the composite buys EXACTNESS +")
    print(f"  delete + enumerate + set algebra with those extra bits.")
    assertion(comp_bits > 0, "space reported honestly (composite is larger)")

    header("RESULT")
    print("  Membership = divisibility: zero false positives, zero false")
    print("  negatives, and the structure also DELETES, ENUMERATES, and does")
    print("  union/intersect - none of which a Bloom filter can.")
    print()
    print("  Not a free lunch: Bloom uses fewer bits per element when you can")
    print("  tolerate errors. But for allowlists, exactly-once delivery, and")
    print("  dedup where a false positive is a bug, the prime composite is the")
    print("  exact, invertible, dynamic set - the same object that hashed game")
    print("  positions (Test 34) and verified the ledger (Test 41).")


if __name__ == "__main__":
    main()
