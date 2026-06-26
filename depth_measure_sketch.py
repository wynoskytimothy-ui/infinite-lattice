"""
Depth-as-MEASURE: test what the verified inclusion-exclusion / erasure-code
property of the AETHOS lattice actually ENABLES as a data-structure.

Grounded facts used:
  - zeta(chain) = sum(chain), invariant of the transgressor n (the lock).
  - inclusion-exclusion zeta(A|B) = zeta(A)+zeta(B)-zeta(A&B) holds EXACTLY.
  - triple is an exact erasure code: missing = sum - pair.

We probe four claims, RUN them, and compare HONESTLY to HyperLogLog/Bloom.
"""
import random
import math
import sys

from aethos_complex_plane import wing_transform, missing_member
from aethos_lattice import BranchKind

random.seed(1234)


def zeta(chain):
    """Depth = the lattice measure. We READ it off the lattice, not just python sum,
    to prove the lattice node carries the measure."""
    if not chain:
        return 0.0
    # interior n: any value strictly inside; use mean (guaranteed interior for >=2 distinct)
    n = sum(chain) / len(chain)
    psi = wing_transform(BranchKind.VA1, chain, n, 1)
    return psi.zeta


# ---------------------------------------------------------------------------
# (c) STREAMING / MERGEABLE ACCUMULATOR  (the measure as a CRDT-style register)
# ---------------------------------------------------------------------------
# A "depth register" over a universe: each element x contributes a fixed token
# tok(x). The register stores the running zeta = sum of tokens of DISTINCT
# elements seen. To make distinct-count exact we must store WHICH elements, OR
# choose tokens so the sum is invertible. Test both framings honestly.

def tok(x):
    # deterministic per-element weight; primes-ish so sums stay distinguishable
    return (x * 2 + 1)


class DepthRegister:
    """Mergeable measure: keeps the EXACT set (for exactness) + its lattice zeta.
    This is a grow-only set CRDT whose 'measure' query is O(1) off the depth."""
    __slots__ = ("members", "_zeta")

    def __init__(self):
        self.members = set()
        self._zeta = 0.0

    def add(self, x):
        if x not in self.members:
            self.members.add(x)
            self._zeta += tok(x)

    def measure(self):
        return self._zeta

    def distinct_count(self):
        return len(self.members)

    def merge(self, other):
        # union; zeta of union via inclusion-exclusion read off membership
        for x in other.members:
            self.add(x)

    def lattice_zeta(self):
        """Recompute the measure THROUGH the lattice node, proving zeta==stored sum."""
        chain = sorted(tok(x) for x in self.members)
        return zeta(chain)


# ---------------------------------------------------------------------------
# (d) SET RECONCILIATION via the missing-member / erasure-code rule.
# ---------------------------------------------------------------------------
# Two shards each hold a set. They want the SYMMETRIC DIFFERENCE by exchanging
# small sketches, NOT the full sets. This is exactly the IBLT / set-reconciliation
# problem. The lattice's erasure-code ("missing = sum - pair") is the k=3 atom.
# We build a multi-cell sketch: hash each element into b cells; each cell keeps
# (count, xor_keysum, depthsum). A cell with count==1 is "pure" -> its single
# element is recoverable as xor_keysum (the missing-member rule generalized:
# erasure-decode the lone survivor). Peel pure cells iteratively.
#
# This is the AETHOS erasure atom (recover-the-missing) scaled from k=3 to a
# whole sketch -> an Invertible-Bloom-Lookup-Table built on the depth measure.

_MIX = 0x9E3779B97F4A7C15

def _hash(x, salt):
    # 64-bit splitmix-style avalanche -> good dispersion (the weak rehash was the bug)
    z = (x ^ (salt * _MIX)) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def cells_of(x, b, k):
    """k DISTINCT cell indices for element x (IBLT requirement: an element must
    not land in the same cell twice or the count parity breaks)."""
    idxs = []
    salt = 1
    while len(idxs) < k:
        c = _hash(x, salt) % b
        if c not in idxs:
            idxs.append(c)
        salt += 1
        if salt > 100 * k:  # b too small; bail with what we have
            break
    return idxs


class ReconSketch:
    __slots__ = ("b", "count", "keysum", "depthsum", "k")

    def __init__(self, b, k=3):
        self.b = b
        self.k = k
        self.count = [0] * b
        self.keysum = [0] * b          # xor of element ids -> erasure recovery
        self.depthsum = [0.0] * b      # the lattice depth/measure per cell

    def add(self, x):
        for c in cells_of(x, self.b, self.k):
            self.count[c] += 1
            self.keysum[c] ^= x
            self.depthsum[c] += tok(x)

    def subtract(self, other):
        """Cell-wise difference -> encodes the symmetric difference."""
        d = ReconSketch(self.b, self.k)
        for c in range(self.b):
            d.count[c] = self.count[c] - other.count[c]
            d.keysum[c] = self.keysum[c] ^ other.keysum[c]
            d.depthsum[c] = self.depthsum[c] - other.depthsum[c]
        return d

    def decode(self):
        """Peel pure cells (|count|==1) -> recover each element via the missing rule.
        Returns (only_in_A, only_in_B, success)."""
        count = self.count[:]
        keysum = self.keysum[:]
        depthsum = self.depthsum[:]
        onlyA, onlyB = set(), set()
        progress = True
        while progress:
            progress = False
            for c in range(self.b):
                if count[c] in (1, -1):
                    x = keysum[c]
                    # erasure-code consistency check: depth must match token of lone survivor
                    if abs(abs(depthsum[c]) - tok(x)) > 1e-9:
                        continue  # not actually pure (collision masquerading)
                    if count[c] == 1:
                        onlyA.add(x)
                    else:
                        onlyB.add(x)
                    sign = count[c]
                    for cc in cells_of(x, self.b, self.k):
                        count[cc] -= sign
                        keysum[cc] ^= x
                        depthsum[cc] -= sign * tok(x)
                    progress = True
        success = all(c == 0 for c in count)
        return onlyA, onlyB, success


# ---------------------------------------------------------------------------
# HyperLogLog (approximate distinct-count) for honest comparison.
# ---------------------------------------------------------------------------
class HLL:
    def __init__(self, p=12):
        self.p = p
        self.m = 1 << p
        self.reg = bytearray(self.m)

    def add(self, x):
        h = hash((x, 0xabcdef)) & 0xffffffffffffffff
        idx = h >> (64 - self.p)
        w = (h << self.p) & 0xffffffffffffffff | (1 << (self.p - 1))
        rank = 1
        while not (w >> 63) & 1:
            w <<= 1
            rank += 1
        if rank > self.reg[idx]:
            self.reg[idx] = rank

    def count(self):
        alpha = 0.7213 / (1 + 1.079 / self.m)
        s = sum(2.0 ** -r for r in self.reg)
        e = alpha * self.m * self.m / s
        if e <= 2.5 * self.m:
            z = self.reg.count(0)
            if z:
                e = self.m * math.log(self.m / z)
        return e


def human_bytes(n):
    for u in ["B", "KB", "MB"]:
        if n < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}GB"


def main():
    print("=" * 70)
    print("(c) STREAMING MERGEABLE MEASURE  — exact distinct-count via depth")
    print("=" * 70)
    N = 50000
    stream = [random.randint(0, 20000) for _ in range(N)]
    true_distinct = len(set(stream))

    reg = DepthRegister()
    hll = HLL(p=12)
    for x in stream:
        reg.add(x)
        hll.add(x)

    print(f"stream length          : {N}")
    print(f"TRUE distinct          : {true_distinct}")
    print(f"DepthRegister distinct : {reg.distinct_count()}  (exact? {reg.distinct_count()==true_distinct})")
    print(f"HLL estimate           : {hll.count():.1f}  (err {100*abs(hll.count()-true_distinct)/true_distinct:.2f}%)")
    # prove the measure is read off the lattice node
    lat = reg.lattice_zeta()
    print(f"depth (stored sum)     : {reg.measure()}")
    print(f"depth (via lattice node): {lat}  (match? {abs(lat-reg.measure())<1e-6})")
    print(f"DepthRegister bytes    : ~{human_bytes(reg.distinct_count()*8)} (stores the SET)")
    print(f"HLL bytes              : {human_bytes(hll.m)} (fixed, regardless of N)")

    print()
    print("MERGEABILITY (CRDT) — split stream across 4 shards, merge, compare to global")
    shards = [DepthRegister() for _ in range(4)]
    for i, x in enumerate(stream):
        shards[i % 4].add(x)
    merged = DepthRegister()
    for s in shards:
        merged.merge(s)
    print(f"  merged distinct  : {merged.distinct_count()}  (== global? {merged.distinct_count()==true_distinct})")
    print(f"  merged depth     : {merged.measure()}  (== global? {merged.measure()==reg.measure()})")
    # idempotent + commutative spot-check (CRDT laws)
    m2 = DepthRegister(); m2.merge(shards[2]); m2.merge(shards[0]); m2.merge(shards[2]); m2.merge(shards[1]); m2.merge(shards[3])
    print(f"  reorder+dup merge: distinct {m2.distinct_count()} depth {m2.measure()}  (CRDT idempotent/commutative? {m2.distinct_count()==true_distinct and m2.measure()==reg.measure()})")

    print()
    print("=" * 70)
    print("(d) SET RECONCILIATION — recover SYMMETRIC DIFFERENCE from a small sketch")
    print("    (the erasure 'missing-member' atom scaled to a whole sketch = IBLT)")
    print("=" * 70)
    universe = 200000
    base = set(random.sample(range(universe), 40000))
    A = set(base)
    B = set(base)
    # introduce d differences
    for d_target in [50, 200, 1000, 3000]:
        OVERHEAD = 2.0  # IBLT load factor; sized to the DIFFERENCE not the set
        A = set(base); B = set(base)
        onlyA_true = set(random.sample(sorted(set(range(universe)) - base), d_target // 2))
        onlyB_true = set(random.sample(sorted(set(range(universe)) - base - onlyA_true), d_target // 2))
        A |= onlyA_true
        B |= onlyB_true
        true_symdiff = (A - B) | (B - A)

        # sketch sized to OVERHEAD x the difference (IBLT rule of thumb), NOT to set size
        b = max(16, int(len(true_symdiff) * OVERHEAD))
        sa = ReconSketch(b, k=4); sb = ReconSketch(b, k=4)
        for x in A: sa.add(x)
        for x in B: sb.add(x)
        diff = sa.subtract(sb)
        recA, recB, ok = diff.decode()
        recovered = recA | recB
        exact = (recovered == true_symdiff) and ok
        sketch_bytes = b * (8 + 8 + 8)  # count,key,depth per cell
        full_bytes = (len(A) + len(B)) * 8
        print(f"  |symdiff|={len(true_symdiff):5d}  sketch cells={b:5d} "
              f"recovered={len(recovered):5d}  EXACT={exact}  decoded_clean={ok}  "
              f"sketch={human_bytes(sketch_bytes)} vs ship-both-sets={human_bytes(full_bytes)}")

    print()
    print("=" * 70)
    print("HONEST COMPARISON")
    print("=" * 70)
    print("""
 distinct-count:
   DepthRegister  = EXACT but stores the full set (O(distinct)) -> same space as a
                    hash-set; the depth gives O(1) measure + clean CRDT merge, but
                    HLL beats it MASSIVELY on space (4-16KB FIXED for billions of
                    items, ~1% error). Exactness here costs space; it is NOT a free
                    win over HLL for pure cardinality.
 set reconciliation:
   ReconSketch    = sketch size scales with the DIFFERENCE, not the set. Two 40k
                    sets that differ in 200 elements reconcile from a few-KB sketch
                    instead of shipping 640KB of sets. This is the REAL edge: the
                    erasure-decode (missing-member rule) recovers the EXACT symmetric
                    difference. Same family as IBLT/Minisketch (used in Bitcoin
                    Erlay). The lattice's k=3 erasure atom IS this primitive.
""")


if __name__ == "__main__":
    main()
