"""
Two-sided audit: does the AETHOS lattice genuinely SOLVE additive number theory?

Tests vs sympy ground truth:
 (a) PARTITIONS  — do distinct meet-nodes at depth zeta=N count p(N)/q(N)?
 (b) SUBSET-SUM  — does a meet/factor search find a subset summing to a target?
 (c) GOLDBACH    — enumerate all (p,q) prime pairs with p+q=E via 2-way meet preimage.

Ground truth: sympy.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import sympy
from sympy import primerange, isprime
from sympy.functions.combinatorial.numbers import partition as sympy_partition

from aethos_lattice import BranchKind
from aethos_sequences import canon_on_chain
from aethos_complex_plane import swap_meet, wing_transform


def banner(t):
    print("=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------------------
# Reference q(N): partitions of N into DISTINCT parts
# ---------------------------------------------------------------------------
def q_distinct(N: int) -> int:
    """Number of partitions of N into distinct positive parts (ground truth via DP)."""
    # dp[s] = number of ways to write s with distinct parts seen so far
    dp = [0] * (N + 1)
    dp[0] = 1
    for part in range(1, N + 1):
        for s in range(N, part - 1, -1):
            dp[s] += dp[s - part]
    return dp[N]


def all_distinct_partitions(N: int):
    """Yield every partition of N into distinct positive parts (as sorted tuples)."""
    def rec(remaining, maxpart, cur):
        if remaining == 0:
            yield tuple(cur)
            return
        for part in range(min(remaining, maxpart), 0, -1):
            cur.append(part)
            yield from rec(remaining - part, part - 1, cur)
            cur.pop()
    yield from rec(N, N, [])


# ===========================================================================
# (a) PARTITIONS via meet-node depth zeta = N
# ===========================================================================
def test_partitions(Nmax=14):
    banner("(a) PARTITIONS — distinct meet-nodes at depth zeta=N vs q(N)/p(N)")
    print("Lattice claim: a 2-chain (a,p) deposits zeta = a+p+n (k<=2, no lock).")
    print("A 'partition' of N is a multiset of parts summing to N. We ask: do the")
    print("DISTINCT (X,Y,zeta) meet-nodes generated at depth N count any partition fn?\n")

    print(f"{'N':>3} {'q(N)':>8} {'p(N)':>10} {'#2chainNodes':>13} {'#distNodeXYZ':>13}")
    match_q = match_p = 0
    for N in range(4, Nmax + 1):
        qN = q_distinct(N)
        pN = int(sympy_partition(N))

        # Enumerate every UNORDERED pair {a,p} of distinct positive ints with a+p < N,
        # transgress an interior n so that zeta = a+p+n = N  => n = N-(a+p), need a<n<p? No:
        # for 2-chain (a<p), interior seg means a<=n<p. zeta = a+p+n. Set zeta=N.
        # Collect the resulting meet-node coords.
        nodes = set()
        cnt = 0
        for a in range(1, N):
            for p in range(a + 1, N):
                n = N - (a + p)
                if n <= 0:
                    continue
                # interior requires a <= n < p (seg index 1 of 2-chain)
                seg_interior = (a <= n < p)
                # We allow any seg but record what coord results
                coord = canon_on_chain(BranchKind.VA1, (a, p), n)
                if coord[2] == N:  # depth lands on N
                    cnt += 1
                    nodes.add(coord)
        print(f"{N:>3} {qN:>8} {pN:>10} {cnt:>13} {len(nodes):>13}")
        if len(nodes) == qN:
            match_q += 1
        if len(nodes) == pN:
            match_p += 1

    print(f"\n  distinct-node-count == q(N): {match_q}/{Nmax-3} N values")
    print(f"  distinct-node-count == p(N): {match_p}/{Nmax-3} N values")
    print("  NOTE: 2-chains only give 2-part sums. Real partitions need k-part chains.")

    # Now the HONEST version: use k-chains of arbitrary length = distinct partitions directly.
    # A distinct partition of N IS a strictly-increasing chain summing to N.
    # zeta for a k-chain at the TOP segment (n>=ak) = sum(chain)+n; at interior lock = sum(chain).
    # The natural lattice handle: each distinct partition is a chain; sum(chain)=N is its depth
    # when we read zeta at an INTERIOR locked node (k>=3) where zeta = sum(chain) = N exactly.
    print("\n  --- k-chain reading: distinct partition = strictly-increasing chain, sum=N ---")
    print(f"{'N':>3} {'q(N)':>8} {'#chains(sum=N,distinct,k>=1)':>28} {'#zetaLockNodes':>15}")
    for N in range(4, Nmax + 1):
        qN = q_distinct(N)
        parts = list(all_distinct_partitions(N))
        # For each chain with k>=3, interior locked zeta = sum = N. Record the locked node.
        lock_nodes = set()
        for chain in parts:
            if len(chain) >= 3:
                c = tuple(sorted(chain))
                # pick an interior n strictly between a1 and ak that is not an anchor
                # interior seg => zeta locked to sum = N
                interior_n = None
                for n in range(c[0], c[-1] + 1):
                    seg = sum(1 for a in c if n >= a)
                    if 0 < seg < len(c):
                        interior_n = n
                        break
                if interior_n is not None:
                    coord = canon_on_chain(BranchKind.VA1, c, interior_n)
                    lock_nodes.add((coord, c))  # tag with chain to keep distinct
        print(f"{N:>3} {qN:>8} {len(parts):>28} {len(lock_nodes):>15}")
    print("  => #distinct partitions == q(N) BY CONSTRUCTION (we enumerated them).")
    print("  => The lattice STORES each on a node; it does not COUNT them by a formula.")


# ===========================================================================
# (b) SUBSET-SUM via meet/search
# ===========================================================================
def test_subset_sum(trials=200, setsize=12, maxval=60, seed=0):
    banner("(b) SUBSET-SUM — does a meet/factor search find a subset hitting target?")
    import random
    rng = random.Random(seed)

    def brute_subset_sum(vals, target):
        """Ground truth: any subset summing to target? return one or None."""
        n = len(vals)
        for r in range(1, n + 1):
            for combo in itertools.combinations(range(n), r):
                if sum(vals[i] for i in combo) == target:
                    return combo
        return None

    def lattice_subset_sum(vals, target):
        """
        'Lattice' approach: the meet of a chain deposits zeta = sum(chain)+n (or sum at
        interior lock). The only additive handle the lattice gives is: a chain's depth
        equals its element-sum. So 'find subset summing to target' = 'find a sub-chain
        whose sum(chain) == target'. That is EXACTLY subset-sum; the lattice gives no
        shortcut beyond reading zeta = sum. So the search is identical brute force.
        We implement it via the lattice's zeta read to be faithful.
        """
        n = len(vals)
        for r in range(1, n + 1):
            for combo in itertools.combinations(range(n), r):
                sub = sorted(vals[i] for i in combo)
                if len(set(sub)) != len(sub):
                    continue
                csum = sum(sub)
                # read depth via lattice: for k>=3 interior lock zeta==sum; else sum+n with n chosen 0-equiv
                # Use the closed identity zeta_top = sum + n; at n such that we read pure sum we set the
                # lattice depth = csum. The lattice literally returns sum in zeta.
                if csum == target:
                    return combo
        return None

    found_agree = 0
    exact_match = 0
    for _ in range(trials):
        vals = [rng.randint(1, maxval) for _ in range(setsize)]
        # ensure distinct for chain validity in some trials
        vals = list(dict.fromkeys(vals))
        target = rng.randint(1, sum(vals))
        gt = brute_subset_sum(vals, target)
        lt = lattice_subset_sum(vals, target)
        gt_has = gt is not None
        lt_has = lt is not None
        if gt_has == lt_has:
            found_agree += 1
        # check lattice's returned subset actually sums right
        if lt is not None and sum(vals[i] for i in lt) == target:
            exact_match += 1
        elif lt is None and gt is None:
            exact_match += 1
    print(f"  trials={trials}  decision agreement with brute force: {found_agree}/{trials}")
    print(f"  lattice-returned subset valid (or correct 'no'): {exact_match}/{trials}")
    print("  => Lattice 'subset-sum' = read zeta=sum of a sub-chain. SAME exponential search.")
    print("  => No algebraic speedup: the meet does not invert sum without enumerating subsets.")


# ===========================================================================
# (c) GOLDBACH via 2-way meet preimage
# ===========================================================================
def test_goldbach(Emax=120):
    banner("(c) GOLDBACH — meet preimage X=a+p=E enumerates all prime pairs")
    print("Lattice fact: swap_meet(a,p) => X-coord = a+p. So the meet of solo chains")
    print("{a},{p} lands at X = a+p. Fix even E; the preimage {(a,p): meet.X==E,")
    print("both prime} == all Goldbach pairs. We VERIFY meet.X==a+p, then compare the")
    print("enumerated pair set to sympy.\n")

    # First verify the meet really gives X = a+p and is the same node both ways
    meet_ok = 0
    meet_tot = 0
    for a in range(3, 40, 2):
        for p in range(a, 40, 2):
            left, right = swap_meet(a, p)
            meet_tot += 1
            # X coord == a+p, and both banks agree (the 'meet')
            if abs(left.z.real - (a + p)) < 1e-9 and left.coord == right.coord:
                meet_ok += 1
    print(f"  meet identity  meet.X == a+p AND bank(a)@p==bank(p)@a : {meet_ok}/{meet_tot}")

    def goldbach_via_meet(E):
        """Enumerate (a,p) a<=p, both prime, meet.X == E."""
        pairs = []
        primes = list(primerange(2, E))
        pset = set(primes)
        for a in primes:
            p = E - a
            if p < a:
                break
            if p in pset:
                # confirm via the lattice meet that X lands on E
                left, right = swap_meet(a, p)
                if abs(left.z.real - E) < 1e-9 and left.coord == right.coord:
                    pairs.append((a, p))
        return pairs

    def goldbach_sympy(E):
        primes = list(primerange(2, E))
        pset = set(primes)
        return [(a, E - a) for a in primes if E - a >= a and (E - a) in pset]

    allmatch = 0
    total = 0
    mism = []
    for E in range(4, Emax + 1, 2):
        gm = goldbach_via_meet(E)
        gs = goldbach_sympy(E)
        total += 1
        if gm == gs:
            allmatch += 1
        else:
            mism.append((E, gm, gs))
    print(f"  Goldbach pair-set match (E=4..{Emax}): {allmatch}/{total} even numbers")
    if mism:
        print(f"  mismatches: {mism[:5]}")
    # show a couple
    for E in (28, 100):
        print(f"  E={E}: pairs={goldbach_via_meet(E)}")
    print("\n  => meet.X == a+p is EXACT (re-encodes addition). Enumerating the preimage")
    print("     still requires testing each a for primality of E-a (sympy does the work).")
    print("     The lattice provides the ADDITION as a node coord, not the prime sieve.")


if __name__ == "__main__":
    test_partitions(14)
    print()
    test_subset_sum(200, 12, 60, 0)
    print()
    test_goldbach(120)
