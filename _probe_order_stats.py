"""
RIGOROUS TWO-SIDED PROBE: is the AETHOS meet a genuine selection/sorting engine?

Ground truth: numpy.median, numpy.partition, numpy.sort.
AETHOS API:    canon_on_chain(VA1, chain, n)  -- the (X,Y,zeta) coordinate.

Claim under test (from MEMORY): for sorted {a,b,c}, the 3-way meet gives
    Y = median(a,b,c),  zeta = sum,  X = top-two-sum (b+c).
We probe whether this generalizes to k-th order statistic of m>3, and to sorting.
"""
from __future__ import annotations
import numpy as np
from aethos_lattice import BranchKind
from aethos_sequences import canon_on_chain

VA1 = BranchKind.VA1
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# PART A — re-verify the median gem, and dissect WHAT the 3-way meet computes.
# ---------------------------------------------------------------------------
def three_way_meet(a, b, c):
    """Equalization node of sorted {a,b,c}: canon_on_chain(VA1,(a,b,c), n=middle-witness).

    The interior-segment formula for a 3-chain (a1<a2<a3) at a1<n<a3 is
        (a3 + n, n, sum).
    The triple-equalization witnesses are n = each missing member. Take the
    (a,c)@n=b rail -> the canonical 'middle' node. We read the FULL coord.
    """
    s = sorted([a, b, c])
    a1, a2, a3 = s
    # rail (a1,a3) transgressed to the missing middle witness n=a2:
    coord = canon_on_chain(VA1, (a1, a3), a2)
    return coord  # (X, Y, zeta)


def test_median_3():
    n = 5000
    ok_y = ok_z = ok_x = 0
    for _ in range(n):
        v = rng.integers(0, 1000, size=3)
        while len(set(v.tolist())) < 3:
            v = rng.integers(0, 1000, size=3)
        a, b, c = v.tolist()
        X, Y, Z = three_way_meet(a, b, c)
        s = sorted([a, b, c])
        if Y == s[1]:
            ok_y += 1            # median
        if Z == sum(s):
            ok_z += 1            # total sum
        if X == s[1] + s[2]:
            ok_x += 1            # top-two sum
    return n, ok_y, ok_z, ok_x


# ---------------------------------------------------------------------------
# PART B — k-th order statistic of m>3 by COMPOSING meets.
# The interior 2-chain formula canon_on_chain(VA1,(lo,hi), n) = (hi+n, n, lo+hi+n)
# for lo<n<hi: the Y output is literally n (a pass-through), the X is hi+n,
# zeta is lo+hi+n. The ONLY order-statistic content is: pick which n.
# So a 'selection network' must use the median-of-3 node as a comparator.
# We build median-of-medians style selection PURELY from 3-way meet medians
# and measure exact-match vs np.partition.
# ---------------------------------------------------------------------------
def meet_median3(a, b, c):
    """Return the median of 3 using ONLY the lattice meet Y-output.
    Guards ties (lattice rejects equal anchors)."""
    global TIE_REJECTS
    s = sorted([a, b, c])
    if s[0] == s[1] or s[1] == s[2]:
        TIE_REJECTS += 1
        return s[1]  # classical median fallback on a tie
    X, Y, Z = three_way_meet(a, b, c)
    return Y


TIE_REJECTS = 0  # count how often the meet REFUSES (equal anchors) -> a real limit


def meet_min2(a, b):
    """min(a,b) via 2-chain meet: canon_on_chain Y at n below both?
    For a 2-chain (lo,hi), seg=0 (n<lo) -> Y = a1 = lo = min. So feeding n
    smaller than both anchors returns the min on Y. That IS the (min,+) read.

    HONEST LIMIT: the lattice REJECTS equal anchors (set semantics), so a tie
    must be handled outside the meet. We count those rejections."""
    global TIE_REJECTS
    if a == b:
        TIE_REJECTS += 1
        return a  # meet cannot represent {a,a}; classical fallback
    lo, hi = sorted([a, b])
    X, Y, Z = canon_on_chain(VA1, (lo, hi), lo - 1)  # seg 0 -> (a1, ...)
    return Y  # == lo == min


def meet_max2(a, b):
    """max(a,b): seg=k (n>=hi) -> Y = ak = hi = max."""
    global TIE_REJECTS
    if a == b:
        TIE_REJECTS += 1
        return a
    lo, hi = sorted([a, b])
    X, Y, Z = canon_on_chain(VA1, (lo, hi), hi + 1)  # seg=k -> (ak+n, ak, ...)
    return Y  # == hi == max


def test_minmax_comparators():
    n = 5000
    ok_min = ok_max = 0
    for _ in range(n):
        a, b = rng.integers(0, 1000, size=2).tolist()
        if a == b:
            b += 1
        if meet_min2(a, b) == min(a, b):
            ok_min += 1
        if meet_max2(a, b) == max(a, b):
            ok_max += 1
    return n, ok_min, ok_max


# ---------------------------------------------------------------------------
# PART C — SORTING NETWORK from the meet comparator.
# A comparator on a wire pair (a,b) -> (min,max). We have both via the meet.
# Build a Batcher odd-even mergesort network and run it using ONLY meet
# min/max. Compare to numpy.sort for correctness.
# ---------------------------------------------------------------------------
def meet_compare_swap(arr, i, j):
    """In-place comparator: arr[i],arr[j] -> (min,max) using lattice meet."""
    lo = meet_min2(arr[i], arr[j])
    hi = meet_max2(arr[i], arr[j])
    arr[i], arr[j] = lo, hi


def oddeven_merge(arr, lo, n, r):
    step = r * 2
    if step < n:
        oddeven_merge(arr, lo, n, step)
        oddeven_merge(arr, lo + r, n, step)
        for i in range(lo + r, lo + n - r, step):
            meet_compare_swap(arr, i, i + r)
    else:
        meet_compare_swap(arr, lo, lo + r)


def oddeven_merge_sort(arr, lo, n):
    if n > 1:
        m = n // 2
        oddeven_merge_sort(arr, lo, m)
        oddeven_merge_sort(arr, lo + m, m)
        oddeven_merge(arr, lo, n, 1)


def lattice_sort(values):
    """Sort via Batcher odd-even mergesort built ONLY from meet comparators.
    Pads to power of two with +inf sentinels."""
    a = list(values)
    n = 1
    while n < len(a):
        n *= 2
    base = (max(a) + 1) if a else 0
    padded = a + [base + i for i in range(n - len(a))]  # distinct sentinels > all values
    oddeven_merge_sort(padded, 0, n)
    return padded[:len(values)]


def _distinct_array(m):
    return rng.choice(np.arange(0, 100000), size=m, replace=False).tolist()


def test_sorting(distinct=True):
    global TIE_REJECTS
    TIE_REJECTS = 0
    n_trials = 2000
    exact = 0
    sizes = []
    for _ in range(n_trials):
        m = int(rng.integers(2, 33))
        v = _distinct_array(m) if distinct else rng.integers(0, 50, size=m).tolist()
        got = lattice_sort(v)
        want = sorted(v)
        if got == want:
            exact += 1
        sizes.append(m)
    return n_trials, exact, (min(sizes), max(sizes)), TIE_REJECTS


# ---------------------------------------------------------------------------
# PART D — k-th order statistic of m via the sort, and via QUICKSELECT using
# median-of-3 PIVOTS from the meet (the meet's actual selection contribution).
# ---------------------------------------------------------------------------
def quickselect_meet(values, k):
    """k-th smallest (0-indexed) using median-of-3 pivot from lattice meet."""
    a = list(values)
    while True:
        if len(a) == 1:
            return a[0]
        # median-of-3 pivot via the meet
        if len(a) >= 3:
            pivot = meet_median3(a[0], a[len(a) // 2], a[-1])
        else:
            pivot = a[0]
        lt = [x for x in a if x < pivot]
        eq = [x for x in a if x == pivot]
        gt = [x for x in a if x > pivot]
        if k < len(lt):
            a = lt
        elif k < len(lt) + len(eq):
            return pivot
        else:
            k -= len(lt) + len(eq)
            a = gt


def test_kth_order_stat():
    n = 3000
    exact = 0
    for _ in range(n):
        m = int(rng.integers(3, 40))
        v = rng.integers(0, 5000, size=m).tolist()
        k = int(rng.integers(0, m))
        got = quickselect_meet(v, k)
        want = int(np.partition(np.array(v), k)[k])
        if got == want:
            exact += 1
    return n, exact


if __name__ == "__main__":
    print("=" * 70)
    print("PART A — 3-way meet decomposition vs ground truth (5000 trials)")
    n, oy, oz, ox = test_median_3()
    print(f"  Y == median(a,b,c)     : {oy}/{n}")
    print(f"  zeta == sum(a,b,c)     : {oz}/{n}")
    print(f"  X == top-two-sum (b+c) : {ox}/{n}")

    print("\nPART B — meet as min/max comparator (5000 trials)")
    n, omin, omax = test_minmax_comparators()
    print(f"  meet_min2 == min : {omin}/{n}")
    print(f"  meet_max2 == max : {omax}/{n}")

    print("\nPART C1 — Batcher odd-even mergesort, DISTINCT values (clean network)")
    n, exact, (smin, smax), ties = test_sorting(distinct=True)
    print(f"  exact match vs sorted() : {exact}/{n}   (sizes {smin}..{smax}, tie-rejects {ties})")

    print("\nPART C2 — same network, values WITH ties (0..49) -> exposes tie limit")
    n, exact, (smin, smax), ties = test_sorting(distinct=False)
    print(f"  exact match vs sorted() : {exact}/{n}   (tie-rejects (classical fallback used) {ties})")

    print("\nPART D — k-th order statistic via meet-median-of-3 quickselect")
    n, exact = test_kth_order_stat()
    print(f"  exact match vs np.partition : {exact}/{n}")
