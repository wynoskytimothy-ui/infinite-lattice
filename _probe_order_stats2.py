"""
TWO-SIDED part 2: does the LATTICE do the comparison, or do I do it before calling?

In probe 1, meet_min2(a,b) called sorted([a,b]) FIRST -> the comparison was mine,
not the lattice's. To claim the lattice is a *selection engine* the comparison must
happen INSIDE the meet algebra, not in Python's sorted().

Test: build the meet WITHOUT pre-sorting. normalize_chain() sorts internally, so the
chain (a,b) in EITHER order yields the SAME canon. Then read off min/max/median from
the OUTPUT coordinate. If that works, the lattice's normalize+formula IS the comparator.
But note: normalize_chain literally calls sorted(). So the comparison lives in
normalize_chain, which is... a classical sort. We quantify exactly where the work is.
"""
from __future__ import annotations
import numpy as np
from aethos_lattice import BranchKind
from aethos_sequences import canon_on_chain, normalize_chain

VA1 = BranchKind.VA1
rng = np.random.default_rng(1)


def min_max_from_meet_unsorted(a, b):
    """Pass anchors in ARBITRARY order; let the lattice's own normalize+formula
    surface min and max. We do NOT pre-sort. n is chosen below/above both via
    a lattice-internal probe (we use the chain's own endpoints AFTER normalize)."""
    if a == b:
        return a, a
    # normalize_chain does the sort internally -> chain[0]=min, chain[-1]=max
    chain = normalize_chain((a, b))   # <-- this is where the compare happens
    lo_probe = chain[0] - 1
    hi_probe = chain[-1] + 1
    _, ymin, _ = canon_on_chain(VA1, chain, lo_probe)   # seg0 Y = a1 = min
    _, ymax, _ = canon_on_chain(VA1, chain, hi_probe)   # segk Y = ak = max
    return ymin, ymax


def test_unsorted():
    n = 5000
    ok = 0
    for _ in range(n):
        a, b = rng.integers(0, 1000, size=2).tolist()
        mn, mx = min_max_from_meet_unsorted(a, b)
        if mn == min(a, b) and mx == max(a, b):
            ok += 1
    return n, ok


# ---------------------------------------------------------------------------
# The DECISIVE test: a comparator that uses NO Python comparison at all.
# Can the meet ALGEBRA alone (sum/min on the OUTPUT, no sorted()) separate
# min from max? The 2-chain interior formula at a FIXED n between them is
# (hi+n, n, lo+hi+n). From X=hi+n and zeta=lo+hi+n we can RECOVER:
#   hi = X - n   ;   lo = zeta - X   ;   no comparison used, pure arithmetic.
# So given the PAIR already as a normalized chain, the meet RECOVERS both
# order statistics by ARITHMETIC. The sort is upstream (normalize). The meet
# itself is an invertible ENCODING, not the comparator.
# ---------------------------------------------------------------------------
def recover_by_arithmetic(lo, hi):
    """Given chain (lo,hi) ALREADY ordered, pick any interior n, recover lo,hi
    with pure arithmetic from the meet coordinate (no comparison)."""
    n = (lo + hi) / 2.0          # any value strictly between
    X, Y, Z = canon_on_chain(VA1, (lo, hi), n)   # (hi+n, n, lo+hi+n)
    hi_rec = X - n
    lo_rec = Z - X
    return lo_rec, hi_rec


def test_arithmetic_recovery():
    n = 5000
    ok = 0
    for _ in range(n):
        a, b = sorted(rng.integers(0, 1000, size=2).tolist())
        if a == b:
            b += 1
        lo_r, hi_r = recover_by_arithmetic(a, b)
        if abs(lo_r - a) < 1e-9 and abs(hi_r - b) < 1e-9:
            ok += 1
    return n, ok


# ---------------------------------------------------------------------------
# Timing: lattice_sort vs numpy.sort (is it competitive, or 1000x slower?)
# ---------------------------------------------------------------------------
def timing():
    import time
    from _probe_order_stats import lattice_sort
    sizes = [8, 16, 32]
    out = {}
    for m in sizes:
        v = rng.choice(np.arange(0, 1_000_000), size=m, replace=False).tolist()
        t0 = time.perf_counter()
        for _ in range(200):
            lattice_sort(v)
        t_lat = (time.perf_counter() - t0) / 200
        arr = np.array(v)
        t0 = time.perf_counter()
        for _ in range(200):
            np.sort(arr)
        t_np = (time.perf_counter() - t0) / 200
        out[m] = (t_lat * 1e6, t_np * 1e6)
    return out


if __name__ == "__main__":
    print("=" * 70)
    print("Where does the comparison live? (decisive two-sided test)")
    n, ok = test_unsorted()
    print(f"  min/max via lattice normalize+formula : {ok}/{n}")
    print("    (NB: normalize_chain internally calls sorted() -> the COMPARE is classical)")

    n, ok = test_arithmetic_recovery()
    print(f"\n  recover lo,hi from meet coord by ARITHMETIC (no compare) : {ok}/{n}")
    print("    -> the meet is an INVERTIBLE ENCODING of an already-ordered pair,")
    print("       not the thing that orders them.")

    print("\nTiming  lattice_sort vs numpy.sort  (microseconds/call):")
    for m, (tl, tn) in timing().items():
        print(f"  size {m:3d}:  lattice {tl:9.1f}us   numpy {tn:7.2f}us   ratio {tl/tn:8.0f}x")
