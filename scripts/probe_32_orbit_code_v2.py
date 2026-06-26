"""
Corrected 32-chamber code test.

REALITY (measured, corrects the handed facts):
  - zeta = +/- sum(chain). Wing z-flip flips zeta's SIGN. |zeta| = sum(chain)
    is the conserved invariant, carried INDEPENDENTLY by all 32 chambers.
  - The (X,Y) plane has only 16 distinct points (z-flip changes zeta not X,Y),
    so the 32 slots = 16 (X,Y) x 2 zeta-signs.

CODE INTERPRETATION:
  - Each chamber is a codeword symbol carrying the SAME number |zeta|=sum(chain),
    a 32x repetition code on that one scalar.
  - Test erasure + corruption recovery of |zeta| by majority vote.
  - Test whether the chain ITSELF (not just its sum) is recoverable from chambers.
"""
import os, sys, random
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind, VECTORS

random.seed(1); np.random.seed(1)
BRANCHES = list(BranchKind)
WINGS = list(range(1, 9))
SLOTS = [(b, w) for b in BRANCHES for w in WINGS]


def orbit(chain, n):
    return {(b, w): wing_transform(b, chain, n, w) for (b, w) in SLOTS}


def rand_chain(k, lo=2, hi=40):
    return sorted(random.sample(range(lo, hi), k))


def abs_zeta_votes(orb):
    return [round(abs(p.zeta), 6) for p in orb.values()]


# (1) Confirm |zeta| repetition: all 32 carry sum(chain)
def test_abs_zeta(trials=300):
    all32 = 0
    distinct_xy = []
    for _ in range(trials):
        chain = rand_chain(random.randint(3, 6))
        n = (chain[0] + chain[-1]) / 2.0
        orb = orbit(chain, n)
        votes = abs_zeta_votes(orb)
        if len(set(votes)) == 1 and votes[0] == round(sum(chain), 6):
            all32 += 1
        xy = {(round(p.z.real, 6), round(p.z.imag, 6)) for p in orb.values()}
        distinct_xy.append(len(xy))
    return all32, trials, np.mean(distinct_xy), min(distinct_xy), max(distinct_xy)


# (2) ERASURE of |zeta|: 1 survivor recovers it (repetition code, distance 32)
def test_erasure_abs(trials=500):
    ok = {k: 0 for k in range(1, 33)}
    for _ in range(trials):
        chain = rand_chain(random.randint(3, 6))
        n = (chain[0] + chain[-1]) / 2.0
        orb = orbit(chain, n)
        truth = round(sum(chain), 6)
        slots = list(orb.keys()); random.shuffle(slots)
        for k in range(1, 33):
            surv = [round(abs(orb[s].zeta), 6) for s in slots[:k]]
            if len(set(surv)) == 1 and surv[0] == truth:
                ok[k] += 1
    return ok, trials


# (3) CORRUPTION of |zeta|: majority vote corrects up to floor((32-1)/2)=15 errors
def test_corruption_abs(trials=500, levels=(1, 5, 10, 14, 15, 16, 17, 20)):
    res = {}
    for c in levels:
        detect = correct = 0
        for _ in range(trials):
            chain = rand_chain(random.randint(3, 6))
            n = (chain[0] + chain[-1]) / 2.0
            orb = orbit(chain, n)
            truth = round(sum(chain), 6)
            votes = [round(abs(p.zeta), 6) for p in orb.values()]
            idx = list(range(32)); random.shuffle(idx)
            for i in idx[:c]:
                votes[i] = truth + random.choice([1, 2, 3, -1, -2, 5])
            if len(set(votes)) > 1:
                detect += 1
            best, cnt = Counter(votes).most_common(1)[0]
            if best == truth:
                correct += 1
        res[c] = (detect / trials, correct / trials)
    return res, trials


# (4) Does the orbit recover the WHOLE chain (not just sum)?
#     Test injectivity of the full 16-point (X,Y) set + |zeta| over many chains.
def sig(orb):
    xy = tuple(sorted({(round(p.z.real, 6), round(p.z.imag, 6)) for p in orb.values()}))
    return (round(abs(next(iter(orb.values())).zeta), 6), xy)


def test_full_recovery(trials=4000):
    seen = {}
    coll = 0; ex = []
    for _ in range(trials):
        chain = rand_chain(random.randint(3, 6))
        n = (chain[0] + chain[-1]) / 2.0
        s = sig(orbit(chain, n))
        if s in seen and seen[s] != tuple(chain):
            coll += 1
            if len(ex) < 4:
                ex.append((seen[s], tuple(chain)))
        seen[s] = tuple(chain)
    return coll, trials, ex


# (5) Fix the transgressor n so it doesn't leak chain info: does the orbit
#     still pin the chain? (n was set from chain endpoints above; test n fixed.)
def test_full_recovery_fixed_n(trials=4000, n=0.0):
    seen = {}; coll = 0; ex = []
    for _ in range(trials):
        chain = rand_chain(random.randint(3, 6))
        s = sig(orbit(chain, n))
        if s in seen and seen[s] != tuple(chain):
            coll += 1
            if len(ex) < 6:
                ex.append((seen[s], tuple(chain)))
        seen[s] = tuple(chain)
    return coll, trials, ex


if __name__ == "__main__":
    print("=" * 70)
    print("(1) |zeta| repetition + distinct (X,Y) count")
    a, T, mxy, mn, mx = test_abs_zeta()
    print(f"  all 32 chambers carry |zeta|=sum: {a}/{T}")
    print(f"  distinct (X,Y) points per orbit: mean={mxy:.1f} min={mn} max={mx} (expect 16)")

    print("=" * 70)
    print("(2) ERASURE of |zeta| (repetition code)")
    ok, T = test_erasure_abs()
    for k in [1, 2, 8, 31, 32]:
        print(f"  k={k:2d} survivors -> |zeta| recovered {ok[k]}/{T}")

    print("=" * 70)
    print("(3) CORRUPTION of |zeta| -> majority vote correction")
    res, T = test_corruption_abs()
    for c, (d, cor) in res.items():
        print(f"  corrupted={c:2d}: detect={d*100:5.1f}%  correct={cor*100:5.1f}%")
    print("  (32x repetition code: corrects <=15 errors, detects <=31)")

    print("=" * 70)
    print("(4) FULL CHAIN recovery (n from endpoints): collisions")
    c, T, ex = test_full_recovery()
    print(f"  collisions {c}/{T}; examples {ex}")

    print("=" * 70)
    print("(5) FULL CHAIN recovery with FIXED n=0 (no n leak): collisions")
    c, T, ex = test_full_recovery_fixed_n()
    print(f"  collisions {c}/{T}; examples {ex}")
