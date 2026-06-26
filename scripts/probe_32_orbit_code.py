"""
The 32-chamber orbit as an ERROR-CORRECTING CODE.

One chain -> 32 chambers (4 branches x 8 wings), each a complex z=(X+iY),
all sharing zeta = sum(chain) (the conserved invariant / lock).

We test, by RUNNING:
(a) Is the 32-orbit a GROUP under wing/branch operations (closed, invertible)?
(b) Given a CORRUPTED subset (drop or perturb some vectors), can we recover the
    chain / detect corruption from the conserved invariant + symmetry constraints?
(c) How many of the 32 can we lose and still uniquely recover? -> redundancy number.
"""
import itertools
import os
import sys
import random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind, VECTORS

random.seed(0)
np.random.seed(0)

BRANCHES = list(BranchKind)          # 4
WINGS = list(range(1, 9))            # 8  (1..8)
SLOTS = [(b, w) for b in BRANCHES for w in WINGS]  # 32 ordered slots


def orbit(chain, n):
    """Return dict slot->(X, Y, zeta) for the 32 chambers of one (chain, n).
    NOTE (corrected from handed facts): zeta = +/- sum(chain); the wing z-flip
    flips the SIGN of zeta. The conserved invariant is |zeta| = sum(chain)."""
    out = {}
    for (b, w) in SLOTS:
        psi = wing_transform(b, chain, n, w)
        out[(b, w)] = (psi.z.real, psi.z.imag, psi.zeta)
    return out


def rand_chain(k, lo=2, hi=40):
    """Strictly increasing integer chain of length k."""
    vals = sorted(random.sample(range(lo, hi), k))
    return vals


# ----------------------------------------------------------------------------
# (0) Basic structure: confirm 32 distinct vectors, one conserved zeta
# ----------------------------------------------------------------------------
def test_structure(trials=200):
    n_distinct = []
    zeta_conserved = 0
    for _ in range(trials):
        k = random.randint(3, 6)
        chain = rand_chain(k)
        n = (chain[0] + chain[-1]) / 2.0   # interior transgressor
        orb = orbit(chain, n)
        vecs = {(round(x, 9), round(y, 9)) for (x, y, z) in orb.values()}
        zetas = {round(z, 9) for (x, y, z) in orb.values()}
        n_distinct.append(len(vecs))
        if len(zetas) == 1 and abs(list(zetas)[0] - sum(chain)) < 1e-6:
            zeta_conserved += 1
    return np.mean(n_distinct), min(n_distinct), max(n_distinct), zeta_conserved, trials


# ----------------------------------------------------------------------------
# (a) GROUP test. The 8 wings + 4 branches act on the (X,Y) plane.
#     A wing/branch is a sign-flip on (x,y,z) per VECTORS table. We test whether
#     the maps slot_i -> slot_j compose into a closed, invertible group on the
#     orbit (a permutation group on the 32 slots induced by flip composition).
# ----------------------------------------------------------------------------
def flip_signs(v):
    """Sign triple (sx, sy, sz) for a VECTORS entry (wing) -> +/-1."""
    return (-1 if v.flip_x else 1, -1 if v.flip_y else 1, -1 if v.flip_z else 1)


def test_group():
    # Each wing 1..8 maps to a sign triple in {+1,-1}^3 with sy always +1.
    triples = [flip_signs(VECTORS[w - 1]) for w in WINGS]
    sx = sorted({t[0] for t in triples})
    sy = sorted({t[1] for t in triples})
    sz = sorted({t[2] for t in triples})
    # Closure under componentwise multiplication (the sign group {+1,-1}^3 subset)?
    triple_set = set(triples)
    closed = True
    has_identity = (1, 1, 1) in triple_set
    inverses = True
    for a in triples:
        for b in triples:
            prod = (a[0] * b[0], a[1] * b[1], a[2] * b[2])
            if prod not in triple_set:
                closed = False
        # in sign group every element is its own inverse; check a*a = identity present
        if (a[0] * a[0], a[1] * a[1], a[2] * a[2]) != (1, 1, 1):
            inverses = False
    return {
        "n_wings": len(triples),
        "distinct_triples": len(triple_set),
        "sy_values": sy,            # if only [1], the y-axis is fixed -> not full 2^3
        "sx_values": sx,
        "sz_values": sz,
        "closed_under_mult": closed,
        "has_identity": has_identity,
        "self_inverse": inverses,
        "is_group": closed and has_identity and inverses,
        "group_order_if_group": len(triple_set),
    }


# ----------------------------------------------------------------------------
# (b)+(c) RECOVERY. The code "word" for a chain is its 32 (X,Y) pairs + the
#     shared zeta. Question: from a corrupted/partial set of chambers, recover
#     the chain (i.e. recover zeta=sum and the chain values), and detect/correct
#     corrupted chambers.
#
#  The chain is recoverable from any single uncorrupted chamber IF we can invert
#  wing_transform. We test recovery via the conserved zeta + the wing/branch
#  redundancy: each of the 32 chambers independently encodes zeta=sum(chain).
# ----------------------------------------------------------------------------
def recover_zeta_majority(corrupted_orbit):
    """Each chamber reports a zeta. True zeta = majority vote. Returns (zeta, votes)."""
    zetas = [round(z, 6) for (x, y, z) in corrupted_orbit.values()]
    from collections import Counter
    c = Counter(zetas)
    best, count = c.most_common(1)[0]
    return best, count, len(zetas)


def test_erasure(trials=300):
    """How many of 32 chambers can we ERASE and still recover zeta?
    Since every chamber carries zeta independently, 1 survivor suffices.
    We confirm: recover zeta from k survivors for k=1..32."""
    ok_at_k = {k: 0 for k in range(1, 33)}
    for _ in range(trials):
        chain = rand_chain(random.randint(3, 6))
        n = (chain[0] + chain[-1]) / 2.0
        orb = orbit(chain, n)
        true_zeta = round(sum(chain), 6)
        slots = list(orb.keys())
        random.shuffle(slots)
        for k in range(1, 33):
            survivors = {s: orb[s] for s in slots[:k]}
            zeta_vals = {round(z, 6) for (x, y, z) in survivors.values()}
            if len(zeta_vals) == 1 and list(zeta_vals)[0] == true_zeta:
                ok_at_k[k] += 1
    return ok_at_k, trials


def test_corruption(trials=300, n_corrupt_levels=(1, 5, 10, 14, 15, 16, 20)):
    """Corrupt c of 32 chambers by perturbing their reported zeta to a wrong value.
    Detection: any disagreement in zeta votes -> corruption flagged.
    Correction: majority vote on zeta recovers true sum.
    Report detection rate + correction rate vs number corrupted."""
    results = {}
    for c in n_corrupt_levels:
        detected = 0
        corrected = 0
        for _ in range(trials):
            chain = rand_chain(random.randint(3, 6))
            n = (chain[0] + chain[-1]) / 2.0
            orb = orbit(chain, n)
            true_zeta = round(sum(chain), 6)
            orb2 = dict(orb)
            slots = list(orb2.keys())
            random.shuffle(slots)
            for s in slots[:c]:
                x, y, z = orb2[s]
                # corrupt zeta to a distinct wrong value
                orb2[s] = (x, y, z + random.choice([1, 2, 3, -1, -2]))
            zeta_vals = {round(z, 6) for (x, y, z) in orb2.values()}
            if len(zeta_vals) > 1:
                detected += 1  # disagreement -> corruption detected
            zeta_hat, count, total = recover_zeta_majority(orb2)
            if zeta_hat == true_zeta:
                corrected += 1
        results[c] = {
            "detect_rate": detected / trials,
            "correct_rate": corrected / trials,
            "majority_threshold": "needs <16 corrupted for majority",
        }
    return results, trials


# ----------------------------------------------------------------------------
# (c-strong) FULL chain recovery from chambers: does (X,Y) of chambers + zeta
#     pin down the chain uniquely? Test: do two DIFFERENT chains ever produce the
#     same full 32-orbit (collision)? If never, the orbit is an injective code.
# ----------------------------------------------------------------------------
def signature(orb):
    return tuple(sorted((round(x, 6), round(y, 6)) for (x, y, z) in orb.values()))


def test_injectivity(trials=2000):
    seen = {}
    collisions = 0
    examples = []
    for _ in range(trials):
        k = random.randint(3, 6)
        chain = rand_chain(k)
        n = (chain[0] + chain[-1]) / 2.0
        orb = orbit(chain, n)
        sig = (round(sum(chain), 6), signature(orb))
        if sig in seen and seen[sig] != tuple(chain):
            collisions += 1
            if len(examples) < 5:
                examples.append((seen[sig], tuple(chain)))
        seen[sig] = tuple(chain)
    return collisions, trials, examples, len(seen)


if __name__ == "__main__":
    print("=" * 70)
    print("(0) STRUCTURE: 32 distinct vectors, conserved zeta")
    mean_d, min_d, max_d, zcons, T = test_structure()
    print(f"  distinct vectors per orbit: mean={mean_d:.2f} min={min_d} max={max_d}")
    print(f"  zeta conserved & == sum(chain): {zcons}/{T}")

    print("=" * 70)
    print("(a) GROUP structure of the wing/branch flips")
    g = test_group()
    for k, v in g.items():
        print(f"  {k}: {v}")

    print("=" * 70)
    print("(b/c) ERASURE: recover zeta from k survivors")
    ok, T = test_erasure()
    # report the smallest k that always works and a few points
    for k in [1, 2, 4, 8, 16, 32]:
        print(f"  k={k:2d} survivors -> zeta recovered {ok[k]}/{T}")

    print("=" * 70)
    print("CORRUPTION: detect + correct vs #corrupted chambers (of 32)")
    res, T = test_corruption()
    for c, r in res.items():
        print(f"  corrupted={c:2d}: detect={r['detect_rate']*100:5.1f}%  "
              f"correct(majority)={r['correct_rate']*100:5.1f}%")

    print("=" * 70)
    print("(c-strong) INJECTIVITY: do distinct chains ever share a 32-orbit?")
    coll, T, ex, uniq = test_injectivity()
    print(f"  collisions: {coll}/{T} trials   unique orbits seen: {uniq}")
    if ex:
        print(f"  example collisions: {ex}")
