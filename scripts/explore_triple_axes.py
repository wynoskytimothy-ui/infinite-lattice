"""
Build HIGHER DIMENSIONS by COMPOSING 3-way meets.

Vision: each triple = one orthogonal axis ("90 degrees away"). An item is a SET
of triples. Each triple's MEET-NODE (conserved sum + its 32-orbit chamber vector)
is one coordinate. Test three concrete questions:

  (a) encode a D-dim vector as D triples -> recover all D coords independently?
  (b) how many independent axes coexist before collisions force ambiguity?
  (c) is each triple-axis truly orthogonal (independent) of the others?

The meet of triple (a,p,q) lives at node n=missing, with z and zeta=sum(a,p,q).
KEY mechanic verified: any 2 of 3 recover the 3rd (erasure code), and all 3
drop-one witnesses collide on ONE node. So a triple is a self-checking address.
"""
import itertools
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aethos_complex_plane as cp
from aethos_lattice import BranchKind

random.seed(0)


def triple_meet(a, p, q, branch=BranchKind.VA1, wing=1):
    """Return the canonical meet-node of triple {a,p,q}: (z, zeta, missing-recovers)."""
    chain = sorted([a, p, q])
    # zeta (depth) is the conserved invariant = sum
    s = cp.sum_chain(tuple(float(x) for x in chain))
    # The meet z: use any drop-one witness (all 3 collide). Recover each member.
    recovered = []
    for drop in range(3):
        subset = [chain[i] for i in range(3) if i != drop]
        miss = cp.missing_member(chain, subset)
        recovered.append(miss)
    n, Psi = cp.equalize_witness(chain, [chain[1], chain[2]], branch=branch, wing=wing)
    return Psi.z, Psi.zeta, sorted(recovered)


def encode_vector(values, base=1000):
    """Encode a D-dim integer vector as D triples.

    Axis i carries coordinate value v_i. We pick a triple {a,p,q} on a private
    integer band for axis i so triples on different axes live on disjoint number
    ranges (the 'orthogonal' construction). The coordinate is read from the
    conserved sum (zeta) of the triple, offset-decoded against the band base.
    """
    triples = []
    for i, v in enumerate(values):
        band = (i + 1) * base  # private band start for axis i
        # represent v: a=band, p=band+1, q=band+1+v  => sum encodes v, all distinct
        a, p, q = band, band + 1, band + 2 + v
        triples.append((i, band, (a, p, q)))
    return triples


def decode_vector(triples):
    """Recover each coordinate independently from its triple meet (no cross-talk)."""
    out = {}
    for i, band, (a, p, q) in triples:
        z, zeta, recovered = triple_meet(a, p, q)
        # zeta = a+p+q = band + (band+1) + (band+2+v) = 3*band + 3 + v
        v = int(round(zeta)) - (3 * band + 3)
        out[i] = (v, z, zeta)
    return out


print("=" * 70)
print("(a) ENCODE D-dim vector as D triples, recover all D coords independently")
print("=" * 70)
D = 8
true_vec = [random.randint(0, 50) for _ in range(D)]
triples = encode_vector(true_vec)
dec = decode_vector(triples)
rec_vec = [dec[i][0] for i in range(D)]
print("true :", true_vec)
print("recov:", rec_vec)
ok = (true_vec == rec_vec)
print("ALL %d coords recovered independently: %s" % (D, ok))
print("(each axis decoded ONLY from its own triple's meet -- no other axis read)")

print()
print("=" * 70)
print("(c) ORTHOGONALITY: change ONE axis -> do OTHER meet-nodes move?")
print("=" * 70)
# Perturb axis 3, re-decode, check axes != 3 are byte-identical (z, zeta)
base_nodes = {i: (dec[i][1], dec[i][2]) for i in range(D)}
true_vec2 = list(true_vec)
true_vec2[3] += 17  # bump axis 3
dec2 = decode_vector(encode_vector(true_vec2))
moved = []
for i in range(D):
    z1, zeta1 = base_nodes[i]
    z2, zeta2 = dec2[i][1], dec2[i][2]
    if (z1, zeta1) != (z2, zeta2):
        moved.append(i)
print("perturbed axis 3 by +17")
print("axes whose meet-node MOVED:", moved)
print("orthogonal (only axis 3 moved):", moved == [3])
print("axis-3 coord still correct:", dec2[3][0] == true_vec2[3])

print()
print("=" * 70)
print("(b) CAPACITY: how many axes coexist before meet-NODES collide?")
print("=" * 70)
# Two regimes:
#  REGIME 1 (disjoint bands): nodes never collide by construction -> unbounded.
#  REGIME 2 (shared small integer pool): triples drawn from pool 1..M, a meet
#  node is identified by (z, zeta). Count distinct triples that map to DISTINCT
#  meet-nodes. Collision = two different triples landing on the same (z,zeta).
for M in [20, 40, 80, 160, 320]:
    pool = list(range(1, M + 1))
    seen = {}
    n_triples = 0
    n_collide = 0
    for a, p, q in itertools.combinations(pool, 3):
        z, zeta, _ = triple_meet(a, p, q)
        key = (z, zeta)
        n_triples += 1
        if key in seen:
            n_collide += 1
        else:
            seen[key] = (a, p, q)
    distinct_nodes = len(seen)
    print("pool 1..%-4d : %7d triples -> %7d DISTINCT meet-nodes, %6d collisions (%.1f%% unique)"
          % (M, n_triples, distinct_nodes, n_collide, 100.0 * distinct_nodes / n_triples))

print()
print("=" * 70)
print("(b2) WITH 32-orbit chambers: each node x 32 (4 branch x 8 wing) -> capacity x32?")
print("=" * 70)
# Does putting the same triple through all 32 chambers give 32 DISTINCT z's
# (sharing zeta)? If so each meet-node fans out to 32 sub-addresses.
a, p, q = 7, 11, 19
chain = sorted([a, p, q])
chambers = set()
zetas = set()
for br in list(BranchKind):
    for wing in range(1, 9):
        n, Psi = cp.equalize_witness(chain, [chain[1], chain[2]], branch=br, wing=wing)
        chambers.add(Psi.z)
        zetas.add(Psi.zeta)
print("triple {7,11,19}: distinct z across 32 chambers =", len(chambers),
      "| distinct zeta =", len(zetas), "(should be 1 conserved)")

print()
print("=" * 70)
print("(b3) COLLISION-FREE CAPACITY of (z, zeta, branch, wing) full address")
print("=" * 70)
for M in [40, 80, 160]:
    pool = list(range(1, M + 1))
    seen = set()
    n_triples = 0
    n_collide = 0
    for a, p, q in itertools.combinations(pool, 3):
        chain = sorted([a, p, q])
        for br in list(BranchKind):
            for wing in range(1, 9):
                n, Psi = cp.equalize_witness(chain, [chain[1], chain[2]], branch=br, wing=wing)
                key = (Psi.z, Psi.zeta, br, wing)
                n_triples += 1
                if key in seen:
                    n_collide += 1
                else:
                    seen.add(key)
    print("pool 1..%-4d x32 chambers: %8d addresses -> %8d DISTINCT, %6d collisions (%.2f%% unique)"
          % (M, n_triples, len(seen), n_collide, 100.0 * len(seen) / n_triples))
