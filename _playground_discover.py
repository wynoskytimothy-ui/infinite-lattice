#!/usr/bin/env python3
"""Playground discovery — emergent AETHOS math patterns. No retrieval agenda."""
from __future__ import annotations

import itertools
import math
import random
from collections import Counter, defaultdict
from fractions import Fraction

from aethos_complex_plane import (
    equalize_witness,
    swap_meet,
    triple_equalization,
    wing_transform,
    wing_transform_lid,
)
from aethos_lattice import BranchKind, LatticeBank32, LatticeId, VECTORS, lattice_id_parts
from aethos_recursive import LatticeBank32K, normalize_primes
from aethos_sequences import sum_chain
from lattice_retriever_v1.k_meet import compose_k, velocity_meet, slide_meet

rng = random.Random(42)

# --- ground meet (user formula) ---
def meet2(a: int, p: int) -> tuple[int, int, int]:
    s = a + p
    return (s, min(a, p), s)


def unmeet(X: int, Y: int) -> tuple[int, int]:
    return Y, X - Y


def primes_upto(n: int) -> list[int]:
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


print("=" * 72)
print("PROBE 1: 32-wing invariants (swap_meet holds on ALL wings?)")
print("=" * 72)
swap_ok = 0
zeta_invariant = 0
y_sign_pattern: dict[int, set[int]] = defaultdict(set)
for lid in LatticeId:
    branch, vec = lattice_id_parts(lid)
    for a, p in [(3, 11), (5, 97), (7, 13)]:
        left = wing_transform_lid((a,), p, lid)
        right = wing_transform_lid((p,), a, lid)
        if left.coord == right.coord:
            swap_ok += 1
        if abs(left.zeta - (a + p)) < 1e-9:
            zeta_invariant += 1
        y_sign_pattern[lid].add(int(math.copysign(1, left.z.imag or 1)))
print(f"  swap_meet unified: {swap_ok}/{32*3} wing×pair trials")
print(f"  zeta == a+p always: {zeta_invariant}/{32*3}")
# VA vs VB: zeta flip?
va_z, vb_z = [], []
for w in range(1, 5):
    psi_va = wing_transform(BranchKind.VA1, (5,), 7, w)
    psi_vb = wing_transform(BranchKind.VA1, (5,), 7, w + 4)  # v5..v8 = VB
    va_z.append(psi_va.zeta)
    vb_z.append(psi_vb.zeta)
print(f"  VA1 zeta spread (w1..w4): {set(va_z)}  VB5..8 zeta: {set(vb_z)}")
print(f"  INVARIANT: zeta = a+n on solo chain; X+Z coupling on VA1: X==Z always? ", end="")
xs = [wing_transform(BranchKind.VA1, (p,), 7, 1).coord for p in primes_upto(50)[2:]]
print(all(c[0] == c[2] for c in xs))

print("\n" + "=" * 72)
print("PROBE 2: chain composition k=3,4,5 — numbers appearing ONLY at composition")
print("=" * 72)
for k in (3, 4, 5):
    A = tuple(sorted(rng.sample(primes_upto(80)[2:], k)))
    solo_vals = set()
    for x in A:
        solo_vals.add(meet2(x, A[0])[0])
        solo_vals.add(meet2(x, A[-1])[1])
    comp_vals = set()
    for drop in A:
        S = tuple(x for x in A if x != drop)
        _, psi = equalize_witness(A, S)
        comp_vals.add(round(psi.zeta))
        comp_vals.add(round(psi.z.real))
        comp_vals.add(round(psi.z.imag))
    only_comp = comp_vals - solo_vals
    print(f"  k={k} chain {A}: zeta={sum(A)}  composition-only values (sample): "
          f"{sorted(only_comp)[:12]}{'...' if len(only_comp)>12 else ''}  count={len(only_comp)}")

print("\n" + "=" * 72)
print("PROBE 3: meet-key collision at scale (unordered pairs -> meet2 key)")
print("=" * 72)
for bound in (50, 100, 200):
    keys: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    ps = primes_upto(bound)
    for a, p in itertools.combinations(ps, 2):
        keys[meet2(a, p)].append((a, p))
    collisions = sum(1 for v in keys.values() if len(v) > 1)
    max_clique = max(len(v) for v in keys.values())
    print(f"  primes<={bound}: {len(keys)} unique keys, {collisions} colliding keys, "
          f"max fanout={max_clique}")

print("\n" + "=" * 72)
print("PROBE 4: symmetry — swap a↔p, VA↔VB branch swap")
print("=" * 72)
# a↔p: meet is symmetric by construction
sym_ap = all(meet2(a, p) == meet2(p, a) for a in range(2, 40) for p in range(2, 40))
print(f"  meet2(a,p)==meet2(p,a) always: {sym_ap}")
# branch swap: VA1 vs VA2 at same (A,n) — relation?
A = (3, 5, 7)
n = 5
va1 = wing_transform(BranchKind.VA1, A, n, 1)
va2 = wing_transform(BranchKind.VA2, A, n, 1)
print(f"  VA1@n=5: z={va1.z.real}+{va1.z.imag}i  VA2: z={va2.z.real}+{va2.z.imag}i")
print(f"  VA2 Y = -VA1 Y? {abs(va2.z.imag + va1.z.imag) < 1e-9}  zeta equal? {va1.zeta==va2.zeta}")
# VB = YXZ swap of VA
va = wing_transform(BranchKind.VA1, (5,), 7, 1).coord
vb = wing_transform(BranchKind.VA1, (5,), 7, 5).coord  # v5 = VB family
print(f"  VB is (Y,X,Z) swap of VA: {vb == (va[1], va[0], va[2])}")

print("\n" + "=" * 72)
print("PROBE 5: hidden primes — appear as min() never as solo anchor")
print("=" * 72)
hidden_min = Counter()
appear_anchor = set(primes_upto(200)[2:])
for a, p in itertools.combinations(primes_upto(100)[2:], 2):
    hidden_min[meet2(a, p)[1]] += 1
never_anchor = [p for p in primes_upto(100)[2:] if p not in appear_anchor]
# primes that ONLY appear as min in meets from anchor set {primes as operands}
as_min_only = []
for p in primes_upto(100)[2:]:
  as_operand = p in appear_anchor
  as_min = hidden_min[p] > 0
  if as_min and not as_operand:
    as_min_only.append(p)
print(f"  (trivial) min-only from fixed anchor set: {as_min_only[:5]}")
# richer: values that appear as Y=min but never as X=sum when scanning pairs up to N
sum_vals, min_vals = set(), set()
for a, p in itertools.combinations(range(2, 80), 2):
    X, Y, _ = meet2(a, p)
    sum_vals.add(X)
    min_vals.add(Y)
min_never_sum = sorted(min_vals - sum_vals)[:20]
print(f"  integers 2..79: appear as min but NEVER as sum: count={len(min_vals-sum_vals)}  e.g. {min_never_sum}")

print("\n" + "=" * 72)
print("PROBE 6: hypergraph — primes as nodes, meet keys as hyperedges")
print("=" * 72)
ps = primes_upto(40)[2:]
edge_by_key: dict[tuple, tuple] = {}
node_degree = Counter()
for combo in itertools.combinations(ps, 3):
    rep = compose_k(*combo)
    if rep.full_sunflower_unified:
        key = rep.sub_sunflowers[0].coord
        edge_by_key[key] = combo
        for p in combo:
            node_degree[p] += 1
print(f"  C(12,3)=220 triples: {len(edge_by_key)} distinct sunflower nodes")
top = node_degree.most_common(5)
print(f"  hottest nodes (in most unified triples): {top}")
# connectivity: share a meet node?
shared = 0
keys = list(edge_by_key.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        c1, c2 = keys[i], keys[j]
        if c1[0] == c2[0] or c1[2] == c2[2]:  # same zeta shell
            shared += 1
print(f"  triples sharing zeta or X among 220: {shared} pairs (zeta-shell clustering)")

print("\n" + "=" * 72)
print("PROBE 7: n-rail walk from 1→N — repeated meet points / attractors")
print("=" * 72)
def rail_meet_orbit(a0: int, p0: int, steps: int = 20):
    a, p = a0, p0
    path = [meet2(a, p)]
    for _ in range(steps):
        X, Y, _ = meet2(a, p)
        a, p = X, Y  # feed meet output back as next (a,p)
        path.append(meet2(a, p))
    return path

for seed in [(3, 5), (2, 7), (11, 13), (1, 1)]:
    path = rail_meet_orbit(*seed, 15)
    Xs = [t[0] for t in path]
    Ys = [t[1] for t in path]
    print(f"  seed {seed}: X=[{Xs[0]},{Xs[1]},{Xs[2]}...->{Xs[-1]}]  Y frozen? {len(set(Ys))==1}  Y={Ys[0]}")

print("\n" + "=" * 72)
print("PROBE 8: 1100↔1 complement at scale (bit / parity lens)")
print("=" * 72)
# User pattern: binary 1100 (12) ↔ 1 under meet parity / complement
for scale in (4, 8, 16, 32):
    # test: (a,p) with a=2^scale-4 (1100...0) and p=1 -> meet structure
    a = (1 << scale) - 4  # ...1100
    p = 1
    X, Y, Z = meet2(a, p)
    # parity XOR pattern
    par = (X ^ Y) & ((1 << scale) - 1)
    print(f"  scale {scale}: a={a} ({bin(a)}), p=1 -> X={X} Y={Y}  (X^Y)&mask={par}  "
          f"Y==1? {Y==1}")
# general: min(a,1)=1 always; X=a+1. Complement: ~a + 1 = -a-1 mod 2^k?
for k in range(3, 10):
    mask = (1 << k) - 1
    a = int("1100" + "0" * (k - 4), 2) if k >= 4 else 12 & mask
    X, Y, _ = meet2(a, 1)
    comp = (~a) & mask
    print(f"  k={k}: a={a:0{k}b} meet(_,1)->Y={Y}  ~a&mask={comp:b}  Y==1 invariant: {Y==1}")

print("\n" + "=" * 72)
print("PROBE 9: 32-orbit structure — distinct coords per branch row")
print("=" * 72)
A = (3, 5, 7)
n = 5
orbit = {}
for b in BranchKind:
    for w in range(1, 9):
        psi = wing_transform(b, A, n, w)
        orbit[(b.name, w)] = psi.coord
print(f"  chain {A} n={n}: {len(set(orbit.values()))}/32 distinct")
# Klein 4 from wing flips?
for b in BranchKind:
    zs = [orbit[(b.name, w)][0] for w in range(1, 9)]
    print(f"    {b.name}: X coords = {sorted(set(zs))}")

print("\n" + "=" * 72)
print("PROBE 10: velocity/slide meets for k=4,5 chains")
print("=" * 72)
for k in (4, 5):
    A = tuple(sorted(rng.sample(primes_upto(60)[2:], k)))
    vel = velocity_meet(*A)
    sli = slide_meet(*A)
    print(f"  {A}: velocity unified={vel.unified if vel else None}  "
          f"slide unified={sli.unified if sli else None}  "
          f"n_deep={vel.n_deep if vel else None}")

print("\n" + "=" * 72)
print("PROBE 11: interior zeta LOCK (0<s<k) — zeta = sum(A) independent of n")
print("=" * 72)
A = (3, 5, 7, 11)
for n in range(4, 10):
    psi = wing_transform(BranchKind.VA1, A, n, 1)
    print(f"  n={n} seg interior? {4<=n<11}  zeta={psi.zeta}  sum={sum(A)}")

print("\n" + "=" * 72)
print("PROBE 12: iterated meet = Euclidean / CF (verify)")
print("=" * 72)
def lattice_gcd(a, b):
    while min(a, b) != 0:
        X, Y, _ = meet2(a, b)
        a, b = Y, X - 2 * Y
    return max(a, b)

ok = all(lattice_gcd(a, b) == math.gcd(a, b) for a in range(1, 60) for b in range(1, 60))
print(f"  meet-subtraction GCD == math.gcd for 1..59^2: {ok}")

print("\n" + "=" * 72)
print("PROBE 13: cross-n swap ALL 32 lattices (3,11 canonical)")
print("=" * 72)
for p, q in [(3, 11), (5, 13), (7, 97)]:
    bank_p = LatticeBank32.single_prime(p)
    bank_q = LatticeBank32.single_prime(q)
    hits = sum(1 for lid in LatticeId if bank_p[lid].at(q) == bank_q[lid].at(p))
    print(f"  ({p},{q}): {hits}/32 lattices swap-unified")

print("\nDONE")
