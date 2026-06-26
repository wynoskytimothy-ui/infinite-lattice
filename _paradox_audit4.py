"""
Final probe: does the lattice's OWN promote/walk_down give a constructive,
collision-free, INVERTIBLE encoding of nested guest-lists (the thing Hilbert
scenario C actually needs)? And is the FTA round-trip exact?  Ground truth: FTA.
"""
from __future__ import annotations
import sys, itertools
import numpy as np
from sympy import factorint, prod
sys.path.insert(0, ".")
from aethos_recursive_lattice import RecursiveLattice
from core.primes import chain_primes

print("="*74)
print("PROBE D -- prime-PRODUCT (set) encoding round-trip via FTA (Godel for SETS)")
print("="*74)
# The lattice represents a guest-list (a SET of primes) as their product; FTA
# decodes it back EXACTLY. This is the squarefree-Godel set encoder.
prs = chain_primes(60)
rng = np.random.default_rng(0)
ok = 0; T = 3000; maxsz = 0
for _ in range(T):
    ksz = int(rng.integers(1, 9))
    members = sorted(set(int(x) for x in rng.choice(prs, size=ksz, replace=False)))
    code = 1
    for p in members: code *= p
    dec = sorted(factorint(code).keys())
    if dec == members: ok += 1
    maxsz = max(maxsz, ksz)
print(f"  set->product->FTA round-trip exact: {ok}/{T}  (sets up to size {maxsz})")
print("  => squarefree prime-product IS a bijection {finite prime sets} <-> {squarefree N}.")
print("     This is the genuine constructive injection. It is classical (Godel/FTA).")

print()
print("="*74)
print("PROBE E -- nested promotion: is the lattice tower itself collision-free,")
print("           and does walk_down invert promote (the constructive part)?")
print("="*74)
lat = RecursiveLattice()
for p in chain_primes(8): lat.register_base(p)
# build a LINEAR tower (walk_down stays linear) and check promote is injective
ids = [lat.promote([3,5,7], "root")]
fresh = [p for p in chain_primes(120) if p not in (3,5,7)]
for i in range(60):
    ids.append(lat.promote([ids[-1], fresh[i]], f"L{i}"))
# (i) promote assigns DISTINCT primes (no two nodes share an id)
distinct = len(set(ids)) == len(ids)
# (ii) walk_down(top) recovers EXACTLY the multiset of base leaves used
top = ids[-1]
leaves = lat.walk_down(top)
expected_leaves = sorted([3,5,7] + fresh[:60])
got = sorted(leaves)
print(f"  promote assigns distinct primes (collision-free ids): {distinct}")
print(f"  walk_down(top) leaf-set == base primes deposited: {got == expected_leaves}  "
      f"(#leaves={len(leaves)})")
# (iii) each level strictly above its children (the type-stratification, again)
mono = all(lat.resolve(c).level < lat.resolve(p).level
           for p in ids for c in lat.resolve(p).sub_chain)
print(f"  strict level monotonicity (type stratification): {mono}")
print()
print("  VERDICT on 'new vs known':")
print("   * collision-free routing: REAL but supplied by FTA/Godel, NOT by the")
print("     (X,Y,zeta) coordinate (which collides 72.8% -- that's the swap-MEET).")
print("   * invertibility: REAL on UNORDERED multisets (meet = (sum,min)); the")
print("     ordered/labelled bijection still needs Godel/Cantor.")
print("   * Russell: the level invariant = ZF foundation / 1908 type theory,")
print("     re-encoded geometrically; resolves by FORBIDDING x in x, not by a new")
print("     consistent answer under naive comprehension.")
print("DONE")
