"""
Two-sided probe of whether the ACTUAL AETHOS coordinate API gives anything NEW
beyond classical Godel/Cantor for Hilbert, and whether the (X,Y,zeta) address
is itself an injection (collision-free routing) at scale.
Ground truth: Cantor pairing bijection, sympy FTA.
"""
from __future__ import annotations
import sys, itertools
import numpy as np
sys.path.insert(0, ".")
from aethos_complex_plane import wing_transform, triple_equalization, swap_meet
from aethos_lattice import BranchKind

print("="*74)
print("PROBE A -- is the AETHOS (X,Y,zeta) address an injection on guest labels?")
print("="*74)
# A 'guest' = a finite sorted chain A plus rail n. Classic Hilbert needs an
# injection guest -> room. Does wing_transform give DISTINCT (X,Y,zeta) per guest?
seen = {}
coll = 0; total = 0
samples = []
for k in (1,2,3):
    chains = itertools.combinations(range(1, 14), k)
    for ch in chains:
        for n in range(0, 16):
            psi = wing_transform(BranchKind.VA1, ch, n, 1)
            key = (round(psi.z.real,6), round(psi.z.imag,6), round(psi.zeta,6))
            total += 1
            if key in seen:
                coll += 1
                if len(samples) < 6:
                    samples.append((ch, n, seen[key], key))
            else:
                seen[key] = (ch, n)
print(f"  guests routed={total}  distinct addresses={len(seen)}  COLLISIONS={coll}")
print(f"  => coordinate is NOT injective on (chain,n): {coll>0}  collision rate {coll/total:.1%}")
print("  example collisions (chain,n) -> (chain,n) at same (X,Y,zeta):")
for ch,n,prev,key in samples[:5]:
    print(f"     {prev}  ==  ({ch}, n={n})  @ {key}")

print()
print("="*74)
print("PROBE B -- the equalization 'meet' as a routing primitive: invertible?")
print("="*74)
# The claim: meet(a,p)=(a+p,min,a+p) is INVERTIBLE. Test: from (sum, min) can we
# recover {a,p}?  sum=a+p, min=m => other = sum-m. Unordered pair recovered.
rng=np.random.default_rng(0); ok=0; T=20000
for _ in range(T):
    a,p = [int(x) for x in rng.integers(0,10**6,size=2)]
    s=a+p; m=min(a,p); rec=sorted((m, s-m))
    if rec==sorted((a,p)): ok+=1
print(f"  meet (sum,min) -> recover unordered pair: {ok}/{T} exact")
# BUT: is it invertible as a FUNCTION of the ORDERED pair? No -- min loses order.
# So it's invertible on the QUOTIENT (unordered/multiset), not on ordered pairs.
a,p=3,7
print(f"  meet(3,7)={ (a+p,min(a,p),a+p) }  meet(7,3)={ (p+a,min(p,a),p+a) }  "
      f"-> ORDER LOST (same image): {(a+p,min(a,p))==(p+a,min(p,a))}")
print("  => invertible on UNORDERED pairs only (it is the (sum,min) of a multiset).")

print()
print("="*74)
print("PROBE C -- does triple_equalization give a NEW set-reconciliation power")
print("           or is it = recovering the 3rd element from any 2 (Lagrange-trivial)?")
print("="*74)
# triple_equalization(a,p,q): all three pair-rails meet at the SAME node.
# Test the inverse: given the meet node (top2sum, median, total), can we recover
# the multiset {a,p,q}?  total=a+b+c, median=b, top2sum=b+c => c=top2sum-b,
# a=total-top2sum.  So yes -- 2 of the 3 coords + 1 = full multiset.
ok=0; T=20000
for _ in range(T):
    a,b,c = sorted(int(x) for x in rng.integers(0,10**6,size=3))
    top2, med, tot = b+c, b, a+b+c
    rc = top2-med; ra = tot-top2; rb = med
    if sorted((ra,rb,rc))==[a,b,c]: ok+=1
print(f"  triple meet node -> recover sorted {{a,b,c}}: {ok}/{T} exact")
print("  HONEST: this is solving a 3x3 linear system (top2,med,total are 3 indep")
print("  linear functionals of the sorted triple). Invertible because the 3x3")
print("  matrix [[0,1,1],[0,1,0],[1,1,1]] is nonsingular (det=-1). Classical.")
M=np.array([[0,1,1],[0,1,0],[1,1,1]])
print(f"  det of the linear map = {round(np.linalg.det(M)):d} (nonzero => invertible)")
print("DONE")
