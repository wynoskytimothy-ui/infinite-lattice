"""Investigate the b2 surprise: 16 distinct z (not 32) and 2 distinct zeta (not 1)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aethos_complex_plane as cp
from aethos_lattice import BranchKind

a, p, q = 7, 11, 19
chain = sorted([a, p, q])
print("chain", chain, "sum=", cp.sum_chain(tuple(map(float, chain))))
print("%-8s %-5s %-18s %-8s" % ("branch", "wing", "z", "zeta"))
rows = []
for br in list(BranchKind):
    for wing in range(1, 9):
        n, Psi = cp.equalize_witness(chain, [chain[1], chain[2]], branch=br, wing=wing)
        rows.append((br.name, wing, Psi.z, Psi.zeta))
        print("%-8s %-5d %-18s %-8s" % (br.name, wing, Psi.z, Psi.zeta))

zset = set(r[2] for r in rows)
zetaset = set(r[3] for r in rows)
print("\ndistinct z =", len(zset), " distinct zeta =", len(zetaset), "->", sorted(zetaset))

# Is zeta conserved when n is INTERIOR? equalize_witness recovers n=missing=7 here
# (smallest), which is NOT interior of [7,11,19] -- the lock said zeta=sum only
# when n is INTERIOR. Test recovering the INTERIOR member instead.
print("\n--- recover INTERIOR member (drop the middle, n=11 is interior) ---")
inner_rows = []
for br in list(BranchKind):
    for wing in range(1, 9):
        # subset = the two OUTER members; missing = interior 11
        n, Psi = cp.equalize_witness(chain, [chain[0], chain[2]], branch=br, wing=wing)
        inner_rows.append((br.name, wing, Psi.z, Psi.zeta))
zset2 = set(r[2] for r in inner_rows)
zetaset2 = set(r[3] for r in inner_rows)
print("recovering interior n=11: distinct z =", len(zset2),
      " distinct zeta =", len(zetaset2), "->", sorted(zetaset2))

# Now the honest capacity claim: per triple, how many DISTINCT (z,zeta) chambers
# does it actually produce? Use full canon over all 32 + both recovery directions.
print("\n--- distinct (z,zeta) chambers per triple, all 32, interior recovery ---")
ch = set((r[2], r[3]) for r in inner_rows)
print("distinct (z,zeta) =", len(ch))
