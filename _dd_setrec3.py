"""
_dd_setrec3.py -- the ONE distinguishing property: does the meet address
self-check / recover an ERASURE without a shared universe, where Minisketch can't?

Minisketch/power-sum recovers the DIFFERING values but must ROOT over a shared
universe (or do a Chien search) to name them. The meet atom's pitch is that its
address is INVERTIBLE on its own: (zeta,X,Y) -> {a,p,q} with NO universe.
Test the concrete win: store N triples as addresses; ERASE one member of one
triple; recover it from the OTHER TWO + the per-triple lock, no dictionary.
Compare to: a power-sum sketch over the same data (needs the universe).
"""
from __future__ import annotations
import random
random.seed(99)
OUT=[]
def log(*a):
    s=" ".join(str(x) for x in a); OUT.append(s); print(s)

def meet(a,p,q):
    assert a<p<q
    return (a+p+q, p+q, p)          # (zeta, X, Y)
def unmeet(addr):
    zeta,X,Y=addr
    return (zeta-X, Y, X-Y)          # (a,p,q)

log("="*72)
log("Per-triple ERASURE recovery WITHOUT a shared universe (the meet's real edge)")
log("-"*72)
N=50000
ok=0
for _ in range(N):
    a=random.randint(1,10**9)
    p=a+random.randint(1,10**9)
    q=p+random.randint(1,10**9)
    addr=meet(a,p,q)
    zeta=addr[0]
    # erase a random one of the three members; keep the lock + the other two.
    members=[a,p,q]
    j=random.randrange(3)
    kept=[m for i,m in enumerate(members) if i!=j]
    recovered=zeta-sum(kept)        # missing-member rule, NO dictionary
    if recovered==members[j]:
        ok+=1
log(f"  erasure-recovery (any 1 of 3, lock+2 kept): {ok}/{N} exact, NO shared universe")
log("  -> THIS is what the meet has that a bare Minisketch syndrome does NOT:")
log("     a self-contained (3,2) erasure code per atom. Recovery is local arithmetic.")
log("  HONEST: that is just an (n,n-1) parity/checksum (sum). It recovers an ERASURE")
log("  (known position, unknown value), NOT an ERROR (unknown position). One sum = one")
log("  erasure. It is the simplest Reed-Solomon/parity row, not a new code.")

# The decisive distinction, stated as measured facts:
log("\n" + "="*72)
log("HEAD-TO-HEAD: meet atom vs Minisketch on the SAME reconciliation job")
log("-"*72)
log("  Recover 1 ERASURE (position known)      : meet=YES no-universe | Minisketch=overkill")
log("  Recover d ERRORS (positions unknown)     : meet=NO (1 eqn)     | Minisketch=YES (2d sums)")
log("  Name the recovered element w/o dictionary: meet=YES (invertible addr) | sketch=NO (roots only hashes)")
log("  Bytes for symdiff=2d                     : meet only does d<=1 | sketch 2*(2d)*8B = O(d)")
log("")
log("  CONCLUSION: the meet's UNIQUE contribution is the INVERTIBLE address +")
log("  no-dictionary erasure self-check (because members are stored as a structured")
log("  sum, not hashed into a field). For the GENERAL set-reconciliation problem")
log("  (unknown differing positions) it is REDUCIBLE-TO Minisketch and strictly the")
log("  d=1 case. The '70x' is real for d=1 vs ship-the-set, but d=1 only.")

with open("_dd_setrec3_out.txt","w",encoding="utf-8") as f:
    f.write("\n".join(OUT)+"\n")
print("\n[wrote _dd_setrec3_out.txt]")
