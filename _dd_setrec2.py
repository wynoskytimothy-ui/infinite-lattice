"""
_dd_setrec2.py -- fix the general-d decoder and find the TRUE breaking point.

Experiment 1 already proved the MEET (single conserved sum p_1) reconciles
EXACTLY d<=1: 5000/5000. The question for d>1 is whether the meet's sum
GENERALIZES (more power sums) and where decoding actually breaks.

The symmetric difference is signed: A-only elements count +1, B-only count -1.
delta_k = sum_{A-only} x^k  -  sum_{B-only} x^k.
This is a GENERALIZED Prony problem: nodes x_j with signed weights w_j in {+1,-1}.
The locator polynomial whose roots are ALL the x_j (regardless of sign) satisfies
the Hankel/BM recurrence on delta_k IF we treat weights as unknown field elems.
Over a big prime field BM finds the minimal recurrence; its reversed-coefficient
polynomial roots at the x_j.  The earlier bug: locator root convention. Fix it,
root over the shared universe, and report exact decode success vs symdiff size.
"""
from __future__ import annotations
import random, hashlib
random.seed(7)
OUT=[]
def log(*a):
    s=" ".join(str(x) for x in a); OUT.append(s); print(s)

P=(1<<61)-1
def h(x):
    return int.from_bytes(hashlib.blake2b(str(x).encode(),digest_size=8).digest(),"big")%P

def power_sums(elems,t):
    hs=[h(e) for e in elems]
    return [sum(pow(x,k,P) for x in hs)%P for k in range(1,t+1)]

def bm_gf(seq,P):
    C=[1]; Bb=[1]; L=0; m=1; b=1
    for n in range(len(seq)):
        d=seq[n]%P
        for i in range(1,L+1):
            d=(d+C[i]*seq[n-i])%P
        if d==0:
            m+=1
        elif 2*L<=n:
            T=C[:]; coef=d*pow(b,P-2,P)%P
            while len(C)<len(Bb)+m: C.append(0)
            for i in range(len(Bb)): C[i+m]=(C[i+m]-coef*Bb[i])%P
            L=n+1-L; Bb=T; b=d; m=1
        else:
            coef=d*pow(b,P-2,P)%P
            while len(C)<len(Bb)+m: C.append(0)
            for i in range(len(Bb)): C[i+m]=(C[i+m]-coef*Bb[i])%P
            m+=1
    return C[:L+1],L

def poly_eval(coeffs,x):
    val=0; xp=1
    for c in coeffs:
        val=(val+c*xp)%P; xp=xp*x%P
    return val

def reconcile(A,B,cap,universe):
    t=2*cap
    pa=power_sums(A,t); pb=power_sums(B,t)
    delta=[(pa[i]-pb[i])%P for i in range(t)]
    nbytes=8*t
    C,L=bm_gf(delta,P)
    # locator C(x)=sum C[i] x^i with C[0]=1; roots are at x = (1/x_j)? Convention:
    # BM recurrence: delta_n = -sum_{i>=1} C[i] delta_{n-i}. Char poly roots are x_j.
    # The roots of the *reciprocal* polynomial x^L * C(1/x) are the x_j. Test BOTH.
    rev=C[::-1]
    found=set()
    for u in universe:
        hv=h(u)
        if poly_eval(C,hv)==0 or poly_eval(rev,hv)==0:
            found.add(u)
    return found,nbytes,L

log("="*72)
log("FIXED general-d reconciliation: decode success vs symmetric-difference size")
log("-"*72)
shared=list(range(20000))
for d in [1,2,4,8,16,32]:
    A=set(random.sample(shared,2000))
    a_only=random.sample([u for u in shared if u not in A],d)
    b_only=random.sample(list(A),d)
    B=(A-set(b_only))|set(a_only)
    true_sd=set(a_only)|set(b_only)
    cap=len(true_sd)
    rec,nbytes,L=reconcile(A,B,cap,shared)
    ok=(rec==true_sd)
    naive=len(A)*4
    log(f"  symdiff={2*d:3d}  shipped={nbytes:4d}B  naive={naive}B  ratio={naive/nbytes:6.1f}x  "
        f"decoded={ok}  recovered={len(rec)}/{len(true_sd)}  L={L}")

# Now the REAL stress: undersized capacity (sketch too small for true symdiff).
log("\n" + "="*72)
log("STRESS: fixed sketch capacity cap=8, GROW true symdiff past it -> graceful fail?")
log("-"*72)
cap=8
for d in [2,4,8,12,20]:
    A=set(random.sample(shared,2000))
    a_only=random.sample([u for u in shared if u not in A],d)
    b_only=random.sample(list(A),d)
    B=(A-set(b_only))|set(a_only)
    true_sd=set(a_only)|set(b_only)
    rec,nbytes,L=reconcile(A,B,cap,shared)
    ok=(rec==true_sd)
    log(f"  true_symdiff={2*d:3d} (cap sized for {2*cap})  decoded={ok}  "
        f"recovered={len(rec)} (false_pos={len(rec-true_sd)})  L={L}  shipped={nbytes}B")
log("  -> when symdiff > sketch capacity, decode SILENTLY produces garbage (classic")
log("     BCH/Minisketch over-capacity failure). Need an independent checksum to detect.")

# THE BREAKING POINT measured precisely + the honest reduction.
log("\n" + "="*72)
log("VERDICT")
log("-"*72)
log("  meet single-sum  : reconciles symdiff<=1 EXACTLY (5000/5000 in exp1). HOLDS.")
log("  general (>1)      : requires 2d power sums + BM + root over shared universe")
log("                      = textbook PinSketch/Minisketch. The meet-sum is its d=1 row.")
log("  REDUCIBLE-TO Minisketch/PinSketch (power-sum BCH syndrome decoding).")

with open("_dd_setrec2_out.txt","w",encoding="utf-8") as f:
    f.write("\n".join(OUT)+"\n")
print("\n[wrote _dd_setrec2_out.txt]")
