"""
_dd_break_meet_float.py -- pin the EXACT float53 breaking point of the meet.

The only numeric failure found is float precision (Z = a+p+q in float53).
Integers / Python bignums never lose precision (arbitrary precision).
Floats lose the low bits once a+p+q exceeds 2^53. Find the wall.
"""
import os, random, math

OUT=[]
def log(s=""): OUT.append(str(s))

def meet(a,p,q):
    a,p,q=sorted((a,p,q)); return (p+q,p,a+p+q)
def decode(X,Y,Z): return (Z-X,Y,X-Y)

log("="*70)
log("FLOAT53 wall: round-trip error vs magnitude (worst over 2000 random/decade)")
log("="*70)
log(f"  2^53 = {2**53} ~ 9.0e15  (the float53 integer-exact ceiling)")
for e in range(0,20):
    base=10.0**e
    worst=0.0
    for _ in range(2000):
        a=base*random.random(); p=base*random.random(); q=base*random.random()
        X,Y,Z=meet(a,p,q); da,dp,dq=decode(X,Y,Z)
        a,p,q=sorted((a,p,q))
        err=max(abs(da-a),abs(dp-p),abs(dq-q))
        worst=max(worst,err)
    flag = "EXACT" if worst==0 else ("<1e-3" if worst<1e-3 else "*** LOSSY ***")
    log(f"  1e{e:>2}: worst abs round-trip err = {worst:.6g}   {flag}")

log("")
log("="*70)
log("INTEGER path (Python int = arbitrary precision): same magnitudes")
log("="*70)
worst_int=0
for e in range(0,40,4):
    base=10**e
    bad=0
    for _ in range(5000):
        a=random.randint(0,base); p=random.randint(0,base); q=random.randint(0,base)
        X,Y,Z=meet(a,p,q); da,dp,dq=decode(X,Y,Z)
        if (da,dp,dq)!=tuple(sorted((a,p,q))): bad+=1
    log(f"  1e{e:>2}: integer round-trip failures = {bad}/5000")
    worst_int=max(worst_int,bad)
log(f"  -> integer path total failures across all magnitudes: {worst_int}")

log("")
log("="*70)
log("EXACT threshold: smallest a+p+q where float53 first drops a bit")
log("="*70)
# walk Z upward; first Z where (Z+1)-Z != 1 in float means we lost the unit
z=2.0**52
while (z+1.0)-z == 1.0:
    z*=2
log(f"  first power-of-two Z where float (Z+1)-Z != 1: Z = {z:.0f} = 2^{int(math.log2(z))}")
log(f"  => meet stays bit-exact in float iff a+p+q < 2^53 ({2**53}).")
log(f"     Above that, the depth lock Z aliases neighbouring sums -> COLLISIONS")
log(f"     between triples whose totals differ only in the lost low bits.")

# demonstrate an ACTUAL float collision above 2^53
log("")
log("  Constructed float collision above 2^53:")
A=2.0**53
# two triples with totals differing by 1 -- indistinguishable in float53
t1=(A, A+2, A+4)         # total 3A+6
t2=(A, A+2, A+5)         # total 3A+7  (differs by 1)
m1=meet(*t1); m2=meet(*t2)
log(f"    t1={t1} -> meet Z={m1[2]!r}")
log(f"    t2={t2} -> meet Z={m2[2]!r}")
log(f"    Z equal? {m1[2]==m2[2]}  X equal? {m1[0]==m2[0]}  -> "
    f"{'COLLISION (distinct triples, same float address)' if m1==m2 else 'distinct'}")

txt="\n".join(OUT)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"_dd_break_meet_float_out.txt"),
          "w",encoding="utf-8") as f: f.write(txt)
print(txt)
