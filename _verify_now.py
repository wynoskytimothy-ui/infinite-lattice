import numpy as np, random

# ---- CLAIM 1: meet is unimodular bijection (det=-1), 0 collisions ----
# encode (a,p,q) sorted a<=p, transgressor q -> (X,Y,Z) = (p+q, p, a+p+q)
def enc(a,p,q): return (p+q, p, a+p+q)
def dec(X,Y,Z): return (Z-X, Y, X-Y)  # a=Z-X, p=Y, q=X-Y
M = np.array([[0,1,1],[0,1,0],[1,1,1]])
print("det(M) =", int(round(np.linalg.det(M))))

# round-trip + collision test
seen={}; coll=0; rtfail=0
random.seed(0)
for _ in range(2_000_000):
    a=random.randint(0,10_000_000); p=random.randint(0,10_000_000); q=random.randint(0,10_000_000)
    X,Y,Z=enc(a,p,q)
    if dec(X,Y,Z)!=(a,p,q): rtfail+=1
    k=(X,Y,Z)
    if k in seen and seen[k]!=(a,p,q): coll+=1
    else: seen[k]=(a,p,q)
print(f"roundtrip_fail={rtfail}  distinct-triple_collisions={coll}  of 2,000,000")

# ---- CLAIM 2: float wall at 2^53 ----
def enc_f(a,p,q): return (float(p+q), float(p), float(a+p+q))
def dec_f(X,Y,Z): return (Z-X, Y, X-Y)
for e in [50,52,53,54,55]:
    a=2**e; p=1; q=1
    X,Y,Z=enc_f(a,p,q); r=dec_f(X,Y,Z)
    ok = (r==(float(a),float(p),float(q)))
    print(f"2^{e}: encode->decode exact? {ok}  recovered a={r[0]:.0f} (true {a})")

# ---- CLAIM 3 (red team): tamper one coordinate -> still a valid integer triple, undetected ----
random.seed(1)
caught=0; N=50000
for _ in range(N):
    a=random.randint(1,1000); p=random.randint(a,2000); q=random.randint(0,2000)
    X,Y,Z=enc(a,p,q)
    # corrupt one coordinate
    which=random.randint(0,2); delta=random.choice([-3,-2,-1,1,2,3])
    XX,YY,ZZ=(X+delta if which==0 else X, Y+delta if which==1 else Y, Z+delta if which==2 else Z)
    aa,pp,qq=dec(XX,YY,ZZ)
    # "detected" only if decode lands on non-integer or violates a<=p ordering invariant
    valid_int = all(float(v).is_integer() for v in (aa,pp,qq))
    if not valid_int: caught+=1
print(f"tamper caught by integrality alone: {caught}/{N}")

# add the ordering invariant a<=p<= (a+? ) as a weak syndrome
caught2=0
random.seed(1)
for _ in range(N):
    a=random.randint(1,1000); p=random.randint(a,2000); q=random.randint(0,2000)
    X,Y,Z=enc(a,p,q)
    which=random.randint(0,2); delta=random.choice([-3,-2,-1,1,2,3])
    XX,YY,ZZ=(X+delta if which==0 else X, Y+delta if which==1 else Y, Z+delta if which==2 else Z)
    aa,pp,qq=dec(XX,YY,ZZ)
    if not (aa<=pp and qq>=0 and aa>=0): caught2+=1
print(f"tamper caught WITH ordering+nonneg invariant: {caught2}/{N}")
