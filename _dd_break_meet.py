"""
_dd_break_meet.py  --  ADVERSARIAL stress of the AETHOS invertible 3-way meet.

Meet (canonical, from aethos_address_store._meet / aethos_complex_plane):
  for sorted a<=p<=q, subset=(p,q), transgressor n=a (missing member):
    X = p+q,  Y = p,  Z(zeta) = a+p+q
  spec inversion: a = Z - X,  p = Y,  q = X - Y
  (q = X - Y = (p+q) - p = q  OK ; a = Z - X = (a+p+q)-(p+q) = a OK)

We push to destruction:
  T1  distinct-triple -> distinct-(X,Y,Z) at MILLIONS of random triples, big magnitude
  T2  ADVERSARIAL collision construction / exact algebraic proof of injectivity
  T3  decode under negatives, zeros, a==p dups, floats, huge ints
  T4  does invertibility NEED a<=p<=q sorted -- what breaks unsorted
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aethos_complex_plane import equalize_witness, BranchKind

OUT = []
def log(s=""):
    OUT.append(str(s))

# ---- the REAL meet via library code (no re-derivation) ------------------
def meet_lib(a, p, q, branch=BranchKind.VA1, wing=1):
    chain = sorted((a, p, q))
    _, psi = equalize_witness(chain, [chain[1], chain[2]], branch=branch, wing=wing)
    x, y, z = psi.coord
    return (x, y, z)

def meet_lib_safe(a, p, q):
    """Returns None if the library rejects (duplicate anchors), else the addr."""
    try:
        return meet_lib(a, p, q)
    except ValueError:
        return None

# ---- the spec closed form (what we claim it equals) ---------------------
def meet_spec(a, p, q):
    a, p, q = sorted((a, p, q))
    return (p + q, p, a + p + q)   # (X, Y, Z)

# ---- the spec inversion -------------------------------------------------
def decode_spec(X, Y, Z):
    a = Z - X
    p = Y
    q = X - Y
    return (a, p, q)

log("="*70)
log("STEP 0: confirm library meet == spec closed form (sorted positive)")
log("="*70)
mism = 0
for _ in range(20000):
    a = random.randint(0, 10**6); p = random.randint(0, 10**6); q = random.randint(0, 10**6)
    if meet_lib(a,p,q) != meet_spec(a,p,q):
        mism += 1
        if mism <= 5:
            log(f"  MISMATCH a,p,q={sorted((a,p,q))} lib={meet_lib(a,p,q)} spec={meet_spec(a,p,q)}")
log(f"  lib-vs-spec mismatches over 20000 random sorted triples: {mism}")
log(f"  -> closed form {'CONFIRMED' if mism==0 else 'BROKEN'}; rest of probe uses fast spec form\n")

# =========================================================================
log("="*70)
log("T1: distinct-triple -> distinct-address, scaled to MILLIONS, big magnitude")
log("="*70)
for (N, MAG) in [(1_000_000, 10**3), (3_000_000, 10**6), (5_000_000, 10**7)]:
    seen = {}
    coll_distinct = 0   # two DIFFERENT triples -> same address  (real collision)
    examples = []
    t0 = time.time()
    for _ in range(N):
        a = random.randint(0, MAG); p = random.randint(0, MAG); q = random.randint(0, MAG)
        tri = tuple(sorted((a, p, q)))
        addr = meet_spec(a, p, q)
        if addr in seen:
            if seen[addr] != tri:
                coll_distinct += 1
                if len(examples) < 5:
                    examples.append((seen[addr], tri, addr))
        else:
            seen[addr] = tri
    dt = time.time() - t0
    log(f"  N={N:>9,} MAG=1e{len(str(MAG))-1}: distinct-triple collisions = {coll_distinct} "
        f"({dt:.1f}s, {len(seen):,} unique addrs)")
    for e in examples:
        log(f"      COLLISION {e[0]} and {e[1]} -> {e[2]}")
log("")

# =========================================================================
log("="*70)
log("T2: ADVERSARIAL collision construction + exact injectivity proof")
log("="*70)
log("  Try to FORCE two distinct sorted triples (a1,p1,q1)!=(a2,p2,q2)")
log("  to the SAME (X,Y,Z)=(p+q, p, a+p+q).")
log("  Algebra: Y=p fixes p. X=p+q fixes q=X-Y. Z=a+p+q fixes a=Z-X.")
log("  => (a,p,q) is a BIJECTIVE linear map of (X,Y,Z). Determinant:")
log("       [X]   [0 1 1][a]")
log("       [Y] = [0 1 0][p]   det = 1*(0*1-1*1)... compute below")
log("       [Z]   [1 1 1][q]")
import itertools
# matrix rows for (X,Y,Z) in terms of (a,p,q)
M = [[0,1,1],[0,1,0],[1,1,1]]
def det3(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
           -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
           +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))
d = det3(M)
log(f"  det(M) = {d}   -> {'INVERTIBLE (no collisions possible on Z^3)' if d!=0 else 'SINGULAR'}")
log("  Adversarial brute force: search ALL sorted triples in [0,200]^3 for a collision...")
addr_map = {}
brute_coll = 0
for a in range(0, 201):
    for p in range(a, 201):
        for q in range(p, 201):
            ad = meet_spec(a,p,q)
            if ad in addr_map and addr_map[ad] != (a,p,q):
                brute_coll += 1
                if brute_coll <= 5:
                    log(f"      BRUTE COLLISION {addr_map[ad]} vs {(a,p,q)} -> {ad}")
            else:
                addr_map[ad] = (a,p,q)
log(f"  exhaustive sorted triples in [0,200]^3 = {len(addr_map):,} addrs, collisions = {brute_coll}")
log("")

# =========================================================================
log("="*70)
log("T3: decode under negatives / zeros / a==p dups / floats / huge ints")
log("="*70)
cases = [
    ("all zero",        (0,0,0)),
    ("a==p==q",         (7,7,7)),
    ("a==p dup",        (5,5,9)),
    ("p==q dup",        (3,8,8)),
    ("one negative",    (-4, 2, 9)),
    ("two negative",    (-9,-2, 5)),
    ("all negative",    (-9,-7,-3)),
    ("float vals",      (1.5, 2.5, 9.25)),
    ("huge ints",       (10**30, 10**30+1, 10**30+99)),
    ("mixed sign+float",(-3.25, 0.0, 11.5)),
]
fail = 0
for name, raw in cases:
    a,p,q = sorted(raw)
    X,Y,Z = meet_spec(a,p,q)
    da,dp,dq = decode_spec(X,Y,Z)
    ok = (da,dp,dq) == (a,p,q)
    if not ok: fail += 1
    log(f"  {name:18s} sorted={(a,p,q)} -> meet={(X,Y,Z)} -> decode={(da,dp,dq)}  {'OK' if ok else '*** BREAK ***'}")
log(f"  decode failures (abstract closed form): {fail}/{len(cases)}")
log("  Now the SAME cases through the LIBRARY meet (equalize_witness):")
libfail = 0; librej = 0
for name, raw in cases:
    a,p,q = sorted(raw)
    res = meet_lib_safe(a,p,q)
    if res is None:
        librej += 1
        log(f"  {name:18s} sorted={(a,p,q)} -> LIBRARY REJECTS (duplicate/invalid anchors)")
        continue
    X,Y,Z = res
    da,dp,dq = decode_spec(X,Y,Z)
    ok = abs(da-a)<1e-9 and abs(dp-p)<1e-9 and abs(dq-q)<1e-9
    if not ok: libfail += 1
    log(f"  {name:18s} sorted={(a,p,q)} -> LIB meet={(X,Y,Z)} decode={(da,dp,dq)}  {'OK' if ok else '*** BREAK ***'}")
log(f"  library: rejected={librej}, decode-failures={libfail}")
# float exactness stress: does float meet round-trip at large magnitude?
log("  float magnitude stress (does float arithmetic stay exact?):")
fbreak = 0
for e in [3, 6, 9, 12, 15, 17, 18, 20]:
    base = 10.0**e
    a,p,q = sorted((base+0.5, base+1.5, base+9.25))
    X,Y,Z = meet_spec(a,p,q)
    da,dp,dq = decode_spec(X,Y,Z)
    ok = abs(da-a)<1e-6 and abs(dp-p)<1e-6 and abs(dq-q)<1e-6
    if not ok: fbreak += 1
    log(f"    1e{e:>2}: roundtrip {'EXACT' if ok else 'LOST PRECISION'}  (err a={da-a:.3g})")
log(f"  float precision breaks: {fbreak}")
log("")

# =========================================================================
log("="*70)
log("T4: does invertibility NEED a<=p<=q sorted? break unsorted")
log("="*70)
log("  meet_spec SORTS internally, so encode is order-free. The question is")
log("  whether the meet PRESERVES which input was which (does it remember order?).")
log("  Test: feed 6 permutations of the SAME multiset {a,p,q}.")
for tri in [(3,8,20),(1,1,5),(0,4,4)]:
    addrs = set()
    for perm in itertools.permutations(tri):
        addrs.add(meet_spec(*perm))
    log(f"  multiset {tri}: distinct addrs over 6 perms = {len(addrs)}  "
        f"-> {'order LOST (meet is symmetric, decode gives sorted only)' if len(addrs)==1 else 'order kept'}")
log("  CONSEQUENCE: meet is a SET operator. If the application needs the ordered")
log("  tuple (who is a vs q), the meet alone CANNOT recover it -- only the sorted")
log("  multiset. decode_spec returns sorted (a,p,q); original permutation is GONE.")
log("")
log("  Now: what if we DON'T sort before decode? Feed raw (a,p,q) to meet_spec")
log("  without sorting and try the literal closed form X=p+q,Y=p,Z=a+p+q (no sort):")
def meet_nosort(a,p,q): return (p+q, p, a+p+q)
nfail=0
for _ in range(100000):
    a=random.randint(-50,50); p=random.randint(-50,50); q=random.randint(-50,50)
    X,Y,Z = meet_nosort(a,p,q)
    da,dp,dq = decode_spec(X,Y,Z)  # a=Z-X, p=Y, q=X-Y
    if (da,dp,dq)!=(a,p,q): nfail+=1
log(f"  UNSORTED literal-form roundtrip failures over 100000: {nfail}")
log("  -> If you keep the SAME no-sort closed form on both sides, decode is")
log("     EXACT even unsorted/negative. Sorting is a CONVENTION for the lattice")
log("     read, NOT a requirement of the linear algebra.")
log("")
log("  KEY ADVERSARIAL POINT: the canonical lattice meet (equalize_witness)")
log("  REQUIRES n=a be the MIN (missing member of the (p,q) subset). If a is NOT")
log("  the min, the 'missing member' picked changes -> different X,Y. Test that:")
disagree = 0
ex=[]
dup_rejected = 0
for _ in range(50000):
    a=random.randint(0,1000); p=random.randint(0,1000); q=random.randint(0,1000)
    # library always sorts; literal no-sort form does not
    lib = meet_lib_safe(a,p,q)
    if lib is None:   # library rejected duplicate anchors
        dup_rejected += 1
        continue
    lit = meet_nosort(a,p,q)
    if lib != lit:
        disagree += 1
        if len(ex)<5: ex.append(((a,p,q), lib, lit))
log(f"  library REJECTED (duplicate-anchor ValueError): {dup_rejected}/50000")
log(f"  library(sorted) vs literal(unsorted) disagree: {disagree}/50000")
for e in ex:
    log(f"      {e[0]}: lib={e[1]} literal={e[2]}")
log("  -> The LATTICE meet is only invertible-as-specified when fed sorted input;")
log("     feed it unsorted and X,Y are computed off the wrong median -> the spec")
log("     decode a=Z-X,p=Y,q=X-Y returns WRONG members. Sorting IS load-bearing")
log("     for the lattice path (not for the abstract linear map).")

txt = "\n".join(OUT)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dd_break_meet_out.txt"),
          "w", encoding="utf-8") as f:
    f.write(txt)
print(txt)
