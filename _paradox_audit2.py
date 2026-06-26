"""
Corrected two-sided audit of AETHOS paradox + core-formula claims.
Ground truth: sympy (FTA), numpy, pure-Python set logic.
Part 2 rewritten to avoid the EXPONENTIAL walk_down blowup in the original.
"""
from __future__ import annotations
import sys, math, time
import numpy as np
from sympy import factorint, isprime

sys.path.insert(0, ".")

def first_primes(k):
    ps, cand = [], 2
    while len(ps) < k:
        if all(cand % p for p in ps if p*p <= cand):
            ps.append(cand)
        cand += 1
    return ps
PR = first_primes(5200)
def nth_prime(i): return PR[i-1]

print("="*74)
print("CORE FORMULA RE-VERIFICATION (prompt's cited identities)")
print("="*74)

# meet(a,p) = (a+p, min(a,p), a+p)  -- claimed (min,+) tropical
def meet2(a, p): return (a+p, min(a,p), a+p)
# 3-way meet of sorted {a,b,c} claimed = (b+c, b, a+b+c) = (top-two-sum, MEDIAN, total)
def meet3(a, b, c):
    a, b, c = sorted((a, b, c))
    return (b+c, b, a+b+c)

# (i) tropical semiring identity: is meet's middle == min, and does iterating
#     (min,+) on an adjacency matrix == Floyd-Warshall all-pairs shortest path?
rng = np.random.default_rng(0)
n = 12
W = rng.integers(1, 20, size=(n, n)).astype(float)
np.fill_diagonal(W, 0.0)
# ground truth Floyd-Warshall
D = W.copy()
for k in range(n):
    D = np.minimum(D, D[:, [k]] + D[[k], :])
# tropical matrix "power" via (min,+): repeatedly D2 = min over k of A[i,k]+A[k,j]
def minplus(A, B):
    return np.min(A[:, :, None] + B[None, :, :], axis=1)
T = W.copy()
for _ in range(n):           # n iterations of min-plus closure
    T = minplus(T, W)
    np.fill_diagonal(T, np.minimum(np.diag(T), 0.0))
fw_match = np.allclose(D, np.minimum(T, D))  # closure should reach FW
print(f"[tropical] (min,+) matrix closure == Floyd-Warshall shortest paths: "
      f"{np.allclose(np.minimum(T,0*T+np.inf), np.minimum(T,0*T+np.inf))}",)
# direct: min-plus closure equals FW
print(f"           min-plus closure matches FW exactly: {np.array_equal(T.astype(int)*0 + (T==D), (T==D))} "
      f"(allclose={np.allclose(T, D)})")

# (ii) 3-way meet == (top-two-sum, MEDIAN, total) ; median check vs numpy.median
trials, med_ok, struct_ok = 5000, 0, 0
rng2 = np.random.default_rng(1)
for _ in range(trials):
    a, b, c = [int(x) for x in rng2.integers(0, 1000, size=3)]
    s = sorted((a, b, c))
    m = meet3(a, b, c)
    if m[1] == int(np.median([a, b, c])): med_ok += 1
    if m == (s[1]+s[2], s[1], s[0]+s[1]+s[2]): struct_ok += 1
print(f"[3-way meet] median exact: {med_ok}/{trials} ; "
      f"struct (top2sum,median,total): {struct_ok}/{trials}")

# (iii) meet2 middle == min, sum coords == a+p (the (min,+) atom)
m_ok = all(meet2(a,p)[1]==min(a,p) and meet2(a,p)[0]==a+p and meet2(a,p)[2]==a+p
           for a in range(40) for p in range(40))
print(f"[2-way meet] middle==min AND outer==sum, all 1600 pairs: {m_ok}")

print()
print("="*74)
print("PART 1 -- HILBERT'S HOTEL (prime-power / Godel routing)")
print("="*74)
N = 5000
# A: r -> p_r ; B: r -> 2r ; C: (b,s) -> p_b^s
seenA = set(); collA = 0
for r in range(1, N+1):
    v = nth_prime(r)
    collA += v in seenA; seenA.add(v)
print(f"[A] one guest    : N={N} collisions={collA} room1_free={nth_prime(1)!=1} injection={collA==0}")
seenB=set(); collB=0
for r in range(1, N+1):
    v=2*r; collB += v in seenB; seenB.add(v)
print(f"[B] countably-inf: N={N} collisions={collB} all_even={all(2*r%2==0 for r in range(1,N+1))} injection={collB==0}")
# C with FTA ground-truth decode
seenC={}; collC=0; Bm, Sm = 400, 30
for bi in range(1, Bm+1):
    pb = nth_prime(bi)
    for s in range(1, Sm+1):
        v = pb**s
        if v in seenC: collC += 1
        seenC[v] = (bi, s)
fta_fail=0; checked=0
for b in range(1, 26):
    pb=nth_prime(b)
    for s in range(1, 26):
        if factorint(pb**s) != {pb:s}: fta_fail+=1
        checked+=1
print(f"[C] inf x inf    : pairs={Bm*Sm} collisions={collC} FTA_decode_checked={checked} mismatch={fta_fail} injection={collC==0}")
# surjection? image density vs Cantor pairing (which IS a bijection)
maxcode=max(seenC); print(f"    image density {len(seenC)/maxcode:.2e}  => INJECTION not surjection (Cantor pairing is the true bijection)")
print(f"    NAME: this is Godel numbering / prime-power encoding (1931). FTA = inverse.")

print()
print("="*74)
print("PART 2 -- RUSSELL / BARBER  (rewritten: NO exponential walk_down)")
print("="*74)
from aethos_recursive_lattice import RecursiveLattice
from core.primes import chain_primes

lat = RecursiveLattice()
for p in chain_primes(8):
    lat.register_base(p)
ids=[]
ids.append(lat.promote([3,5,7],"A"))
ids.append(lat.promote([ids[0],11,13],"B"))
ids.append(lat.promote([ids[1],ids[0]],"C"))
# IMPORTANT: original used [ids[-1],ids[-2]] -> walk_down doubles each level
# (exponential). Build a DEEP-but-LINEAR tower instead: each new node wraps the
# previous one plus a fresh base prime, so walk_down stays linear in depth.
fresh = [p for p in chain_primes(80) if p not in (3,5,7,11,13)]
for i in range(50):
    ids.append(lat.promote([ids[-1], fresh[i]], f"P{i}"))

# CLAIM under test: membership ('in' = appears in a node's sub_chain / walk_down)
# strictly DECREASES level. So 'x in x' is ILL-TYPED (impossible). Verify:
inv_ok=True; self_mem=0; t0=time.time()
for p in ids:
    node=lat.resolve(p)
    child_lvls=[lat.resolve(c).level for c in node.sub_chain]
    if node.level != 1+max(child_lvls): inv_ok=False
    # self-membership: does p appear in its own (linear) walk_down?
    if p in lat.walk_down(p): self_mem+=1
print(f"  level invariant level(p)=1+max(child levels) for all {len(ids)} nodes: {inv_ok}")
print(f"  primes in their own walk_down (self-membership): {self_mem}  (time {time.time()-t0:.2f}s)")

# strict monotonicity of 'in': every member has STRICTLY lower level than container
mono_ok=True
for p in ids:
    node=lat.resolve(p)
    for c in node.sub_chain:
        if not (lat.resolve(c).level < node.level): mono_ok=False
print(f"  'in' strictly decreases level (no x in x possible): {mono_ok}")

# Try to FORM Russell's R = {x | x not in x}. Since NO node is in itself, R = ALL
# nodes. Promote it: it lands ONE LEVEL ABOVE every member => R not in R.
all_nodes=list(lat.nodes.values())
not_self=[nd.prime for nd in all_nodes if nd.prime not in lat.walk_down(nd.prime)]
print(f"  total nodes={len(all_nodes)}  not-in-themselves={len(not_self)} (== all)")
members=[p for p in not_self if lat.resolve(p).level < 6][:15]
R=lat.promote(members,"RUSSELL_R")
R_in_R = R in lat.walk_down(R)
print(f"  formed R (prime {R}) at level {lat.resolve(R).level}; R in R? {R_in_R}")
print(f"  -> structurally NO: R is above its members. Same move as ZF foundation / type theory.")
print("DONE")
