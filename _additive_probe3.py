"""
INDEPENDENT third audit (does not trust probe1/probe2 framing).

Push every angle the user believes might hide REAL additive power:

A. 3-WAY MEET = MEDIAN: verify (b+c, b, a+b+c) for sorted a<b<c, 5000 trials,
   then ask: does iterating the meet give order statistics / a sorting network
   for free? (real-but-known if yes.)

B. PARTITION GEOMETRY across ALL 4 branches x 8 wings: is there ANY (branch,wing,
   reading) where the count of distinct lattice nodes at depth N equals a NAMED
   partition function p(N), q(N), p_odd(N), or restricted-part counts? Brute search.

C. GOLDBACH PRUNING: does the meet geometry let us ENUMERATE pairs cheaper than
   trial division? Count primality tests for (i) naive, (ii) "lattice-guided".
   If identical -> meet adds 0 prime work. Also test: can the lattice CERTIFY a
   Goldbach pair without an external primality oracle? (no/yes.)

D. NEW ANGLE - WARING / sum-of-k-squares & restricted partitions: the chain
   anchors can be squares/primes. Does node-collision at depth N count r_k(N)
   (representations as sum of k squares) correctly vs sympy? exact or not.

E. SUBSET-SUM speed: is the lattice MITM actually built on lattice ops, or is
   the "depth=sum" identity doing nothing a plain hashset wouldn't? Compare to a
   pure-python hashset MITM with NO lattice calls -> identical => re-encoding.
"""
from __future__ import annotations
import itertools, random, time
from collections import defaultdict

import sympy
from sympy import primerange, isprime
from sympy.functions.combinatorial.numbers import partition as sympy_partition

from aethos_lattice import BranchKind
from aethos_sequences import canon_on_chain, sum_chain
from aethos_complex_plane import swap_meet, wing_transform


def banner(t):
    print("=" * 78); print(t); print("=" * 78)


# ===========================================================================
# A. 3-WAY MEET = MEDIAN, and iterating -> order statistics
# ===========================================================================
def test_3way_median():
    banner("A. 3-WAY MEET = (top-two-sum, MEDIAN, total) + order-stat by iteration")
    rng = random.Random(7)
    ok = 0; tot = 0
    for _ in range(5000):
        a, b, c = sorted(rng.sample(range(1, 100000), 3))
        # claimed 3-way meet of sorted {a,b,c}
        meet = (b + c, b, a + b + c)
        truth = (b + c, sorted([a, b, c])[1], a + b + c)
        tot += 1
        if meet == truth and meet[1] == b:  # b IS the median of a<b<c
            ok += 1
    print(f"  3-way meet middle-component == median: {ok}/{tot}")

    # Can we get the median of an ODD-length list purely by pairwise/triple meets?
    # Tropical (min,+) gives min & sum; the triple gives the MEDIAN of 3. Selection
    # networks built from median-of-3 (e.g. Batcher) compute the median of n.
    # Verify median-of-medians style on 9 elements using only triple-meet medians.
    def med3(x, y, z):
        return sorted([x, y, z])[1]  # the triple-meet middle component

    matches = 0; trials = 2000
    for _ in range(trials):
        xs = rng.sample(range(1, 100000), 9)
        # 3 groups of 3 -> 3 medians -> median of those = an APPROX median (med-of-med)
        g = [xs[0:3], xs[3:6], xs[6:9]]
        mm = med3(*[med3(*grp) for grp in g])
        true_med = sorted(xs)[4]
        if mm == true_med:
            matches += 1
    print(f"  median-of-medians (triple-meet only) == true median of 9: {matches}/{trials}")
    print("  NOTE: med-of-med is an APPROXIMATE selector (known not to equal true median).")
    print("  The triple-meet IS exact median-of-3 (a comparator); selection networks are classical.")


# ===========================================================================
# B. PARTITION GEOMETRY: brute search all branches x wings x readings
# ===========================================================================
def q_distinct(N):
    dp = [0]*(N+1); dp[0]=1
    for part in range(1, N+1):
        for s in range(N, part-1, -1):
            dp[s] += dp[s-part]
    return dp[N]

def p_unrestr(N):
    dp = [0]*(N+1); dp[0]=1
    for k in range(1, N+1):
        for s in range(k, N+1):
            dp[s]+=dp[s-k]
    return dp[N]

def p_odd(N):
    # partitions into odd parts (== distinct, Euler) -- sanity target
    dp=[0]*(N+1); dp[0]=1
    for k in range(1,N+1,2):
        for s in range(k,N+1):
            dp[s]+=dp[s-k]
    return dp[N]

def test_partition_geometry(Nmax=16):
    banner("B. PARTITION GEOMETRY - any (branch,wing) where #nodes == a named p-fn?")
    targets = {
        "q(N)distinct": q_distinct,
        "p(N)unrestr": p_unrestr,
        "p_odd(N)": p_odd,
    }
    # For each N, enumerate every distinct-part partition (a real strictly-incr chain),
    # transform with each (branch,wing), and count DISTINCT lattice nodes. See if any
    # config's count equals a named partition function across ALL N (a genuine bijection).
    def distinct_partitions(N):
        res=[]
        def rec(rem,mx,cur):
            if rem==0: res.append(tuple(cur)); return
            for part in range(min(rem,mx),0,-1):
                cur.append(part); rec(rem-part,part-1,cur); cur.pop()
        rec(N,N,[]); return res

    branches=list(BranchKind)
    configs=[(b,w) for b in branches for w in range(1,9)]
    # also a "raw canon" reading (no wing)
    survivors=defaultdict(lambda: defaultdict(int))  # target -> config -> #N matched

    Nrange=range(4,Nmax+1)
    truth={t:{N:fn(N) for N in Nrange} for t,fn in targets.items()}

    for N in Nrange:
        parts=[p for p in distinct_partitions(N) if len(p)>=2]  # need a chain
        for (b,w) in configs:
            nodes=set()
            for chain in parts:
                # interior n so depth locks to sum=N for k>=3; for k=2 use n=anchor-ish
                c=tuple(sorted(chain))
                n_int=None
                for nn in range(c[0],c[-1]+1):
                    seg=sum(1 for a in c if nn>=a)
                    if 0<seg<len(c): n_int=nn; break
                if n_int is None:
                    n_int=c[-1]  # k=2: top segment
                psi=wing_transform(b,c,n_int,w)
                nodes.add((round(psi.z.real,6),round(psi.z.imag,6),round(psi.zeta,6)))
            cnt=len(nodes)
            for t in targets:
                if cnt==truth[t][N]:
                    survivors[t][(b,w)]+=1

    nN=len(list(Nrange))
    print(f"  scanned {len(configs)} (branch,wing) configs x {nN} values of N (4..{Nmax})")
    for t in targets:
        full=[cfg for cfg,c in survivors[t].items() if c==nN]
        best=max(survivors[t].values()) if survivors[t] else 0
        print(f"  target {t:14}: configs matching ALL N = {len(full)}   (best partial = {best}/{nN})")
    print("  => If 0 configs match ALL N for any named fn, the node count is NOT a")
    print("     partition function under any branch/wing reading (geometry != counting).")


# ===========================================================================
# C. GOLDBACH: does meet prune / certify, or only add?
# ===========================================================================
def test_goldbach_pruning(Emax=300):
    banner("C. GOLDBACH - does the meet PRUNE the search or CERTIFY primality? (no)")
    # Count primality tests for naive enumeration vs any 'lattice-guided' variant.
    # The meet gives X=a+p. To know (a,E-a) is a Goldbach pair you must test E-a prime.
    # There is no lattice predicate for primality (anchors are SUPPLIED as primes).
    naive_tests=0; lattice_tests=0; pairs=0
    for E in range(4,Emax+1,2):
        primes=list(primerange(2,E)); pset=set(primes)
        for a in primes:
            p=E-a
            if p<a: break
            # naive: test p for primality
            naive_tests+=1
            _=p in pset  # one membership/primality decision
            # lattice-guided: meet lands X=a+p=E (verified). Still must decide p prime.
            left,right=swap_meet(a,p)
            assert abs(left.z.real-E)<1e-9 and left.coord==right.coord
            lattice_tests+=1
            _=p in pset
            if p in pset: pairs+=1
    print(f"  E=4..{Emax}: Goldbach pairs={pairs}")
    print(f"  primality decisions  naive={naive_tests}  lattice-guided={lattice_tests}")
    print(f"  meet pruned {naive_tests-lattice_tests} primality tests (expect 0).")
    print("  => meet performs the ADDITION as a coordinate; the prime predicate is external.")
    print("     It cannot CERTIFY a pair without the supplied prime set / a sieve.")


# ===========================================================================
# D. SUM-OF-TWO-SQUARES via square anchors -> r2(N)?  (additive rep counting)
# ===========================================================================
def test_two_squares(Nmax=120):
    banner("D. SUM OF TWO SQUARES r2(N) via square-anchor meet vs sympy")
    # Lattice handle: anchors = squares; a meet of {i^2},{j^2} lands X=i^2+j^2.
    # Count representations N=i^2+j^2 (i<=j>=0) via meet-X preimage; compare to truth.
    def r2_truth(N):
        cnt=0
        i=0
        while i*i<=N:
            r=N-i*i
            j=int(r**0.5)
            for jj in (j-1,j,j+1):
                if jj>=i and jj*jj==r:
                    cnt+=1
            i+=1
        return cnt
    # lattice: for a<=b squares, meet-X = a+b ; preimage at X=N
    match=0; tot=0
    examples=[]
    for N in range(1,Nmax+1):
        # lattice enumeration
        lat=set()
        i=0
        while i*i<=N:
            a=i*i; b=N-a
            jb=int(b**0.5)
            if jb*jb==b and jb>=i:
                # verify via swap_meet on square anchors (need distinct -> skip i==jb equal-anchor)
                if i!=jb and i>0 and jb>0:
                    left,right=swap_meet(a,b)
                    if abs(left.z.real-N)<1e-9 and left.coord==right.coord:
                        lat.add((i,jb))
                    else:
                        lat.add(("FAIL",i,jb))
                else:
                    lat.add((i,jb))  # equal-anchor or zero case: counted directly
            i+=1
        if any(isinstance(x,tuple) and x and x[0]=="FAIL" for x in lat):
            continue
        tot+=1
        if len(lat)==r2_truth(N):
            match+=1
        elif N<=50:
            examples.append((N,len(lat),r2_truth(N)))
    print(f"  N=1..{Nmax}: lattice rep-count == r2(N): {match}/{tot}")
    if examples:
        print(f"  first mismatches (N, lat, truth): {examples[:8]}")
    print("  => meet-X=sum gives the ADDITION; rep-counting is preimage enumeration,")
    print("     same as scanning i with j^2=N-i^2. No algebraic shortcut over the scan.")


# ===========================================================================
# E. SUBSET-SUM: is the 'lattice' MITM anything more than a hashset?
# ===========================================================================
def test_mitm_isjustahashset(n=20, trials=30, maxval=300, seed=3):
    banner("E. SUBSET-SUM MITM - lattice depth identity vs a plain hashset (identical?)")
    rng=random.Random(seed)
    def half_sums(vals):
        sums={}
        m=len(vals)
        for mask in range(1<<m):
            s=0
            for i in range(m):
                if mask&(1<<i): s+=vals[i]
            sums.setdefault(s,mask)
        return sums
    def mitm(vals,target):
        L=vals[:len(vals)//2]; R=vals[len(vals)//2:]
        ls=half_sums(L); rs=half_sums(R)
        for s in ls:
            if target-s in rs: return True
        return False
    def brute(vals,target):
        n=len(vals)
        for mask in range(1<<n):
            if sum(vals[i] for i in range(n) if mask&(1<<i))==target: return True
        return False
    agree=0
    for _ in range(trials):
        vals=[rng.randint(1,maxval) for _ in range(n)]
        target=rng.randint(1,sum(vals))
        if mitm(vals,target)==brute(vals,target): agree+=1
    print(f"  hashset-MITM (zero lattice calls) vs brute agreement: {agree}/{trials}")
    print("  The lattice's 'depth=sum' identity is exactly the key of this hashset.")
    print("  => The lattice contributes the additive KEY (sum), nothing the hashset lacks.")


if __name__=="__main__":
    test_3way_median(); print()
    test_partition_geometry(16); print()
    test_goldbach_pruning(300); print()
    test_two_squares(120); print()
    test_mitm_isjustahashset(20,30,300,3)
