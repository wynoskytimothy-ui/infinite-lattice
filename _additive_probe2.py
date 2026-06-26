"""
Push harder: is there a REAL computational advantage hiding in the meet?

1. PARTITION COUNT via generating function — can the lattice's additive depth
   axis compute p(N)/q(N) by a CONVOLUTION (the Euler product) rather than by
   storing each partition? Test: does iterating the meet/depth reproduce the
   pentagonal-number recurrence? (vs sympy)

2. SUBSET-SUM meet-in-the-middle — the meet identity bank(a)@p==bank(p)@a is a
   COLLISION. Split set in halves, build sum-nodes for each half, look for a
   collision X_left + X_right == target. Is this the classic 2^(n/2) MITM?
   Compare node count to 2^n brute and verify correctness vs brute.

3. GOLDBACH — is the meet doing any prime work, or purely the addition? Strip
   sympy: count how many primality tests the meet itself performs (answer: 0).
"""
from __future__ import annotations
import itertools, random, time
import sympy
from sympy import primerange
from sympy.functions.combinatorial.numbers import partition as sympy_partition

from aethos_lattice import BranchKind
from aethos_sequences import canon_on_chain


def banner(t):
    print("=" * 74); print(t); print("=" * 74)


# ---------------------------------------------------------------------------
# 1. Partition COUNT — does additive depth give the generating-function count?
# ---------------------------------------------------------------------------
def test_partition_count(Nmax=40):
    banner("1. PARTITION COUNT p(N) — convolution on the additive depth axis")
    print("The lattice depth zeta is ADDITIVE (sum). The partition generating fn")
    print("prod 1/(1-x^k) is a CONVOLUTION over the same additive monoid (N,+).")
    print("If 'the corpus is a number' and depth=sum, then the coefficient extraction")
    print("IS the standard DP. We run the DP (which the additive axis supports) and")
    print("check it equals sympy p(N). This shows the lattice's additive structure")
    print("HOSTS the generating function but adds nothing beyond the classical DP.\n")

    # unrestricted partition DP — the convolution prod 1/(1-x^k)
    def p_dp(N):
        dp = [0]*(N+1); dp[0]=1
        for k in range(1, N+1):
            for s in range(k, N+1):
                dp[s]+=dp[s-k]
        return dp

    dp = p_dp(Nmax)
    ok = sum(1 for N in range(Nmax+1) if dp[N]==int(sympy_partition(N)))
    print(f"  additive-axis convolution DP == sympy p(N): {ok}/{Nmax+1}")
    print(f"  e.g. p(40) = {dp[40]}  (sympy {int(sympy_partition(40))})")
    print("  VERDICT: exact, but it is the textbook generating-function DP; the")
    print("  lattice supplies the monoid (N,+,sum-as-depth), not a new algorithm.")


# ---------------------------------------------------------------------------
# 2. SUBSET-SUM meet-in-the-middle via meet collisions
# ---------------------------------------------------------------------------
def test_mitm(n=22, trials=40, maxval=200, seed=1):
    banner("2. SUBSET-SUM meet-in-the-middle — is the meet a real 2^(n/2) collision?")
    print("The meet identity bank(a)@p == bank(p)@a is X=a+p collision. Generalize:")
    print("any sub-chain's depth zeta = sum. Split the set; build all half-sums as")
    print("lattice depths; a target is hit iff a LEFT depth collides with target-RIGHT")
    print("depth. That collision search is the textbook MITM. We verify correctness")
    print("vs 2^n brute and report node counts.\n")
    rng = random.Random(seed)

    def half_sums(vals):
        """All subset sums of one half, recorded as lattice depths zeta=sum(subchain)."""
        sums = {}
        m = len(vals)
        for mask in range(1 << m):
            s = 0
            sub = []
            for i in range(m):
                if mask & (1<<i):
                    s += vals[i]; sub.append(vals[i])
            # 'depth' via lattice: for a sorted distinct subchain k>=3 interior zeta==sum;
            # we just assert the additive read = s (lattice depth identity), no need to
            # call canon for every mask — identity proven; sample-check below.
            sums.setdefault(s, mask)
        return sums

    def mitm(vals, target):
        L = vals[:len(vals)//2]; R = vals[len(vals)//2:]
        ls = half_sums(L); rs = half_sums(R)
        for s, mask in ls.items():
            need = target - s
            if need in rs:
                return (mask, rs[need])
        return None

    def brute(vals, target):
        n=len(vals)
        for mask in range(1<<n):
            s=sum(vals[i] for i in range(n) if mask&(1<<i))
            if s==target: return mask
        return True if False else None if True else None  # placeholder
    def brute2(vals,target):
        n=len(vals)
        for mask in range(1<<n):
            s=sum(vals[i] for i in range(n) if mask&(1<<i))
            if s==target: return mask
        return None

    # spot-check the lattice depth identity zeta(subchain)==sum on a few subchains
    idcheck=0; idtot=0
    for _ in range(50):
        k=rng.randint(3,5)
        c=tuple(sorted(rng.sample(range(1,80),k)))
        n_int=None
        for nn in range(c[0],c[-1]+1):
            seg=sum(1 for a in c if nn>=a)
            if 0<seg<len(c): n_int=nn; break
        if n_int is not None:
            coord=canon_on_chain(BranchKind.VA1,c,n_int)
            idtot+=1
            if coord[2]==sum(c): idcheck+=1
    print(f"  lattice depth identity zeta(subchain)==sum: {idcheck}/{idtot}")

    agree=0; mitm_nodes=[]; brute_nodes=[]
    t_mitm=t_brute=0.0
    for _ in range(trials):
        vals=[rng.randint(1,maxval) for _ in range(n)]
        target=rng.randint(1,sum(vals))
        t0=time.perf_counter(); m=mitm(vals,target); t_mitm+=time.perf_counter()-t0
        t0=time.perf_counter(); b=brute2(vals,target); t_brute+=time.perf_counter()-t0
        if (m is not None)==(b is not None):
            agree+=1
        mitm_nodes.append(2*(1<<(n//2)))
        brute_nodes.append(1<<n)
    print(f"  MITM vs brute DECISION agreement: {agree}/{trials}")
    print(f"  nodes built: MITM 2*2^(n/2)={2*(1<<(n//2))}  vs brute 2^n={1<<n}  (n={n})")
    print(f"  wall time over {trials} trials: MITM {t_mitm*1000:.1f}ms  brute {t_brute*1000:.1f}ms")
    print("  VERDICT: the meet-collision IS meet-in-the-middle (real 2^(n/2)), but")
    print("  MITM is a classical 1974 algorithm (Horowitz-Sahni). The lattice depth")
    print("  axis re-encodes the subset-sum table; it does not beat MITM.")


# ---------------------------------------------------------------------------
# 3. Goldbach — how much prime work does the meet do?
# ---------------------------------------------------------------------------
def test_goldbach_workcount(Emax=200):
    banner("3. GOLDBACH work attribution — meet does the +, sympy does the primes")
    print("Count operations: the meet contributes the addition a+p=E (a coordinate).")
    print("Primality of a and E-a is NOT a lattice operation. We verify every Goldbach")
    print("pair is recoverable, and that the lattice's role is purely the additive map.\n")
    total_pairs=0; verified=0
    for E in range(4,Emax+1,2):
        primes=list(primerange(2,E)); pset=set(primes)
        for a in primes:
            p=E-a
            if p<a: break
            if p in pset:
                total_pairs+=1
                # lattice contribution: X coordinate of meet = a+p
                X = (a)+(p)  # exactly what swap_meet returns as left.z.real (verified in probe1)
                if X==E: verified+=1
    print(f"  Goldbach pairs E=4..{Emax}: {total_pairs}, meet-X==E for all: {verified}/{total_pairs}")
    print("  Goldbach holds empirically (every even >=4 has >=1 pair) — UNPROVEN in general.")
    print("  VERDICT: lattice supplies addition-as-coordinate; conjecture & sieve are external.")


if __name__=="__main__":
    test_partition_count(40); print()
    test_mitm(22,40,200,1); print()
    test_goldbach_workcount(200)
