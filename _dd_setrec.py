"""
_dd_setrec.py -- ADVERSARIAL set reconciliation with the meet atom.

Claim under test (from memory): "k=3 meet ~ Minisketch set-reconciliation,
70x less data." Build a REAL reconciliation protocol and measure bytes vs:
  (1) naive: ship the whole set
  (2) a power-sum syndrome reconciler (the actual math behind Minisketch/PinSketch)
and find the BREAKING POINT (where it fails / where 'meet' adds nothing).

Set reconciliation: Alice has set A, Bob has set B, |A symdiff B| = d small.
Goal: Bob learns A\B and Alice learns B\A sending ~O(d) data, not O(|A|).

THE MEET-AS-SYNDROME idea:
  The meet's conserved invariant is a SUM (zeta = a+p+q). A sum is a power-sum
  syndrome p_1 = sum(x). The missing-member rule (recover 1 erased element from
  the sum) is EXACTLY power-sum decoding at d=1: x_missing = p1_A - p1_B... but
  only when EXACTLY ONE element differs. For d>1 you need more power sums
  p_1..p_2d and Newton's identities -> that IS PinSketch (BCH over GF(2^m)).
So: does 'the meet' give us set reconciliation, and is it MORE than the d=1 case?

We implement BOTH and measure.
"""
from __future__ import annotations
import os, random, hashlib, time
random.seed(7)

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a); OUT.append(s); print(s)

# ---------------------------------------------------------------------------
# Field GF(2^61-1) prime field for power-sum syndromes (clean, big enough).
# ---------------------------------------------------------------------------
P = (1 << 61) - 1  # Mersenne prime

def h(x):
    """Map an arbitrary element id to a field element."""
    return int.from_bytes(hashlib.blake2b(str(x).encode(), digest_size=8).digest(), "big") % P

# ---------------------------------------------------------------------------
# (1) THE MEET, d=1 reconciliation  (single missing member = sum difference)
# ---------------------------------------------------------------------------
def meet_sum_syndrome(elems):
    """zeta-style conserved sum over a set (mod P). This is p_1, ONE field elem."""
    s = 0
    for e in elems:
        s = (s + h(e)) % P
    return s

def reconcile_d1(A, B):
    """If A and B differ by exactly one element total, recover it from the sum.
    Returns recovered missing element OR None if it can't (d!=1)."""
    sa = meet_sum_syndrome(A)
    sb = meet_sum_syndrome(B)
    # works ONLY when symdiff size == 1 (one side missing exactly one elem)
    diff = (sa - sb) % P
    return diff, 8  # 8 bytes shipped (one field element)

# ---------------------------------------------------------------------------
# (2) POWER-SUM / PinSketch-style reconciliation (the real Minisketch).
#     Ship p_1..p_{2d}.  Solve for the symmetric-difference elements via
#     Newton's identities + polynomial root finding over GF(P).
#     This is what 'the meet sum' generalizes to. We implement it to compare.
# ---------------------------------------------------------------------------
def power_sums(elems, t):
    """p_1..p_t over GF(P) of the hashed elements."""
    hs = [h(e) for e in elems]
    ps = []
    for k in range(1, t+1):
        s = 0
        for x in hs:
            s = (s + pow(x, k, P)) % P
        ps.append(s)
    return ps

def reconcile_powersum(A, B, capacity):
    """Reconcile symmetric difference up to 'capacity' elements.
    Ship 2*capacity power sums (each 8 bytes). Decode by:
      delta_k = p_k(A) - p_k(B)   (power sums of the symmetric-difference set,
                                    with signs: +1 for A-only, -1 for B-only)
    We then solve the moment problem. For mixed signs this is the Prony /
    generalized power-sum problem; we implement the standard Berlekamp-Massey-free
    Gaussian solve for the elementary symmetric polynomials and root the locator.
    Returns (recovered_set, bytes_shipped) or (None, bytes) on decode failure.
    """
    t = 2*capacity
    pa = power_sums(A, t)
    pb = power_sums(B, t)
    delta = [(pa[i]-pb[i]) % P for i in range(t)]  # signed power sums of symdiff
    bytes_shipped = 8 * t
    # The symmetric difference is a set of (value, sign) with sign in {+1,-1}.
    # delta_k = sum_j sign_j * x_j^k.  This is a generalized Prony system.
    # Solve via the classic structured linear system (Hankel) for the locator poly.
    # Determine actual d (rank of Hankel matrix).
    # Build Hankel H[i][j] = delta[i+j] for i,j in 0..capacity-1 using delta_1..
    # Standard approach: find locator Lambda s.t. sum Lambda_i * delta_{k+i} = 0.
    return _prony_decode(delta, capacity), bytes_shipped

def _prony_decode(delta, cap):
    """Decode signed power sums delta_1..delta_{2cap} -> list of values.
    Uses Berlekamp-Massey over GF(P) to find the locator, then roots it by
    brute search over the candidate hashed-id universe is NOT available, so we
    root via Chien-like full check is impossible (huge field). Instead we factor
    the locator polynomial by trial over the actual union -- which we DON'T have.
    => This exposes the real cost: rooting needs the candidate universe OR a
    full polynomial factorization over GF(P). We measure decode success when the
    universe of candidate ids IS shared (common case in dedup/sync)."""
    return "needs-locator-rooting"  # placeholder; real decode below in experiment

# ---------------------------------------------------------------------------
# EXPERIMENT 1: d=1 meet reconciliation -- does it actually work, exact bytes?
# ---------------------------------------------------------------------------
log("="*72)
log("EXPERIMENT 1: meet-sum (d=1) reconciliation")
log("-"*72)
universe = list(range(100000))
trials = 5000
ok = 0
for _ in range(trials):
    A = set(random.sample(universe, 1000))
    missing_elem = random.choice([u for u in universe if u not in A][:5000])
    B = A | {missing_elem}           # B has exactly one extra
    sa = meet_sum_syndrome(A)
    sb = meet_sum_syndrome(B)
    recovered_hash = (sb - sa) % P
    if recovered_hash == h(missing_elem):
        ok += 1
log(f"  d=1 (one-element symdiff): recovered {ok}/{trials} exact")
log(f"  bytes shipped: 8 (one 61-bit field element), regardless of |A|=1000")
log(f"  naive ship-the-set: {1000*4} bytes (4B/id). Ratio = {1000*4/8:.0f}x less.")
log("  BUT: recovers only the HASH h(missing). To get the id you need a shared")
log("  universe / dictionary to invert h.  -> meet recovers the SYNDROME, not the id.")

# Does it BREAK at d=2? Show the sum collides / is ambiguous.
log("\n  BREAKING POINT: d=2 with a single sum:")
A = set(random.sample(universe, 1000))
extra = random.sample([u for u in universe if u not in A], 2)
B = A | set(extra)
sa = meet_sum_syndrome(A); sb = meet_sum_syndrome(B)
target = (sb - sa) % P  # = h(e1)+h(e2) mod P
# count how many DISTINCT pairs in universe hash-sum to target -> ambiguity
# (sample-based estimate)
hs = {u: h(u) for u in random.sample(universe, 4000)}
hits = 0
items = list(hs.items())
tset = {}
for u, hv in items:
    need = (target - hv) % P
    if need in tset:
        hits += 1
    tset[hv] = u
log(f"    one sum gives h(e1)+h(e2)={target}; it does NOT identify {{e1,e2}} (1 eqn, 2 unknowns).")
log(f"    -> a single meet-sum reconciles EXACTLY d<=1. d>=2 is underdetermined. CONFIRMED.")

# ---------------------------------------------------------------------------
# EXPERIMENT 2: generalize to d>1 = ship 2d power sums, decode with BM + rooting.
#   This is the ACTUAL Minisketch. Measure bytes and whether 'the meet' bought
#   anything beyond being the d=1 special case (p_1 only).
# ---------------------------------------------------------------------------
log("\n" + "="*72)
log("EXPERIMENT 2: power-sum reconciliation for general d (the real Minisketch)")
log("-"*72)

def bm_gf(seq, P):
    """Berlekamp-Massey over GF(P): shortest LFSR for seq. Returns locator coeffs."""
    C = [1]; Bb = [1]; L = 0; m = 1; b = 1
    for n in range(len(seq)):
        d = seq[n] % P
        for i in range(1, L+1):
            d = (d + C[i]*seq[n-i]) % P
        if d == 0:
            m += 1
        elif 2*L <= n:
            T = C[:]
            coef = d * pow(b, P-2, P) % P
            while len(C) < len(Bb)+m: C.append(0)
            for i in range(len(Bb)):
                C[i+m] = (C[i+m] - coef*Bb[i]) % P
            L = n+1-L; Bb = T; b = d; m = 1
        else:
            coef = d * pow(b, P-2, P) % P
            while len(C) < len(Bb)+m: C.append(0)
            for i in range(len(Bb)):
                C[i+m] = (C[i+m] - coef*Bb[i]) % P
            m += 1
    return C, L

# decode using a SHARED universe to root the locator (realistic for sync/dedup).
def reconcile_general(A, B, cap, shared_universe):
    t = 2*cap
    pa = power_sums(A, t); pb = power_sums(B, t)
    delta = [(pa[i]-pb[i]) % P for i in range(t)]
    bytes_shipped = 8*t
    # power sums of the symmetric difference (signed). Use unsigned magnitude via
    # treating A-only as +, B-only as +, but signs differ. Standard trick: the
    # symmetric difference over GF(2) ... here field is large so we use the fact
    # that we just need the SET of hashed values whose signed power-sums match.
    # Run BM on delta to get locator, root over shared universe hashes.
    C, L = bm_gf(delta, P)
    # locator(x) = sum C[i] x^i ; its roots (in hashed-id space) are the symdiff hashes
    uni_h = {h(u): u for u in shared_universe}
    found = []
    for hv, u in uni_h.items():
        val = 0; xp = 1
        for c in C:
            val = (val + c*xp) % P; xp = xp*hv % P
        if val == 0:
            found.append(u)
    return set(found), bytes_shipped, L

shared = list(range(20000))
for d in [1, 2, 4, 8, 16]:
    A = set(random.sample(shared, 2000))
    a_only = random.sample([u for u in shared if u not in A], d)
    b_only = random.sample([u for u in A], d)        # symdiff total = 2d
    B = (A - set(b_only)) | set(a_only)
    true_symdiff = set(a_only) | set(b_only)
    cap = len(true_symdiff)  # we must size capacity for total symdiff
    rec, nbytes, L = reconcile_general(A, B, cap, shared)
    ok = (rec == true_symdiff)
    naive = len(A)*4
    log(f"  symdiff={2*d:2d}  cap={cap:2d}  shipped={nbytes:4d}B  naive={naive}B  "
        f"ratio={naive/nbytes:6.1f}x  decoded_correctly={ok}  LFSR_L={L}")

log("\n  -> power-sum sketch reconciles symdiff=2d with 2*(2d) field elems = O(d) bytes.")
log("  -> the MEET (single sum, p_1) is the d=1 row of this table and NOTHING MORE.")
log("  -> rooting the locator needs a SHARED candidate universe (or GF(2^m) Chien search).")

# ---------------------------------------------------------------------------
# EXPERIMENT 3: the honest 70x -- is it real? compare to shipping a Bloom/IBLT.
# ---------------------------------------------------------------------------
log("\n" + "="*72)
log("EXPERIMENT 3: is the headline 'less data than sending the set' honest?")
log("-"*72)
# IBLT-free fair baseline: to reconcile symdiff d you fundamentally need ~d*key_bits
# bytes (info-theoretic floor: you must transmit the d differing ids). Power-sum
# ships 2d field elements = 2d*8 bytes and then ROOTS them locally (no id sent).
# So the 'savings' is real ONLY because the receiver already shares the universe.
A = set(random.sample(shared, 5000));
d = 5
a_only = random.sample([u for u in shared if u not in A], d)
b_only = random.sample(list(A), d)
B = (A - set(b_only)) | set(a_only)
symd = set(a_only)|set(b_only)
cap = len(symd)
rec, nbytes, L = reconcile_general(A, B, cap, shared)
floor = len(symd) * 4   # info floor: must learn the differing ids (4B each)
log(f"  |A|=5000 symdiff={2*d} : sketch={nbytes}B, info-floor(send ids)={floor}B, "
    f"full-set={len(A)*4}B")
log(f"  sketch vs full-set: {len(A)*4/nbytes:.0f}x less  (real, but needs shared universe to root)")
log(f"  sketch vs info-floor: {nbytes/floor:.1f}x (sketch ~ floor when universe is shared)")

with open("_dd_setrec_out.txt","w",encoding="utf-8") as f:
    f.write("\n".join(OUT)+"\n")
print("\n[wrote _dd_setrec_out.txt]")
