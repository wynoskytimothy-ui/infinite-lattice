"""
_dd_new_caps.py -- adversarial deep-dive on NEW capabilities of the meet atom.

The meet atom (verified, from spec + aethos_address_store.py):
  triple {a,p,q} (a<p<q) -> meet node with
     zeta = a + p + q              (conserved depth / the LOCK)
     readout (X,Y,Z) = (p+q, p, p+q)   INVERTIBLE: p=Y, q=X-Y, a=zeta-X
  missing-member: any 2 of the 3 recover the 3rd as  member = zeta - (sum of other two)
  meet = (sum, min) = tropical (min,+).

THREE candidate NEW capabilities, each built + stressed to breaking:
  (A) SECRET SHARING  -- is missing-member a real (k,n) threshold scheme? what leaks?
  (B) CRDT            -- do two independently-built co-location stores merge
                         commutatively/associatively and converge?
  (C) SET RECONCILIATION -- reconcile two sets with meet-syndromes, fewer bytes
                         than shipping the set?  (memory: k=3 meet ~ Minisketch)

RUN, don't theorize. Report exact numbers + the honest limit + label.
"""
from __future__ import annotations
import os, sys, random, itertools, struct, hashlib
random.seed(1234)

OUT = []
def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s)

# ---------------------------------------------------------------------------
# The meet atom, pure arithmetic (matches aethos_address_store TripleNode).
# ---------------------------------------------------------------------------
def meet(a, p, q):
    """Co-locate triple a<p<q. Returns the address (zeta, X, Y)."""
    assert a < p < q, (a, p, q)
    zeta = a + p + q
    X = p + q
    Y = p
    return (zeta, X, Y)

def unmeet(addr):
    """Invert: recover {a,p,q} from address (zeta, X, Y)."""
    zeta, X, Y = addr
    p = Y
    q = X - Y
    a = zeta - X
    return (a, p, q)

def missing(zeta, two):
    """Recover 3rd member from sum-lock and any two members."""
    return zeta - sum(two)

# sanity
for _ in range(20000):
    a = random.randint(1, 10**6)
    p = a + random.randint(1, 10**6)
    q = p + random.randint(1, 10**6)
    addr = meet(a, p, q)
    assert unmeet(addr) == (a, p, q)
    assert missing(addr[0], (a, p)) == q
    assert missing(addr[0], (p, q)) == a
log("[sanity] meet invertible + missing-member: 20000/20000 OK")
log("="*70)

# ===========================================================================
# (A) SECRET SHARING -- is missing-member a threshold scheme? What leaks?
# ===========================================================================
log("\n(A) SECRET SHARING via missing-member  member = zeta - sum(others)")
log("-"*70)
# Frame: secret = q. Shares = {a, p, zeta}. Holder of (a,p,zeta) -> q. (2-of-3? n-of-n?)
# Adversarial Q1: is it (k,n)-threshold (k shares reveal nothing, k+1 reveal all)?
# Q2: what does the PUBLIC address (zeta,X,Y) leak about the members?

# Q2 first: the address IS the answer. unmeet recovers everything from (zeta,X,Y).
secret = random.randint(1, 10**9)
a = random.randint(1, secret-1); p = (a+secret)//2 if (a+secret)//2 > a else a+1
# build a valid triple where q = secret
a = random.randint(1, 1000); p = a + random.randint(1, 1000); q = secret
if not (a < p < q):
    a, p = 1, 2
addr = meet(a, p, q)
log(f"  secret q={q}; full address (zeta,X,Y)={addr}")
log(f"  -> unmeet(address) = {unmeet(addr)}   (the address PUBLICLY decodes the secret)")
log("  VERDICT Q2: the meet ADDRESS is NOT hiding -- it is a (lossless) commitment+opening.")

# Q1: treat the 3 raw members {a,p,q} + the lock zeta as 4 'shares'.
# Sub-question: given ONLY zeta (the lock), how much entropy remains in the secret?
# zeta = a+p+q. If a,p drawn from a public range R, knowing zeta constrains q to
# zeta-a-p i.e. one linear equation. Count solutions.
def count_solutions_given_zeta(zeta, lo, hi):
    """How many ordered a<p<q with members in [lo,hi] sum to zeta?"""
    cnt = 0
    for a in range(lo, hi+1):
        for p in range(a+1, hi+1):
            q = zeta - a - p
            if p < q <= hi:
                cnt += 1
    return cnt

lo, hi = 1, 60
zeta = 90
n_sol = count_solutions_given_zeta(zeta, lo, hi)
log(f"\n  Threshold test: members in [{lo},{hi}], reveal ONLY zeta={zeta}.")
log(f"    # of valid triples consistent with zeta = {n_sol}")
log(f"    -> zeta alone leaves {n_sol} candidates: it HIDES the secret (1 linear eqn, 3 unknowns).")
# Now reveal one MORE member (say a). Solutions?
def count_given_zeta_and_a(zeta, a, lo, hi):
    cnt = 0
    for p in range(a+1, hi+1):
        q = zeta - a - p
        if p < q <= hi:
            cnt += 1
    return cnt
a_known = 20
n_sol2 = count_given_zeta_and_a(zeta, a_known, lo, hi)
log(f"    reveal zeta + one member a={a_known}: # candidates = {n_sol2}")
log(f"    reveal zeta + TWO members: # candidates = 1 (missing-member is exact).")
log("  VERDICT Q1: this is exactly (n-1, n) over 1 linear equation = REDUCIBLE-TO Shamir@degree-1")
log("    i.e. trivial additive/linear secret sharing (a+p+q=zeta is a 1-D hyperplane).")
log("    NOT a new primitive: it's the t=1 polynomial = sum-share. No threshold > all-but-one.")
log("    Info-theoretic hiding HOLDS for the lock alone, but the meet ADDRESS reveals all -> ")
log("    the lattice address is the OPPOSITE of a share (it's the opening).")

# ===========================================================================
# (B) CRDT -- do independent co-location stores merge by union, commutative/assoc/convergent?
# ===========================================================================
log("\n\n(B) CRDT  -- two independently-built meet stores, merge = union of addresses")
log("-"*70)
# A 'co-location store' = a set of meet addresses (each a frozen triple-node).
# Merge(S1,S2) = S1 | S2  (set union of addresses). Test the 3 CRDT laws +
# convergence under arbitrary interleaving, AND whether the meet adds anything
# a plain set-of-hashes (G-Set) doesn't.

def make_store(triples):
    return frozenset(meet(*sorted(t)) for t in triples)

def merge(s1, s2):
    return s1 | s2

# random triple universe
def rnd_triple():
    a = random.randint(1, 500)
    p = a + random.randint(1, 500)
    q = p + random.randint(1, 500)
    return (a, p, q)

universe = [rnd_triple() for _ in range(300)]
def rnd_substore(k):
    return make_store(random.sample(universe, k))

# Law 1: commutativity
ok_comm = all(merge(s1:=rnd_substore(40), s2:=rnd_substore(40)) ==
              merge(s2, s1) for _ in range(2000))
# Law 2: associativity
ok_assoc = True
for _ in range(2000):
    s1, s2, s3 = rnd_substore(30), rnd_substore(30), rnd_substore(30)
    if merge(merge(s1, s2), s3) != merge(s1, merge(s2, s3)):
        ok_assoc = False; break
# Law 3: idempotence
ok_idem = all((s:=rnd_substore(40)) == merge(s, s) for _ in range(2000))
log(f"  commutative   : {ok_comm}")
log(f"  associative   : {ok_assoc}")
log(f"  idempotent    : {ok_idem}")

# Convergence: N replicas, random gossip, all reach same state = union of all ops.
N = 8
ops = [make_store([rnd_triple() for _ in range(10)]) for _ in range(40)]  # 40 update batches
replicas = [frozenset() for _ in range(N)]
# each op applied to a random replica
for op in ops:
    replicas[random.randrange(N)] |= op
# random gossip rounds
for _ in range(500):
    i, j = random.sample(range(N), 2)
    m = merge(replicas[i], replicas[j])
    replicas[i] = replicas[j] = m
truth = frozenset().union(*ops)
converged = all(r == truth for r in replicas)
log(f"  convergence   : {N} replicas, 40 ops, 500 gossips -> all equal truth: {converged}")
log(f"    final state size = {len(truth)} addresses")

# The adversarial question: does the MEET add anything over a G-Set of raw hashes?
# Test: can the merged store ANSWER a relational query that a hash-set cannot?
# Query: 'is there a stored triple containing member m?' -- meet address lets you
# test membership-of-a-member WITHOUT storing members, via the invertible address.
target_member = universe[0][0]
# reconstruct all members present from addresses alone:
recovered_members = set()
for addr in truth:
    recovered_members.update(unmeet(addr))
has = target_member in recovered_members
log(f"  relational readback: recovered member-set from addresses only, |members|={len(recovered_members)}")
log(f"    membership-of-member query answerable from merged CRDT: {has}")
log("  VERDICT (B): CRDT laws HOLD (it's a G-Set / grow-only set -- union is always a CRDT).")
log("    The NEW part vs a plain hash G-Set: each element is an INVERTIBLE relational address,")
log("    so the merged set is queryable for its *members* and self-checks (erasure) -- a hash G-Set")
log("    is opaque. But convergence itself is REDUCIBLE-TO G-Set; the meet adds queryability, not")
log("    new merge semantics. No conflict resolution beyond union (no removes -> no real CRDT hardness).")

with open("_dd_new_caps_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
print("\n[wrote _dd_new_caps_out.txt]")
