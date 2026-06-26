"""
Two-sided audit of the AETHOS paradox claims.

(1) HILBERT'S HOTEL via prime-power (Godel) routing — verify genuine injection
    for the three classic scenarios, collision-check at scale vs sympy FTA.
(2) RUSSELL/BARBER — test whether the n=p / level-invariant boundary FORMALLY
    resolves Russell's paradox, or is a structural ANALOGY to type theory.

Ground truth: sympy (factorint = FTA), pure-Python set logic.
"""
from __future__ import annotations
import sys, itertools, math
from sympy import factorint   # ground-truth FTA inverse (small grid only)

# fast prime list (avoid sympy.prime() global cache blowup)
def first_primes(k: int) -> list[int]:
    ps = []
    cand = 2
    while len(ps) < k:
        if all(cand % p for p in ps if p * p <= cand):
            ps.append(cand)
        cand += 1
    return ps

_PRIMES = first_primes(2100)
def nth_prime(i: int) -> int:   # 1-indexed
    return _PRIMES[i - 1]

# ----------------------------------------------------------------------------
# PART 1: HILBERT'S HOTEL — prime-power guest routing
# ----------------------------------------------------------------------------
# A guest is identified by an occupied room (a positive integer >= 1).
# The "hotel" is the set of occupied rooms. We must accommodate new guests
# with an INJECTION old_rooms -> new_rooms that frees space, with NO collisions.
#
# The AETHOS lattice's actual mechanism (aethos_recursive_lattice.promote +
# PROMOTION_POOL = the primes, and walk_down/FTA) is prime-power / Godel
# numbering. So we formalize guest routing the way the lattice already encodes
# nested chains: a guest-list (finite or a bus-index) maps to an integer by
# prime-power encoding, and FTA guarantees the decode is unique => injection.

print("=" * 74)
print("PART 1 — HILBERT'S HOTEL as prime-power (Godel) routing")
print("=" * 74)

# --- Scenario A: one new guest. Classic shift n -> n+1.
# Prime-flavored version: room r -> p_r (the r-th prime). Frees room 1 (and more).
def routeA(r: int) -> int:
    return nth_prime(r)

# --- Scenario B: countably infinite new guests. Classic n -> 2n (frees odds).
def routeB(r: int) -> int:
    return 2 * r

# --- Scenario C: infinitely many infinite buses.
# Guest = (bus b >= 1, seat s >= 1). Hotel guest = bus 0.
# Prime-power routing: room = 2^? ... The canonical bijection N x N -> N.
# We use the prime-power Godel map that the lattice's FTA decode inverts:
#   (b, s) -> 2^b * 3^s     is an INJECTION (FTA: unique factorization).
# Existing hotel guest r -> 2^0 * 3^? no; keep hotel on a disjoint prime.
# Cleanest: map bus b seat s -> p_b ^ s where p_b is the b-th prime.
#   FTA => the pair (p_b, s) is recoverable => injection on (b,s).
def routeC(b: int, s: int) -> int:
    return nth_prime(b) ** s

N = 2000  # scale for collision checks

# ---- Scenario A collision check
seenA = {}
collisionsA = 0
for r in range(1, N + 1):
    v = routeA(r)
    if v in seenA:
        collisionsA += 1
    seenA[v] = r
print(f"[A] one new guest  : routed {N} guests, collisions={collisionsA}, "
      f"min image={min(seenA)} (room 1 free? {1 not in seenA.values() or routeA(1) != 1})")
print(f"    injection holds: {collisionsA == 0}")

# ---- Scenario B
seenB = set()
collisionsB = 0
for r in range(1, N + 1):
    v = routeB(r)
    if v in seenB:
        collisionsB += 1
    seenB.add(v)
odds_free = all(routeB(r) % 2 == 0 for r in range(1, N + 1))
print(f"[B] countably-inf  : collisions={collisionsB}, all images even (odds freed)={odds_free}")
print(f"    injection holds: {collisionsB == 0}")

# ---- Scenario C: infinitely many infinite buses, p_b^s
# Two checks: (i) collision check at scale (no factoring needed),
#             (ii) FTA-decode ground-truth on a smaller grid (sympy factorint).
seenC = {}
collisionsC = 0
B_max, S_max = 300, 40   # 12000 (bus,seat) pairs — collision scale
primes_cache = [nth_prime(b) for b in range(1, B_max + 1)]
for bi, pb in enumerate(primes_cache, start=1):
    for s in range(1, S_max + 1):
        v = pb ** s
        if v in seenC:
            collisionsC += 1
        seenC[v] = (bi, s)

# (ii) ground-truth FTA decode on a tractable subgrid
fta_fail = 0
fta_checked = 0
for b in range(1, 31):          # 30 buses
    pb = nth_prime(b)
    for s in range(1, 31):      # 30 seats  -> 900 sympy factorizations
        v = pb ** s
        f = factorint(v)        # sympy = Fundamental Theorem of Arithmetic
        fta_checked += 1
        if f != {pb: s}:
            fta_fail += 1
print(f"[C] inf x inf buses: pairs={B_max*S_max}, collisions={collisionsC}, "
      f"FTA-decode checked={fta_checked} mismatches={fta_fail}")
print(f"    injection holds: {collisionsC == 0}   FTA inverts every code: {fta_fail == 0}")

# Is routeC a BIJECTION onto N? No — its image is the prime-POWERS only.
# It is an INJECTION N x N -> N (accommodation succeeds), NOT a surjection.
# The standard Cantor pairing IS a bijection. Compare densities:
max_code = max(seenC.keys())
print(f"    largest code used = {max_code} for {len(seenC)} pairs "
      f"=> density {len(seenC)/max_code:.2e} (Cantor pairing density ~ O(1)).")
print(f"    image is prime-powers only => INJECTION not surjection onto N.")
print(f"    (classic Cantor pairing is the true bijection; Godel code wastes N.)")

# Honest name check: this IS Godel numbering / prime-power encoding.
print("\n  NAME (honest): prime-power encoding == Godel numbering. Not new;")
print("  it is the 1931 construction. FTA (sympy factorint) is the inverse.")

# ----------------------------------------------------------------------------
# PART 2: RUSSELL / BARBER
# ----------------------------------------------------------------------------
print()
print("=" * 74)
print("PART 2 — RUSSELL'S PARADOX vs the level-invariant boundary")
print("=" * 74)

sys.path.insert(0, ".")
from aethos_recursive_lattice import RecursiveLattice, RecursiveNode
from core.primes import chain_primes

lat = RecursiveLattice()
base = chain_primes(8)
for p in base:
    lat.register_base(p)

# Build a hierarchy and CHECK the invariant directly: for every promoted prime,
# level(p) = 1 + max(level(c) for c in sub_chain), and p not in walk_down(p).
ids = []
ids.append(lat.promote([3,5,7], "A"))
ids.append(lat.promote([ids[0],11,13], "B"))
ids.append(lat.promote([ids[1],ids[0]], "C"))
for _ in range(50):
    ids.append(lat.promote([ids[-1], ids[-2]] if len(ids) >= 2 else [3,5], f"P{len(ids)}"))

inv_ok = True
self_mem = 0
for p in ids:
    node = lat.resolve(p)
    child_lvls = [lat.resolve(c).level for c in node.sub_chain]
    if node.level != 1 + max(child_lvls):
        inv_ok = False
    if p in lat.walk_down(p):
        self_mem += 1
print(f"  level invariant holds for all {len(ids)} promoted primes: {inv_ok}")
print(f"  primes appearing in their own walk_down (self-membership): {self_mem}")

# THE CRUX (two-sided): Russell's set is S = {x | x not in x}.
# The lattice forbids x in x by a TYPE/LEVEL stratification. So you literally
# CANNOT FORM the predicate "x not in x" at a single level: membership only
# crosses ONE level down. Does that "resolve" Russell? Let's be precise.
#
# Russell's paradox needs UNRESTRICTED COMPREHENSION: {x | phi(x)} is a set for
# any phi. The contradiction: R = {x | x notin x}; ask R in R?.
# Stratified/typed set theory BLOCKS the comprehension (phi must be stratified;
# "x in x" is NOT stratifiable). The lattice does the SAME thing GEOMETRICALLY:
# "in" only relates level L to level L+1, so "x in x" is ill-typed.
#
# TEST: can we even *write down* R in the lattice? Try to build the set of all
# nodes that do not contain themselves, then ask if it contains itself.
all_nodes = list(lat.nodes.values())
not_self = [nd for nd in all_nodes if nd.prime not in lat.walk_down(nd.prime)]
print(f"\n  nodes total={len(all_nodes)}, nodes NOT in themselves={len(not_self)} "
      f"(== all, since self-membership is impossible)")
# To "form R as a node" we'd promote a chain = [every non-self-member].
# But promote() puts R at level max+1, ABOVE all its members. So R is NOT a
# member of itself by construction. The paradoxical question 'R in R?' is
# answered structurally: NO, because R lives one level up. No contradiction —
# but ALSO: R fails to capture itself, exactly the typed-set-theory outcome.
try:
    R = lat.promote([nd.prime for nd in not_self if nd.level < 50][:20], "RUSSELL_R")
    R_in_R = R in lat.walk_down(R)
    print(f"  formed R (prime {R}) at level {lat.resolve(R).level}; is R in R? {R_in_R}")
    print(f"  -> NO contradiction: R is one level above its members (typed).")
except Exception as e:
    print(f"  could not form R: {e}")

print("""
  VERDICT (two-sided):
   PRO: The level invariant IS a faithful geometric model of Russell's OWN
        resolution (1908 theory of types / stratification). 'x in x' is
        ill-typed because membership strictly decreases level. Verified above.
   CON: It does NOT 'dissolve' Russell in any new way. It is the SAME move as
        ZF's axiom of foundation / Russell's type theory: forbid self-membership.
        It resolves the paradox by DISALLOWING the offending comprehension,
        not by giving a consistent answer to 'R in R' under naive comprehension.
        A coordinate boundary 'n=p regime switch' is window-dressing on
        stratification; the work is done by 'level strictly increases', which
        is exactly the foundation/type axiom.
""")
print("DONE")
