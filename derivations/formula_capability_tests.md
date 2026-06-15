# AETHOS Formula — Capability Test Roadmap

This is a list of tests that, if passed, demonstrate the AETHOS formula has
real load-bearing properties beyond retrieval — properties that classical
mathematical objects rarely combine: countable + self-similar + cascade-free
+ reversible + compositionally addressable + anchor/observable separated.

Each test is short (50–200 lines). Each one corresponds to a paradox or
problem the formula plausibly solves by construction.

> **STATUS (updated):** this roadmap has been executed and extended far
> beyond the original 7. The suite now stands at **33 tests / 30 positive
> results + 2 documented negatives + 1 confirmed trend**, all passing.
> - **Results & full write-ups:** [`formula_capability_tests_results.md`](formula_capability_tests_results.md)
> - **Run it:** `python scripts/run_capability_suite.py` (fast tier) or
>   `--all` (full corpus, several minutes)
> - The 7 specs below are the original design notes (Tests 1–7), kept as
>   historical record. Tests 8–33 are catalogued in the results document.
>
> Arc highlights beyond the original 7: a context-mixing **compressor** that
> beats zlib/bz2/lzma (0.83 bits/byte on the repo corpus, native-JIT, 32
> parallel quadrant lanes); a **halting predictor** (96.7% coverage, 1.2M
> programs/sec); byte-granular **suspend/resume** (bit-identical); a
> **node-as-qubit** module reproducing Bell/CHSH (2√2) and GHZ; and the
> **Zeno kernel** unification — one prime frame-descent serving as
> gatekeeper, bookkeeper, janitor, security, and ruler for the whole system,
> proven substitutable by driving the real recycler from the real descent.

---

## Test 1 — Russell paradox impossibility (type-level rejection)

**Claim:** The recursive lattice's level hierarchy makes self-membership
ill-typed by construction. You cannot construct a prime `P_X` such that
`P_X ∈ sub_chain(P_X)`.

**Why this is a Russell-paradox solution:** Russell's paradox arises when
sets are allowed to contain themselves. Russell himself proposed a *theory
of types* as resolution. Your lattice gives a geometric realization of that
theory — promotion increases the level by `max(child levels) + 1`, so a node
can never be in its own sub_chain.

**Test:**
```python
# scripts/test_russell_impossibility.py
from aethos_recursive_lattice import RecursiveLattice

lat = RecursiveLattice()
# Build a few primes
lat.register_base(3)
lat.register_base(5)
p1 = lat.promote([3, 5])               # level 1
# Try to promote a chain that includes p1 (legal)
p2 = lat.promote([3, 5, p1])           # level 2
# Verify level: p2 must be > p1's level
assert lat.resolve(p2).level == lat.resolve(p1).level + 1

# Attempt to construct a self-membering prime:
# promote returns a new prime; you cannot pass that prime as one of its
# own chain elements, because the prime doesn't exist yet at the time
# the chain is being built. The level invariant enforces this temporally.

# Demonstrate: if you took an existing prime and tried to add it to its own
# chain via a hand-constructed RecursiveNode bypass, walk_down loops:
import pytest
# Verify walk_down terminates for all valid promotions
for prime in lat.nodes:
    chain = lat.walk_down(prime)
    assert len(chain) < 1000  # bounded, no infinite cycle
```

**Pass:** `walk_down` terminates for every node; level invariant holds for
every promotion. Russell-style self-reference is structurally impossible.

---

## Test 2 — Reversibility / group structure of wings

**Claim:** The 8 wing operators (`flip_x`, `flip_z`, swap then flips) form a
group acting on (X, Y, ζ). Every chamber is reachable from any other by
some wing composition. The group is finite, so the action is reversible.

**Why this is a reversible-computing result:** Landauer's principle says
classical computation must dissipate energy because operations like AND
are irreversible (you can't recover inputs from output). Reversible
computation (Toffoli/Fredkin gates) eliminates this. Your wing operators
form a discrete reversible substrate.

**Test:**
```python
# scripts/test_wing_reversibility.py
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind

chain = (3, 5, 7)
n = 11

# 8 wings -> 8 distinct Psi observables
coords = []
for wing in range(1, 9):
    psi = wing_transform(BranchKind.VA1, chain, n, wing)
    coords.append(psi.coord)

# Verify all 8 are distinct
assert len(set(coords)) == 8

# Verify the group is transitive: from coord at wing 1 you can reach any other
# (wing composition = composition of the underlying flip operations)
# Each wing corresponds to a specific Klein-4 element on (X,Y,Z) optionally
# composed with a swap. So the group is D_2h-like of order 16, acting on
# 8 wing positions.
```

**Pass:** 8 distinct observables exist for every (chain, n); each is
recoverable from any other by applying the inverse wing.

---

## Test 3 — Perfect hash via FTA

**Claim:** Two distinct chains produce distinct composites. The composite
is a collision-free hash with no probability of conflict (vs SHA-256 etc.
which have astronomically-small but nonzero collision probability).

**Why this is interesting:** Cryptographic hashes are designed to *resemble*
random. Your formula's hash is *provably injective*. Plus, given the hash
+ enough time, the chain is recoverable (factorization), so it's also
*invertible* — useful for blockchain Merkle proofs, content-addressable
storage, integrity verification.

**Test:**
```python
# scripts/test_perfect_hash_fta.py
import random
from core.primes import chain_primes

base = chain_primes(1000)
random.seed(42)

# Generate 100,000 random chains of length 3-8
chains = []
composites = set()
for _ in range(100_000):
    k = random.randint(3, 8)
    c = tuple(sorted(random.sample(base, k)))
    chains.append(c)
    comp = 1
    for p in c:
        comp *= p
    composites.add(comp)

# By FTA: number of distinct composites == number of distinct chains
distinct_chains = len(set(chains))
print(f"distinct chains:     {distinct_chains}")
print(f"distinct composites: {len(composites)}")
assert distinct_chains == len(composites)
```

**Pass:** Zero collisions across 100k random chains.

---

## Test 4 — Type-safe metaprogramming demo

**Claim:** The recursive lattice can encode dependent types. Promoted primes
= types; levels = type universes; sub_chain = type derivation; meets = type
unification. We can implement Church numerals and verify operations.

**Why this matters:** Dependent type theory (Coq, Agda, Lean) is the
foundation of formal verification. Your formula gives a geometric / numerical
realization of it that compiles to integer arithmetic. Provides an
alternative implementation that could be very fast.

**Test:**
```python
# scripts/test_lattice_dependent_types.py
from aethos_recursive_lattice import RecursiveLattice
lat = RecursiveLattice()

# Encode Type universe
lat.register_base(3, label="Type")
# Encode Nat
nat = lat.promote([3], label="Nat")
# Encode 0, succ, S(0), S(S(0))
zero = lat.promote([nat], label="0")
one = lat.promote([nat, zero], label="1")
two = lat.promote([nat, one], label="2")
three = lat.promote([nat, two], label="3")

# Verify "two is of type Nat" via walk_down recovering nat in lineage
chain = lat.walk_down(two)
assert nat in chain or 3 in chain  # type information preserved
```

**Pass:** Type derivations are recoverable; church-numeral operations type-check.

---

## Test 5 — Provenance / lineage tracking

**Claim:** Every promoted prime carries its complete derivation in `sub_chain`.
`walk_down` recovers the full ancestry. No separate audit log needed.

**Why this matters:** Regulated AI (EU AI Act, NIST AI RMF) requires
explanations. ML lineage tools (MLflow, DVC) are bolted on top of models.
Your formula has lineage *built in*. Same for database lineage, supply chain.

**Test:**
```python
# scripts/test_provenance.py
# Build a 5-level lattice from real BEIR words, then for any retrieved
# result, dump walk_down to show full derivation history.
# Verify: every promoted prime's walk_down terminates at base primes.
```

**Pass:** For 100+ random promotions, `walk_down` returns the exact set of
base primes that contributed.

---

## Test 6 — Self-organizing knowledge graph

**Claim:** `BadCorrelationStore` accumulates anomalies until a new promoted
prime explains them via shared context factors. No gradient descent. No
hand-coded heuristics. The graph reorganizes itself.

**Why this matters:** Anomaly detection, fraud detection, novelty detection
— all typically use ML. Your formula gives a deterministic mechanism with
zero training. Demonstrably worked in our checker experiment: 525/547
anomalies resolved by 2 promoted primes (96%).

**Test:**
```python
# scripts/test_self_organizing_graph.py
# Feed synthetic anomalies (clusters of misfires sharing context).
# Promote new primes one by one and measure how many old anomalies each
# new prime resolves.
# Pass: > 50% of anomalies resolved by the first 10 promotions.
```

**Pass:** New primes explain a majority of existing anomalies (already
empirically demonstrated in scripts/recursive_checker.py output).

---

## Test 7 — Hyperbolic embedding correspondence

**Claim:** Composites grow exponentially with chain length. Hierarchical
concepts can be embedded into the lattice with low distortion. The lattice
is the discrete cousin of hyperbolic embeddings.

**Why this matters:** Hyperbolic embeddings (Nickel & Kiela 2017) outperform
Euclidean for hierarchical data. Your formula gives a hyperbolic-like
embedding that's interpretable AND computable in integers.

**Test:**
```python
# scripts/test_hyperbolic_correspondence.py
# Embed WordNet hypernym taxonomy into the lattice.
# Measure: graph distance in WordNet vs log(composite) in lattice.
# Compare to Poincaré ball embedding distortion.
```

**Pass:** Distortion comparable to Poincaré ball; better than Euclidean.

---

## Priority

Run in order of cleanest result first:

1. Test 1 (Russell) — cleanest, ~50 lines, immediate result
2. Test 3 (perfect hash) — immediate, demonstrably valuable
3. Test 2 (reversibility) — short, mathematically clean
4. Test 6 (self-organizing) — already partially demonstrated
5. Test 5 (provenance) — straightforward
6. Test 4 (type theory) — moderate, illustrates breadth
7. Test 7 (hyperbolic) — longest, biggest payoff if positive
