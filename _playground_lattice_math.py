#!/usr/bin/env python3
"""Unbiased lattice-math playground — symmetries, invariants, emergent structure."""
import itertools
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_complex_plane import wing_transform, equalize_witness, swap_meet
from aethos_lattice import BranchKind, LatticeBank32, LatticeId, VECTORS, prime_pair_case
from aethos_promotion import PROMOTION_POOL, intersection_prime, letter_chain
from aethos_sequences import sum_chain, make_chain, SequenceKind
from aethos_words import letter_to_prime
from lattice_retriever_v1.deep_branch_codec import triple_case_stream
from lattice_retriever_v1.electron_lattice_codec import (
    build_electron_alphabet,
    entangle_imag,
    entangle_witness,
    electrons_opposite,
)
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.stage02_intersections import lattice_signature
from lattice_retriever_v1.stage02_intersections import lattice_coords_32, intersect_primes
from lattice_retriever_v1.stage06_composites import section5_triple_roles
from lattice_retriever_v1.trigger_formula_codec import _ambiguous_locks
from lattice_retriever_v1.k_meet import swap_meet_primes as k_swap_meet

rng = random.Random(42)
BRANCHES = list(BranchKind)

print("=" * 72)
print("AETHOS LATTICE PLAYGROUND — symmetries & emergent structure")
print("=" * 72)

# ── 1. 32 chambers: equivalence classes / quotient ─────────────────────
print("\n[1] 32-CHAMBER QUOTIENT STRUCTURE")
A = (3, 5, 7)
n = 5
bank = LatticeBank32.prime_pair(3, 5)  # pair bank for (3,5) anchors
collisions = bank.find_same_n_collisions(n)
print(f"  pair (3,5) @ n={n}: {len(collisions)} collision classes among 32 chambers")
for coord, lids in collisions[:4]:
    print(f"    coord {coord}  <- chambers {[int(x) for x in lids]} ({len(lids)} fold)")

# orbit under wing_transform (complex plane)
orbit_zeta = set()
orbit_xy = set()
orbit_full = set()
for b in BRANCHES:
    for w in range(1, 9):
        psi = wing_transform(b, A, n, w)
        orbit_zeta.add(round(psi.zeta, 6))
        orbit_xy.add((round(psi.z.real, 3), round(psi.z.imag, 3)))
        orbit_full.add((round(psi.z.real, 3), round(psi.z.imag, 3), round(psi.zeta, 3)))
print(f"  chain {A} @ n={n}: zeta invariant={len(orbit_zeta)} value(s)={orbit_zeta}")
print(f"    (X,Y) quotient: {len(orbit_xy)} distinct / 32  |  full (X,Y,zeta): {len(orbit_full)}")

# VA vs VB: swap symmetry
va_coords = [bank[LatticeId(w)].at(n) for w in range(1, 5)]
vb_coords = [bank[LatticeId(w + 4)].at(n) for w in range(1, 5)]
vb_from_va = [(c[1], c[0], c[2]) for c in va_coords]  # Y-X swap
print(f"  VA1-4 vs VB5-8 Y<->X swap match: {va_coords == vb_from_va}")

# ── 2. Electron entanglement + wing channels ───────────────────────────
print("\n[2] ELECTRON ENTANGLEMENT — binding rules")
for data in (b"abab", b"abcabc", b"the cat"):
    cat = build_electron_alphabet(data)
    alpha = SymbolAlphabet.from_bytes(data)
    coin_map = {s.byte: s.coin for s in cat}
    pairs = []
    for i in range(len(data) - 1):
        a, b = data[i], data[i + 1]
        w = entangle_witness(a, b, alpha, cat)
        pairs.append((chr(a), chr(b), w.intersection_imag, w.opposite))
    print(f"  {data!r}: {pairs[:5]}")

# opposite membrane rule: count how often adjacent opposites share imag band
data = b"the quick brown fox jumps"
cat = build_electron_alphabet(data)
alpha = SymbolAlphabet.from_bytes(data)
imag_vals = []
for i in range(len(data) - 1):
    w = entangle_witness(data[i], data[i + 1], alpha, cat)
    imag_vals.append(w.intersection_imag)
print(f"  'the quick...' imag range: min={min(imag_vals)} max={max(imag_vals)} "
      f"distinct={len(set(imag_vals))}/{len(imag_vals)}")

# ── 3. trigger_formula ambiguous locks ─────────────────────────────────
print("\n[3] TRIGGER_FORMULA — when n alone fails")
cases = [
    b"a",
    b"ab",
    b"aba",
    b"abab",
    b"abcabc",
    b"aabb",
    b"aaabbb",
    b"abcba",
]
for data in cases:
    alpha = SymbolAlphabet.from_bytes(data)
    locks = _ambiguous_locks(data, alpha)
    pct = 100 * len(locks) / max(1, len(data) - 1)
    print(f"  {data!r} (|sym|={alpha.n}): {len(locks)}/{len(data)-1} steps need 3-way lock ({pct:.0f}%)")
    if locks and len(data) <= 8:
        print(f"    locks at walk steps: {locks}")

# minimal ambiguous alphabet
for sym_n in (3, 4, 5):
    syms = bytes(range(ord("a"), ord("a") + sym_n))
    # cyclic string forces repeated pair_n
    data = syms * 3
    alpha = SymbolAlphabet.from_bytes(data)
    locks = _ambiguous_locks(data, alpha)
    print(f"  cyclic {sym_n}-symbol alphabet: {len(locks)}/{len(data)-1} ambiguous "
          f"({'ALL need lock' if len(locks)==len(data)-1 else 'partial'})")

# ── 4. deep_branch triple case parity ──────────────────────────────────
print("\n[4] DEEP_BRANCH — triple case parity / emergent structure")
for data in (b"abc", b"ababab", b"aaa", b"abccba"):
    alpha = SymbolAlphabet.from_bytes(data)
    cases = triple_case_stream(data, alpha)
    parity = sum(c % 2 for c in cases)
    print(f"  {data!r}: cases={cases}  odd-count={parity}  len={len(cases)}")

# case distribution over random triples
case_hist = Counter()
parity_hist = Counter()
primes = make_chain(SequenceKind.PRIMES, 30)[5:20]
for a, b, c in itertools.combinations(primes, 3):
    _, _, n, case = section5_triple_roles((a, b, c))
    case_hist[case] += 1
    parity_hist[case % 2] += 1
print(f"  prime triples (n=15 choose 3): case dist {dict(case_hist)}")
print(f"  case parity (odd/even): {dict(parity_hist)}")

# case vs n mod 2 for fixed corridor
a, p = 3, 11
case_by_n_mod = defaultdict(set)
for n in range(1, 50):
    c = prime_pair_case(a, p, n)
    case_by_n_mod[n % 2].add(c)
print(f"  corridor (3,11): cases at even-n={case_by_n_mod[0]} odd-n={case_by_n_mod[1]}")

# ── 5. Promotion ladder 2→1099→1 ─────────────────────────────────────
print("\n[5] PROMOTION LADDER — fixed points & cycles")
pool0 = PROMOTION_POOL[0]
pool1098 = PROMOTION_POOL[1098] if len(PROMOTION_POOL) > 1098 else None
print(f"  PROMOTION_POOL len={len(PROMOTION_POOL)}  [0]={pool0}  [1098]={pool1098}")

# L2→L3→L1: thing paths
words = {
    "th": intersection_prime("th"),
    "ing": intersection_prime("ing"),
    "thing_letters": intersection_prime("thing"),
}
print(f"  letter intersections: th={words['th']} ing={words['ing']} thing={words['thing_letters']}")

# meet composite products (Stage 05)
from lattice_retriever_v1.stage05_free_token import meet_composite

th_p, ing_p = letter_to_prime("t"), letter_to_prime("h")  # wrong - use actual
from aethos_words import word_to_order
th_primes = word_to_order("th")
ing_primes = word_to_order("ing")
# simulate pool promotion primes
p_th = pool0
p_ing = PROMOTION_POOL[1]
prod_th_ing = meet_composite(p_th, p_ing)
prod_t_hing = meet_composite(letter_to_prime("t"), PROMOTION_POOL[2])
print(f"  pool th×ing={prod_th_ing}  t×hing={prod_t_hing}  thing_letters={words['thing_letters']}")
print(f"  all three distinct: {len({prod_th_ing, prod_t_hing, words['thing_letters']})==3}")

# transgressor n cycle: does n=1 ever equal n=1099 placement?
bank_p = LatticeBank32.single_prime(pool0)
c1 = bank_p[LatticeId.L01].at(1)
c1099 = bank_p[LatticeId.L01].at(1099) if 1099 < 5000 else None
print(f"  pool prime {pool0} L01: n=1 -> {c1}")
if c1099:
    print(f"  n=1099 -> {c1099}  (fixed point: {c1==c1099})")

# promotion attractor: intersection_prime always ≤ sum of letters
samples = ["cat", "dog", "thing", "the", "ing"]
for w in samples:
    ic = intersection_prime(w)
    lc = sum(letter_chain(w))
    print(f"  intersection({w!r})={ic}  sum(letters)={lc}  ratio={ic/lc:.3f}")

# ── 6. entangle_imag partitions ────────────────────────────────────────
print("\n[6] entangle_imag — imaginary axis regions")
alpha = SymbolAlphabet.from_bytes(b"abcdefghijklmnopqrstuvwxyz")
imag_grid = {}
for a in alpha.symbols[:8]:
    for b in alpha.symbols[:8]:
        pa, pb = alpha.prime_for(a), alpha.prime_for(b)
        im = entangle_imag(pa, pb)
        imag_grid[(a, b)] = im
vals = list(imag_grid.values())
print(f"  8×8 symbol grid: imag distinct={len(set(vals))} min={min(vals)} max={max(vals)}")
# partition by sign of imag contribution
pos = sum(1 for v in vals if v > 0)
neg = sum(1 for v in vals if v < 0)
zero = sum(1 for v in vals if v == 0)
print(f"  sign partition: pos={pos} neg={neg} zero={zero}")

# do opposite coins cluster in same imag band?
data = b"the cat sat on the mat"
cat = build_electron_alphabet(data)
alpha2 = SymbolAlphabet.from_bytes(data)
by_opposite = {"opp": [], "same": []}
for i in range(len(data) - 1):
    w = entangle_witness(data[i], data[i + 1], alpha2, cat)
    key = "opp" if w.opposite else "same"
    by_opposite[key].append(w.intersection_imag)
print(f"  opposite-membrane pairs imag mean={sum(by_opposite['opp'])/max(1,len(by_opposite['opp'])):.1f} "
      f"same-membrane mean={sum(by_opposite['same'])/max(1,len(by_opposite['same'])):.1f}")

# ── 7. (a,p) pairs → same meet vector? invertibility ───────────────────
print("\n[7] MEET VECTOR COLLISIONS — (a,p) invertibility")
# 2-way swap: bank(a)@n=p vs bank(p)@n=a
collisions_ap = []
for a, p in itertools.combinations(make_chain(SequenceKind.PRIMES, 20)[3:12], 2):
    w1 = k_swap_meet(a, p)
    w2 = k_swap_meet(p, a)
    if w1.unified and w2.unified:
        collisions_ap.append((a, p, w1.coord))
print(f"  swap_meet(a,p) unified for {len(collisions_ap)}/36 pairs; all symmetric: "
      f"{all(c[2]==k_swap_meet(c[1],c[0]).coord for c in collisions_ap)}")

# different (a,p) same 32-signature?
sig_map = defaultdict(list)
test_primes = make_chain(SequenceKind.PRIMES, 25)[5:18]
for a, p in itertools.combinations(test_primes, 2):
    sig = lattice_signature((a, p), n=7)
    sig_map[sig].append((a, p))
dup_sigs = [(k, v) for k, v in sig_map.items() if len(v) > 1]
print(f"  32-lattice signatures @ n=7: {len(sig_map)} unique, {len(dup_sigs)} collisions")
if dup_sigs:
    print(f"    example collision: {dup_sigs[0][1]}")

# anchor_sum degeneracy (metadata not identity)
sum_map = defaultdict(list)
for a, b, c in itertools.combinations(test_primes, 3):
    s = a + b + c
    sum_map[s].append((a, b, c))
degen = sum(1 for v in sum_map.values() if len(v) > 1)
print(f"  anchor_sum (a+p+n) collisions among triples: {degen}/{len(sum_map)} sums shared")

# meet_composite IS invertible for semiprimes
from lattice_retriever_v1.stage05_free_token import factor_pair_composite
semiprime_collisions = 0
for pairs in sig_map.values():
    for a, p in pairs:
        comp = a * p
        fa, fp = factor_pair_composite(comp)
        if (fa, fp) != (min(a, p), max(a, p)):
            semiprime_collisions += 1
print(f"  meet_composite factor-back failures: {semiprime_collisions} (expect 0)")

# ── 8. zeta=height, min=depth landscape ────────────────────────────────
print("\n[8] HEIGHT (zeta) vs DEPTH (min) LANDSCAPE")
chain = (3, 5, 7, 11)
for n in range(1, 20):
    psi = wing_transform(BranchKind.VA1, chain, n, 1)
    seg_min = min(chain)
    height = psi.zeta
    depth_feat = height - sum_chain(chain)  # excess over sum at interior
    print(f"  n={n:2d}  zeta={height:5.0f}  sum={sum_chain(chain)}  excess={depth_feat:5.0f}  "
          f"seg={('below' if n<min(chain) else 'interior' if n<max(chain) else 'above')}")

# wells: local minima of |z| across n rail
ns = range(1, 100)
moduli = [abs(wing_transform(BranchKind.VA1, (3, 11), n, 1).z) for n in ns]
wells = [n for n in ns[1:-1] if moduli[n - ns.start] < moduli[n - 1 - ns.start] and moduli[n - ns.start] < moduli[n + 1 - ns.start]]
print(f"  corridor (3,11) spring modulus wells at n={wells[:8]}... ({len(wells)} total in 1..99)")

# interior lock plateau
interior_zeta = set()
for n in range(5, 10):  # between 3 and 11
    psi = wing_transform(BranchKind.VA1, chain, n, 1, lock_interior=True)
    interior_zeta.add(round(psi.zeta, 3))
print(f"  interior lock (3,5,7,11) n=5..9: zeta plateau values={interior_zeta}")

print("\n" + "=" * 72)
print("DONE")
