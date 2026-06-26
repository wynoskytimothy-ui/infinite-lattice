#!/usr/bin/env python3
"""PI + wave + 4-6D playground — numerical probes for higher-dim oscillation vision."""
from __future__ import annotations

import itertools
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_complex_plane import equalize_witness, swap_meet, wing_transform
from aethos_lattice import BranchKind, LatticeBank32, LatticeId, lattice_id_parts, prime_pair_case
from aethos_pi_bridge import (
    compare_pi_vertex_to_spring_i_act,
    pi_branch_bits_to_wing_mask,
    pi_dyadic_point,
    pi_layer0_direction_matches,
)
from aethos_promotion import CorrelationLink, LatticeTier, PromotedToken
from aethos_spring_complex import SpringPoint, i_act, swap_xy
from aethos_words import letter_to_prime
from lattice_retriever_v1.stage03_rotation import (
    NUM_QUADRANTS,
    wing_and_branch_from_quadrant,
    wing_from_frequency_profile,
)
from lattice_retriever_v1.stage07_semantic_light import correlation_dims, rotation_quadrant_l4
from lattice_retriever_v1.wing_channel_codec import wing_channel_at
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from pi.constructive_pi import pi_recurrence, point_on_circle_complex, sin_cos_table

TWO_PI = 2 * math.pi
PI_OVER_2 = math.pi / 2


def primes_upto(n: int) -> list[int]:
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]


def header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ── Q1: 32 wings ↔ 2π, frequency periodicity ─────────────────────────────
header("Q1: 32-WING ROTATION <-> 2pi / CONSTRUCTIVE PI")

# Pi bisection: N_k = 4 * 2^k → N_3 = 32 vertices on unit circle
pi_levels = list(pi_recurrence(6))
for k, N, A, B, C, area in pi_levels[:6]:
    print(f"  pi level k={k}: N={N} vertices, half-angle={180/N:.4f}°, "
          f"arc/step = {TWO_PI/N:.6f} rad")

print(f"\n  MATCH: N at k=3 equals NUM_QUADRANTS={NUM_QUADRANTS}? "
      f"{pi_levels[3][1] == NUM_QUADRANTS}")
print(f"  MATCH: 4 branches × 8 wings = {4*8} = NUM_QUADRANTS")

# Map quadrant q → angle if uniform on circle
angles_q = [TWO_PI * (q - 1) / NUM_QUADRANTS for q in range(1, NUM_QUADRANTS + 1)]
# Pi dyadic vertex j at k=3 (N=32)
pi_k3_angles = []
for j in range(32):
    re, im = point_on_circle_complex(3, j)
    pi_k3_angles.append(math.atan2(float(im), float(re)) % TWO_PI)

# Nearest pi-vertex to each quadrant slot
nearest = []
for ang in angles_q:
    best = min(range(32), key=lambda j: min(abs(pi_k3_angles[j] - ang),
                                            TWO_PI - abs(pi_k3_angles[j] - ang)))
    nearest.append(best)
print(f"  Uniform 32-slot → nearest pi-k3 vertex indices (first 8): {nearest[:8]}")
print(f"  Unique pi vertices hit by uniform slots: {len(set(nearest))}/32")

# Frequency profile → quadrant: periodicity / sensitivity
print("\n  Frequency → wing periodicity (mod 32):")
profiles = [
    ((100,),),
    ((100, 1),),
    ((1, 100),),
    ((50, 50),),
    ((50, 51),),
]
for prof in profiles:
    q = wing_from_frequency_profile(prof[0])
    w, b = wing_and_branch_from_quadrant(q)
    print(f"    profile={prof[0]} → q={q} wing={w} branch={b.name}")

# Sweep one-symbol df: how many distinct quadrants in 1..500?
qs = {wing_from_frequency_profile((d,)) for d in range(1, 501)}
print(f"  Single-symbol df 1..500 → {len(qs)} distinct quadrants (of 32)")

# Two-symbol: position weight creates phase offset (anagram = different q)
anagram_pairs = [("cat", "act"), ("tar", "rat"), ("stop", "pots")]
print("  Anagram quadrant separation (synthetic df=10 each char):")
for a, b in anagram_pairs:
    pa = tuple(10 for _ in a)
    pb = tuple(10 for _ in b)
    qa, qb = wing_from_frequency_profile(pa), wing_from_frequency_profile(pb)
    print(f"    {a!r} q={qa}  {b!r} q={qb}  Δ={abs(qa-qb)}")


# ── Q2: L4-L6 as wave harmonics on frozen zeta mesa ─────────────────────
header("Q2: L4-L6 CORRELATION AS WAVE HARMONICS ON ZETA MESA")

ps = primes_upto(60)[2:15]  # small primes for letter mapping
# Build dim triples for all pairs
triples: list[tuple[int, int, float, float, float]] = []
for a, b in itertools.combinations(ps, 2):
    d4, d5, d6 = correlation_dims(a, b, strength=1)
    triples.append((a, b, d4, d5, d6))

# "Mesa" = zeta locked at sum(chain) for interior n
A = (3, 5, 7, 11)
zeta_mesa = sum(A)
interior_zetas = []
boundary_zetas = []
for n in range(1, 20):
    psi = wing_transform(BranchKind.VA1, A, n, 1)
    if 3 < n < 11:  # interior between anchors
        interior_zetas.append(psi.zeta)
    else:
        boundary_zetas.append(psi.zeta)
print(f"  Chain {A}: interior zeta values (should lock): {set(interior_zetas)}")
print(f"  Mesa target sum(A)={zeta_mesa}, locked? {all(abs(z - zeta_mesa) < 1e-9 for z in interior_zetas)}")
print(f"  Boundary zeta spread (sample): {sorted(set(round(z, 2) for z in boundary_zetas))[:8]}")

# Treat dim4,dim5,dim6 as (A,B,C) wave amplitudes; look for harmonic structure
d4s = [t[2] for t in triples]
d5s = [t[3] for t in triples]
d6s = [t[4] for t in triples]
print(f"\n  L4–L6 ranges over prime pairs: dim4 [{min(d4s):.2f},{max(d4s):.2f}] "
      f"dim5 [{min(d5s):.2f},{max(d5s):.2f}] dim6 [{min(d6s):.2f},{max(d6s):.2f}]")

# Phase proxy: atan2(dim6, dim4) as angular coordinate on correlation plane
phases = [math.atan2(d6, d4) for _, _, d4, _, d6 in triples]
# dim5 as radial / envelope
radii = [math.sqrt(d4 * d4 + d6 * d6) for _, _, d4, _, d6 in triples]
print(f"  Phase atan2(d6,d4): {len(set(round(p, 2) for p in phases))} distinct buckets (0.01 rad)")

# Harmonic test: dim5 ≈ f(dim4, dim6) — correlation
mean_d5 = sum(d5s) / len(d5s)
ss_res = sum((d5 - (d4 + d6) / 2) ** 2 for d4, d5, d6 in zip(d4s, d5s, d6s))
ss_tot = sum((d5 - mean_d5) ** 2 for d5 in d5s)
r2_envelope = 1 - ss_res / ss_tot if ss_tot else 0
print(f"  dim5 vs (dim4+dim6)/2 envelope R²={r2_envelope:.4f} (standing-wave midline?)")

# Oscillation across prime gap: dim values vs |a-b|
gap_corr = []
for a, b, d4, d5, d6 in triples:
    gap = abs(a - b)
    gap_corr.append((gap, d4 + d5 + d6))
gaps = [g for g, _ in gap_corr]
amps = [s for _, s in gap_corr]
# simple sign of oscillation: count peaks
sorted_by_gap = sorted(gap_corr)
amp_diffs = [sorted_by_gap[i + 1][1] - sorted_by_gap[i][1]
             for i in range(len(sorted_by_gap) - 1)]
sign_changes = sum(1 for i in range(len(amp_diffs) - 1) if amp_diffs[i] * amp_diffs[i + 1] < 0)
print(f"  Total amplitude (d4+d5+d6) vs prime-gap: {sign_changes} sign-changes "
      f"(oscillation proxy) in {len(amp_diffs)} steps")


# ── Q3: 96 wing-channel states as wave modes ────────────────────────────
header("Q3: 96 WING-CHANNEL STATES = 3 CASES × 32 WINGS")

alpha = SymbolAlphabet.from_bytes(b"abcde")
data = b"abcde" * 20
counts = Counter(data)
channels: Counter[int] = Counter()
case_wing: Counter[tuple[int, int]] = Counter()
for i in range(len(data) - 1):
    ch = wing_channel_at(alpha, data[i], data[i + 1], n=i + 1, sym_counts=counts)
    channels[ch.channel_id] += 1
    case_wing[(ch.case, ch.wing)] += 1

print(f"  Distinct channel_ids used on sample: {len(channels)}/96 possible")
print(f"  channel_id range: 0..95 = (case-1)*32 + wing")
# Decompose: 3 regimes × 32 wings
cases_used = {c for c, _ in case_wing}
wings_used = {w for _, w in case_wing}
print(f"  Cases seen: {sorted(cases_used)}  Wings seen: {len(wings_used)}/8 unique wings")
print(f"  Full grid coverage: {len(case_wing)}/96 cells occupied")

# Mode number: channel_id as angular frequency index
mode_freqs = [ch_id / 96 * TWO_PI for ch_id in range(96)]
print(f"  Mode 0 phase=0, mode 24 phase=pi, mode 48 phase=pi, mode 72 phase=3pi/2 "
      f"(uniform grid on [0,2pi))")

# Case boundaries = rail regime switches (wave packet envelope)
for a, p in [(3, 5), (3, 11), (5, 97)]:
    boundaries = []
    for n in range(1, max(a, p) + 5):
        c = prime_pair_case(a, p, n)
        if not boundaries or boundaries[-1][1] != c:
            boundaries.append((n, c))
    print(f"  Regime switches (a,p)=({a},{p}): {boundaries}")


# ── Q4: VB Y↔X swap as π/2 phase shift ──────────────────────────────────
header("Q4: VB Y<->X SWAP <-> pi/2 PHASE (i_act)")

print(f"  PROVEN layer0 diagonal = pi/4: {pi_layer0_direction_matches()}")

# VA wing 1 vs VB wing 5 on same (chain, n)
A_solo = (5,)
n = 7
va = wing_transform(BranchKind.VA1, A_solo, n, 1)
vb = wing_transform(BranchKind.VA1, A_solo, n, 5)
print(f"  VA1 w1: z={va.z.real:.4f}+{va.z.imag:.4f}i  zeta={va.zeta}")
print(f"  VB  w5: z={vb.z.real:.4f}+{vb.z.imag:.4f}i  zeta={vb.zeta}")
print(f"  VB coord == (Y,X,Z) swap of VA: {vb.coord == (va.coord[1], va.coord[0], va.coord[2])}")

# Angle change VA → VB
ang_va = math.atan2(va.z.imag, va.z.real)
ang_vb = math.atan2(vb.z.imag, vb.z.real)
delta = (ang_vb - ang_va) % TWO_PI
print(f"  arg(z) VA→VB delta = {delta:.4f} rad = {math.degrees(delta):.2f}° "
      f"(pi/2={PI_OVER_2:.4f})")

# Compare to i_act on spring plane
sp_va = SpringPoint(va.z.real, va.z.imag)
sp_swap = swap_xy(sp_va)
sp_i = i_act(SpringPoint(1, 0))
print(f"  i_act(1,0) = ({sp_i.x:.4f}, {sp_i.y:.4f})  [unit i, angle pi/2]")

# Sweep all 8 wing pairs VA w ↔ VB w+4
deltas = []
for w in range(1, 5):
    va_w = wing_transform(BranchKind.VA1, A_solo, n, w)
    vb_w = wing_transform(BranchKind.VA1, A_solo, n, w + 4)
    av = math.atan2(va_w.z.imag, va_w.z.real)
    bv = math.atan2(vb_w.z.imag, vb_w.z.real)
    d = (bv - av) % TWO_PI
    deltas.append(d)
print(f"  VA↔VB arg deltas (w1..4): {[f'{d:.3f}' for d in deltas]}")
near_half_pi = sum(1 for d in deltas if abs(d - PI_OVER_2) < 0.15 or abs(d - 3 * PI_OVER_2) < 0.15)
print(f"  Near pi/2 or 3pi/2: {near_half_pi}/4 wing pairs")

# Pi branch bits → wing mask (partial functor)
for bits in [(0,), (1,), (0, 1), (1, 0, 1), (1, 1, 1)]:
    mask = pi_branch_bits_to_wing_mask(bits)
    print(f"  pi bits {bits} → wing_mask={mask} (VB/flip bits)")


# ── Q5: Composed meets as standing waves ──────────────────────────────────
header("Q5: COMPOSED MEETS — PLATEAU (NODE) VS BOUNDARY (ANTINODE)")

def zeta_profile(chain: tuple[int, ...], n_max: int = 15) -> list[tuple[int, float, float]]:
    """(n, zeta, |z|) along rail."""
    out = []
    for n in range(1, n_max + 1):
        psi = wing_transform(BranchKind.VA1, chain, n, 1)
        out.append((n, psi.zeta, abs(psi.z)))
    return out


for chain in [(3, 5, 7), (3, 5, 7, 11), (2, 3, 5, 7, 11)]:
    prof = zeta_profile(chain)
    zetas = [z for _, z, _ in prof]
    mags = [m for _, _, m in prof]
    interior = [z for n, z, _ in prof if chain[0] < n < chain[-1]]
    boundary = [z for n, z, _ in prof if n <= chain[0] or n >= chain[-1]]
    var_int = max(interior) - min(interior) if interior else 0
    var_bnd = max(boundary) - min(boundary) if boundary else 0
    # |z| oscillation
    mag_range_int = max(mags[1:-1]) - min(mags[1:-1]) if len(mags) > 2 else 0
    print(f"\n  chain {chain}:")
    print(f"    zeta interior variance={var_int:.6f}  boundary variance={var_bnd:.6f}")
    print(f"    |z| interior swing={mag_range_int:.4f} (antinode proxy at boundaries)")

# Triple equalization: composed meet plateau
A = (3, 5, 7)
for drop in A:
    sub = tuple(x for x in A if x != drop)
    n_w, psi = equalize_witness(A, sub)
    print(f"  equalize {A} \\ {{{drop}}} → n_w={n_w} zeta={psi.zeta} z={psi.z}")

# Swap meet: left/right same coord = standing node
print("\n  Swap-meet nodes (left.coord == right.coord):")
for a, p in [(3, 11), (5, 13), (7, 97)]:
  for w in [1, 2, 5, 8]:
    left, right = swap_meet(a, p, BranchKind.VA1, w)
    match = left.coord == right.coord
    if w == 1:
      print(f"    ({a},{p}) w={w}: unified={match} zeta_L={left.zeta} zeta_R={right.zeta}")

# 32-lattice bank: collision classes = nodes where wings interfere to same coord
bank = LatticeBank32.prime_pair(3, 5)
n = 5
collisions = bank.find_same_n_collisions(n)
print(f"\n  pair (3,5) @ n=5: {len(collisions)} collision classes (standing nodes)")
for coord, lids in collisions[:3]:
    print(f"    node {coord} ← {len(lids)} wings fold together")


# ── SYNTHESIS: 4D/5D/6D as oscillating lifts ─────────────────────────────
header("SYNTHESIS: 4D/5D/6D WAVE COMPOSITION — BUILD STRUCTURES")

# Structure 1: Phase lattice — (dim4, dim5, dim6) + rotation_quadrant = 4D cage coords
print("  BUILD 1 — Phase cage per prime-pair edge:")
sample_words = ["cat", "dog", "quantum"]
word_primes = {w: letter_to_prime(w[0]) * letter_to_prime(w[-1]) for w in sample_words}
for w1, w2 in itertools.combinations(sample_words, 2):
    p1, p2 = word_primes[w1], word_primes[w2]
    d4, d5, d6 = correlation_dims(p1, p2)
    rq = rotation_quadrant_l4(p1, p2)
    phase = math.atan2(d6, d4)
    print(f"    {w1}—{w2}: phase={phase:.3f} rad, rq={rq}, envelope d5={d5:.3f}")

# Structure 2: 32-wing carrier × 3-case envelope = 96-mode alphabet
print("\n  BUILD 2 — 96-mode symbol codec (case=envelope, wing=carrier):")
print("    Each byte-pair → (case, wing) = complex amplitude on frozen zeta doc shell")
print("    Decompress = lazy_read_channel() regenerates symbol from formula branch")

# Structure 3: Pi level k indexes transgressor depth; wing mask from bisection path
print("\n  BUILD 3 — Pi depth ↔ rail depth ↔ wing selector:")
for k in range(5):
    N = 4 * (2 ** k)
    n_rail = 2 ** k
    print(f"    pi k={k}: N={N} vertices, demo rail n=2^k={n_rail}, "
          f"angle step=2π/{N}={TWO_PI/N:.5f} rad")

# Structure 4: Miniverse interference — 4 branches as quarter-wave plate
print("\n  BUILD 4 - 4 VA branches as pi/2 phase fan on same zeta mesa:")
n = 5
A = (3, 5, 7)
branch_angles = []
for b in BranchKind:
    psi = wing_transform(b, A, n, 1)
    branch_angles.append((b.name, math.atan2(psi.z.imag, psi.z.real), psi.zeta))
for name, ang, zeta in branch_angles:
    print(f"    {name}: arg(z)={ang:.4f}, zeta={zeta} (mesa locked={zeta==sum(A)})")

# Structure 5: Triple meet spawns L4-L6 wing cage (miniverse birth)
print("\n  BUILD 5 — Triple meet → wing cage (higher dim lift at locked node):")
from lattice_retriever_v1.stage06_composites import ing_three_way_demo
addr = ing_three_way_demo(quadrant=6)
print(f"    ing triple: meet={addr.meet_composite} quadrant={addr.quadrant}")
print("    L4=d4 witness A, L5=d5 witness B, L6=d6 witness C — oscillate strength, not address")

# Pi ↔ spring alignment sample
print("\n  Pi vertex ↔ spring plane alignment (partial functor):")
for k, j in [(1, 1), (2, 3), (3, 7), (3, 15)]:
    cmp = compare_pi_vertex_to_spring_i_act(k, j)
    print(f"    k={k} j={j}: match={cmp['angles_match']} diff={cmp['angle_diff']:.4f}")

print("\nDONE — see sections above for emergent BUILD patterns.")
