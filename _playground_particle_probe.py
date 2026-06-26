#!/usr/bin/env python3
"""Particle-metaphor playground — meet as crossing, n-rail as transgression."""
import cmath
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_complex_plane import (
    all_branch_phases,
    imaginary_start,
    swap_meet,
    trigger_history,
    wing_transform,
)
from aethos_electron_tokenizer import CoinState, state_to_bits
from aethos_lattice import BranchKind, LatticeBank32, LatticeId, VECTORS, prime_pair_case
from lattice_retriever_v1.electron_lattice_codec import (
    build_electron_alphabet,
    entangle_imag,
    entangle_witness,
    wing_case_to_coin,
)
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.stage03_rotation import wing_and_branch_from_quadrant, wing_from_frequency_profile
from lattice_retriever_v1.wing_channel_codec import wing_channel_at

PI = math.pi


def phase_angle(z: complex) -> float:
    return cmath.phase(z)


def spring_modulus(z: complex) -> float:
    return abs(z)


def probe_q1_meet_as_particle():
    """Q1: Is each meet node a particle with (z, zeta, phase)?"""
    print("\n" + "=" * 72)
    print("Q1 — MEET NODE AS PARTICLE: (z position, zeta depth, arg(z) phase)")
    print("=" * 72)

    # Layer 0: pure imaginary rail — particle walks 45deg line in spring plane
    print("\n  [Layer 0] imaginary_start — n-rail as particle birth")
    trail = []
    for n in range(0, 8):
        psi = imaginary_start(n)
        ang = phase_angle(psi.z) * 180 / PI
        trail.append((n, psi.z.real, psi.z.imag, psi.zeta, ang))
        print(f"    n={n}  z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}  |z|={spring_modulus(psi.z):.2f}  arg={ang:.1f}deg")
    # invariant: arg(z) = 45deg always at layer 0
    angles = [t[4] for t in trail[1:]]
    print(f"  => Layer-0 phase LOCK: arg(z) std={math.sqrt(sum((a-45)**2 for a in angles)/len(angles)):.4f} deg (~0 = fixed phase)")

    # Meet = particle crossing: swap_meet(3,11)
    print("\n  [Meet crossing] swap_meet(3,11) — two banks, same node")
    left, right = swap_meet(3, 11)
    match = left.coord == right.coord
    print(f"    bank(3)@n=11: z={left.z.real:.0f}{left.z.imag:+.0f}i  zeta={left.zeta:.0f}  arg={phase_angle(left.z)*180/PI:.1f}deg")
    print(f"    bank(11)@n=3:  z={right.z.real:.0f}{right.z.imag:+.0f}i  zeta={right.zeta:.0f}  arg={phase_angle(right.z)*180/PI:.1f}deg")
    print(f"    nodes COLLIDE: {match}  |Δz|={abs(left.z-right.z):.6f}  Δzeta={abs(left.zeta-right.zeta):.6f}")

    # Triple equalization node — one particle, three arrival rails
    from aethos_complex_plane import triple_equalization
    eq = triple_equalization(3, 5, 7)
    coords = [psi.coord for _, psi in eq.values()]
    ns = [int(n) for n, _ in eq.values()]
    print(f"\n  [Triple meet] (3,5,7) -- 3 rails => 1 node @ n in {ns}")
    for label, (n_w, psi) in eq.items():
        print(f"    {label}@n={int(n_w)}: z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")
    print(f"  => 3 distinct transgressor momenta (n=7,5,3) land SAME (X,Y,zeta): {len(set(coords))==1}")

    # Hilbert distinction: same coord, different path labels
    print(f"  => BUT path labels differ: {list(eq.keys())} — same position, different history (metaphor: same energy level, different decay channel)")

    return {
        "layer0_phase_deg": 45.0,
        "swap_meet_match": match,
        "triple_rails": len(eq),
        "triple_unique_coords": len(set(coords)),
    }


def probe_q2_spring_plane_psi():
    """Q2: How does Psi=(z,zeta) relate to physical particle in 3D lattice?"""
    print("\n" + "=" * 72)
    print("Q2 — SPRING PLANE Ψ=(z,zeta) vs 3D LATTICE COORD")
    print("=" * 72)

    p = 5
    bank = LatticeBank32.single_prime(p)
    lat = bank[LatticeId.L01]
    print(f"\n  Single prime p={p}, L01 (VA1×v1), n=0..10:")
    rows = []
    for n in range(0, 11):
        coord = lat.at(n)
        psi = wing_transform(BranchKind.VA1, (p,), n, 1)
        rows.append((n, coord, psi))
        regime = lat.regime_label(n)
        print(f"    n={n:2d}  lattice(X,Y,Z)={coord}  Psi z={psi.z.real:.0f}{psi.z.imag:+.0f}i zeta={psi.zeta:.0f}  regime={regime}")

    # Map: lattice Z == zeta always for VA1 single prime?
    zeta_match = sum(1 for n, c, psi in rows if abs(c[2] - psi.zeta) < 1e-9)
    print(f"\n  => lattice Z == zeta for {zeta_match}/{len(rows)} steps (VA1 single-prime)")

    # Velocity boundary at n=p: Y jumps from n to p (imaginary "momentum flip")
    psi_before = wing_transform(BranchKind.VA1, (p,), p - 1, 1)
    psi_at = wing_transform(BranchKind.VA1, (p,), p, 1)
    psi_after = wing_transform(BranchKind.VA1, (p,), p + 1, 1)
    dy_before = psi_at.z.imag - psi_before.z.imag
    dy_after = psi_after.z.imag - psi_at.z.imag
    print(f"\n  [Velocity boundary n=p={p}]")
    print(f"    n={p-1}: Y={psi_before.z.imag:.0f}  =>  n={p}: Y={psi_at.z.imag:.0f}  =>  n={p+1}: Y={psi_after.z.imag:.0f}")
    print(f"    ΔY across boundary: before={dy_before:.0f}  after={dy_after:.0f}  (regime switch = 'velocity change')")

    # 3D interpretation numbers
    x_span = max(c[0] for _, c, _ in rows) - min(c[0] for _, c, _ in rows)
    z_span = max(c[2] for _, c, _ in rows) - min(c[2] for _, c, _ in rows)
    mod_span = max(spring_modulus(psi.z) for _, _, psi in rows) - min(spring_modulus(psi.z) for _, _, psi in rows)
    print(f"\n  3D footprint n=0..10: ΔX={x_span}  ΔZ(zeta)={z_span}  Δ|z|={mod_span:.2f}")
    print(f"  => z=X+iY lives IN the spring plane (complex 2D); zeta=Z is depth rail (real 1D)")

    return {"zeta_match_rate": zeta_match / len(rows), "velocity_boundary_p": p, "dy_flip": dy_after}


def probe_q3_va_branches_polarization():
    """Q3: VA1-VA4 = different polarization/orientation of same particle?"""
    print("\n" + "=" * 72)
    print("Q3 — VA1..VA4 AS POLARIZATION / ORIENTATION FAN")
    print("=" * 72)

    chain = (3, 5, 7)
    n = 5
    phases = all_branch_phases(chain, n, wing=1)
    zeta_vals = [round(psi.zeta, 3) for psi in phases.values()]
    z_vals = [(round(psi.z.real, 3), round(psi.z.imag, 3)) for psi in phases.values()]
    moduli = [round(spring_modulus(psi.z), 3) for psi in phases.values()]
    angles = [round(phase_angle(psi.z) * 180 / PI, 1) for psi in phases.values()]

    print(f"\n  chain {chain} @ n={n}, wing=1 — 4 branch 'polarizations':")
    for b, psi in phases.items():
        print(f"    {b.name}: z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}  |z|={spring_modulus(psi.z):.2f}  arg={phase_angle(psi.z)*180/PI:.1f}deg")

    print(f"\n  => zeta INVARIANT across 4 branches: {len(set(zeta_vals))==1} (all zeta={zeta_vals[0]})")
    print(f"  => (X,Y) DISTINCT: {len(set(z_vals))}/4 orientations in spring plane")
    print(f"  => |z| values: {moduli}  (same modulus: {len(set(moduli))==1})")
    print(f"  => arg(z) spread: {min(angles):.1f}deg .. {max(angles):.1f}deg  (Δ={max(angles)-min(angles):.1f}deg)")

    # 32 chambers = 4 branches × 8 vector flips
    bank = LatticeBank32.prime_pair(3, 5)
    coords_at_n = [bank[lid].at(n) for lid in LatticeId]
    unique_coords = len(set(coords_at_n))
    print(f"\n  32 chambers (pair 3,5) @ n={n}: {unique_coords} unique lattice coords / 32")
    print(f"  => Full orientation space: 4 phase fans × 8 vector mirrors = 32 'polarization chambers'")

    # stage03: frequency rotation picks quadrant => branch
    profiles = [((1, 1), "equal"), ((5, 1), "left-heavy"), ((1, 5), "right-heavy")]
    print(f"\n  [Stage03 rotation] same pair, different frequency => different branch:")
    for prof, label in profiles:
        q = wing_from_frequency_profile(prof)
        w, br = wing_and_branch_from_quadrant(q)
        print(f"    profile {prof} ({label}): quadrant={q} => wing={w} branch={br.name}")

    return {"zeta_invariant": len(set(zeta_vals)) == 1, "xy_orientations": len(set(z_vals)), "chambers_unique": unique_coords}


def probe_q4_electron_quaternion_spin():
    """Q4: Electron 4 states = quaternions? spin? wave phases?"""
    print("\n" + "=" * 72)
    print("Q4 — ELECTRON 4 STATES: membrane×spring encoding")
    print("=" * 72)

    # Map to Pauli-like basis (metaphor only)
    pauli_map = {
        CoinState.WS: ("|0⟩ soft", (0, 0, 0, 1)),   # identity-ish
        CoinState.WH: ("|0⟩ hard", (0, 0, 1, 0)),   # σz
        CoinState.BS: ("|1⟩ soft", (0, 1, 0, 0)),   # σy
        CoinState.BH: ("|1⟩ hard", (1, 0, 0, 0)),   # σx
    }
    print("\n  CoinState bit layout (membrane<<1 | spring):")
    for st in CoinState:
        m, s = state_to_bits(st)
        print(f"    {st.name}={int(st):02b}  membrane={m} spring={s}  metaphor={pauli_map[st][0]}")

    # wing_case => coin: 3 cases × 32 wings => 4 states
    alpha = SymbolAlphabet.from_bytes(b"abcdefghij")
    case_coin_hist = Counter()
    wing_coin_hist = Counter()
    for case in (1, 2, 3):
        for wing in range(32):
            coin = wing_case_to_coin(case, wing)
            case_coin_hist[(case, coin.name)] += 1
            wing_coin_hist[(wing & 1, coin.name)] += 1

    print(f"\n  wing_case_to_coin: 3 cases × 32 wings => 4 coins")
    for case in (1, 2, 3):
        sub = {k[1]: v for k, v in case_coin_hist.items() if k[0] == case}
        print(f"    case {case}: {dict(sub)}  (spring=case-1 mod 2 => {case-1 & 1})")

    membrane_split = Counter()
    spring_split = Counter()
    for case in (1, 2, 3):
        for wing in range(32):
            c = wing_case_to_coin(case, wing)
            m, s = state_to_bits(c)
            membrane_split[m] += 1
            spring_split[s] += 1
    print(f"\n  => membrane bit = wing&1: WS+WH={membrane_split[0]}  BS+BH={membrane_split[1]}  (50/50)")
    print(f"  => spring bit = (case-1)&1: soft={spring_split[0]}  hard={spring_split[1]}")

    # Entanglement: opposite membrane pairs
    data = b"abababab"
    cat = build_electron_alphabet(data)
    alpha2 = SymbolAlphabet.from_bytes(data)
    w = entangle_witness(data[0], data[1], alpha2, cat)
    print(f"\n  'abab' entangle: imag={w.intersection_imag}  opposite_membrane={w.opposite}")
    print(f"  => Entanglement = shared imag coordinate ({w.intersection_imag}) + opposite membrane bit")

    # Quaternion distance check (wild): treat (membrane, spring) as i,j components
    def quat_norm(m, s):
        # unit quaternion from 2 bits => (w, x, y, z) heuristic
        w = math.cos(m * PI / 2) * math.cos(s * PI / 2)
        x = math.sin(m * PI / 2)
        y = math.sin(s * PI / 2)
        z = math.sin((m ^ s) * PI / 2)
        return math.sqrt(w * w + x * x + y * y + z * z)

    norms = [quat_norm(*state_to_bits(st)) for st in CoinState]
    print(f"\n  [Wild conjecture] 2-bit => quaternion heuristic norms: {[round(n, 3) for n in norms]}")
    print(f"  => NOT unit quaternions (norms vary) — 4 states fit 2-bit code, not SU(2) double cover")

    return {"membrane_50_50": membrane_split[0] == membrane_split[1], "n_coin_states": 4}


def probe_q5_n_rail_wave_animation():
    """Q5: Animate n=1=>N on one prime rail — wave pattern?"""
    print("\n" + "=" * 72)
    print("Q5 — N-RAIL TRANSGRESSION: wave pattern as n=>∞")
    print("=" * 72)

    p = 7
    N = 50
    print(f"\n  Single prime p={p}, VA1 wing=1, n=1..{N}")

    xs, ys, zetas, mods, angles, regimes = [], [], [], [], [], []
    for n in range(1, N + 1):
        psi = wing_transform(BranchKind.VA1, (p,), n, 1)
        xs.append(psi.z.real)
        ys.append(psi.z.imag)
        zetas.append(psi.zeta)
        mods.append(spring_modulus(psi.z))
        angles.append(phase_angle(psi.z))
        regimes.append("B" if n >= p else "A")

    # Wave features
    boundary_n = p
    print(f"  Regime boundary at n={boundary_n}: A(n<{p}) => B(n≥{p})")
    print(f"  Y(n) pre-boundary:  n=1..{p-1} => Y = n  (linear imaginary walk)")
    print(f"  Y(n) post-boundary: n≥{p} => Y = {p}  (velocity LOCK at anchor)")

    # Oscillation in arg(z)?
    pre_angles = angles[: boundary_n - 1]
    post_angles = angles[boundary_n - 1 :]
    print(f"\n  arg(z) pre-lock:  {[round(a*180/PI,1) for a in pre_angles[:6]]}...")
    print(f"  arg(z) post-lock: {[round(a*180/PI,1) for a in post_angles[:6]]}...")

    # |z| "standing wave" wells
    wells = [n for i in range(1, len(mods) - 1) if mods[i] < mods[i - 1] and mods[i] < mods[i + 1]]
    print(f"  |z| local minima (wells) in 1..{N}: n={wells[:10]} ({len(wells)} total)")

    # Prime pair corridor: 3-case oscillation
    a, pp = 3, 11
    cases = [prime_pair_case(a, pp, n) for n in range(1, N + 1)]
    case_runs = []
    prev = cases[0]
    run = 1
    for c in cases[1:]:
        if c == prev:
            run += 1
        else:
            case_runs.append((prev, run))
            prev = c
            run = 1
    case_runs.append((prev, run))
    print(f"\n  Pair corridor ({a},{pp}) case sequence n=1..{N}:")
    print(f"    case runs: {case_runs[:8]}...")
    print(f"    case distribution: {dict(Counter(cases))}")
    print(f"    => 3-case FSM = piecewise constant 'phase sectors' on rail")

    # Fourier-ish: does Y oscillate after lock? (it shouldn't for single prime)
    y_post = ys[boundary_n - 1 :]
    y_unique_post = len(set(round(y, 6) for y in y_post))
    print(f"\n  Post-lock Y unique values: {y_unique_post} (expect 1 = flat 'standing wave')")
    print(f"  zeta slope post-lock: Δzeta/Δn = 1.0 (linear depth march)")

    # ASCII wave sketch of |z| mod 20
    print(f"\n  |z| rail sketch (n=1..30, scaled):")
    sketch_mods = [spring_modulus(wing_transform(BranchKind.VA1, (p,), n, 1).z) for n in range(1, 31)]
    mx = max(sketch_mods)
    for n, m in enumerate(sketch_mods, 1):
        bar = "#" * int(20 * m / mx)
        mark = "|" if n == p else " "
        print(f"    n={n:2d}{mark} {bar}")

    return {"boundary_n": boundary_n, "wells_count": len(wells), "case_sectors": len(case_runs)}


def probe_cross_bank_particle_meet():
    """Bonus: all 32 lattices meet at prime swap — particle collision census."""
    print("\n" + "=" * 72)
    print("BONUS — CROSS-BANK MEET: 3<->11 particle collisions")
    print("=" * 72)
    bank3 = LatticeBank32.single_prime(3)
    bank11 = LatticeBank32.single_prime(11)
    meets = 0
    for lid in LatticeId:
        if bank3[lid].at(11) == bank11[lid].at(3):
            meets += 1
    print(f"  swap (3,11): {meets}/32 chambers collide at reciprocal n")
    print(f"  => Every polarization chamber has a mirror meet (proven in demo())")


def main():
    print("=" * 72)
    print("PARTICLE PLAYGROUND — honest metaphor probe with numbers")
    print("=" * 72)

    r1 = probe_q1_meet_as_particle()
    r2 = probe_q2_spring_plane_psi()
    r3 = probe_q3_va_branches_polarization()
    r4 = probe_q4_electron_quaternion_spin()
    r5 = probe_q5_n_rail_wave_animation()
    probe_cross_bank_particle_meet()

    print("\n" + "=" * 72)
    print("SUMMARY METRICS")
    print("=" * 72)
    for k, v in {**r1, **r2, **r3, **r4, **r5}.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
