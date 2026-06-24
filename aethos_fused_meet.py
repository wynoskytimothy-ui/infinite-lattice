#!/usr/bin/env python3
r"""
aethos_fused_meet.py  --  TARGET 4: THE FUSED COMPLEX OPERATOR
==============================================================
The two-temperature synthesis (scripts/maslov_unification.py,
scripts/bench_pi_roots_interferometer.py) proved two facts but left ONE gap:

  PARTICLE  : softmin_T over REAL action costs  ->  min (argmin / shortest path)
              as T -> 0.                       [Maslov dequantization, real Boltzmann]
  WAVE      : Sigma_k a_k * exp(i*phi_k) over the constructive-pi roots of unity
              = 0 (full orbit), subset sums = Dirichlet diffraction. [complex phases]

They are CONSISTENT but live on TWO different operators over TWO different objects
(real costs vs complex phases).  This file attempts ONE complex-weighted meet
operator M_beta over chamber amplitudes  a_k * exp(i*phi_k)  with an inverse
temperature beta, and tests BOTH limits in the SAME operator:

  THE OPERATOR (analytic continuation of softmin into the complex plane):

      Z(beta) = Sigma_k  a_k * exp( -beta * c_k + i * phi_k )          (partition fn)
      M_beta  = -(1/beta) * log Z(beta)                                (free energy)

  where c_k are REAL action / meet costs (the particle ledger) and phi_k are the
  EXACT constructive-pi phases (the wave ledger).  This is one operator with one
  inverse-temperature knob beta.  Its two faces:

    * COLD  (beta -> inf):  Z is dominated by the single smallest-cost term;
        Re(M_beta) -> min_k c_k  (the tropical (min,+) meet, the argmin particle
        path) and Im(M_beta) -> phi_{argmin}/beta -> 0.  The PARTICLE.

    * WAVE  (the on-shell / degenerate-cost slice, c_k == const):  the real
        Boltzmann weights are uniform, exp(-beta*c) factors OUT, and
        Z(beta) = exp(-beta*c) * Sigma_k a_k * exp(i*phi_k)  =  the pure phase
        interferometer.  Full orbit sum = 0 (dark), subset sums = Dirichlet.

Every number below is COMPUTED, not asserted.  Two-sided: the test reports where
the single operator delivers both limits AND the precise place it does NOT (the
honest obstruction).

    cd "C:/Users/wynos/New folder (3)" && python aethos_fused_meet.py
"""
from __future__ import annotations

import cmath
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pi"))

import numpy as np

# constructive-pi roots: exact phases from {+,-,*,/,sqrt}, no transcendentals
from pi.constructive_pi import primitive_root_of_unity, cpower  # noqa: E402


# ===========================================================================
# THE ONE FUSED OPERATOR
# ===========================================================================
def fused_meet(costs, phases, beta, amps=None):
    r"""ONE complex operator M_beta over chamber amplitudes a_k * exp(i*phi_k).

        Z(beta) = Sigma_k a_k * exp(-beta*c_k + i*phi_k)
        M_beta  = -(1/beta) * log Z(beta)

    costs  : real action costs c_k  (the particle ledger; drives the argmin meet).
    phases : real phases phi_k      (the wave ledger; the constructive-pi roots).
    beta   : inverse temperature.   beta->inf = cold/particle; finite = warm/wave.
    amps   : optional real magnitudes a_k (default 1 = unit roots of unity).

    Returns the complex M_beta and the complex partition function Z.
    Re(M_beta) is the (soft) meet cost; -Im(M_beta)*beta is the surviving phase.
    """
    c = np.asarray(costs, dtype=np.float64)
    phi = np.asarray(phases, dtype=np.float64)
    a = np.ones_like(c) if amps is None else np.asarray(amps, dtype=np.float64)
    # stabilize by the real minimum cost (the cold attractor)
    cmin = float(np.min(c))
    # exponent = -beta*(c - cmin) + i*phi  ->  |term| <= a_k, no overflow
    expo = (-beta * (c - cmin)) + 1j * phi
    z = np.sum(a * np.exp(expo))                 # = exp(+beta*cmin) * Z(beta)
    if z == 0:                                   # perfectly dark interferometer
        return complex("nan"), complex(0.0)
    # M = -(1/beta) log Z = cmin - (1/beta) log( z )   (undo the cmin shift)
    M = cmin - (1.0 / beta) * cmath.log(z)
    Z_true = z * cmath.exp(-beta * cmin)
    return M, Z_true


def fused_partition_only(amps, phases, beta_cost_uniform=0.0):
    """The WAVE face: when all costs are equal, beta*c factors out and the
    partition function IS the pure phase sum Sigma a_k exp(i phi_k).  Returns it."""
    a = np.asarray(amps, dtype=np.float64)
    phi = np.asarray(phases, dtype=np.float64)
    return np.sum(a * np.exp(1j * phi))


# ===========================================================================
# constructive-pi phases for N chambers (exact roots of unity)
# ===========================================================================
def pi_phases(n):
    """Phase angles of the n-th roots of unity, taken from the constructive-pi
    recurrence (no transcendental pi).  Returns the EXACT arg() of each root."""
    # primitive_root_of_unity(k) = e^{i pi / 2^{k+1}}.  For the n=8 wing-grid use
    # the pi/4 generator (8th roots); generalize by reading arg of cpower(g, j).
    # We expose arctan2 of the constructive (re,im) tuples so the phases come from
    # the {+,-,*,/,sqrt} pi, not math.pi.
    if n == 8:
        g = primitive_root_of_unity(1)           # e^{i pi/4}, built constructively
        out = []
        for j in range(8):
            z = cpower(g, j)
            out.append(math.atan2(float(z[1]), float(z[0])))
        return np.array(out)
    if n == 32:
        g = primitive_root_of_unity(3)           # e^{i pi/16} -> 32nd roots
        out = []
        for j in range(32):
            z = cpower(g, j)
            out.append(math.atan2(float(z[1]), float(z[0])))
        return np.array(out)
    # generic fallback (still exact roots of unity, just via uniform spacing)
    return np.array([2 * math.pi * j / n for j in range(n)])


# ===========================================================================
# TEST 1 -- COLD LIMIT: beta -> inf collapses M_beta to the tropical (min,+) meet
# ===========================================================================
def test_cold_collapses_to_particle():
    print("=" * 78)
    print("[1] COLD beta->inf : ONE operator M_beta -> tropical (min,+) MEET (argmin)")
    print("=" * 78)
    # eight competing route costs through eight chambers (the particle ledger),
    # each carrying a constructive-pi phase (the wave ledger). Same a_k=1.
    a, p = 3, 5
    costs = np.array([a + p, a + p + 2, a + p + 1, a + p + 7,
                      a + p + 4, a + p + 3, a + p + 9, a + p + 6], dtype=float)
    phases = pi_phases(8)
    true_meet = float(costs.min())               # tropical (min,+) value = a+p = 8
    argmin = int(costs.argmin())
    print(f"  meet(a={a},p={p}) route costs c_k = {[int(c) for c in costs]}")
    print(f"  tropical (min,+) meet = min_k c_k = {int(true_meet)}  (argmin chamber = {argmin})")
    print(f"  carrying constructive-pi phases phi_k (8th roots) on the SAME chambers\n")
    print(f"  {'beta':>10} {'Re(M_beta)':>16} {'cost gap':>14} "
          f"{'Im(M_beta)':>14} {'phase->0':>12}")
    re_err = im_err = None
    for beta in [0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 1000.0]:
        M, _ = fused_meet(costs, phases, beta)
        re_err = abs(M.real - true_meet)
        im_err = abs(M.imag)                     # surviving phase should -> 0
        print(f"  {beta:>10.1f} {M.real:>16.10f} {re_err:>14.3e} "
              f"{M.imag:>14.3e} {im_err:>12.3e}")
    cold_ok = re_err < 1e-2 and im_err < 1e-2
    print(f"\n  COLD VERDICT: Re(M_beta)->meet ({re_err:.2e}) AND Im(M_beta)->0 "
          f"({im_err:.2e})  -> {'PASS' if cold_ok else 'FAIL'}")
    print("  => the SAME operator, cold, IS the particle: argmin meet, phase extinguished.")
    return cold_ok, re_err, im_err


# ===========================================================================
# TEST 2 -- WAVE FACE: the on-shell (degenerate-cost) slice of the SAME operator
#            reproduces the full-orbit NULL (sum = 0) and the Dirichlet grating.
# ===========================================================================
def test_wave_face_null_and_grating():
    print()
    print("=" * 78)
    print("[2] WAVE face : on-shell slice (c_k==const) of M_beta = pure phase sum")
    print("=" * 78)
    print("  When all costs are EQUAL the real Boltzmann factor exp(-beta*c) factors")
    print("  OUT of Z, so Z(beta) = exp(-beta*c) * Sigma_k a_k exp(i phi_k).  The phase")
    print("  interferometer IS the partition function of the same operator.\n")

    # (a) FULL 8-orbit and 32-orbit NULL: sum of roots of unity = 0 (dark fringe)
    for n in (8, 32):
        phi = pi_phases(n)
        const_cost = np.full(n, 4.2)             # any constant on-shell cost
        # The partition function at ANY beta is exp(-beta*4.2) * Sigma exp(i phi).
        beta = 0.7
        _, Z = fused_meet(const_cost, phi, beta)
        # divide out the (real, positive) Boltzmann prefactor to read the pure phase sum
        pref = math.exp(-beta * 4.2)
        phase_sum = Z / pref
        print(f"  n={n:>2} full orbit: |Sigma_k exp(i phi_k)| (from M_beta's Z) = "
              f"{abs(phase_sum):.3e}   (== 0 = destructive / dark)")
    print()

    # (b) GRATING: open a SUBSET of wings (amps mask) -> Dirichlet diffraction.
    phi8 = pi_phases(8)
    print("  Dirichlet grating: open first M wings (amps mask), constant cost.")
    print(f"  {'M slits':>8} {'|partition Z|/pref':>20} {'|Dirichlet kernel|':>20} "
          f"{'rel err':>12}")
    worst = 0.0
    for M in (2, 3, 5, 8):
        amps = np.array([1.0] * M + [0.0] * (8 - M))
        const_cost = np.full(8, 1.5)
        beta = 1.3
        _, Z = fused_meet(const_cost, phi8, beta, amps=amps)
        pref = math.exp(-beta * 1.5)
        meas = abs(Z / pref)
        # M consecutive 8th-root phases -> Dirichlet kernel with per-slit phase pi/4
        phistep = math.pi / 4
        s = math.sin(phistep / 2)
        D = M if abs(s) < 1e-12 else abs(math.sin(M * phistep / 2) / s)
        # at a DARK point (D ~ 0, e.g. the full 8-slit null) relative error is
        # undefined; use absolute error there. Both measured and ref are ~0.
        rel = abs(meas - D) if D < 1e-9 else abs(meas - D) / D
        worst = max(worst, rel)
        print(f"  {M:>8} {meas:>20.8f} {D:>20.8f} {rel:>12.3e}")
    wave_ok = worst < 1e-9
    print(f"\n  worst rel err vs Dirichlet = {worst:.3e}")
    print(f"  WAVE VERDICT: on-shell slice of M_beta = interferometer (NULL + grating) "
          f"-> {'PASS' if wave_ok else 'FAIL'}")
    return wave_ok, worst


# ===========================================================================
# TEST 3 -- THE CRUX: does ONE beta-sweep on the SAME (cost, phase) config move
#            continuously from WAVE (interference visible) to PARTICLE (argmin)?
#            This is where the honest obstruction lives.
# ===========================================================================
def test_continuous_sweep_crux():
    print()
    print("=" * 78)
    print("[3] CRUX : ONE beta-sweep, SAME config -- wave-visibility vs particle-pick")
    print("=" * 78)
    print("  Config: 8 chambers, costs SLIGHTLY spread, constructive-pi phases.")
    print("  Track (i) particle sharpness = max Boltzmann weight (->1 = one path),")
    print("        (ii) wave visibility = |Im part of Z's phase content| that survives.\n")

    # nearly-degenerate costs so the wave can be seen warm, the particle cold
    costs = np.array([8.0, 8.05, 8.10, 8.02, 8.08, 8.03, 8.06, 8.09])
    phi = pi_phases(8)
    true_meet = float(costs.min())

    print(f"  {'beta':>10} {'maxBoltz':>10} {'Re(M)-meet':>12} "
          f"{'|phaseZ|':>12} {'regime':>20}")
    rows = []
    for beta in [0.05, 0.2, 0.5, 1.0, 3.0, 10.0, 50.0, 300.0]:
        c = costs
        cmin = c.min()
        w = np.exp(-beta * (c - cmin))
        w = w / w.sum()
        maxb = float(w.max())
        M, Z = fused_meet(costs, phi, beta)
        # the phase content of Z normalized by the real Boltzmann mass (visibility)
        real_mass = float(np.sum(np.exp(-beta * (c - cmin))))
        phaseZ = abs(np.sum(np.exp(-beta * (c - cmin)) * np.exp(1j * phi))) / real_mass
        regime = "WAVE (spread)" if maxb < 0.3 else ("PARTICLE (picked)" if maxb > 0.9
                                                     else "crossover")
        rows.append((beta, maxb, M.real - true_meet, phaseZ))
        print(f"  {beta:>10.2f} {maxb:>10.4f} {M.real - true_meet:>12.3e} "
              f"{phaseZ:>12.4f} {regime:>20}")

    # The honest reading: at warm beta the phase sum interferes (visibility != trivial),
    # at cold beta one path is picked (maxBoltz->1) and the phase visibility collapses
    # to 1 (single surviving phasor, |.|=1). Report both ends.
    warm = rows[0]
    cold = rows[-1]
    print(f"\n  WARM end (beta={warm[0]}): maxBoltz={warm[1]:.3f} (spread, wave-like), "
          f"phase-visibility |phaseZ|={warm[3]:.3f}")
    print(f"  COLD end (beta={cold[0]}): maxBoltz={cold[1]:.3f} (one path, particle), "
          f"Re(M)-meet={cold[2]:.2e}, |phaseZ|={cold[3]:.3f} (single phasor -> 1)")
    # crossover exists if maxBoltz rises monotonically across the sweep
    maxb_seq = [r[1] for r in rows]
    monotone = all(maxb_seq[i] <= maxb_seq[i + 1] + 1e-12 for i in range(len(maxb_seq) - 1))
    print(f"  particle-sharpness rises monotonically with beta: {monotone}")
    return monotone, rows


# ===========================================================================
# TEST 4 -- THE OBSTRUCTION, MADE PRECISE.  Can ONE beta give BOTH the
#   FULL-NULL wave (sum=0) AND the argmin particle on the SAME costs?
#   We sweep beta on EQUAL costs (so the null is exact) and watch the particle
#   limit; then on SPREAD costs (so the particle is sharp) and watch the null.
# ===========================================================================
def test_obstruction_precise():
    print()
    print("=" * 78)
    print("[4] OBSTRUCTION (precise) : can ONE beta give full-NULL AND sharp-argmin?")
    print("=" * 78)
    phi = pi_phases(8)

    print("  (A) EQUAL costs (the wave is exact, |Sigma exp(i phi)|=0 at ALL beta):")
    eqc = np.full(8, 5.0)
    for beta in [0.1, 1.0, 10.0, 100.0]:
        _, Z = fused_meet(eqc, phi, beta)
        pref = math.exp(-beta * 5.0)
        null = abs(Z / pref)
        # particle sharpness on equal costs = 1/8 forever (no path is preferred)
        w = np.ones(8) / 8
        print(f"     beta={beta:>7.1f}: |phase sum|={null:.2e}  maxBoltz={w.max():.4f} "
              f"(no argmin -- all costs tie, particle UNDEFINED)")

    print("\n  (B) SPREAD costs (the particle is sharp; watch the wave NULL DIE):")
    spc = np.array([8.0, 8.2, 8.1, 8.7, 8.4, 8.3, 8.9, 8.6])
    cmin = spc.min()
    for beta in [0.05, 0.5, 5.0, 50.0]:
        w = np.exp(-beta * (spc - cmin)); w /= w.sum()
        # the phase NULL now uses the Boltzmann-WEIGHTED phasors, not equal ones
        weighted_phase = abs(np.sum(w * np.exp(1j * phi)))
        print(f"     beta={beta:>7.2f}: maxBoltz={w.max():.4f}  "
              f"|weighted phase sum|={weighted_phase:.4f} "
              f"({'wave alive' if weighted_phase < 0.5 else 'wave collapsing/gone'})")

    print("\n  PRECISE OBSTRUCTION:")
    print("  The SAME operator gives the particle in the cold limit (always) and the")
    print("  wave on the equal-cost (on-shell) slice (always).  But the FULL-orbit")
    print("  NULL (Sigma exp(i phi)=0) and a SHARP argmin are MUTUALLY EXCLUSIVE on")
    print("  one beta+cost config: the null needs EQUAL real weights (cost degeneracy),")
    print("  which is exactly the regime where the argmin is UNDEFINED (all paths tie).")
    print("  Spreading the costs to define an argmin re-weights the phasors and DESTROYS")
    print("  the exact null. So one operator -- YES; one beta giving both extreme")
    print("  signatures at once -- NO. They are the two ASYMPTOTES, not a single point.")
    return True


# ===========================================================================
# TEST 5 -- consistency cross-check against the TWO original operators.
#   Confirm M_beta reproduces (a) the real softmin from maslov_unification and
#   (b) the complex phase sum from the interferometer -- so it genuinely SUBSUMES
#   both, it is not a third unrelated thing.
# ===========================================================================
def test_subsumes_both_originals():
    print()
    print("=" * 78)
    print("[5] SUBSUMPTION : M_beta reproduces BOTH original operators exactly")
    print("=" * 78)
    # (a) real softmin: M_beta with phi=0 == softmin_{T=1/beta}
    costs = np.array([5.0, 6.0, 9.0, 12.0])
    zero_phase = np.zeros(4)
    print(f"  {'beta':>8} {'Re(M_beta), phi=0':>20} {'softmin_T=1/beta':>20} {'err':>12}")
    worst_a = 0.0
    for beta in [0.1, 1.0, 10.0]:
        M, _ = fused_meet(costs, zero_phase, beta)
        T = 1.0 / beta
        m = costs.min()
        softmin = m - T * math.log(np.sum(np.exp(-(costs - m) / T)))
        err = abs(M.real - softmin)
        worst_a = max(worst_a, err)
        print(f"  {beta:>8.1f} {M.real:>20.10f} {softmin:>20.10f} {err:>12.2e}")
    a_ok = worst_a < 1e-9

    # (b) complex phase sum: Z of M_beta on equal costs == the interferometer sum
    print()
    phi8 = pi_phases(8)
    eqc = np.full(8, 0.0)
    _, Z = fused_meet(eqc, phi8, 1.0)
    direct = np.sum(np.exp(1j * phi8))
    err_b = abs(Z - direct)
    print(f"  Z(M_beta, equal cost) vs direct Sigma exp(i phi): "
          f"|{abs(Z):.3e}| vs |{abs(direct):.3e}|  err={err_b:.2e}")
    b_ok = err_b < 1e-9
    print(f"\n  SUBSUMES softmin (err {worst_a:.1e}) AND phase sum (err {err_b:.1e}) "
          f"-> {'PASS' if a_ok and b_ok else 'FAIL'}")
    return a_ok and b_ok, worst_a, err_b


def main():
    print("#" * 78)
    print("# aethos_fused_meet.py  --  ONE complex operator M_beta = -(1/beta) log Z")
    print("#   Z(beta) = Sigma_k a_k exp(-beta*c_k + i*phi_k)  (cost + constructive-pi phase)")
    print("#" * 78)
    print()

    cold_ok, re_err, im_err = test_cold_collapses_to_particle()
    wave_ok, grating_err = test_wave_face_null_and_grating()
    monotone, _ = test_continuous_sweep_crux()
    test_obstruction_precise()
    subs_ok, sa, sb = test_subsumes_both_originals()

    print()
    print("#" * 78)
    print("# VERDICT (all numerical)")
    print("#" * 78)
    print(f"  [1] COLD beta->inf -> particle meet : Re err {re_err:.2e}, Im err {im_err:.2e}"
          f"  -> {'PASS' if cold_ok else 'FAIL'}")
    print(f"  [2] WAVE on-shell slice (NULL+grating): grating err {grating_err:.2e}"
          f"  -> {'PASS' if wave_ok else 'FAIL'}")
    print(f"  [3] particle-sharpness monotone in beta: {monotone}")
    print(f"  [5] subsumes softmin AND phase sum    : {sa:.1e} / {sb:.1e}"
          f"  -> {'PASS' if subs_ok else 'FAIL'}")
    print()
    print("  ANSWER: ONE complex operator M_beta = -(1/beta) log Sigma a_k exp(-beta c_k + i phi_k)")
    print("  DOES give the particle in one limit (cold beta->inf = argmin (min,+) meet,")
    print("  phase extinguished) and the wave in another (equal-cost on-shell slice =")
    print("  the constructive-pi interferometer: full NULL + Dirichlet grating).")
    print()
    print("  HONEST OBSTRUCTION: the two extreme SIGNATURES (exact full-orbit NULL and a")
    print("  SHARP argmin) cannot coexist at a single (beta, cost) point -- the NULL needs")
    print("  cost degeneracy (equal real weights), which is exactly where the argmin is")
    print("  undefined.  They are the two ASYMPTOTES of one operator, not one shared point.")
    print("  The quantum lattice is REACHABLE as one analytic operator; it is NOT a single")
    print("  state that is simultaneously maximally-particle and maximally-wave (that IS")
    print("  complementarity -- the obstruction is physical, not a defect of the operator).")
    allok = cold_ok and wave_ok and subs_ok and monotone
    return allok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
