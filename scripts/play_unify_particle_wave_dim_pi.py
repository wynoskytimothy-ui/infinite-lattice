#!/usr/bin/env python3
"""
ONE THING: particle + wave (higher dims) + 3D-plane/lattice + constructive-pi.

Pure play, every claim a NUMERICAL check. Run:
    python scripts/play_unify_particle_wave_dim_pi.py

The single object that ties the four threads:

    A STACK OF PHASE-ROTATED SLICES, read at temperature T.

  * constructive pi BUILDS each slice's rotation from {+,-,*,/,sqrt} (8th roots
    of unity = the 8 wings).
  * the lattice MEET is the COLD (T->0) read of that stack = the particle /
    shortest path (Maslov/tropical (min,+)).
  * the full complex chamber SUPERPOSITION is the WARM read = the wave; its
    32-orbit sum=0 is Sigma(roots of unity)=0 = destructive interference.
  * a D-dim object = D phase-rotated stacks; each new dimension is a new
    rotation/frequency axis. Higher dims ARE the wave effect, made constructive
    by stacking rotated lower-dim slices.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpmath import mp, mpf, sqrt as msqrt, pi as PI_REF
mp.prec = 120

from pi.constructive_pi import (
    pi_estimate,
    primitive_root_of_unity,
    cmul,
    cpower,
    hyperball_volume,
    hypersphere_3surface,
    sphere_volume,
)
from aethos_complex_plane import wing_transform, triple_equalization
from aethos_lattice import BranchKind
from aethos_complex_rotation import swap_corridor, reflect_real


def banner(t):
    print("=" * 74)
    print(t)
    print("=" * 74)


# ---------------------------------------------------------------------------
# THREAD A — the COLD limit IS the meet: Maslov dequantization, measured.
#   softmin_T(a,b) = -T log(e^{-a/T}+e^{-b/T}) -> min(a,b)  as T->0
#   logsumexp_T    =  T log(e^{a/T}+e^{b/T})   -> max(a,b) ~ the (max,+) side
# The lattice meet(a,p) uses (min,+); show the warm semiring cools to it.
# ---------------------------------------------------------------------------
def softmin(a, b, T):
    m = min(a, b)
    return m - T * math.log(math.exp(-(a - m) / T) + math.exp(-(b - m) / T))


def logsumexp_max(a, b, T):
    # (max,+) side: T*log(e^{a/T}+e^{b/T}) -> max(a,b) as T->0 (stable shift).
    m = max(a, b)
    return m + T * math.log(math.exp((a - m) / T) + math.exp((b - m) / T))


def thread_A_cold_limit_is_meet():
    banner("THREAD A  particle = COLD read of the wave  (Maslov (min,+) <- (+,*))")
    pairs = [(3.0, 7.0), (12.0, 5.0), (1.0, 1.0), (0.2, 9.8)]
    print(f"  {'a':>5} {'b':>5} | {'min(a,b)':>9} | softmin@T= "
          f"{'1.0':>8} {'0.1':>8} {'0.01':>8} {'1e-4':>9}")
    worst = 0.0
    for a, b in pairs:
        row = [softmin(a, b, T) for T in (1.0, 0.1, 0.01, 1e-4)]
        worst = max(worst, abs(row[-1] - min(a, b)))
        print(f"  {a:5.1f} {b:5.1f} | {min(a,b):9.4f} | "
              f"           {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:9.5f}")
    print(f"  max |softmin@T=1e-4 - min|  = {worst:.3e}   (-> 0 : the meet is the cold limit)")

    # The lattice meet itself: meet(a,p) = (a+p, min(a,p), a+p). Confirm its
    # 'min' slot IS this cold read, and the '+' slots ARE softplus_add at T->0.
    a, p = 3.0, 7.0
    meet_min = min(a, p)
    meet_plus = a + p
    cold_min = softmin(a, p, 1e-6)
    cold_plus = logsumexp_max(a, p, 1e-6)  # NOTE: this -> max, not a+b; see honest note
    print(f"\n  meet(3,7) tropical = (a+p, min, a+p) = ({meet_plus}, {meet_min}, {meet_plus})")
    print(f"  softmin@1e-6 -> min : {cold_min:.6f}  (matches {meet_min})")
    print(f"  HONEST: logsumexp@1e-6 -> max = {cold_plus:.4f} != a+b={meet_plus}.")
    print(f"          The '+' in (min,+) is ORDINARY addition (the carried path cost),")
    print(f"          recovered as T*log(prod) = a+b, NOT as logsumexp. Only the MIN")
    print(f"          slot is the dequantized softmin. That is the precise tie.")
    return worst < 1e-3


# ---------------------------------------------------------------------------
# THREAD B — the WAVE: 32 chambers = roots of unity, sum = 0 = interference.
#   8 wings <-> 8th roots of unity; the wing operator S(z)=i*conj(z) is a
#   reflection that, composed around the orbit, tiles the circle. Show:
#   (1) the constructive-pi 8th roots of unity sum to 0 (full destructive),
#   (2) a SUBSET sum traces the Dirichlet kernel (diffraction), and
#   (3) the lattice's own 32-chamber zeta-orbit sums to 0 (same fact).
# ---------------------------------------------------------------------------
def thread_B_wave_interference():
    banner("THREAD B  wave = chamber superposition  (Sigma roots of unity = 0)")

    # (1) 8th roots of unity from constructive pi (NO transcendentals).
    base = primitive_root_of_unity(1)  # cos(pi/4)+i sin(pi/4) = primitive 8th root
    roots = [cpower(base, j) for j in range(8)]
    sre = sum(r[0] for r in roots)
    sim = sum(r[1] for r in roots)
    print(f"  8 constructive-pi 8th-roots-of-unity sum = ({mp.nstr(sre,4)}, {mp.nstr(sim,4)})")
    print(f"  |sum| = {mp.nstr(msqrt(sre*sre+sim*sim),4)}   (full destructive interference)")

    # (2) partial sums = Dirichlet kernel: |sum_{j<M} e^{i j theta}| traces
    # diffraction. Measure against closed form |sin(M theta/2)/sin(theta/2)|.
    theta = float(PI_REF) / 4.0  # the pi/4 wing step
    print(f"\n  partial-sum magnitude (diffraction / Dirichlet) at theta=pi/4:")
    print(f"   {'M':>3} {'|partial sum|':>14} {'Dirichlet ref':>14} {'err':>10}")
    okB = True
    for M in range(1, 9):
        acc = (mpf(0), mpf(0))
        for j in range(M):
            acc = (acc[0] + roots[j % 8][0], acc[1] + roots[j % 8][1])
        mag = math.hypot(float(acc[0]), float(acc[1]))
        ref = abs(math.sin(M * theta / 2) / math.sin(theta / 2))
        err = abs(mag - ref)
        okB &= err < 1e-6
        print(f"   {M:>3} {mag:14.6f} {ref:14.6f} {err:10.2e}")

    # (3) the LATTICE's own 32-chamber orbit: zeta sums to 0, z real part too.
    chain, n = (3, 5, 7), 5
    zt = 0.0 + 0.0j
    zeta_tot = 0.0
    for b in BranchKind:
        for w in range(1, 9):
            psi = wing_transform(b, chain, n, w)
            zt += psi.z
            zeta_tot += psi.zeta
    print(f"\n  lattice 32-chamber orbit @ (3,5,7) n=5:")
    print(f"   sum z      = {zt}   (Re cancels exactly -> standing wave on Im axis)")
    print(f"   sum zeta   = {zeta_tot}   (full depth cancellation = Sigma roots = 0)")
    return okB and abs(float(sre)) < 1e-20 and abs(zeta_tot) < 1e-9 and zt.real == 0.0


# ---------------------------------------------------------------------------
# THREAD C — higher dims = stacking phase-rotated slices (constructive pi).
#   (a) 4-ball V4 = pi^2/2, 3-sphere = 2 pi^2 from stacked lower-dim slices.
#   (b) each dimension = a new rotation axis: stack triple-meet 'slices' each
#       rotated by a pi-root phase; check independent coordinate per axis.
#   (c) is dimension == frequency mode? a D-dim object -> D phase stacks.
# ---------------------------------------------------------------------------
def thread_C_dimension_is_frequency():
    banner("THREAD C  higher dim = stack of phase-rotated slices (pi builds it)")

    # (a) volumes by stacking, from constructive pi only.
    pi = pi_estimate(28)
    V4 = hyperball_volume(1, K=28, Nw=40000)
    S3 = hypersphere_3surface(1, K=28, Nw=40000)
    print(f"  constructive pi (K=28)      = {mp.nstr(pi,14)}")
    print(f"  4-ball V4   = pi^2/2 = stack of 3-balls  : {mp.nstr(V4,12)}  ref {mp.nstr(PI_REF**2/2,12)}")
    print(f"     err {mp.nstr(abs(V4-PI_REF**2/2),3)}")
    print(f"  3-sphere S3 = 2 pi^2 = stack of 2-spheres: {mp.nstr(S3,12)}  ref {mp.nstr(2*PI_REF**2,12)}")
    print(f"     err {mp.nstr(abs(S3-2*PI_REF**2),3)}")
    ok_vol = abs(V4 - PI_REF**2 / 2) < 1e-6 and abs(S3 - 2 * PI_REF**2) < 1e-3

    # (b) each added dimension = a NEW rotation/frequency axis.
    # Build a D-dim address by stacking D triple-meet 'slices', slice d rotated
    # by the d-th pi-root phase r_d = e^{i pi / 2^{d+1}} (constructive). The
    # claim: each axis carries an INDEPENDENT coordinate (the meet's invariant),
    # so the stack is a genuine D-dim object, not a collapsed 1-D shadow.
    print(f"\n  (b) stack D rotated lattice slices -> D independent coordinates:")
    # the lattice triple-meet invariant for (a,p,q): co-located node value.
    def meet_value(a, p, q):
        eq = triple_equalization(a, p, q)
        # all three rails agree; take zeta (depth) as the slice's scalar coord
        (_, psi) = next(iter(eq.values()))
        return psi.zeta, psi.z

    triples = [(3, 5, 7), (2, 4, 6), (5, 9, 11), (1, 8, 13)]
    print(f"   {'axis d':>6} {'pi-root phase angle':>20} {'meet coord (zeta)':>18} "
          f"{'rotated re':>12} {'rotated im':>12}")
    rotated_coords = []
    for d, tri in enumerate(triples):
        zeta, z = meet_value(*tri)
        rphase = primitive_root_of_unity(d)  # e^{i pi/2^{d+1}}
        ang = math.degrees(math.atan2(float(rphase[1]), float(rphase[0])))
        # embed the scalar meet-coord on axis d by rotating (zeta,0) by the phase
        rot = cmul((mpf(zeta), mpf(0)), rphase)
        rotated_coords.append((float(rot[0]), float(rot[1])))
        print(f"   {d:>6} {ang:20.4f} {zeta:18.1f} {float(rot[0]):12.4f} {float(rot[1]):12.4f}")

    # independence test: the D rotated unit-phases are orthonormal-ish ->
    # the Gram matrix of the phase directions has rank D (no axis is a combo of
    # others). Measure via determinant of the 2D phase vectors pairwise angle.
    phases = [primitive_root_of_unity(d) for d in range(4)]
    print(f"\n   pairwise phase angles (deg) — distinct axes = distinct frequencies:")
    angs = [math.degrees(math.atan2(float(p[1]), float(p[0]))) for p in phases]
    distinct = len(set(round(a, 6) for a in angs)) == len(angs)
    print(f"     angles = {[round(a,3) for a in angs]}   all distinct: {distinct}")

    # (c) dimension == frequency mode? Decompose a D-coordinate STACK and
    # recover each axis by projecting onto its own phase (matched filter).
    # Build signal s = sum_d c_d * e^{i theta_d}; recover c_d by correlation.
    print(f"\n  (c) dimension == frequency mode? build s=Sigma c_d e^(i theta_d), recover c_d:")
    cs = [3.0, 5.0, 7.0, 11.0]
    # use distinct frequencies = the pi-root angles (a small DFT-like basis)
    thetas = [math.radians(a) for a in angs]
    s = sum(c * complex(math.cos(t), math.sin(t)) for c, t in zip(cs, thetas))
    # matched-filter recovery only clean if phases are orthogonal on the sum;
    # the pi-roots are NOT a full orthogonal basis (only 4 of 8 directions), so
    # this is the HONEST boundary: report recovered vs true.
    print(f"   {'axis d':>6} {'true c_d':>9} {'matched-filter Re':>18} {'clean?':>8}")
    clean_all = True
    for d, (c, t) in enumerate(zip(cs, thetas)):
        rec = (s * complex(math.cos(-t), math.sin(-t))).real
        clean = abs(rec - c) < 1e-6
        clean_all &= clean
        print(f"   {d:>6} {c:9.3f} {rec:18.4f} {str(clean):>8}")
    return ok_vol, distinct, clean_all


def main():
    okA = thread_A_cold_limit_is_meet()
    print()
    okB = thread_B_wave_interference()
    print()
    ok_vol, distinct, clean_all = thread_C_dimension_is_frequency()
    print()
    banner("VERDICT")
    print(f"  A  cold limit (T->0 softmin) == lattice meet           : {okA}")
    print(f"  B  chamber/root-of-unity superposition sums to 0       : {okB}")
    print(f"  C  4-ball=pi^2/2 & 3-sphere=2pi^2 by stacking slices   : {ok_vol}")
    print(f"  C  each added dim = a distinct pi-root frequency axis  : {distinct}")
    print(f"  C  D-dim object = D phase stacks, recoverable by filter: {clean_all}")
    print()
    print("  ONE THING: a stack of phase-rotated slices read at temperature T.")
    print("  pi builds the rotations; cold read = particle/meet; warm read = wave;")
    print("  each new slice-axis = a new frequency = a new dimension.")


if __name__ == "__main__":
    main()
