#!/usr/bin/env python3
"""
maslov_unification.py
=====================
ONE knob (temperature T) ties the user's four objects together:

  PARTICLE  = the tropical meet  meet(a,p) = (a+p, min, a+p)   [classical / shortest path / least action]
  WAVE      = the soft meet      softmin_T over sum-over-paths   [quantum / superposition / interference]
  3D-PLANE  = the lattice meet   swap_meet / Floyd-Warshall on the chamber graph
  PI        = the chamber phases = roots of unity from the constructive-pi recurrence (no transcendentals)

Maslov dequantization:  softmin_T(a,b) = -T*log(exp(-a/T)+exp(-b/T))  ->  min(a,b)  as T->0.
The "+" of (min,+) is the T->0 limit of log-sum-exp's additive accumulation.
So the PARTICLE meet is the COLD (T->0) limit of the WAVE sum-over-paths on the SAME operation.

Every number printed here is computed, not asserted. Run:
    python scripts/maslov_unification.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pi"))

# ---------------------------------------------------------------------------
# 0. THE KNOB: soft meet (wave) and its T->0 limit (particle)
# ---------------------------------------------------------------------------

NEG = float("inf")  # tropical "zero" of (min,+) is +inf (additive identity for min)


def hard_meet(a: float, p: float) -> tuple[float, float, float]:
    """Tropical / particle meet: (a+p, min(a,p), a+p).  The exact least-action node.
    The middle slot min(a,p) is the (min,+) combine that drives shortest path."""
    return (a + p, min(a, p), a + p)


def softmin_T(values: np.ndarray, T: float, axis=None) -> np.ndarray:
    """softmin_T(x) = -T log sum exp(-x/T).  WAVE sum-over-paths at temperature T.
    Numerically stabilized. As T->0 -> min(x); as T->inf -> mean-ish / spreads."""
    x = np.asarray(values, dtype=np.float64)
    if axis is None:
        m = np.min(x)
        if not np.isfinite(m):  # all paths absent -> tropical +inf survives
            return float("inf")
        s = np.sum(np.exp(-(x - m) / T))
        return float(m - T * math.log(s))
    m = np.min(x, axis=axis, keepdims=True)
    finite_m = np.where(np.isfinite(m), m, 0.0)
    z = np.exp(-(x - finite_m) / T)
    s = np.sum(z, axis=axis, keepdims=True)
    out = np.where(np.isfinite(m), m - T * np.log(s), np.inf)
    return np.squeeze(out, axis=axis)


def soft_meet_T(a: float, p: float, T: float) -> tuple[float, float, float]:
    """WAVE meet: the middle (min) slot becomes softmin_T(a,p).  As T->0 -> hard_meet."""
    soft = softmin_T(np.array([a, p]), T)
    # the a+p slot is the (max,+)-free additive accumulator; it is exact at all T.
    return (a + p, float(soft), a + p)


# ---------------------------------------------------------------------------
# 1. WAVE -> PARTICLE convergence on the raw operation
# ---------------------------------------------------------------------------

def test_softmin_convergence():
    print("=" * 78)
    print("[1] softmin_T(a,p) -> min(a,p) as T->0  (wave meet -> particle meet)")
    print("=" * 78)
    a, p = 3.0, 7.0
    true_min = min(a, p)
    print(f"  a={a}  p={p}   exact particle min = {true_min}")
    print(f"  {'T':>10} {'softmin_T':>16} {'error vs min':>16}")
    rows = []
    for T in [10.0, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        s = softmin_T(np.array([a, p]), T)
        err = abs(s - true_min)
        rows.append((T, s, err))
        print(f"  {T:>10.0e} {s:>16.10f} {err:>16.3e}")
    # also show T->inf spreads toward the average (full superposition, no choice made)
    big = [softmin_T(np.array([a, p]), T) for T in [1e2, 1e3, 1e6]]
    print(f"  T->inf: softmin -> {big}  (spreads BELOW min toward -T*log(2)-type blur; no path chosen)")
    final_err = rows[-1][2]
    return final_err, true_min


# ---------------------------------------------------------------------------
# 2. The real shortest-path test: soft (min,+) matrix powers -> Floyd-Warshall
# ---------------------------------------------------------------------------

def tropical_matmul_hard(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(min,+) matrix product: C[i,j] = min_k A[i,k] + B[k,j].  PARTICLE composition."""
    n = A.shape[0]
    C = np.full((n, n), NEG)
    for i in range(n):
        for j in range(n):
            C[i, j] = np.min(A[i, :] + B[:, j])
    return C


def tropical_matmul_soft(A: np.ndarray, B: np.ndarray, T: float) -> np.ndarray:
    """WAVE composition: C[i,j] = softmin_T_k( A[i,k] + B[k,j] ).
    This is EXACTLY log-sum-exp of products in the (+,*) semiring under z=exp(-./T):
       exp(-C/T) = sum_k exp(-A[i,k]/T) * exp(-B[k,j]/T)   (a real matrix product!)
    i.e. partition-function / sum-over-paths.  As T->0 it deforms to (min,+)."""
    n = A.shape[0]
    C = np.full((n, n), NEG)
    for i in range(n):
        for j in range(n):
            C[i, j] = softmin_T(A[i, :] + B[:, j], T)
    return C


def floyd_warshall(W: np.ndarray) -> np.ndarray:
    """Reference all-pairs shortest path (the ground-truth particle distances)."""
    n = W.shape[0]
    D = W.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i, k] + D[k, j] < D[i, j]:
                    D[i, j] = D[i, k] + D[k, j]
    return D


def tropical_closure_hard(W: np.ndarray) -> np.ndarray:
    """All-pairs shortest path as the (min,+) matrix closure: D = W^(n) (tropical powers).
    Repeated tropical squaring of (I (+) W) gives Floyd-Warshall distances."""
    n = W.shape[0]
    D = W.copy()
    np.fill_diagonal(D, np.minimum(np.diag(D), 0.0))  # tropical identity on diagonal
    # n-1 relaxations via repeated squaring (path length doubling)
    steps = max(1, math.ceil(math.log2(max(2, n))))
    for _ in range(steps):
        D = tropical_matmul_hard(D, D)
    return D


def tropical_closure_soft(W: np.ndarray, T: float) -> np.ndarray:
    """Same closure with the WAVE soft product at temperature T."""
    n = W.shape[0]
    D = W.copy()
    diag_soft = np.array([softmin_T(np.array([D[i, i], 0.0]), T) for i in range(n)])
    for i in range(n):
        D[i, i] = diag_soft[i]
    steps = max(1, math.ceil(math.log2(max(2, n))))
    for _ in range(steps):
        D = tropical_matmul_soft(D, D, T)
    return D


def test_shortest_path_dequantization():
    print()
    print("=" * 78)
    print("[2] soft (min,+) matrix powers -> Floyd-Warshall as T->0  (paths condense)")
    print("=" * 78)
    rng = np.random.default_rng(7)
    n = 6
    # random weighted DAG-ish graph, positive edge weights, inf for absent
    W = np.full((n, n), NEG)
    np.fill_diagonal(W, 0.0)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < 0.45:
                w = float(rng.integers(1, 9))
                W[i, j] = w
                edges.append((i, j, w))
    print(f"  graph: n={n} nodes, {len(edges)} directed edges, weights in [1,8]")

    D_ref = floyd_warshall(W)
    D_hard = tropical_closure_hard(W)
    hard_match = np.allclose(D_hard[np.isfinite(D_ref)], D_ref[np.isfinite(D_ref)])
    print(f"  tropical closure (hard min,+) == Floyd-Warshall ?  {hard_match}")

    finite = np.isfinite(D_ref)
    print(f"  {'T':>10} {'max|soft-FW|':>16} {'mean|soft-FW|':>16}")
    errs = []
    for T in [5.0, 1.0, 0.3, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3]:
        D_soft = tropical_closure_soft(W, T)
        diff = np.abs(D_soft[finite] - D_ref[finite])
        errs.append((T, float(diff.max()), float(diff.mean())))
        print(f"  {T:>10.0e} {diff.max():>16.6e} {diff.mean():>16.6e}")
    # show the wave at high T blurs distances DOWN (counts alternative paths)
    D_hot = tropical_closure_soft(W, 5.0)
    print(f"  at T=5 the soft distance is systematically BELOW FW (extra paths add up):")
    print(f"     mean(soft - FW) at T=5 = {(D_hot[finite] - D_ref[finite]).mean():+.4f}  (negative = path multiplicity)")
    return hard_match, errs, D_ref, D_hard


# ---------------------------------------------------------------------------
# 3. PI: the chamber phases are roots of unity from the constructive recurrence;
#    their full sum = 0 (destructive interference); subset sums = Dirichlet kernel.
#    Tie to the SAME wave: superposition of phases vs the picked (cold) path.
# ---------------------------------------------------------------------------

def test_pi_roots_interference():
    print()
    print("=" * 78)
    print("[3] PI: chamber phases = roots of unity (constructive-pi, no transcendentals)")
    print("=" * 78)
    from constructive_pi import primitive_root_of_unity, cpower
    from mpmath import mpf

    # 8 wings <-> 8th roots of unity = cpower of the pi/4 root (primitive_root_of_unity(1))
    base = primitive_root_of_unity(1)  # cos(pi/4)+i sin(pi/4), built from {+,-,*,/,sqrt}
    roots8 = [cpower(base, j) for j in range(8)]
    re_sum = sum(float(z[0]) for z in roots8)
    im_sum = sum(float(z[1]) for z in roots8)
    print(f"  8 wing-phases = (cos pi/4 + i sin pi/4)^j, j=0..7   built with NO trig/pi:")
    for j, z in enumerate(roots8):
        print(f"    wing {j+1}: {float(z[0]):+.6f} {float(z[1]):+.6f}i")
    print(f"  FULL 8-orbit sum = {re_sum:+.3e} {im_sum:+.3e}i   (== 0 = destructive interference)")

    # 32 chambers = 4 branches x 8 wings -> 32nd roots of unity, full sum still 0
    base32 = primitive_root_of_unity(3)  # cos(pi/16)+i sin(pi/16) -> 32nd roots
    roots32 = [cpower(base32, j) for j in range(32)]
    re32 = sum(float(z[0]) for z in roots32)
    im32 = sum(float(z[1]) for z in roots32)
    print(f"  FULL 32-chamber sum = {re32:+.3e} {im32:+.3e}i   (== 0)")

    # subset / partial sum = Dirichlet kernel (diffraction): sum_{j=0}^{M-1} e^{i j theta}
    print(f"  partial sums (diffraction / Dirichlet kernel), 32-chamber, theta=2pi/32:")
    theta = 2 * math.pi / 32
    for M in [1, 4, 8, 16, 24, 32]:
        # closed-form Dirichlet magnitude
        if M % 32 == 0:
            mag = 0.0
        else:
            mag = abs(math.sin(M * theta / 2) / math.sin(theta / 2))
        # actual partial sum from the constructive roots
        rs = sum(float(roots32[j][0]) for j in range(M))
        is_ = sum(float(roots32[j][1]) for j in range(M))
        amag = math.hypot(rs, is_)
        print(f"    M={M:>2}: |partial sum| = {amag:8.4f}   Dirichlet |sin(M*th/2)/sin(th/2)| = {mag:8.4f}")
    return abs(re_sum) + abs(im_sum), abs(re32) + abs(im32)


# ---------------------------------------------------------------------------
# 4. THE UNIFICATION: ONE temperature knob over BOTH the (min,+) meet AND the
#    phase superposition.  Wave (T=inf): all phases alive, sum=0, no path picked.
#    Particle (T=0): softmin collapses to the single least-action meet path.
#    Show: the SAME softmin_T that condenses shortest paths ALSO governs how much
#    of the phase orbit survives -- both are "1/T sharpens the argmin".
# ---------------------------------------------------------------------------

def soft_argmin_weights(values: np.ndarray, T: float) -> np.ndarray:
    """Boltzmann weights w_k = exp(-x_k/T)/Z.  T->0: one-hot on argmin (particle picks ONE path).
    T->inf: uniform (wave: every path equally alive -> the roots-of-unity superposition)."""
    x = np.asarray(values, dtype=np.float64)
    z = np.exp(-(x - x.min()) / T)
    return z / z.sum()


def test_one_knob_unifies():
    print()
    print("=" * 78)
    print("[4] ONE KNOB: temperature sharpens the meet (particle) AND collapses the")
    print("    phase superposition (wave). Same softmin governs both.")
    print("=" * 78)
    # Take the candidate path costs into one node of the shortest-path graph and
    # show the Boltzmann weights condense from uniform (wave) to one-hot (particle).
    costs = np.array([5.0, 6.0, 9.0, 12.0])  # competing path costs to a node
    print(f"  competing path costs to a node: {costs.tolist()}  (true particle path = cost {costs.min()})")
    print(f"  {'T':>10} {'softmin':>12} {'weights (Boltzmann over paths)':>40} {'entropy':>10}")
    for T in [100.0, 5.0, 1.0, 0.3, 0.1, 0.01]:
        w = soft_argmin_weights(costs, T)
        s = softmin_T(costs, T)
        ent = -np.sum(w * np.log(w + 1e-300))
        wtxt = "[" + " ".join(f"{x:.3f}" for x in w) + "]"
        print(f"  {T:>10.0e} {s:>12.4f} {wtxt:>40} {ent:>10.4f}")
    print("  T->inf: weights UNIFORM = full superposition (wave, like the 8 roots all alive, sum=0)")
    print("  T->0  : weights ONE-HOT on argmin = the single least-action meet (particle path)")

    # Quantitative link: max Boltzmann weight -> 1 as T->0 (particle), -> 1/k as T->inf (wave)
    k = len(costs)
    wmax_cold = soft_argmin_weights(costs, 1e-6).max()
    wmax_hot = soft_argmin_weights(costs, 1e6).max()
    print(f"  max-weight: cold(T=1e-6)={wmax_cold:.6f} (-> 1, particle)  "
          f"hot(T=1e6)={wmax_hot:.6f} (-> 1/{k}={1/k:.4f}, wave)")
    return wmax_cold, wmax_hot, 1.0 / k


# ---------------------------------------------------------------------------
# 5. CROSS-CHECK against the repo's own lattice meet (swap_meet / tropical claim)
# ---------------------------------------------------------------------------

def test_lattice_meet_is_tropical():
    print()
    print("=" * 78)
    print("[5] repo lattice meet == tropical (min,+) shortest-path  (3D-plane <-> particle)")
    print("=" * 78)
    try:
        from aethos_complex_plane import swap_meet, triple_equalization
        from aethos_lattice import BranchKind
    except Exception as e:  # noqa: BLE001
        print(f"  (lattice import unavailable: {e})")
        return None
    # swap_meet symmetry: bank(a)@p == bank(p)@a  (the 2-way meet co-location)
    a, p = 3.0, 5.0
    left, right = swap_meet(a, p)
    sym = (left.coord == right.coord)
    print(f"  swap_meet(3,5): left z={left.z}  zeta={left.zeta}; right z={right.z}  zeta={right.zeta}")
    print(f"  2-way meet co-locates (bank(a)@p == bank(p)@a) ? {sym}")
    # triple meet: all three 2-way rails equalize to ONE node (least-action k-way meet)
    eq = triple_equalization(3, 5, 7)
    coords = [psi.coord for _, psi in eq.values()]
    triple_same = all(c == coords[0] for c in coords)
    print(f"  triple_equalization(3,5,7): all pair-rails -> {coords[0]} ? {triple_same}")
    # the meet's depth slot zeta = a+p mirrors the tropical (a+p) accumulator slot
    print(f"  meet depth zeta={left.zeta}  vs tropical (a+p)={a+p}  -> additive slot matches: "
          f"{abs(left.zeta-(a+p))<1e-9}")
    return sym, triple_same


def main():
    err_softmin, _ = test_softmin_convergence()
    hard_match, errs, D_ref, D_hard = test_shortest_path_dequantization()
    sum8, sum32 = test_pi_roots_interference()
    wmax_cold, wmax_hot, inv_k = test_one_knob_unifies()
    lattice = test_lattice_meet_is_tropical()

    print()
    print("=" * 78)
    print("VERDICT SUMMARY (all numerical)")
    print("=" * 78)
    print(f"  [1] softmin_T(3,7)-min error at T=1e-6 : {err_softmin:.3e}  (-> 0, EXACT in limit)")
    print(f"  [2] tropical closure == Floyd-Warshall  : {hard_match}")
    print(f"      soft closure max-err at T=1e-3      : {errs[-1][1]:.3e}  (-> FW as T->0)")
    print(f"  [3] 8-root orbit sum |.|                : {sum8:.3e}  (== 0, destructive)")
    print(f"      32-chamber orbit sum |.|            : {sum32:.3e}  (== 0)")
    print(f"  [4] Boltzmann max-weight cold/hot       : {wmax_cold:.4f} / {wmax_hot:.4f} (-> 1 / {inv_k:.4f})")
    print(f"  [5] lattice meet tropical+triple        : {lattice}")
    print()
    print("  ONE KNOB T: T->0 = PARTICLE (single least-action meet = Floyd-Warshall path);")
    print("              T->inf = WAVE (uniform Boltzmann = full root-of-unity superposition, sum=0).")
    print("  The (min,+) meet is the Maslov T->0 dequantization of the (+,*) sum-over-paths.")


if __name__ == "__main__":
    main()
