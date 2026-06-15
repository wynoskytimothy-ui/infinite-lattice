"""
Velocity planes and 4-way branching as rotation around the imaginary line.

Fixed axis: j = n + ni  (critical line, Re = Im).

Perpendicular spring component rotates by i^k (k = 0..3) in the frame
where j is the real axis.  That is the clean 4-way branch fan.

Velocity tiers come from transgressor segment s along anchor chain A:
  s = 0 .. k  as n crosses each anchor  =>  regime / speed change.

Legacy VA1..VA4 canon formulas remain exact; this module is the rotation
picture + velocity-scaled plane instances.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Sequence

from plane3d.lattice import BranchKind
from plane3d.psi import ComplexPlane3D, wing_transform
from plane3d.recursive import segment_index, z_depth
from plane3d.sequences import canon_on_chain, normalize_chain
from plane3d.spring import CRITICAL_J, SpringPoint, is_on_critical_line


_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_J_HAT = complex(_INV_SQRT2, _INV_SQRT2)
_U_HAT = complex(_INV_SQRT2, -_INV_SQRT2)


@dataclass(frozen=True)
class ImaginaryAxes:
    """
    Orthonormal frame around the imaginary line.

    critical  — j = (1, 1) on n+ni (fixed under branch rotation)
    perp_u    — unit perpendicular in spring plane ℂ
    depth_v   — ζ axis (vertical in ℂ×ℝ embedding)
    """

    critical: SpringPoint
    perp_u: SpringPoint
    depth_v: float  # 1.0 = unit depth step

    @classmethod
    def default(cls) -> ImaginaryAxes:
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return cls(
            critical=CRITICAL_J,
            perp_u=SpringPoint(inv_sqrt2, -inv_sqrt2),
            depth_v=1.0,
        )


@dataclass(frozen=True)
class VelocityPlane:
    """
    One complex-plane instance at a velocity tier.

    branch_k     : 0..3 quarter-turn around j
    segment      : transgressor regime index
    velocity     : unit speed along rail (1.0 = nominal)
    """

    chain: tuple[float, ...]
    n: float
    branch_k: int
    segment: int
    velocity: float
    psi: ComplexPlane3D

    @property
    def branch(self) -> BranchKind:
        return BranchKind((self.branch_k % 4) + 1)


def j_hat_complex() -> complex:
    return _J_HAT


def u_hat_complex() -> complex:
    """Unit axis perpendicular to j in the spring plane."""
    return _U_HAT


def decompose_around_critical(z: complex) -> tuple[complex, complex]:
    """
    z = z_parallel + z_perp  where z_parallel lies on j (fixed under rotation).

    Returns (parallel, perp_vector) with perp_vector = b * u_hat.
    """
    a = (z * _J_HAT.conjugate()).real
    b = (z * _U_HAT.conjugate()).real
    return a * _J_HAT, b * _U_HAT


def compose_around_critical(parallel: complex, perp: complex) -> complex:
    return parallel + perp


def rotate_around_critical_line(
    z: complex,
    zeta: float,
    branch_k: int,
) -> tuple[complex, float]:
    """
    Quarter-turn around j in the plane ⊥ j spanned by (u_hat, ζ).

    The critical line j = n+ni is the rotation axis.  Spring parallel to j
    and the perpendicular u component trade with depth ζ under branch_k.
    """
    k = branch_k % 4
    if k == 0:
        return z, zeta
    a = (z * _J_HAT.conjugate()).real
    b = (z * _U_HAT.conjugate()).real
    parallel = a * _J_HAT
    angle = k * math.pi / 2.0
    ca, sa = math.cos(angle), math.sin(angle)
    b_new = b * ca - zeta * sa
    zeta_new = b * sa + zeta * ca
    return parallel + b_new * _U_HAT, zeta_new


def rotate_around_critical(z: complex, branch_k: int, *, zeta: float = 0.0) -> complex:
    """Spring readout after branch_k rotation (depth couples at zeta≠0)."""
    z_rot, _ = rotate_around_critical_line(z, zeta, branch_k)
    return z_rot


def rotate_psi_around_critical(psi: ComplexPlane3D, branch_k: int) -> ComplexPlane3D:
    z_rot, zeta_rot = rotate_around_critical_line(psi.z, psi.zeta, branch_k)
    return ComplexPlane3D(z=z_rot, zeta=zeta_rot)


def segment_velocity(chain: Sequence[float] | tuple[float, ...], n: float) -> tuple[int, float]:
    """
    Regime index and velocity scale from transgressor position.

    Interior segments (between anchors) use locked depth — slower ζ walk.
    End segments use full n + sum(A) — nominal velocity 1.0.
    """
    c = normalize_chain(chain)
    if not c:
        return 0, 1.0
    if all(float(int(x)) == x for x in c):
        primes = tuple(int(x) for x in c)
        seg = segment_index(primes, int(n))
        k = len(primes)
    else:
        from plane3d.sequences import segment_index_chain as _seg_chain

        seg = _seg_chain(c, n)
        k = len(c)
    if k <= 2:
        return seg, 1.0
    if 0 < seg < k:
        return seg, 0.5
    return seg, 1.0


def velocity_plane(
    chain: Sequence[float] | tuple[float, ...],
    n: float,
    branch_k: int,
    *,
    wing: int = 1,
    lock_interior: bool = True,
) -> VelocityPlane:
    """Build one velocity-tier complex plane at branch rotation k."""
    c = normalize_chain(chain)
    seg, vel = segment_velocity(c, n)
    base = wing_transform(BranchKind.VA1, c, n, wing, lock_interior=lock_interior)
    zeta_in = base.zeta if (branch_k % 4) == 0 else base.zeta * vel
    z_rot, zeta_rot = rotate_around_critical_line(base.z, zeta_in, branch_k)
    psi = ComplexPlane3D(z=z_rot, zeta=zeta_rot)
    return VelocityPlane(
        chain=c,
        n=float(n),
        branch_k=branch_k % 4,
        segment=seg,
        velocity=vel,
        psi=psi,
    )


def four_branch_rotation_planes(
    chain: Sequence[float] | tuple[float, ...],
    n: float,
    *,
    wing: int = 1,
    lock_interior: bool = True,
) -> tuple[VelocityPlane, ...]:
    """Four-way branching = four quarter-positions around j."""
    return tuple(
        velocity_plane(chain, n, k, wing=wing, lock_interior=lock_interior)
        for k in range(4)
    )


def branch_rotation_vs_canon(
    chain: Sequence[float] | tuple[float, ...],
    n: float,
    branch_k: int,
    *,
    wing: int = 1,
    tol: float = 1e-6,
) -> bool:
    """
    Compare rotation picture to legacy canon branch at same k.

    Exact match holds at Layer-0 and some single-anchor cases; full k-chain
    VA2..VA4 use different algebras — returns False when formulas diverge.
    """
    c = normalize_chain(chain)
    vp = velocity_plane(c, n, branch_k, wing=wing)
    legacy = wing_transform(BranchKind(branch_k + 1), c, n, wing)
    dz = abs(vp.psi.z - legacy.z)
    dzeta = abs(vp.psi.zeta - legacy.zeta)
    return dz <= tol and dzeta <= tol


def verify_branch_rotation_gates(tol: float = 1e-9) -> dict[str, bool]:
    axes = ImaginaryAxes.default()
    samples: list[tuple[complex, float]] = [
        (1 + 0j, 0.0),
        (3 + 4j, 5.0),
        (2 + 2j, 3.0),
        (-1 + 2j, 2.0),
    ]
    parallel_fixed = True
    four_cycle = True
    j_fixed = True
    for z, zeta in samples:
        par, _ = decompose_around_critical(z)
        z1, zeta1 = rotate_around_critical_line(z, zeta, 1)
        par1, _ = decompose_around_critical(z1)
        if abs(par1 - par) > tol:
            parallel_fixed = False
        z_acc, zeta_acc = z, zeta
        for _ in range(4):
            z_acc, zeta_acc = rotate_around_critical_line(z_acc, zeta_acc, 1)
        if abs(z_acc - z) > tol or abs(zeta_acc - zeta) > tol:
            four_cycle = False

    on_line = SpringPoint(5.0, 5.0)
    z_line = on_line.to_complex()
    z_line_rot, _ = rotate_around_critical_line(z_line, 0.0, 1)
    if abs(z_line_rot - z_line) > tol:
        j_fixed = False
    par_line, _ = decompose_around_critical(z_line)
    par_line_rot, _ = decompose_around_critical(
        rotate_around_critical_line(z_line, 4.0, 1)[0]
    )
    if abs(par_line_rot - par_line) > tol:
        j_fixed = False

    dot = axes.critical.x * axes.perp_u.x + axes.critical.y * axes.perp_u.y
    axes_ok = is_on_critical_line(axes.critical, tol) and abs(dot) <= tol

    vel_interior = True
    c = (3.0, 5.0, 7.0)
    seg_mid, vel_mid = segment_velocity(c, 6.0)
    if not (0 < seg_mid < 3 and vel_mid == 0.5):
        vel_interior = False

    four_distinct = True
    planes = four_branch_rotation_planes((3.0, 5.0, 7.0), 5.0)
    coords = {(p.psi.z, p.psi.zeta) for p in planes}
    if len(coords) < 4:
        four_distinct = False

    return {
        "parallel_invariant_under_rotation": parallel_fixed,
        "four_quarter_cycle": four_cycle,
        "critical_line_fixed": j_fixed,
        "imaginary_axes_frame": axes_ok,
        "interior_segment_slower_velocity": vel_interior,
        "four_branches_distinct_spring": four_distinct,
    }
