"""
Square-root operators on the 3D complex plane.

  sqrt_spring(z, branch)  — normal √ : half-angle / sheet in the spring plane ℂ
  sqrt_depth(psi, branch) — upside-down √ : half-branch on depth ζ (ℝ)

Layer-0 link (PROVEN): |z₀|² = 2n² with ζ₀ = n  ⇒  |z|/√2 = ζ on the n+ni line.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from plane3d.psi import ComplexPlane3D, imaginary_start
from plane3d.spring import SpringPoint, i_act, spring_mul


@dataclass(frozen=True)
class SheetIndex:
    """2D spring sheet (0..1) × 3D depth sheet (0..1) → 4 of 8 wing quadrants in ℂ."""

    spring_branch: int  # 0 = principal, 1 = opposite Arg/2 sheet
    depth_branch: int   # 0 = principal ζ, 1 = opposite ζ half-branch

    def flat(self) -> int:
        return (self.spring_branch & 1) | ((self.depth_branch & 1) << 1)


def sqrt_spring(z: complex, branch: int = 0) -> complex:
    """
    Principal square root in ℂ; branch 1 is the opposite Riemann sheet.

    Arg(√z) = Arg(z)/2.  √(−1) on branch 0 = i (quarter-turn generator).
    """
    root = cmath.sqrt(z)
    if branch & 1:
        root = -root
    return root


def spring_square(root: complex) -> complex:
    """Square a spring root back to z."""
    return root * root


def sqrt_depth(psi: ComplexPlane3D, branch: int = 0) -> ComplexPlane3D:
    """
    Upside-down √ : half-branch on real depth ζ; spring z unchanged.

    For ζ ≥ 0: ±√ζ.  For ζ < 0: ±√|ζ| with sign from sheet (J partner).
    """
    zeta = psi.zeta
    if zeta == 0.0:
        root_zeta = 0.0
    elif zeta > 0.0:
        root_zeta = math.sqrt(zeta)
    else:
        root_zeta = -math.sqrt(-zeta)
    if branch & 1:
        root_zeta = -root_zeta
    return ComplexPlane3D(z=psi.z, zeta=root_zeta)


def depth_square(psi: ComplexPlane3D, branch: int = 0) -> ComplexPlane3D:
    """Square the depth root: (√↓ζ)² = ζ on the principal sheet."""
    root = sqrt_depth(psi, branch)
    return ComplexPlane3D(z=psi.z, zeta=root.zeta * root.zeta)


def layer0_depth_from_spring(z: complex) -> float:
    """|z|² = 2ζ²  ⇒  ζ = |z|/√2 (Layer-0 modulus link)."""
    return abs(z) / math.sqrt(2.0)


def layer0_spring_from_depth(zeta: float) -> complex:
    """Inverse on critical line: z = ζ + ζi."""
    return complex(zeta, zeta)


def apply_sheets(psi: ComplexPlane3D, sheet: SheetIndex) -> ComplexPlane3D:
    """Compose 2D spring √ sheet then 3D depth √ sheet."""
    z_root = sqrt_spring(psi.z, sheet.spring_branch)
    half = ComplexPlane3D(z=z_root, zeta=psi.zeta)
    return sqrt_depth(half, sheet.depth_branch)


def wing_sheet_mask(wing: int) -> SheetIndex:
    """
    Map wing 1..8 bit mask to (spring_sheet, depth_sheet).

    bit0 = depth (J), bit1 = Re flip in spring, bit2 = VB corridor.
    Spring sheet uses bit2|bit1 folded; depth uses bit0.
    """
    if not 1 <= wing <= 8:
        raise ValueError(f"wing must be 1..8, got {wing}")
    mask = wing - 1
    depth_branch = mask & 1
    spring_branch = (mask >> 1) & 1
    return SheetIndex(spring_branch=spring_branch, depth_branch=depth_branch)


def verify_sqrt_spring_square(tol: float = 1e-9) -> dict[str, bool]:
    samples = [
        1 + 0j,
        -1 + 0j,
        3 + 4j,
        -3 - 4j,
        0 + 0j,
        2 + 2j,
    ]
    sq_ok = True
    neg_one_is_i = abs(sqrt_spring(-1.0, 0) - 1j) <= tol
    branches_opposite = True
    for z in samples:
        r0 = sqrt_spring(z, 0)
        r1 = sqrt_spring(z, 1)
        if abs(spring_square(r0) - z) > tol:
            sq_ok = False
        if abs(spring_square(r1) - z) > tol:
            sq_ok = False
        if z != 0 and abs(r0 + r1) > tol:
            branches_opposite = False
    return {
        "sqrt_spring_square": sq_ok,
        "sqrt_neg_one_is_i": neg_one_is_i,
        "spring_branches_opposite": branches_opposite,
    }


def verify_sqrt_depth_square(tol: float = 1e-9) -> dict[str, bool]:
    samples = [imaginary_start(n) for n in (1.0, 3.0, 7.0, 16.0)]
    sq_ok = True
    j_partner = True
    for psi in samples:
        back = depth_square(psi, 0)
        if abs(back.zeta - psi.zeta) > tol:
            sq_ok = False
        if abs(back.z - psi.z) > tol:
            sq_ok = False
        # branch 1 root squares to same ζ (opposite root)
        back1 = depth_square(psi, 1)
        if abs(back1.zeta - psi.zeta) > tol:
            sq_ok = False
        d0 = sqrt_depth(psi, 0).zeta
        d1 = sqrt_depth(psi, 1).zeta
        if psi.zeta != 0 and abs(d0 + d1) > tol:
            j_partner = False
    return {
        "sqrt_depth_square": sq_ok,
        "depth_branches_opposite": j_partner,
    }


def verify_layer0_sqrt2_link(tol: float = 1e-9) -> dict[str, bool]:
    """On n+ni: ζ = n and |z|/√2 = n."""
    link = True
    for n in (1.0, 3.0, 7.0, 100.0):
        psi = imaginary_start(n)
        if abs(layer0_depth_from_spring(psi.z) - psi.zeta) > tol:
            link = False
        if abs(layer0_spring_from_depth(psi.zeta) - psi.z) > tol:
            link = False
    return {"layer0_sqrt2_link": link}


def verify_j_hat_square_is_i_act(tol: float = 1e-9) -> bool:
    """Unit critical vector squared = i_act on real axis (existing spring proof)."""
    j_hat = SpringPoint(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    j_sq = spring_mul(j_hat, j_hat)
    i_unit = i_act(SpringPoint(1.0, 0.0))
    return abs(j_sq.to_complex() - i_unit.to_complex()) <= tol


def verify_sqrt_gates(tol: float = 1e-9) -> dict[str, bool]:
    out: dict[str, bool] = {}
    out.update(verify_sqrt_spring_square(tol))
    out.update(verify_sqrt_depth_square(tol))
    out.update(verify_layer0_sqrt2_link(tol))
    out["j_hat_square_is_i_act"] = verify_j_hat_square_is_i_act(tol)
    return out
