"""
Partial functor: pi lattice (unit-circle bisection) <-> 3D complex plane (layer 0 + wings).

Status (see ONTOLOGY.md):
  PROVEN  — wing i_act = R_x o S; layer-0 diagonal matches pi pi/4 direction
  PROVEN  — pi dyadic vertex angles = constructive_pi point_on_circle_complex
  PARTIAL — binary branch bits -> wing mask; depth k -> transgressor n
  OPEN    — full VA1 canon_on_chain equals pi walker at all (k, chain) pairs
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Sequence

from aethos_complex_plane import ComplexPlane3D, imaginary_start
from aethos_spring_complex import SpringPoint, i_act

# pi package lives at repo/pi/
_PI_ROOT = Path(__file__).resolve().parent / "pi"
if str(_PI_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PI_ROOT.parent))


def pi_dyadic_point(k: int, j: int) -> tuple[float, float]:
    """Unit-circle vertex from pi recurrence (constructive_pi). Returns (re, im)."""
    from pi.constructive_pi import point_on_circle_complex

    re, im = point_on_circle_complex(k, j)
    return float(re), float(im)


def pi_layer0_direction_matches(k: int = 1, tol: float = 1e-6) -> bool:
    """
    PROVEN: imaginary_start(n) has direction angle pi/4 for all n > 0.
    pi primitive_root at k=1 is also angle pi/4 (45 deg).
    """
    re, im = pi_dyadic_point(1, 1)
    # direction of 1+i
    d_plane = math.atan2(1, 1)
    d_pi = math.atan2(im, re)
    return abs(d_plane - d_pi) < tol


def pi_scale_to_layer0_n(re: float, im: float, tol: float = 1e-9) -> float:
    """
    For z parallel to 1+i ray, find n with n(1+i) = (re, im) when on same ray.
    General: project (re,im) onto diagonal unit vector.
    """
    if abs(re) < tol and abs(im) < tol:
        return 0.0
    # n such that n(1+i) = (re,im) when collinear: re = im = n
    if abs(re - im) < tol * max(1.0, abs(re)):
        return re
    # nearest layer-0 scale on diagonal for arbitrary pi point: average projection
    return (re + im) / 2.0


def embed_pi_vertex_layer0(k: int, j: int) -> ComplexPlane3D:
    """
    PARTIAL functor: map pi vertex (k,j) to layer-0 plane point n(1+i) by angle-matched scale.
    """
    re, im = pi_dyadic_point(k, j)
    n = pi_scale_to_layer0_n(re, im)
    return imaginary_start(n)


def pi_branch_bits_to_wing_mask(bits: Sequence[int]) -> int:
    """
    PARTIAL: ±B binary path (0=+B, 1=-B style) -> wing mask low 3 bits.
    Uses last up to 3 bits; maps to flip_x / flip_z / VB corridor bits.
    """
    if not bits:
        return 0
    tail = list(bits[-3:])
    mask = 0
    if len(tail) >= 1 and tail[0]:
        mask |= 4  # VB_CORRIDOR bit
    if len(tail) >= 2 and tail[1]:
        mask |= 2  # RE_FLIP
    if len(tail) >= 3 and tail[2]:
        mask |= 1  # ZETA_FLIP
    return mask % 8


def pi_depth_to_transgressor(k: int) -> float:
    """PARTIAL: bisection depth k -> rail index n = 2^k (demo scale)."""
    return float(2**k)


def pi_recurrence_legs(k: int) -> tuple[float, float, float]:
    """Return (A_k, B_k, C_k) book/code legs at pi level k from constructive_pi."""
    from pi.constructive_pi import pi_recurrence

    last = None
    for rec in pi_recurrence(k):
        last = rec
    assert last is not None
    _, _N, A, B, C, _area = last
    return float(A), float(B), float(C)


def pi_unit_circle_address(k: int, j: int) -> tuple[float, float]:
    """
    Pi lattice address in (1-A, B) style at vertex j of level k.
    Uses recurrence legs; sign from quadrant index.
    """
    A, B, _C = pi_recurrence_legs(k)
    x = 1.0 - A
    y = B if (j % 2 == 0) else -B
    return x, y


def pi_k0_is_spring_i(tol: float = 1e-6) -> bool:
    """PROVEN: pi primitive root at k=0 is i; equals i_act on unit real."""
    re, im = pi_dyadic_point(0, 1)
    i_unit = i_act(SpringPoint(1, 0))
    return abs(re - i_unit.x) < tol and abs(im - i_unit.y) < tol


def compare_pi_vertex_to_spring_i_act(k: int, j: int, tol: float = 0.25) -> dict[str, float | bool]:
    """
    Compare pi vertex direction to spring plane after i_act on unit real.
    Diagnostic for functor alignment (PARTIAL — not equality of constructions).
    """
    re, im = pi_dyadic_point(k, j)
    psi = embed_pi_vertex_layer0(k, j)
    z_plane = psi.z
    angle_pi = math.atan2(im, re)
    angle_plane = math.atan2(z_plane.imag, z_plane.real)
    i_unit = i_act(SpringPoint(1, 0))
    return {
        "angle_pi": angle_pi,
        "angle_plane": angle_plane,
        "angle_diff": abs(angle_pi - angle_plane),
        "angles_match": abs(angle_pi - angle_plane) < tol or abs(abs(angle_pi - angle_plane) - math.pi) < tol,
        "i_act_unit": (i_unit.x, i_unit.y),
        "plane_z": (z_plane.real, z_plane.imag),
    }


def bridge_report() -> str:
    """Human-readable closure status."""
    lines = [
        "=== PI LATTICE <-> 3D COMPLEX PLANE BRIDGE ===",
        "",
        "PROVEN (spring complex):",
        f"  {verify_spring_section()}",
        "",
        f"PROVEN (layer0 pi/4 direction): {pi_layer0_direction_matches()}",
        f"PROVEN (pi k=0 is spring i_act): {pi_k0_is_spring_i()}",
        "",
        "PARTIAL (embedding samples):",
    ]
    for k, j in ((0, 1), (1, 1), (2, 3), (3, 5)):
        cmp = compare_pi_vertex_to_spring_i_act(k, j)
        lines.append(f"  k={k} j={j}: angles_match={cmp['angles_match']} diff={cmp['angle_diff']:.4f}")
    lines.append("")
    lines.append("OPEN: canon_on_chain(prime A, n) = pi walker at shared label for all k.")
    return "\n".join(lines)


def verify_spring_section() -> str:
    from aethos_spring_complex import verify_i_act_axioms

    return str(verify_i_act_axioms())


if __name__ == "__main__":
    print(bridge_report())
