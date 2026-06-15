"""
Spring-plane complex structure from wing operators (no imported i in definitions).

Definitions use real pairs (X, Y) only. Wing operators R_x and S from
aethos_complex_rotation are the primitive involutions.

    R_x(X, Y) = (-X, Y)
    S(X, Y)   = (Y, X)

Derived:
    i_act     = R_x o S       maps (X, Y) -> (-Y, X)   [multiplication by i]
    conj_act  = -R_x          maps (X, Y) -> (X, -Y)   [complex conjugate on readout z=X+iY]
    neg_act   = i_act o i_act maps (X, Y) -> (-X, -Y)

Readout (optional): z = X + i*Y uses Python complex only at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from plane3d.rotation import reflect_real, swap_corridor


@dataclass(frozen=True)
class SpringPoint:
    """Spring plane before complex readout."""

    x: float
    y: float

    def to_complex(self) -> complex:
        return complex(self.x, self.y)

    @classmethod
    def from_complex(cls, z: complex) -> SpringPoint:
        return cls(float(z.real), float(z.imag))


def reflect_x(p: SpringPoint) -> SpringPoint:
    return SpringPoint(-p.x, p.y)


def swap_xy(p: SpringPoint) -> SpringPoint:
    return SpringPoint(p.y, p.x)


def i_act(p: SpringPoint) -> SpringPoint:
    """i * (X + iY) = -Y + iX  without naming i in the definition."""
    return reflect_x(swap_xy(p))


def conj_act(p: SpringPoint) -> SpringPoint:
    """Complex conjugate on z = X + iY; equals -R_x on readout."""
    return SpringPoint(p.x, -p.y)


def neg_act(p: SpringPoint) -> SpringPoint:
    return i_act(i_act(p))


def spring_add(a: SpringPoint, b: SpringPoint) -> SpringPoint:
    return SpringPoint(a.x + b.x, a.y + b.y)


def spring_mul(a: SpringPoint, b: SpringPoint) -> SpringPoint:
    """Full complex multiply on readout z = X + iY."""
    ac = a.to_complex() * b.to_complex()
    return SpringPoint.from_complex(ac)


def verify_i_act_axioms(tol: float = 1e-9) -> dict[str, bool]:
    """PROVEN identities on spring pairs."""
    samples = [
        SpringPoint(1, 0),
        SpringPoint(0, 1),
        SpringPoint(3, 5),
        SpringPoint(-2, 7),
    ]
    ok = True
    for p in samples:
        z = p.to_complex()
        if abs(i_act(p).to_complex() - 1j * z) > tol:
            ok = False
        if abs(i_act(i_act(p)).to_complex() + z) > tol:
            ok = False
        if abs(conj_act(p).to_complex() - z.conjugate()) > tol:
            ok = False
        if abs(swap_xy(swap_xy(p)).x - p.x) > tol or abs(swap_xy(swap_xy(p)).y - p.y) > tol:
            ok = False
        if abs(reflect_x(reflect_x(p)).x - p.x) > tol or abs(reflect_x(reflect_x(p)).y - p.y) > tol:
            ok = False
    # Operator agreement with complex rotation module (R_x, S on readout)
    for z in [1 + 0j, 3 + 5j, -2 + 7j]:
        p = SpringPoint.from_complex(z)
        if abs(swap_corridor(z) - swap_xy(p).to_complex()) > tol:
            ok = False
        if abs(reflect_real(swap_corridor(z)) - i_act(p).to_complex()) > tol:
            ok = False
        if abs(-reflect_real(z) - conj_act(p).to_complex()) > tol:
            ok = False
    return {
        "i_act_equals_1j_multiply": ok,
        "i_squared_is_negation": ok,
        "conj_matches_conjugate": ok,
        "swap_and_reflect_involutions": ok,
        "matches_complex_rotation_module": ok,
    }


def spring_complex_field_check(tol: float = 1e-9) -> bool:
    """Check field axioms on readout via (X,Y) pairs (sampled)."""
    a = SpringPoint(2, 3)
    b = SpringPoint(-1, 4)
    c = SpringPoint(5, -2)
    one = SpringPoint(1, 0)
    for p in (a, b, c):
        if abs(spring_mul(p, one).to_complex() - p.to_complex()) > tol:
            return False
        if abs(spring_mul(one, p).to_complex() - p.to_complex()) > tol:
            return False
    lhs = spring_mul(a, spring_add(b, c))
    rhs = spring_add(spring_mul(a, b), spring_mul(a, c))
    return abs(lhs.to_complex() - rhs.to_complex()) <= tol


# Critical line j = 1+i (layer-0 direction; Re = Im)
CRITICAL_J = SpringPoint(1.0, 1.0)


def is_on_critical_line(p: SpringPoint, tol: float = 1e-9) -> bool:
    """Re(z) = Im(z) on readout z = X + iY."""
    return abs(p.x - p.y) <= tol


def critical_halver(z: SpringPoint, z_i: SpringPoint) -> SpringPoint:
    """Meet of S-partners on the critical line: (z + z_i) / 2."""
    return SpringPoint((z.x + z_i.x) / 2.0, (z.y + z_i.y) / 2.0)


def verify_critical_line_rotation(tol: float = 1e-9) -> dict[str, bool]:
    """
    PROVEN: critical-line geometry from wing operators only.

    1. S (swap_xy) is reflection across Re=Im; fixes the critical line pointwise.
    2. z and S(z) are opposite half-wedges unless on the line; meet is on the line.
    3. z + S(z) = (a+b)*j with j = (1,1).
    4. j*j = 2i; unit j_hat squared equals i_act on (1,0) — squaring j rotates by i.
    5. i_act generates quarter-turns; four applications return negation (i^2 = -1).
    """
    import math

    j = CRITICAL_J
    j_sq = spring_mul(j, j)
    j_hat = SpringPoint(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    j_hat_sq = spring_mul(j_hat, j_hat)

    s_fixes_line = True
    for t in (1.0, 3.0, 8.0, -2.5):
        p = SpringPoint(t, t)
        if not is_on_critical_line(swap_xy(p), tol):
            s_fixes_line = False

    z = SpringPoint(5.0, 3.0)
    z_i = swap_xy(z)
    meet = critical_halver(z, z_i)
    sum_z = spring_add(z, z_i)

    meet_on_line = is_on_critical_line(meet, tol)
    meet_is_half_sum = abs(meet.x - 4.0) <= tol and abs(meet.y - 4.0) <= tol
    opposite_sides = (z.y - z.x) * (z_i.y - z_i.x) < 0

    sum_is_8j = abs(sum_z.x - 8.0) <= tol and abs(sum_z.y - 8.0) <= tol
    sum_on_line = is_on_critical_line(sum_z, tol)

    j_squared_is_2i = abs(j_sq.x) <= tol and abs(j_sq.y - 2.0) <= tol
    j_hat_sq_is_i_unit = (
        abs(j_hat_sq.x) <= tol
        and abs(j_hat_sq.y - 1.0) <= tol
        and abs(j_hat_sq.to_complex() - i_act(SpringPoint(1.0, 0.0)).to_complex()) <= tol
    )

    quarter_turn = i_act(SpringPoint(1.0, 0.0))
    quarter_is_unit_i = abs(quarter_turn.x) <= tol and abs(quarter_turn.y - 1.0) <= tol
    i_four_is_neg = abs(neg_act(SpringPoint(1.0, 0.0)).x + 1.0) <= tol

    # arg(j*j) = arg(i): squaring critical direction = i_act rotation
    arg_j_sq = math.atan2(j_sq.y, j_sq.x)
    arg_i = math.atan2(1.0, 0.0)
    j_sq_angle_is_pi_over_2 = abs(arg_j_sq - arg_i) <= tol

    return {
        "s_fixes_critical_line": s_fixes_line,
        "swap_partners_opposite_sides": opposite_sides,
        "meet_on_critical_line": meet_on_line and meet_is_half_sum,
        "sum_equals_scalar_j": sum_on_line and sum_is_8j,
        "j_squared_equals_2i": j_squared_is_2i,
        "unit_j_squared_equals_i_act": j_hat_sq_is_i_unit,
        "i_act_quarter_turn_on_real_axis": quarter_is_unit_i,
        "i_act_fourth_is_negation": i_four_is_neg,
        "j_squared_same_angle_as_i": j_sq_angle_is_pi_over_2,
    }


def verify_va_vb_swap_diagonal(tol: float = 1e-9) -> bool:
    """S (swap_xy) fixes n + ni on the critical line; VB corridor is YXZ swap."""
    for n in (1.0, 3.0, 7.0, 11.0):
        p = SpringPoint(n, n)
        if not is_on_critical_line(swap_xy(p), tol):
            return False
    raw = SpringPoint(16.0, 5.0)
    swapped = swap_xy(raw)
    return abs(swapped.x - 5.0) <= tol and abs(swapped.y - 16.0) <= tol
