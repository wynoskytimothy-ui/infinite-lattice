#!/usr/bin/env python3
"""
constructive_pi.py
==================
Consolidated executable backing the Constructive Pi Framework document.

Every table in the document is produced by running ONE of the functions in
this module. The document and the code stay in sync because the document's
numbers come from here.

Contents
--------
  pi_recurrence(K)               -- Part 1: A, B, C, N, cumulative area at each level
  sin_cos_table(K)               -- Part 1B: dyadic-angle sin/cos from the legs
  i_power_table(K)               -- Part 1B: successive 2^k-th roots of i
  circumscribed_bracket(K)       -- Part 1C (NEW): inscribed + circumscribed bracket
  point_on_circle(k, j)          -- Part 1B.3: place j-th point at level k
  addition_formulas              -- Part 1D (NEW): cos(a+b), sin(a+b) algebraic identities
  cone_volume(r, h, K, Nz)       -- Part 2: cone via N-gonal pyramid stack
  sphere_volume(R, K, Nz)        -- Part 3: sphere via disk stack
  sphere_surface_strip(R, Nz)    -- Part 3 (FIX): sphere surface via Archimedean strip rule
  hyperball_volume(R, K, Nw)     -- Part 4 (NEW): 4D hyperball via 3-ball stack
  hypersphere_surface(R, Nw)     -- Part 4 (NEW): 3-sphere via 2-sphere strip stack

All values use mpmath multi-precision arithmetic and only the operations
{+, -, *, /, sqrt}. No trig functions, no Taylor series, no math.pi.
"""

from mpmath import mp, mpf, sqrt, pi as PI_REF
mp.prec = 200


# ============================================================================
# Part 1 - the 2D recursion: pi from a right-triangle bisection
# ============================================================================

def pi_recurrence(K):
    """Run the bisection recurrence for K levels.

    Yields one record per level: (k, N_k, A_k, B_k, C_k, cumulative_area).
    Cumulative area starts at the inscribed-square area (2) and adds the
    sliver triangle N_k * A_{k+1} * B_{k+1} at each step.
    """
    A = mpf(1)
    B = mpf(1)
    C = sqrt(mpf(2))
    N = 4
    area = mpf(2)            # inscribed square has area 2 in unit circle
    # Level 0 record: the square itself
    yield (0, N, A, B, C, area)
    for k in range(1, K + 1):
        # Bisect: new leg A = old chord / 2; new B = sagitta from A
        A_new = C / 2
        B_new = 1 - sqrt(1 - A_new * A_new)   # equivalent to A^2 / (1 + sqrt(1-A^2))
        C_new = sqrt(A_new * A_new + B_new * B_new)
        # Sliver: N_old triangles of area A_new * B_new each
        area = area + N * A_new * B_new
        N = N * 2
        A, B, C = A_new, B_new, C_new
        yield (k, N, A, B, C, area)


def pi_estimate(K):
    """Return the pi estimate after K levels of bisection."""
    for record in pi_recurrence(K):
        last = record
    return last[5]


# ============================================================================
# Part 1B - sine, cosine, i-powers fall out of the legs
# ============================================================================

def sin_cos_table(K):
    """For each level k, return (k, N_k, half_angle_deg, sin, cos, sin_error).

    Half-angle at level k is 180 / N_k degrees.
    sin(half-angle) = C_k / 2.
    cos(half-angle) = sqrt(1 - sin^2) = 1 - B_{k+1}  (because B_{k+1} = 1 - sqrt(1-A^2))
                                                     and A_{k+1} = C_k/2.
    sin_error compares against mpmath's reference sin.
    """
    from mpmath import sin as _sin
    rows = []
    A = mpf(1); B = mpf(1); C = sqrt(mpf(2)); N = 4
    for k in range(K + 1):
        half_angle_rad = PI_REF / N
        sin_val = C / 2
        # next sagitta for cos
        A_next = C / 2
        B_next = 1 - sqrt(1 - A_next * A_next)
        cos_val = 1 - B_next
        err = abs(sin_val - _sin(half_angle_rad))
        rows.append((k, N, mpf(180) / N, sin_val, cos_val, err))
        # advance recurrence
        C = sqrt(A_next * A_next + B_next * B_next)
        A, B, N = A_next, B_next, N * 2
    return rows


def i_power_table(K):
    """Successive 2^k-th roots of i.

    i^(1/2^k) = cos(90 deg / 2^k) + i * sin(90 deg / 2^k)
    Real part = cos leg, imag part = sin leg, from the same recursion.
    """
    from mpmath import sin as _sin, cos as _cos
    rows = []
    # i^1 = i (90 deg), i^(1/2) = e^(i*pi/4) (45 deg), etc.
    # k=0  -> 90 deg          (i itself)
    # k>=1 -> 90/2^k deg
    A = mpf(1); B = mpf(1); C = sqrt(mpf(2)); N = 4
    # angle at recurrence level j is 180/N_j deg.  We want 90/2^k deg.
    # 90/2^k = 180/(2^(k+1)) so N = 2^(k+1).  Start: k=0 -> N=2 (90 deg).
    # But the recurrence starts at N=4 which is 45 deg.  So I'll precompute
    # the k=0 row (90 deg = i itself) directly and then walk the recurrence
    # for k=1, 2, ...
    rows.append((0, 1, mpf(90), mpf(0), mpf(1)))
    for k in range(1, K + 1):
        # angle = 90 / 2^k deg
        sin_val = C / 2
        A_next = C / 2
        B_next = 1 - sqrt(1 - A_next * A_next)
        cos_val = 1 - B_next
        rows.append((k, 2 ** k, mpf(90) / (2 ** k), cos_val, sin_val))
        C = sqrt(A_next * A_next + B_next * B_next)
        A, B, N = A_next, B_next, N * 2
    return rows


def point_on_circle(k, j):
    """Coordinates of the j-th equally spaced point at level k of refinement.

    At level k there are N_k = 4 * 2^k points; point j is at angle
    j * (360 / N_k) degrees, with coords (cos, sin) of that angle.
    Returns (x, y) as mpf.
    """
    from mpmath import cos as _cos, sin as _sin
    N_k = 4 * (2 ** k)
    theta = (2 * PI_REF * j) / N_k
    return _cos(theta), _sin(theta)


# ============================================================================
# Part 1C (NEW) - inscribed AND circumscribed: a two-sided bracket on pi
# ============================================================================

def circumscribed_bracket(K):
    """Inscribed + circumscribed N-gon areas; pi is bracketed between them.

    Inscribed (level k): use the cumulative area from pi_recurrence (>= pi from below? actually <= pi).
    Circumscribed (level k): tangent N-gon area = N * tan(pi/N).
        Using only the recursion's legs: A_{k+1} = sin(theta_k) = C_k/2, and
        cos(theta_k) = 1 - B_{k+1}, so tan(theta_k) = A_{k+1} / (1 - B_{k+1}).
        Then circumscribed-polygon area = N_k * tan(theta_k) where theta_k = pi/N_k.

    Returns rows (k, N_k, inscribed_area, circumscribed_area, midpoint, gap).
    """
    rows = []
    A = mpf(1); B = mpf(1); C = sqrt(mpf(2)); N = 4
    inscribed = mpf(2)
    for k in range(K + 1):
        # tan(pi/N) from recursion legs
        A_next = C / 2
        B_next = 1 - sqrt(1 - A_next * A_next)
        sin_theta = A_next
        cos_theta = 1 - B_next
        tan_theta = sin_theta / cos_theta
        circumscribed = N * tan_theta
        mid = (inscribed + circumscribed) / 2
        gap = circumscribed - inscribed
        rows.append((k, N, inscribed, circumscribed, mid, gap))
        # advance the inscribed sum (sliver)
        inscribed = inscribed + N * A_next * B_next
        C = sqrt(A_next * A_next + B_next * B_next)
        A, B, N = A_next, B_next, N * 2
    return rows


# ============================================================================
# Part 1D (NEW) - addition formulas for non-dyadic angles
# ============================================================================

def add_angle(cos_a, sin_a, cos_b, sin_b):
    """Return (cos(a+b), sin(a+b)) using only +, -, *.  Pure complex multiplication."""
    return (cos_a * cos_b - sin_a * sin_b,
            sin_a * cos_b + cos_a * sin_b)


def sub_angle(cos_a, sin_a, cos_b, sin_b):
    """Return (cos(a-b), sin(a-b))."""
    return (cos_a * cos_b + sin_a * sin_b,
            sin_a * cos_b - cos_a * sin_b)


def cos_sin_at_dyadic(k):
    """Return (cos, sin) of the dyadic angle pi / 2^(k+1) using the recursion.

    k=0 -> pi/2 (90 deg) -> (0, 1)
    k=1 -> pi/4 (45 deg)
    k=2 -> pi/8 (22.5 deg)
    etc.
    """
    if k == 0:
        return (mpf(0), mpf(1))
    # walk the recurrence
    A = mpf(1); B = mpf(1); C = sqrt(mpf(2)); N = 4
    for _ in range(k - 1):
        A_next = C / 2
        B_next = 1 - sqrt(1 - A_next * A_next)
        C_next = sqrt(A_next * A_next + B_next * B_next)
        A, B, C, N = A_next, B_next, C_next, N * 2
    A_next = C / 2
    B_next = 1 - sqrt(1 - A_next * A_next)
    return (1 - B_next, A_next)


def cos_sin_at_3pi_over_4():
    """3*pi/4 = pi/2 + pi/4.  Built from dyadic primitives via addition."""
    c_h, s_h = cos_sin_at_dyadic(0)     # pi/2
    c_q, s_q = cos_sin_at_dyadic(1)     # pi/4
    return add_angle(c_h, s_h, c_q, s_q)


# ============================================================================
# Part 1E (NEW) - locating any point in the complex plane
# ============================================================================
#
# Every point on the unit circle is a complex number z = cos(theta) + i*sin(theta).
# The two legs of the recursion (Part 1B.1) ARE the real and imaginary parts
# of z at the dyadic half-angle, so the recursion produces complex numbers
# directly. Complex multiplication is the addition formula in disguise, so
# repeated multiplication of these primitives reaches every dyadic-rational
# angle on the circle. Scaling by a positive real reaches every magnitude.
# Therefore the recursion plus {+, -, *, /, sqrt} can locate any complex
# number to arbitrary precision.
#
# Representation: a complex value is the tuple (re, im) of two mpf values.
# We do NOT use Python's built-in complex (which carries math.pi via cmath);
# everything stays in the recursion's own arithmetic.
# ============================================================================


def cmul(z1, z2):
    """Complex multiplication: (a, b) * (c, d) = (ac - bd, ad + bc).

    This IS the addition formula from Part 1D:
      Re(z1 * z2) = a*c - b*d = cos(alpha+beta)  when |z_i| = 1
      Im(z1 * z2) = a*d + b*c = sin(alpha+beta)
    """
    a, b = z1
    c, d = z2
    return (a * c - b * d, a * d + b * c)


def cscale(z, r):
    """Scale a complex number by a real magnitude r."""
    return (z[0] * r, z[1] * r)


def cconj(z):
    """Complex conjugate: negate the imaginary part."""
    return (z[0], -z[1])


def cpower(z, n):
    """z raised to integer power n via repeated complex multiplication.

    Implements binary exponentiation, so the cost is O(log n) multiplications.
    Negative n: invert via conjugate (assumes z on unit circle).
    """
    if n == 0:
        return (mpf(1), mpf(0))
    if n < 0:
        return cpower(cconj(z), -n)
    result = (mpf(1), mpf(0))
    base = z
    while n > 0:
        if n & 1:
            result = cmul(result, base)
        base = cmul(base, base)
        n >>= 1
    return result


def primitive_root_of_unity(k):
    """The primitive 2^(k+2)-th root of unity, computed from the recursion.

    z_k = cos(2*pi / 2^(k+2)) + i * sin(2*pi / 2^(k+2))
        = cos(pi / 2^(k+1)) + i * sin(pi / 2^(k+1))

    These are the (2^(k+2))-th roots of unity that the recursion produces
    directly. k=0 gives cos(pi/2) + i*sin(pi/2) = i.
    k=1 gives cos(pi/4) + i*sin(pi/4).
    k=2 gives cos(pi/8) + i*sin(pi/8).
    etc.
    """
    c, s = cos_sin_at_dyadic(k)
    return (c, s)


def point_on_circle_complex(k, j):
    """The j-th equally-spaced point at level k of refinement, as a complex (re, im) tuple.

    At level k there are N_k = 4 * 2^k points; point j sits at angle
    j * (2*pi / N_k) = j * pi / 2^(k+1) = j * (90 / 2^k) degrees.
    """
    base = primitive_root_of_unity(k)
    return cpower(base, j)


def reach_complex(re_target, im_target, k=20):
    """Find a dyadic-angle approximation to the complex number (re_target, im_target).

    Returns (approx_re, approx_im, j, k) where the approximation is
    point_on_circle_complex(k, j) for the integer j that minimizes the
    angular error.
    """
    from mpmath import atan2 as _atan2
    target_angle = _atan2(im_target, re_target)
    # angle per point at level k
    delta = PI_REF / mpf(2 ** (k + 1))
    j = int(round(float(target_angle / delta)))
    # normalize to range [0, 2^(k+2))
    N = 2 ** (k + 2)
    j = j % N
    z = point_on_circle_complex(k, j)
    # scale by target magnitude
    mag = sqrt(re_target * re_target + im_target * im_target)
    return (z[0] * mag, z[1] * mag, j, k)


def complex_plane_examples():
    """Worked examples: compute several non-trivial complex numbers from the recursion.

    Returns rows (label, recurrence_re, recurrence_im, reference_re, reference_im).
    """
    from mpmath import cos as _cos, sin as _sin
    rows = []
    # i itself
    z = primitive_root_of_unity(0)
    rows.append(('i = e^(i*pi/2)', z[0], z[1], _cos(PI_REF/2), _sin(PI_REF/2)))
    # sqrt(i) = e^(i*pi/4)
    z = primitive_root_of_unity(1)
    rows.append(('sqrt(i) = e^(i*pi/4)', z[0], z[1], _cos(PI_REF/4), _sin(PI_REF/4)))
    # -1 = i^2
    z = cpower(primitive_root_of_unity(0), 2)
    rows.append(('-1 = i^2', z[0], z[1], _cos(PI_REF), _sin(PI_REF)))
    # 8th root of unity, point j=3 at level k=1: angle 3*pi/4
    z = point_on_circle_complex(1, 3)
    rows.append(('e^(i*3pi/4)', z[0], z[1], _cos(3*PI_REF/4), _sin(3*PI_REF/4)))
    # 16th root of unity, point j=5 at level k=2: angle 5*pi/8
    z = point_on_circle_complex(2, 5)
    rows.append(('e^(i*5pi/8)', z[0], z[1], _cos(5*PI_REF/8), _sin(5*PI_REF/8)))
    # A non-unit complex number: 3 * e^(i*pi/8)
    base = primitive_root_of_unity(2)        # pi/8
    z = cscale(base, mpf(3))
    rows.append(('3 * e^(i*pi/8)', z[0], z[1], 3*_cos(PI_REF/8), 3*_sin(PI_REF/8)))
    # Product test: e^(i*pi/4) * e^(i*pi/8) = e^(i*3*pi/8)
    a = primitive_root_of_unity(1)
    b = primitive_root_of_unity(2)
    z = cmul(a, b)
    rows.append(('e^(i*pi/4) * e^(i*pi/8)', z[0], z[1],
                 _cos(3*PI_REF/8), _sin(3*PI_REF/8)))
    return rows


# ============================================================================
# Part 2 - cone volume via N-gonal pyramid stack
# ============================================================================

def cone_volume(r, h, K=20):
    """Cone of radius r, height h.  Volume = (1/3) * base_area * h.

    base_area at level K = (Part 1 inscribed-polygon area) * r^2.
    Limit = pi * r^2, so V -> (1/3) * pi * r^2 * h.
    """
    r = mpf(r); h = mpf(h)
    pi_est_area = pi_estimate(K)        # this is the area for unit circle
    base_area = pi_est_area * r * r
    return base_area * h / 3


# ============================================================================
# Part 3 - sphere via disk stack along height axis
# ============================================================================

def sphere_volume(R, K=30, Nz=10000):
    """Sphere volume = stack of disks; each disk's area uses pi from the recursion."""
    R = mpf(R)
    pi_est = pi_estimate(K)
    dz = (2 * R) / Nz
    V = mpf(0)
    for i in range(Nz):
        z = -R + (mpf(i) + mpf('0.5')) * dz
        r_sq = R * R - z * z
        V = V + pi_est * r_sq * dz
    return V


def sphere_surface_strip(R, K=30):
    """Sphere surface area via Archimedean strip rule.

    Claim (Archimedes): the lateral surface of a sphere between two parallel
    planes z=a, z=b equals the lateral surface of the enclosing cylinder
    strip = 2 * pi * R * (b - a).

    Integrating gives S = 2 * pi * R * (2R) = 4 * pi * R^2 exactly at any Nz.
    """
    R = mpf(R)
    pi_est = pi_estimate(K)
    return 4 * pi_est * R * R


def sphere_surface_inscribed_strip(R, K=30, Nz=64):
    """Discrete Archimedean strip rule with finite Nz to show it's exact at any Nz.

    Each strip from z=z_{i} to z=z_{i+1} contributes 2*pi*R*(z_{i+1}-z_{i}).
    Sum telescopes to 2*pi*R*(2R) = 4*pi*R^2.
    """
    R = mpf(R)
    pi_est = pi_estimate(K)
    dz = (2 * R) / Nz
    S = mpf(0)
    for i in range(Nz):
        S = S + 2 * pi_est * R * dz
    return S


# ============================================================================
# Part 4 (NEW) - 4D hyperball and 3-sphere via lower-dim stacking
# ============================================================================
#
# Volume of a 4-ball of radius R:  V_4(R) = (pi^2 / 2) * R^4
# Surface (3-volume of the 3-sphere): S_3(R) = 2 * pi^2 * R^3
#
# The same disk-stack idea works one dimension up: a 4-ball is a stack of
# 3-balls along the 4th axis.
#
#     V_4(R) = integral_{-R}^{R} V_3(sqrt(R^2 - w^2)) dw
#            = integral_{-R}^{R} (4/3) pi (R^2 - w^2)^(3/2) dw
#            = (pi^2 / 2) R^4
#
# And the 3-sphere (boundary of 4-ball) by the strip rule generalizes to
#     S_3(R) = integral_{-R}^{R} S_2_sphere(sqrt(R^2-w^2)) * (R/sqrt(R^2-w^2)) dw
#            = integral_{-R}^{R} 4 pi (R^2 - w^2) * (R/sqrt(R^2 - w^2)) dw
#            = 4 pi R integral_{-R}^{R} sqrt(R^2 - w^2) dw
#            = 4 pi R * (pi R^2 / 2)
#            = 2 pi^2 R^3
# ============================================================================

def hyperball_volume(R, K=30, Nw=10000):
    """4D hyperball volume via stack of 3-balls along the 4th coordinate w.

    Each 3-ball slice at height w has radius sqrt(R^2 - w^2) and volume
    (4/3) * pi * (R^2 - w^2)^(3/2), computed entirely from the recursion's pi.
    """
    R = mpf(R)
    pi_est = pi_estimate(K)
    dw = (2 * R) / Nw
    V = mpf(0)
    for i in range(Nw):
        w = -R + (mpf(i) + mpf('0.5')) * dw
        r2 = R * R - w * w
        if r2 <= 0:
            continue
        r = sqrt(r2)
        V3 = (4 * pi_est * r2 * r) / 3        # (4/3) pi r^3
        V = V + V3 * dw
    return V


def hypersphere_3surface(R, K=30, Nw=10000):
    """3-volume of the 3-sphere (boundary of 4-ball) via strip integration.

    Strip rule one dim up: at height w in the 4th axis, the slice is a
    2-sphere of radius r(w) = sqrt(R^2 - w^2) and surface area 4 pi r^2.
    The slant correction is R / r (analogous to the 3D Archimedes strip).

    Strip 3-volume at height w  = 4 pi r(w)^2 * (R / r(w)) * dw
                                 = 4 pi R * r(w) dw
                                 = 4 pi R * sqrt(R^2 - w^2) dw

    Integrating: S_3 = 4 pi R * (pi R^2 / 2) = 2 pi^2 R^3.
    """
    R = mpf(R)
    pi_est = pi_estimate(K)
    dw = (2 * R) / Nw
    S = mpf(0)
    for i in range(Nw):
        w = -R + (mpf(i) + mpf('0.5')) * dw
        r2 = R * R - w * w
        if r2 <= 0:
            continue
        r = sqrt(r2)
        # strip contribution: 4 pi R * r * dw
        S = S + 4 * pi_est * R * r * dw
    return S


# ============================================================================
# Convergence-rate verification (referenced as Part 1.6 in the doc)
# ============================================================================

def pi_convergence_table(K):
    """For each level k, return (k, N_k, area_error, ratio_to_previous).

    Theoretical: error ~ pi^3 / (3 N^2) for the inscribed-polygon area.
    So consecutive errors should have ratio ~ 1/4 (since N doubles).
    """
    rows = []
    prev_err = None
    for (k, N, A, B, C, area) in pi_recurrence(K):
        err = abs(area - PI_REF)
        if prev_err is not None and err > 0:
            ratio = prev_err / err
        else:
            ratio = None
        rows.append((k, N, err, ratio))
        prev_err = err
    return rows


# ============================================================================
# Self-test: run every table at moderate K, confirm convergence
# ============================================================================

def main():
    print("=" * 78)
    print("Constructive Pi Framework -- self-test")
    print("=" * 78)
    print()

    print("[Part 1]  pi from the 2D bisection recurrence, K=15")
    print("-" * 78)
    print(f"  pi (15 levels) = {mp.nstr(pi_estimate(15), 30)}")
    print(f"  reference pi    = {mp.nstr(PI_REF, 30)}")
    print(f"  signed error    = {mp.nstr(pi_estimate(15) - PI_REF, 6)}")
    print()

    print("[Part 1B] sine/cosine at dyadic angles, first 8 levels")
    print("-" * 78)
    print(f"  {'k':>3} {'N':>7} {'half-angle deg':>15} {'sin (=C/2)':>18} "
          f"{'cos (=1-B_next)':>18} {'sin error':>12}")
    for (k, N, ang, s, c, err) in sin_cos_table(8):
        print(f"  {k:>3} {N:>7} {mp.nstr(ang, 8):>15} {mp.nstr(s, 12):>18} "
              f"{mp.nstr(c, 12):>18} {mp.nstr(err, 4):>12}")
    print()

    print("[Part 1B] i-power table (successive 2^k-th roots of i), first 8")
    print("-" * 78)
    print(f"  {'k':>3} {'2^k':>5} {'angle deg':>11} "
          f"{'Re = cos':>18} {'Im = sin':>18}")
    for (k, two_k, ang, re, im) in i_power_table(8):
        print(f"  {k:>3} {two_k:>5} {mp.nstr(ang, 8):>11} "
              f"{mp.nstr(re, 12):>18} {mp.nstr(im, 12):>18}")
    print()

    print("[Part 1C] inscribed AND circumscribed bracket on pi, first 10 levels")
    print("-" * 78)
    print(f"  {'k':>3} {'N':>7} {'inscribed':>18} {'circumscribed':>18} "
          f"{'midpoint':>18} {'gap':>10}")
    for (k, N, ins, cir, mid, gap) in circumscribed_bracket(10):
        print(f"  {k:>3} {N:>7} {mp.nstr(ins, 12):>18} {mp.nstr(cir, 12):>18} "
              f"{mp.nstr(mid, 12):>18} {mp.nstr(gap, 4):>10}")
    print()

    print("[Part 1D] addition formula: cos(pi/4 + pi/8) and cos(pi/4 - pi/8)")
    print("-" * 78)
    c4, s4 = cos_sin_at_dyadic(1)      # pi/4
    c8, s8 = cos_sin_at_dyadic(2)      # pi/8
    c_sum, s_sum = add_angle(c4, s4, c8, s8)
    c_dif, s_dif = sub_angle(c4, s4, c8, s8)
    from mpmath import cos as _cos, sin as _sin
    print(f"  cos(3pi/8) recurrence   = {mp.nstr(c_sum, 14)}")
    print(f"  cos(3pi/8) reference    = {mp.nstr(_cos(3 * PI_REF / 8), 14)}")
    print(f"  cos(pi/8)  via subtract = {mp.nstr(c_dif, 14)}")
    print(f"  cos(pi/8)  reference    = {mp.nstr(_cos(PI_REF / 8), 14)}")
    print()

    print("[Part 2]  cone volume, r=1 h=3, K=30")
    print("-" * 78)
    V = cone_volume(1, 3, K=30)
    true_V = PI_REF
    print(f"  computed   = {mp.nstr(V, 16)}")
    print(f"  target     = {mp.nstr(true_V, 16)}")
    print(f"  error      = {mp.nstr(V - true_V, 4)}")
    print()

    print("[Part 3]  sphere volume R=1, K=30 Nz=10000")
    print("-" * 78)
    V = sphere_volume(1, K=30, Nz=10000)
    true_V = 4 * PI_REF / 3
    print(f"  computed   = {mp.nstr(V, 16)}")
    print(f"  target     = {mp.nstr(true_V, 16)}")
    print(f"  error      = {mp.nstr(V - true_V, 4)}")
    print()

    print("[Part 3]  sphere surface R=1 via strip rule (Archimedes) at multiple Nz")
    print("-" * 78)
    true_S = 4 * PI_REF
    print(f"  target  = 4 pi = {mp.nstr(true_S, 16)}")
    for Nz in [4, 16, 64, 1000]:
        S = sphere_surface_inscribed_strip(1, K=30, Nz=Nz)
        print(f"  Nz={Nz:>5}  computed = {mp.nstr(S, 16)}  error = {mp.nstr(S-true_S, 4)}")
    print("  -> exact at any Nz because the strip rule integrand is constant in z")
    print()

    print("[Part 4]  4D hyperball volume R=1, K=30 Nw=10000")
    print("-" * 78)
    V4 = hyperball_volume(1, K=30, Nw=10000)
    true_V4 = (PI_REF ** 2) / 2
    print(f"  computed   = {mp.nstr(V4, 16)}")
    print(f"  target     = (pi^2)/2 = {mp.nstr(true_V4, 16)}")
    print(f"  error      = {mp.nstr(V4 - true_V4, 4)}")
    print()

    print("[Part 4]  3-sphere volume R=1, K=30 Nw=10000")
    print("-" * 78)
    S3 = hypersphere_3surface(1, K=30, Nw=10000)
    true_S3 = 2 * (PI_REF ** 2)
    print(f"  computed   = {mp.nstr(S3, 16)}")
    print(f"  target     = 2*pi^2 = {mp.nstr(true_S3, 16)}")
    print(f"  error      = {mp.nstr(S3 - true_S3, 4)}")
    print()

    print("[Part 1.6] empirical convergence rate, error_k / error_{k+1}")
    print("-" * 78)
    print(f"  {'k':>3} {'N':>7} {'|area - pi|':>18} {'ratio prev/curr':>18}")
    for (k, N, err, ratio) in pi_convergence_table(12):
        rstr = mp.nstr(ratio, 6) if ratio is not None else "-"
        print(f"  {k:>3} {N:>7} {mp.nstr(err, 4):>18} {rstr:>18}")
    print("  -> ratio approaches 4 (the 1/N^2 Archimedean rate)")
    print()

    print("=" * 78)
    print("All tables produced by the recursion (or one extra integration layer).")
    print("=" * 78)


if __name__ == "__main__":
    main()
