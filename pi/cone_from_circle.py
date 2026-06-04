#!/usr/bin/env python3
"""
cone_from_circle.py
===================
Volume and lateral surface of a cone from your right-triangle recurrence.

NOTEBOOK CONSTRUCTION
---------------------
Each level k of the circle recurrence produces a base polygon (the inscribed
4*2^k-gon).  Lift that polygon to 3-D by adding a height H, with apex directly
above the base center.  This gives an "inscribed pyramid" approximating the cone.

Two things follow at each level, both from {+, -, *, /, sqrt}:

VOLUME:
    V_k = (1/3) * (base area) * H   = (1/3) * S_k * H
    where S_k = sum_{j=0..k} N_j * (1/2) * A_j * B_j  (the disk-area recurrence)
    Limit:   V_inf = (1/3) * pi * r^2 * H  -- volume of cone

LATERAL SURFACE:
    Each lateral face is an isoceles triangle:
        base   = C_k (chord at level k)
        slant  = L   = sqrt(r^2 + H^2)
        face altitude from apex = sqrt(L^2 - C_k^2/4)
        face area = (1/2) * C_k * sqrt(L^2 - C_k^2/4)
    Total lateral area at level k = N_k * face_area
    Limit:   A_inf = pi * r * L  -- cone lateral surface

Both limits emerge from your recurrence + the (H, r, L) Pythagorean triangle.
No pi imported. No trig. Five operations only.
"""
from mpmath import mp, mpf, sqrt, pi
mp.prec = 200

PI_REF = pi  # only for printing the gap; never used in the recurrence

def cone_from_recurrence(r, H, max_level=20, verbose=True):
    """Compute cone volume and lateral surface via the right-triangle recurrence."""
    r = mpf(r); H = mpf(H)
    L = sqrt(r * r + H * H)  # slant length (one Pythagorean step)

    # Initialize the base recurrence in 'r' units
    A = r; B = r; C = r * sqrt(mpf(2)); N = 4
    S_area = mpf(0)  # cumulative inscribed-polygon area

    if verbose:
        print(f"r = {mp.nstr(r, 12)},  H = {mp.nstr(H, 12)},  L = sqrt(r^2 + H^2) = {mp.nstr(L, 12)}")
        print()
        print(f"{'k':>3} | {'N_k':>6} | {'A_k':>14} | {'B_k':>14} | {'C_k':>14} | "
              f"{'V_k = S_k*H/3':>16} | {'A_lat_k':>16}")
        print('-' * 110)

    true_V = (PI_REF * r * r * H) / 3
    true_A = PI_REF * r * L

    for k in range(max_level + 1):
        # Add this level's contribution to base area
        S_area += N * A * B / 2

        # Volume of inscribed pyramid
        V_k = S_area * H / 3

        # Lateral surface: N_k isoceles faces, each with base C_k
        # face altitude = sqrt(L^2 - (C_k/2)^2)
        face_alt = sqrt(L * L - (C * C) / 4)
        face_area = (C * face_alt) / 2
        A_lat_k = N * face_area

        if verbose and k in [0, 1, 2, 3, 4, 5, 8, 12, 16, 20]:
            print(f"{k:>3} | {N:>6} | {mp.nstr(A,12):>14} | {mp.nstr(B,12):>14} | "
                  f"{mp.nstr(C,12):>14} | {mp.nstr(V_k,14):>16} | {mp.nstr(A_lat_k,14):>16}")

        # Advance the recurrence (cancellation-free sagitta for general r)
        # B = r - sqrt(r^2 - A^2)  =  A^2 / (r + sqrt(r^2 - A^2))
        A2 = (C * C) / 4
        B_new = A2 / (r + sqrt(r * r - A2))
        C_new = sqrt(A2 + B_new * B_new)
        A = sqrt(A2); B = B_new; C = C_new; N *= 2

    if verbose:
        print()
        print(f"  TRUE Volume        = (1/3) pi r^2 H = {mp.nstr(true_V, 16)}")
        print(f"  computed V_{max_level:<2}        = {mp.nstr(V_k, 16)}")
        print(f"  gap                = {mp.nstr(abs(V_k - true_V), 4)}")
        print()
        print(f"  TRUE Lateral Area  = pi r L         = {mp.nstr(true_A, 16)}")
        print(f"  computed A_lat_{max_level:<2}    = {mp.nstr(A_lat_k, 16)}")
        print(f"  gap                = {mp.nstr(abs(A_lat_k - true_A), 4)}")

    return V_k, A_lat_k


if __name__ == "__main__":
    print("=" * 110)
    print("CONE FROM CIRCLE — same recurrence, lifted by H, slant L = sqrt(r^2 + H^2)")
    print("=" * 110)
    print()

    print(">>> Cone with r = 1, H = 1 (so L = sqrt(2)):")
    print()
    V, A = cone_from_recurrence(r=1, H=1, max_level=20)

    print()
    print("=" * 110)
    print()
    print(">>> Cone with r = 3, H = 4 (a 3-4-5 right triangle, so L = 5):")
    print()
    V, A = cone_from_recurrence(r=3, H=4, max_level=20)

    print()
    print("=" * 110)
    print()
    print(">>> Tall thin cone, r = 1, H = 10 (L = sqrt(101)):")
    print()
    V, A = cone_from_recurrence(r=1, H=10, max_level=20)
