#!/usr/bin/env python3
"""
sphere_from_circle.py
=====================
Volume and surface of a sphere from your right-triangle recurrence.

CONSTRUCTION (matching the notebook's pole-and-faces hierarchy)
---------------------------------------------------------------
A sphere of radius R can be built as a stack of disks along its height axis:

    at height z  in [-R, R]:   disk of radius r(z) = sqrt(R^2 - z^2)
    disk area at z             = pi * r(z)^2  =  pi * (R^2 - z^2)

The recurrence supplies pi.  Stacking the disks along z gives:

    Volume   = sum_z [pi * (R^2 - z^2) * dz]   ->   (4/3) pi R^3
    Surface  = sum_z [2 pi R * dz]             ->   4 pi R^2

(For surface, the slant correction gives a factor R/r(z), and 2 pi r(z) * (R/r(z))
 = 2 pi R, which doesn't depend on z. So surface integrates trivially.)

Two "iteration loops" total:
    Outer loop: K levels of the right-triangle recurrence to compute pi
    Inner loop: N_z disk slabs along the height axis to integrate volume

Each step uses ONLY {+, -, *, /, sqrt}.  No pi imported.  No trig.
"""
from mpmath import mp, mpf, sqrt, pi as PI_REF
mp.prec = 200

def compute_pi_via_recurrence(K_levels):
    """Run the right-triangle recurrence to get pi for a unit circle."""
    A = mpf(1); B = mpf(1); C = sqrt(mpf(2)); N = 4
    S = mpf(0)
    for _ in range(K_levels):
        S += N * A * B / 2
        A2 = (C * C) / 4
        B_new = A2 / (1 + sqrt(1 - A2))
        C = sqrt(A2 + B_new * B_new)
        A = sqrt(A2); B = B_new; N *= 2
    return S  # for r=1, S -> pi

def sphere_volume(R, K_levels=40, N_z=10000):
    """Sphere volume = stack of disks along z axis. Each disk's area uses pi from the recurrence."""
    R = mpf(R)
    pi_est = compute_pi_via_recurrence(K_levels)
    dz = (2 * R) / N_z
    V = mpf(0)
    for i in range(N_z):
        # Midpoint rule: z at the center of i-th slab
        z = -R + (i + mpf('0.5')) * dz
        r_sq = R * R - z * z  # radius squared at height z
        V = V + pi_est * r_sq * dz
    return V, pi_est

def sphere_surface(R, K_levels=40, N_z=10000):
    """Sphere surface = sum of cylindrical strips of width 2 pi R * dz."""
    R = mpf(R)
    pi_est = compute_pi_via_recurrence(K_levels)
    dz = (2 * R) / N_z
    A_surface = 2 * pi_est * R * (2 * R)  # closed-form integral, no z dependence
    return A_surface, pi_est

if __name__ == "__main__":
    print("=" * 90)
    print("SPHERE FROM CIRCLES - same recurrence, summed along the height axis")
    print("=" * 90)
    print()

    test_radii = [(1, '1.0'), (2, '2.0'), (3, '3.0'), (5, '5.0')]

    for R, R_label in test_radii:
        true_V = (4 * PI_REF * mpf(R)**3) / 3
        true_S = 4 * PI_REF * mpf(R)**2

        # Volume convergence with N_z (height discretization)
        print(f">>> Sphere with R = {R_label}")
        print(f"    True volume   (4/3) pi R^3 = {mp.nstr(true_V, 14)}")
        print(f"    True surface  4 pi R^2     = {mp.nstr(true_S, 14)}")
        print()
        print(f"    {'K_levels':>9} | {'N_z':>8} | {'computed V':>22} | {'V gap':>14} | {'computed S':>22} | {'S gap':>10}")
        print('    ' + '-' * 100)
        for N_z in [10, 100, 1000, 10000]:
            V, _ = sphere_volume(R, K_levels=30, N_z=N_z)
            S, _ = sphere_surface(R, K_levels=30, N_z=N_z)
            v_gap = abs(V - true_V)
            s_gap = abs(S - true_S)
            print(f'    {30:>9} | {N_z:>8} | {mp.nstr(V, 18):>22} | {mp.nstr(v_gap, 4):>14} | {mp.nstr(S, 18):>22} | {mp.nstr(s_gap, 4):>10}')
        print()

    print("=" * 90)
    print("Observe: as N_z grows, the volume converges quadratically (midpoint rule).")
    print("Surface is exact at any N_z because the integrand 2 pi R is constant in z.")
    print("Both come from the SAME pi (computed once via the recurrence) and {+,-,*,/,sqrt}.")
