#!/usr/bin/env python3
"""
sphere_inscribed.py
===================
Inscribed valley-fill sphere construction.

Start: regular octahedron inscribed in unit sphere (8 equilateral faces).
Each iteration: split every triangular face into 4 by inserting midpoints
on the three edges and PROJECTING the midpoints onto the sphere. The
polyhedron stays inscribed at every step.

This is the geometric construction the original v1 framework asked for.
The pyramid-out construction (apex pushed OUTSIDE the sphere) made
surface diverge. The inscribed valley-fill (midpoint projected ONTO the
sphere) makes both volume AND surface converge.

Right-triangle bookkeeping:
  Initial octahedron faces are equilateral (60-60-60), so each face
  decomposes into 6 right (30-60-90) triangles via its medians, giving
  6 right tetrahedra per face from the center -> 48 right tetrahedra total.

  After iteration 0, 1 face -> 4 sub-faces:
    - 1 central sub-face (between the 3 projected midpoints): equilateral
    - 3 corner sub-faces: isoceles, approaching equilateral as the sphere
      flattens at fine scales.

  Face count rule:  N_k = 8 * 4^k
  Edge count rule:  E_k = 12 * 4^k
  Vertex count rule (Euler):  V_k = 2 + 6 * 4^k   (for our subdivision)

Volume of polyhedron at iteration k:
  V_k = sum over faces of signed_tetra_volume(O, a, b, c) = (1/6) a . (b x c)

Surface area at iteration k:
  S_k = sum over faces of (1/2) |(b-a) x (c-a)|

Targets for unit sphere:
  V -> (4/3) pi = 4.188790204786391
  S -> 4 pi     = 12.566370614359172
"""

from mpmath import mp, mpf, sqrt, pi as PI_REF
mp.prec = 200


# ---- vector primitives ---------------------------------------------------

def vadd(u, v):
    return (u[0] + v[0], u[1] + v[1], u[2] + v[2])

def vsub(u, v):
    return (u[0] - v[0], u[1] - v[1], u[2] - v[2])

def vscale(u, s):
    return (u[0] * s, u[1] * s, u[2] * s)

def cross(u, v):
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )

def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

def vnorm(u):
    return sqrt(dot(u, u))

def proj_sphere(u):
    """Project a point onto the unit sphere (normalize)."""
    L = vnorm(u)
    return (u[0] / L, u[1] / L, u[2] / L)

def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2)


# ---- geometric helpers ---------------------------------------------------

def face_area(a, b, c):
    """Area of triangle (a, b, c) = (1/2) |cross(b-a, c-a)|."""
    n = cross(vsub(b, a), vsub(c, a))
    return vnorm(n) / 2

def tetra_signed_volume(a, b, c):
    """Signed volume of tetrahedron O=(0,0,0), a, b, c = (1/6) a . (b x c).

    For an outward-oriented face (a, b, c) on a sphere centered at O, this
    is positive and equals the volume of the slice from the center to the face.
    """
    return dot(a, cross(b, c)) / 6


# ---- main subdivision routine -------------------------------------------

def octahedron_faces():
    """The 8 oriented faces of the unit octahedron (outward CCW)."""
    v = [
        (mpf(1),  mpf(0),  mpf(0)),
        (mpf(-1), mpf(0),  mpf(0)),
        (mpf(0),  mpf(1),  mpf(0)),
        (mpf(0),  mpf(-1), mpf(0)),
        (mpf(0),  mpf(0),  mpf(1)),
        (mpf(0),  mpf(0),  mpf(-1)),
    ]
    # 8 octants - choose vertex ordering so face normal points outward
    # +++:  +x, +y, +z  -> vertices in CCW order from outside the +++ corner
    # Octant signs determine which vertex from each axis is used
    return [
        (v[0], v[2], v[4]),   # +++
        (v[2], v[1], v[4]),   # -++
        (v[1], v[3], v[4]),   # --+
        (v[3], v[0], v[4]),   # +-+
        (v[2], v[0], v[5]),   # ++-
        (v[1], v[2], v[5]),   # -+-
        (v[3], v[1], v[5]),   # ---
        (v[0], v[3], v[5]),   # +--
    ]


def subdivide_face(face):
    """Replace one triangle (a, b, c) with 4 sub-triangles (inscribed midpoint
    subdivision: project midpoints onto unit sphere). Returns list of 4 faces."""
    a, b, c = face
    m_ab = proj_sphere(midpoint(a, b))
    m_bc = proj_sphere(midpoint(b, c))
    m_ca = proj_sphere(midpoint(c, a))
    return [
        (a,    m_ab, m_ca),    # corner at A
        (b,    m_bc, m_ab),    # corner at B
        (c,    m_ca, m_bc),    # corner at C
        (m_ab, m_bc, m_ca),    # central (equilateral)
    ]


def inscribed_sphere_sequence(K_iters):
    """Return rows (iter, n_faces, surface, volume, S_err, V_err) for each level."""
    faces = octahedron_faces()
    target_S = 4 * PI_REF
    target_V = 4 * PI_REF / 3
    rows = []
    for it in range(K_iters + 1):
        S = sum(face_area(a, b, c) for (a, b, c) in faces)
        V = sum(tetra_signed_volume(a, b, c) for (a, b, c) in faces)
        rows.append((it, len(faces), S, V, S - target_S, V - target_V))
        # Subdivide every face
        new_faces = []
        for f in faces:
            new_faces.extend(subdivide_face(f))
        faces = new_faces
    return rows


def right_tetra_count(K_iters):
    """The bookkeeping table the user described:
       face count = 8 * 4^k
       right-tetra count (medians decomposition) = 6 * face count = 48 * 4^k
    """
    rows = []
    for k in range(K_iters + 1):
        n_faces = 8 * (4 ** k)
        n_right_tetra = 6 * n_faces
        rows.append((k, n_faces, n_right_tetra))
    return rows


# ---- self-test -----------------------------------------------------------

if __name__ == "__main__":
    print("=" * 76)
    print("Inscribed valley-fill sphere construction")
    print("=" * 76)
    print()
    print(f"  {'k':>3} {'faces':>8} {'right-tetra':>12} "
          f"{'surface':>16} {'volume':>16} "
          f"{'|S-4pi|':>10} {'|V-(4/3)pi|':>12} "
          f"{'S ratio':>9} {'V ratio':>9}")
    rows = inscribed_sphere_sequence(8)
    target_S = 4 * PI_REF
    target_V = 4 * PI_REF / 3
    print(f"  {'-'*3} {'-'*8} {'-'*12} {'-'*16} {'-'*16} "
          f"{'-'*10} {'-'*12} {'-'*9} {'-'*9}")
    print(f"  {'':>3} {'':>8} {'':>12} {'target ->':>16} "
          f"{mp.nstr(target_V, 12):>16} {mp.nstr(target_S, 12):>10} "
          f"{mp.nstr(target_V, 12):>12} {'':>9} {'':>9}")
    print(f"  {'':>3} {'':>8} {'':>12} {mp.nstr(target_S, 12):>16} "
          f"{mp.nstr(target_V, 12):>16}")
    print()
    prev_S_err = prev_V_err = None
    for (it, n_faces, S, V, S_err, V_err) in rows:
        n_right_tetra = n_faces * 6
        if prev_S_err is not None and abs(S_err) > 0 and abs(V_err) > 0:
            sr = mp.nstr(abs(prev_S_err) / abs(S_err), 5)
            vr = mp.nstr(abs(prev_V_err) / abs(V_err), 5)
        else:
            sr = '-'
            vr = '-'
        print(f"  {it:>3} {n_faces:>8} {n_right_tetra:>12} "
              f"{mp.nstr(S, 12):>16} {mp.nstr(V, 12):>16} "
              f"{mp.nstr(abs(S_err), 4):>10} {mp.nstr(abs(V_err), 4):>12} "
              f"{sr:>9} {vr:>9}")
        prev_S_err = S_err
        prev_V_err = V_err
    print()
    print("Both errors quarter each iteration (ratio -> 4), matching the")
    print("Archimedean 1/N^2 rate from the 2D bisection.")
