"""
Rotation rules for the 32 sub-quadrants of the AETHOS 3D complex plane.

Structure (proven on VA1 canon at any (A, n))
------------------------------------------------
Fix branch b and canonical (X, Y, zeta) from canon_on_chain(b, A, n).
Write z = X + iY.

**8 wings = Klein-4 on C  x  Zeta reflection**

Spring operators on z (order-2 generators):

    R_x(z) = -Re(z) + i Im(z)     flip_x on VA corridor  (= -conj(z) when Y real)
    S(z)   = +Im(z) + i Re(z)     VB axis swap            (= i * conj(z) for z = X+iY)

Klein table (z0 = X + iY):

    id       z0
    R_x      R_x(z0)
    S        S(z0)
    R_x∘S    R_x(S(z0))            (= i*z on readout; see aethos_spring_complex.i_act)

Depth operator (order 2, commutes with above on z):

    J      (z, zeta) -> (z, -zeta)     flip_z on embedding

Wing index w in 1..8 encodes (family, flip_x, flip_z):

    w = 1 + 2*flip_z + 4*flip_x   for VA (family=0)
    w = 5 + 2*flip_z + 4*flip_x   for VB (family=1 applies S then flips)

Equivalently bit mask on w-1 (0..7):

    bit0 = zeta hemisphere (flip_z)
    bit1 = Re reflect (flip_x) after corridor select
    bit2 = corridor: 0=VA direct, 1=VB swap (YXZ)

**4 branches = formula fan, not uniform Arg(z) step**

VA1..VA4 recompute (X,Y,zeta) from segment FSM — not a fixed e^{ik pi/2} rotation
of the same z.  Branch index b in 0..3 steps the four spring algebras.

**32 sub-quadrants**

    lid = 1 + b*8 + (w-1)     b in 0..3, w in 1..8

Single rotation sweep (visit all 32):

    for b in 0..3:
        for mask in 0..7:
            w = 1 + mask   with VA/VB correction per VECTORS table
            emit wing_transform(b+1, w)

Incremental rotation step on flat index k in 0..31:

    k -> (k+1) mod 32   walks branch-inner, wing-outer (branch-major)
    or k -> (k+1) mod 32 with wing-major: w first, then branch

Group presentation (wing level, fixed branch):

    R_x^2 = S^2 = J^2 = 1
    Wing spring set = {z0, R_x z0, S z0, R_x S z0}   (Klein closure; R_x and S do not commute)
    Psi chamber   = spring choice x {zeta, -zeta}        (8 wings)

Branch fan B (VA1..VA4): order-4 re-canonicalize; not a fixed Arg(z) step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence

from aethos_complex_plane import ComplexPlane3D, wing_transform
from aethos_lattice import BranchKind, LatticeId, VECTORS, apply_vector, lattice_id_parts
from aethos_sequences import canon_on_chain


class WingMask(IntEnum):
    """Bit flags for wing 1..8 (mask = w - 1)."""

    ZETA_FLIP = 1   # bit0 -> flip_z
    RE_FLIP = 2     # bit1 -> flip_x
    VB_CORRIDOR = 4  # bit2 -> YXZ swap family


@dataclass(frozen=True)
class SubQuadrant:
    """One of 32 chambers: branch x wing at fixed (A, n)."""

    index: int  # 0..31
    branch: BranchKind
    wing: int
    lattice_id: LatticeId
    psi: ComplexPlane3D
    mask: int

    @property
    def spring_angle_deg(self) -> float:
        return math.degrees(math.atan2(self.psi.z.imag, self.psi.z.real))


# ---------------------------------------------------------------------------
# Generators on C and C x R
# ---------------------------------------------------------------------------

def reflect_real(z: complex) -> complex:
    """R_x: reflect across Im axis."""
    return complex(-z.real, z.imag)


def swap_corridor(z: complex) -> complex:
    """S: VA spring z -> VB lead (Re, Im swap).  S(z) = i * conj(z) for z = X+iY."""
    return complex(z.imag, z.real)


def zeta_flip(psi: ComplexPlane3D) -> ComplexPlane3D:
    """J: depth hemisphere."""
    return ComplexPlane3D(z=psi.z, zeta=-psi.zeta)


def apply_wing_mask_to_canon(
    canon: tuple[float, float, float],
    mask: int,
) -> ComplexPlane3D:
    """Apply wing bit mask to canonical (X,Y,zeta) — matches VECTORS table."""
    family_vb = bool(mask & int(WingMask.VB_CORRIDOR))
    flip_x = bool(mask & int(WingMask.RE_FLIP))
    flip_z = bool(mask & int(WingMask.ZETA_FLIP))
    x, y, zeta = canon
    if family_vb:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_z:
        zeta = -zeta
    return ComplexPlane3D(z=complex(x, y), zeta=float(zeta))


def wing_mask_from_index(wing: int) -> int:
    """wing 1..8 -> mask 0..7 matching aethos_lattice.VECTORS."""
    if not 1 <= wing <= 8:
        raise ValueError(f"wing must be 1..8, got {wing}")
    v = VECTORS[wing - 1]
    mask = 0
    if v.flip_z:
        mask |= int(WingMask.ZETA_FLIP)
    if v.flip_x:
        mask |= int(WingMask.RE_FLIP)
    if v.family == "VB":
        mask |= int(WingMask.VB_CORRIDOR)
    return mask


def wing_index_from_mask(mask: int) -> int:
    """Inverse: mask 0..7 -> wing 1..8."""
    for w in range(1, 9):
        if wing_mask_from_index(w) == mask:
            return w
    raise ValueError(f"invalid mask {mask}")


def klein_four_on_z(z0: complex) -> tuple[complex, complex, complex, complex]:
    """Four spring phases {id, R_x, S, R_x∘S} on fixed branch canon."""
    rx = reflect_real(z0)
    s = swap_corridor(z0)
    rxs = reflect_real(s)
    return (z0, rx, s, rxs)


def verify_klein_identity(z0: complex) -> bool:
    """S^2 = R_x^2 = id; wing orbit closes as {z0, R_x z0, S z0, R_x S z0}."""
    rx = reflect_real
    s = swap_corridor
    if s(s(z0)) != z0:
        return False
    if rx(rx(z0)) != z0:
        return False
    # R_x and S do not commute; wing table uses fixed order S then R_x (VB corridor).
    orbit = {z0, rx(z0), s(z0), rx(s(z0))}
    if len(orbit) != 4:
        return False
    if s(z0) != 1j * z0.conjugate():
        return False
    if rx(z0) != -z0.conjugate():
        return False
    return True


def sub_quadrant_index(branch: BranchKind, wing: int) -> int:
    """Flat index 0..31 (branch-major)."""
    return (int(branch) - 1) * 8 + (wing - 1)


def index_to_branch_wing(k: int) -> tuple[BranchKind, int]:
    k = k % 32
    branch = BranchKind((k // 8) + 1)
    wing = (k % 8) + 1
    return branch, wing


def all_sub_quadrants(
    chain: Sequence[int | float],
    n: float,
    *,
    branch_major: bool = True,
) -> tuple[SubQuadrant, ...]:
    """Enumerate all 32 chambers at (A, n)."""
    out: list[SubQuadrant] = []
    for k in range(32):
        branch, wing = index_to_branch_wing(k)
        lid = LatticeId(k + 1)
        psi = wing_transform(branch, chain, n, wing)
        mask = wing_mask_from_index(wing)
        out.append(
            SubQuadrant(
                index=k,
                branch=branch,
                wing=wing,
                lattice_id=lid,
                psi=psi,
                mask=mask,
            )
        )
    return tuple(out)


def rotate_step(k: int, delta: int = 1) -> int:
    """Increment flat rotation index mod 32."""
    return (k + delta) % 32


def rotate_cycle(
    chain: Sequence[int | float],
    n: float,
    *,
    start: int = 0,
) -> Iterator[tuple[int, SubQuadrant]]:
    """Walk all 32 sub-quadrants once starting at index start."""
    for i in range(32):
        k = rotate_step(start, i)
        branch, wing = index_to_branch_wing(k)
        psi = wing_transform(branch, chain, n, wing)
        sq = SubQuadrant(
            index=k,
            branch=branch,
            wing=wing,
            lattice_id=LatticeId(k + 1),
            psi=psi,
            mask=wing_mask_from_index(wing),
        )
        yield k, sq


def branch_fan_z(
    chain: Sequence[int | float],
    n: float,
    wing: int = 1,
) -> dict[BranchKind, complex]:
    """Four branch spring values at fixed wing (the 'radar sweep' fan)."""
    return {
        b: wing_transform(b, chain, n, wing).z
        for b in BranchKind
    }


def wing_orbit_z(
    chain: Sequence[int | float],
    n: float,
    branch: BranchKind = BranchKind.VA1,
) -> dict[int, complex]:
    """Eight wing spring values at fixed branch (Klein x zeta)."""
    return {w: wing_transform(branch, chain, n, w).z for w in range(1, 9)}


def apply_generator(
    psi: ComplexPlane3D,
    gen: str,
    *,
    z0_ref: complex | None = None,
) -> ComplexPlane3D:
    """
    Apply named generator: 'Rx', 'S', 'J' on Psi.
    Rx/S act on z; J on zeta.  (Branch fan B requires full re-canonicalize.)
    """
    if gen == "Rx":
        return ComplexPlane3D(z=reflect_real(psi.z), zeta=psi.zeta)
    if gen == "S":
        return ComplexPlane3D(z=swap_corridor(psi.z), zeta=psi.zeta)
    if gen == "J":
        return zeta_flip(psi)
    raise ValueError(f"unknown generator {gen!r}")


def match_lattice_apply(
    branch: BranchKind,
    chain: Sequence[int | float],
    n: float,
    mask: int,
) -> bool:
    """apply_wing_mask_to_canon matches wing_transform for given mask."""
    canon = canon_on_chain(branch, chain, n)
    wing = wing_index_from_mask(mask)
    from_a = apply_wing_mask_to_canon(canon, mask)
    from_b = wing_transform(branch, chain, n, wing)
    return from_a.coord == from_b.coord


def demo() -> None:
    chain = (3, 5, 7)
    n = 5
    z0 = wing_transform(BranchKind.VA1, chain, n, 1).z

    print("=" * 72)
    print("32 SUB-QUADRANT ROTATION RULES")
    print("=" * 72)

    print(f"\nVA1 spring z0 = {z0.real:.0f}{z0.imag:+.0f}i")
    print(f"Klein identites OK: {verify_klein_identity(z0)}")

    print("\n--- Klein-4 on z (fixed branch VA1) ---")
    for i, z in enumerate(klein_four_on_z(z0)):
        print(f"  q{i}: {z.real:+.0f}{z.imag:+.0f}i  arg={math.degrees(math.atan2(z.imag,z.real)):.1f} deg")

    print("\n--- 8 wings = Klein-4 x {zeta,+/-} ---")
    for w, z in wing_orbit_z(chain, n, BranchKind.VA1).items():
        psi = wing_transform(BranchKind.VA1, chain, n, w)
        print(f"  v{w}: z={z.real:+.0f}{z.imag:+.0f}i  zeta={psi.zeta:+.0f}")

    print("\n--- 4 branch fan at v1 (formula rotation) ---")
    for b, z in branch_fan_z(chain, n, 1).items():
        print(f"  {b.name}: z={z.real:+.0f}{z.imag:+.0f}i")

    print("\n--- Rotation cycle k=0..7 (VA1, all wings) ---")
    for k, sq in rotate_cycle(chain, n, start=0):
        if k >= 8:
            break
        print(f"  k={k} L{int(sq.lattice_id):02d} mask={sq.mask:03b} z={sq.psi.z.real:+.0f}{sq.psi.z.imag:+.0f}i zeta={sq.psi.zeta:+.0f}")

    print("\n--- All 32 distinct spring z values? ---")
    sqs = all_sub_quadrants(chain, n)
    zs = [sq.psi.z for sq in sqs]
    zetas = [sq.psi.zeta for sq in sqs]
    print(f"  unique z: {len(set(zs))}/32")
    print(f"  unique (z,zeta): {len({(sq.psi.z, sq.psi.zeta) for sq in sqs})}/32")

    print("\n--- Mask table matches apply_vector ---")
    ok = all(match_lattice_apply(b, chain, n, m) for b in BranchKind for m in range(8))
    print(f"  all 32 masks match: {ok}")


if __name__ == "__main__":
    demo()
