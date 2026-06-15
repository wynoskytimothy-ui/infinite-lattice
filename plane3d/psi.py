"""
AETHOS 3D Complex Plane — derived formalism (countable anchors).

State space
-----------
Each lattice address lives in **C x R** (spring plane + depth axis):

    Psi = (z, zeta)   with   z = X + iY  in C,   zeta = Z  in R

Native label (not Cartesian x,y,z search):

    alpha = (A, b, w, n)

    A  = strictly increasing countable anchor chain (any SequenceKind)
    b  = branch in {VA1..VA4}  (4-way phase fan on C)
    w  = wing 1..8             (8 imaginary-axis corridors)
    n  = transgressor in N     (1D rail parameter)

Plot (Re z, Im z, zeta) is one camera on one wing; native arena is Psi in C x R
(the 3D complex plane). See ONTOLOGY.md — distinct from the pi lattice in pi/.

Glossary (MODEL terms — ONTOLOGY.md §B.5–B.6; disambiguate from textbook math)
-------------------------------------------------------------------------------
Imaginary vector   Phase engine: spring (X,Y), i_act=Rx∘S, branch b (4-way fan).
Complex vector     2-way meet path: swap_meet(a,p), trigger_history along rail n.
Complex number     3-way locked node Psi=(z,zeta); triple_equalization destination.
ICN                Prime composite address; factor C -> chain -> Psi (encoding layer).
Complex readout    z = X + iY (standard a+bi at wing readout — not "complex number" above).
Transmission       alpha=(A,b,w,n); 32 chambers = 1 vector x (4 branches x 8 wings).

Layer 0 — imaginary-axis start (|A| = 0)
----------------------------------------
    canon_b(empty, n) = (n, n, n)  for all b  [VA1 exact; others generalize at k=0]

    z_0(n) = n + n*i = n(1 + i)
    zeta_0(n) = n

Layer 1 — single anchor a
-------------------------
Let b_regime = (n >= a).  VA1:

    X = a + n
    Y = n           if not b_regime   (imag axis walks)
    Y = a           if b_regime       (velocity lock at anchor)
    zeta = a + n

Anchor rest on X-Z spoke: O_a = (a, 0, a).  Displacement at n=a: dz = a + a*i.

Layer k — segment FSM on chain A = (a_1, ..., a_k)
--------------------------------------------------
    s = segment_index(A, n)  in {0, ..., k}

    s = 0       : n < a_1
    s = i       : a_i <= n < a_{i+1}
    s = k       : n >= a_k

Depth (third axis):

    zeta(A, n, s) = sum(A) + n           if k <= 2 or s in {0, k}
                  = sum(A)                 if 0 < s < k  (interior lock)

Branch formulas (VA1; VA2-VA4 same segment breaks, different (X,Y) algebra):

    s = 0 :  (a_1 + a_k,  a_1,  zeta)
    0<s<k:  (a_k + n,     n,    zeta)
    s = k :  (a_k + n,     a_k,  zeta)

VA2-VA4: sign / 2n couplings per aethos_recursive.CANON and aethos_sequences.canon_on_chain.

8 wings — imaginary-axis corridors (operators on R^3)
-----------------------------------------------------
Same canonical (X,Y,zeta), then wing w in 1..8:

    VA (w=1..4):  optional flip_x, flip_z on (X, Y, zeta)
    VB (w=5..8):  (Y,X,zeta) swap, then flips  [Y-Z corridor lead]

Spring readout after wing w:

    z_w = X_w + i*Y_w
    zeta_w = Z_w

32 wings = 4 branches x 8 vectors.

2-way meet (swap) — solo chain {a} vs {p}
-----------------------------------------
    bank(a) @ n=p  ==  bank(p) @ n=a   (all 32 wings, integral chains)

k-way meet — missing-variable rule
----------------------------------
For sorted A = {a, p, q, ...} and subset S c A with |S| = k-1, missing m = A \\ S:

    canon_on_chain(S, n=m) == canon_on_chain(A, n=witness)

Example A=(3,5,7):  (3,5)@7, (3,7)@5, (5,7)@3  all equalize to (12,5,15).

Same node, distinct state: path (which S, which n), branch b, wing w, and
prime-by-prime trigger history z at each a_i crossing distinguish Hilbert labels
even when (X,Y,zeta) collide.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterator, Sequence

from plane3d.lattice import (
    BranchKind,
    Coord,
    LatticeId,
    VECTORS,
    apply_vector,
    lattice_id_parts,
)
from plane3d.recursive import canon_recursive, segment_index, z_depth
from plane3d.sequences import canon_on_chain, normalize_chain, segment_index_chain, sum_chain


@dataclass(frozen=True)
class ComplexPlane3D:
    """Point in C x R: spring z = X + iY, depth zeta = Z."""

    z: complex
    zeta: float

    @classmethod
    def from_coord(cls, coord: Coord) -> ComplexPlane3D:
        x, y, z = coord
        return cls(z=complex(x, y), zeta=float(z))

    @property
    def coord(self) -> Coord:
        return (self.z.real, self.z.imag, self.zeta)

    @property
    def modulus_squared(self) -> float:
        return abs(self.z) ** 2


@dataclass(frozen=True)
class LatticeAddress:
    """Native 3D-complex-plane address alpha = (A, b, w, n)."""

    chain: tuple[float, ...]
    branch: BranchKind
    wing: int  # 1..8
    n: float

    @property
    def lattice_id(self) -> LatticeId:
        return LatticeId((int(self.branch) - 1) * 8 + self.wing)

    @classmethod
    def from_lattice_id(
        cls,
        chain: Sequence[int | float],
        n: float,
        lid: LatticeId,
    ) -> LatticeAddress:
        branch, vector = lattice_id_parts(lid)
        return cls(chain=normalize_chain(chain), branch=branch, wing=vector.index, n=n)


def imaginary_start(n: float) -> ComplexPlane3D:
    """Layer 0: |A| = 0  =>  z = n + n*i,  zeta = n."""
    return ComplexPlane3D(z=complex(n, n), zeta=float(n))


def canon_complex(
    branch: BranchKind,
    chain: Sequence[int | float],
    n: float,
    *,
    lock_interior: bool = True,
) -> ComplexPlane3D:
    """Branch formula on any countable chain -> (z, zeta)."""
    return ComplexPlane3D.from_coord(
        canon_on_chain(branch, chain, n, lock_interior=lock_interior)
    )


def wing_transform(
    branch: BranchKind,
    chain: Sequence[int | float],
    n: float,
    wing: int,
    *,
    lock_interior: bool = True,
) -> ComplexPlane3D:
    """Full alpha = (A, b, w, n) -> Psi in C x R after wing w."""
    if not 1 <= wing <= 8:
        raise ValueError(f"wing must be 1..8, got {wing}")
    canon = canon_on_chain(branch, chain, n, lock_interior=lock_interior)
    vec = VECTORS[wing - 1]
    return ComplexPlane3D.from_coord(apply_vector(canon, vec))


def wing_transform_lid(
    chain: Sequence[int | float],
    n: float,
    lid: LatticeId,
    *,
    lock_interior: bool = True,
) -> ComplexPlane3D:
    branch, vector = lattice_id_parts(lid)
    return wing_transform(branch, chain, n, vector.index, lock_interior=lock_interior)


def all_branch_phases(
    chain: Sequence[int | float],
    n: float,
    wing: int = 1,
    *,
    lock_interior: bool = True,
) -> dict[BranchKind, ComplexPlane3D]:
    """Four complex phases at fixed (A, w, n)."""
    return {
        b: wing_transform(b, chain, n, wing, lock_interior=lock_interior)
        for b in BranchKind
    }


def trigger_history(
    chain: Sequence[int | float],
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
    *,
    lock_interior: bool = True,
) -> Iterator[tuple[float, ComplexPlane3D]]:
    """Prime-by-prime (anchor-by-anchor) spring z at each n = a_i."""
    c = normalize_chain(chain)
    for a in c:
        psi = wing_transform(branch, c, a, wing, lock_interior=lock_interior)
        yield a, psi


def missing_member(chain: Sequence[int | float], subset: Sequence[int | float]) -> float:
    """Unique anchor in chain not present in subset (for 2-way -> k-way meet)."""
    full = set(normalize_chain(chain))
    sub = set(normalize_chain(subset))
    diff = full - sub
    if len(diff) != 1:
        raise ValueError(f"expected co-dimension-1 subset, got missing={diff}")
    return next(iter(diff))


def equalize_witness(
    chain: Sequence[int | float],
    subset: Sequence[int | float],
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> tuple[float, ComplexPlane3D]:
    """
    Missing-variable rule: transgress subset-chain until n = missing anchor.
    Returns (n_witness, Psi) on the equalization node.
    """
    full = normalize_chain(chain)
    m = missing_member(full, subset)
    psi = wing_transform(branch, subset, m, wing)
    return m, psi


def triple_equalization(
    a: float,
    p: float,
    q: float,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> dict[str, tuple[float, ComplexPlane3D]]:
    """
    All three 2-way rails -> same node for sorted a < p < q.
    Returns witness n and Psi for each pair subset.
    """
    if not (a < p < q):
        raise ValueError("require a < p < q")
    full = (a, p, q)
    out: dict[str, tuple[float, ComplexPlane3D]] = {}
    for label, sub in [
        ("ap", (a, p)),
        ("aq", (a, q)),
        ("pq", (p, q)),
    ]:
        n_w, psi = equalize_witness(full, sub, branch, wing)
        out[label] = (n_w, psi)
    return out


def swap_meet(
    a: float,
    p: float,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> tuple[ComplexPlane3D, ComplexPlane3D]:
    """Solo 2-way: bank(a)@n=p vs bank(p)@n=a."""
    left = wing_transform(branch, (a,), p, wing)
    right = wing_transform(branch, (p,), a, wing)
    return left, right


def segment_at(chain: Sequence[int | float], n: float) -> int:
    c = normalize_chain(chain)
    if all(float(int(x)) == x for x in c):
        return segment_index(tuple(int(x) for x in c), int(n))
    return segment_index_chain(c, n)


def depth_at(
    chain: Sequence[int | float],
    n: float,
    *,
    lock_interior: bool = True,
) -> float:
    c = normalize_chain(chain)
    seg = segment_at(c, n)
    if all(float(int(x)) == x for x in c):
        return float(z_depth(tuple(int(x) for x in c), int(n), seg))
    # float chain mirror of z_accumulator
    k = len(c)
    s = sum_chain(c)
    if not lock_interior or k <= 2:
        return s + n
    if 0 < seg < k:
        return s
    return s + n


def derive_va1_closed_form(
    chain: Sequence[int | float],
    n: float,
) -> tuple[float, float, float]:
    """
    Explicit VA1 (X, Y, zeta) for documentation / verification.
    Matches canon_on_chain(VA1, ...).
    """
    c = normalize_chain(chain)
    k = len(c)
    if k == 0:
        return (n, n, n)
    if k == 1:
        a = c[0]
        b = n >= a
        return (a + n, a if b else n, a + n)
    a1, ak = c[0], c[-1]
    seg = segment_at(c, n)
    zeta = depth_at(c, n)
    if seg == 0:
        return (a1 + ak, a1, zeta)
    if seg < k:
        return (ak + n, n, zeta)
    return (ak + n, ak, zeta)


def demo() -> None:
    print("=" * 72)
    print("AETHOS 3D COMPLEX PLANE  —  Psi = (z, zeta)  in  C x R")
    print("=" * 72)

    print("\n--- Layer 0: imaginary start z = n + ni ---")
    for n in (0, 1, 3, 5):
        psi = imaginary_start(n)
        print(f"  n={n}  z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")

    print("\n--- Layer 1: single anchor a=5, VA1 ---")
    for n in (3, 5, 7):
        psi = canon_complex(BranchKind.VA1, (5,), n)
        print(f"  n={n}  z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")

    print("\n--- Triple equalization (3,5,7): missing-variable rule ---")
    eq = triple_equalization(3, 5, 7)
    ref = None
    for label, (n_w, psi) in eq.items():
        print(f"  {label}: n={n_w}  z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")
        ref = ref or psi
    assert all(psi.coord == ref.coord for _, psi in eq.values())

    print("\n--- Same node, 4 branch phases @ (3,5,7) n=5 ---")
    for b, psi in all_branch_phases((3, 5, 7), 5).items():
        print(f"  {b.name}: z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")

    print("\n--- Countable set: EVENS (2,4,6,8) @ n=10 ---")
    psi = canon_complex(BranchKind.VA1, (2, 4, 6, 8), 10)
    print(f"  z={psi.z.real:.0f}{psi.z.imag:+.0f}i  zeta={psi.zeta:.0f}")

    print("\n--- Prime-by-prime trigger history (3,5,7) ---")
    for a, psi in trigger_history((3, 5, 7)):
        print(f"  cross a={a:.0f}: z={psi.z.real:.0f}{psi.z.imag:+.0f}i")


if __name__ == "__main__":
    demo()
