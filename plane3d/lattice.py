"""
AETHOS 3D complex plane — lattice formula (32 chambers per bank).

The arena is Psi = (z, zeta) in C x R. This module implements branch x wing
coordinates. (Deprecated prose name: "phi-prime lattice" — see ONTOLOGY.md.)

Each bank chamber is one of:
  4 branch families (VA1–VA4) × 8 base vectors (v1–v8) = 32

Every lattice has its own transgressor n : 0 → ∞, regime/velocity boundaries,
and coordinate stream. Lattices intersect naturally (same-n or cross-n) when
paths share an address.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Iterator, Literal

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vector:
    index: int  # 1–8
    name: str
    family: Literal["VA", "VB"]
    flip_x: bool
    flip_y: bool
    flip_z: bool

VECTORS: tuple[Vector, ...] = (
    Vector(1, "v1", "VA", False, False, False),
    Vector(2, "v2", "VA", False, False, True),
    Vector(3, "v3", "VA", True, False, False),
    Vector(4, "v4", "VA", True, False, True),
    Vector(5, "v5", "VB", False, False, False),
    Vector(6, "v6", "VB", False, False, True),
    Vector(7, "v7", "VB", True, False, False),
    Vector(8, "v8", "VB", True, False, True),
)

Coord = tuple[int, int, int]
Mode = Literal["single", "pair"]


class BranchKind(IntEnum):
    VA1 = 1
    VA2 = 2
    VA3 = 3
    VA4 = 4


# Sub-branch labels for prime-by-prime (Section 5.1); map 1:1 to VA1–VA4 lattices
PP_SUB = {BranchKind.VA1: "VA1A", BranchKind.VA2: "VA1B", BranchKind.VA3: "VA1C", BranchKind.VA4: "VA1D"}


def yxz_to_xyz(c: Coord) -> Coord:
    y, x, z = c
    return (x, y, z)


def apply_vector(c: Coord, v: Vector) -> Coord:
    if v.family == "VB":
        c = yxz_to_xyz(c)
    x, y, z = c
    if v.flip_x:
        x = -x
    if v.flip_y:
        y = -y
    if v.flip_z:
        z = -z
    return (x, y, z)


# ---------------------------------------------------------------------------
# Branch formulas (canonical before vector transform)
# ---------------------------------------------------------------------------

def single_prime_canon(branch: BranchKind, p: int, n: int) -> Coord:
    b = n >= p
    if branch == BranchKind.VA1:
        return (p + n, p if b else n, p + n)
    if branch == BranchKind.VA2:
        return (p + n, -p if b else -n, p + n)
    if branch == BranchKind.VA3:
        return (n if b else p, 0, p + n)
    return (n + 2 * p if b else p + 2 * n, 0, p + n)


def prime_pair_case(a: int, p: int, n: int) -> int:
    if n < a:
        return 1
    if n < p:
        return 2
    return 3


def prime_pair_canon(sub: str, a: int, p: int, n: int) -> Coord:
    c = prime_pair_case(a, p, n)
    table: dict[str, list[Coord]] = {
        "VA1A": [(a + p, a, a + p + n), (p + n, n, a + p + n), (p + n, p, a + p + n)],
        "VA1B": [
            (2 * n + a + p, -a, a + p + n),
            (2 * a + n + p, -n, a + p + n),
            (2 * a + n + p, -p, a + p + n),
        ],
        "VA1C": [
            (n + p, -n, a + p + n),
            (a + p, -a, a + p + n),
            (a + n, -a, a + p + n),
        ],
        "VA1D": [
            (2 * a + p + n, n, a + p + n),
            (2 * n + p + a, a, a + p + n),
            (2 * p + a + n, a, a + p + n),
        ],
    }
    return table[sub][c - 1]


# ---------------------------------------------------------------------------
# One of 32 independent lattices
# ---------------------------------------------------------------------------

class LatticeId(IntEnum):
    """L01 = VA1×v1 … L32 = VA4×v8."""
    L01 = 1
    L02 = 2
    L03 = 3
    L04 = 4
    L05 = 5
    L06 = 6
    L07 = 7
    L08 = 8
    L09 = 9
    L10 = 10
    L11 = 11
    L12 = 12
    L13 = 13
    L14 = 14
    L15 = 15
    L16 = 16
    L17 = 17
    L18 = 18
    L19 = 19
    L20 = 20
    L21 = 21
    L22 = 22
    L23 = 23
    L24 = 24
    L25 = 25
    L26 = 26
    L27 = 27
    L28 = 28
    L29 = 29
    L30 = 30
    L31 = 31
    L32 = 32


def lattice_id_parts(lid: LatticeId) -> tuple[BranchKind, Vector]:
    i = int(lid) - 1
    branch = BranchKind((i // 8) + 1)
    vector = VECTORS[i % 8]
    return branch, vector


@dataclass
class Lattice:
    """
    A single independent lattice: one branch family on one base vector.
    Transgressor n moves 0 → ∞ along this lattice only.
    """

    id: LatticeId
    branch: BranchKind
    vector: Vector
    mode: Mode
    p: int
    a: int | None = None
    pair_p: int | None = None

    @property
    def name(self) -> str:
        return f"{self.id.name}_{self.branch.name}_{self.vector.name}"

    @classmethod
    def from_id(cls, lid: LatticeId, p: int) -> Lattice:
        branch, vector = lattice_id_parts(lid)
        return cls(id=lid, branch=branch, vector=vector, mode="single", p=p)

    @classmethod
    def from_id_pair(cls, lid: LatticeId, a: int, p: int) -> Lattice:
        if a > p:
            a, p = p, a
        branch, vector = lattice_id_parts(lid)
        return cls(id=lid, branch=branch, vector=vector, mode="pair", p=a, a=a, pair_p=p)

    def _canon(self, n: int) -> Coord:
        if self.mode == "single":
            return single_prime_canon(self.branch, self.p, n)
        assert self.a is not None and self.pair_p is not None
        sub = PP_SUB[self.branch]
        return prime_pair_canon(sub, self.a, self.pair_p, n)

    def at(self, n: int) -> Coord:
        """Address on this lattice at transgressor n."""
        return apply_vector(self._canon(n), self.vector)

    def transgress(self, n_min: int = 0, n_max: int = 100) -> Iterator[tuple[int, Coord]]:
        for n in range(n_min, n_max + 1):
            yield n, self.at(n)

    def velocity_boundaries(self) -> list[int]:
        """n values where formula regime switches (velocity change)."""
        if self.mode == "single":
            return [self.p]
        assert self.a is not None and self.pair_p is not None
        return [self.a, self.pair_p]

    def regime_label(self, n: int) -> str:
        if self.mode == "single":
            return "B" if n >= self.p else "A"
        assert self.a is not None and self.pair_p is not None
        c = prime_pair_case(self.a, self.pair_p, n)
        return f"case{c}"

    def anchor(self) -> Coord:
        """Prime anchor with n = p on this vector (Section 2)."""
        ap = self.pair_p if self.mode == "pair" else self.p
        base = (ap, 0, ap)
        return apply_vector(base, self.vector)


# ---------------------------------------------------------------------------
# Bank of all 32 independent lattices
# ---------------------------------------------------------------------------

class LatticeBank32:
    """Holds exactly 32 lattices for one prime configuration."""

    def __init__(self, lattices: list[Lattice]) -> None:
        if len(lattices) != 32:
            raise ValueError(f"expected 32 lattices, got {len(lattices)}")
        self._by_id = {lat.id: lat for lat in lattices}
        self.lattices = lattices

    @classmethod
    def single_prime(cls, p: int) -> LatticeBank32:
        return cls([Lattice.from_id(LatticeId(i), p) for i in range(1, 33)])

    @classmethod
    def prime_pair(cls, a: int, p: int) -> LatticeBank32:
        return cls([Lattice.from_id_pair(LatticeId(i), a, p) for i in range(1, 33)])

    def __getitem__(self, lid: LatticeId | int) -> Lattice:
        key = LatticeId(lid) if isinstance(lid, int) else lid
        return self._by_id[key]

    def all_at(self, n: int) -> dict[LatticeId, Coord]:
        return {lat.id: lat.at(n) for lat in self.lattices}

    def find_same_n_collisions(self, n: int) -> list[tuple[Coord, list[LatticeId]]]:
        bucket: dict[Coord, list[LatticeId]] = {}
        for lat in self.lattices:
            bucket.setdefault(lat.at(n), []).append(lat.id)
        return [(c, ids) for c, ids in bucket.items() if len(ids) > 1]

    def find_cross_n_meetings(
        self,
        other: LatticeBank32,
        n_min: int = 1,
        n_max: int = 500,
        lattice_filter: LatticeId | None = None,
    ) -> list[tuple[Coord, list[tuple[LatticeId, LatticeId, int, int]]]]:
        """
        Natural intersection between two banks (e.g. prime 3 bank vs prime 11 bank).
        Same lattice id on both sides — L07 on p=3 meets L07 on p=11 at cross-n.
        """
        meetings: dict[Coord, list[tuple[LatticeId, LatticeId, int, int]]] = {}
        ids = [lattice_filter] if lattice_filter else list(LatticeId)
        for lid in ids:
            la, lb = self[lid], other[lid]
            for na in range(n_min, n_max + 1):
                ca = la.at(na)
                for nb in range(n_min, n_max + 1):
                    if lb.at(nb) == ca:
                        meetings.setdefault(ca, []).append((lid, lid, na, nb))
        # dedupe entries per coord
        out: list[tuple[Coord, list[tuple[LatticeId, LatticeId, int, int]]]] = []
        for c, rows in meetings.items():
            uniq = list({(a, b, na, nb) for a, b, na, nb in rows})
            if len(uniq) >= 1:
                out.append((c, uniq))
        return sorted(out, key=lambda x: min(r[2] + r[3] for r in x[1]))

    def cross_prime_meet(
        self,
        p_left: int,
        p_right: int,
        lid: LatticeId = LatticeId.L01,
        n_max: int = 2000,
    ) -> list[tuple[int, int, Coord]]:
        """Meetings where left bank (p_left) at n=p_right matches right at n=p_left."""
        left = LatticeBank32.single_prime(p_left)
        right = LatticeBank32.single_prime(p_right)
        hits: list[tuple[int, int, Coord]] = []
        for n_left in range(1, n_max + 1):
            c = left[lid].at(n_left)
            for n_right in range(1, n_max + 1):
                if n_right == p_left and n_left == p_right and right[lid].at(n_right) == c:
                    hits.append((n_left, n_right, c))
        return hits


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    bank5 = LatticeBank32.single_prime(5)
    print("=== 32 independent lattices (prime p=5) ===\n")
    print(f"  Count: {len(bank5.lattices)}")
    print(f"  L01: {bank5[LatticeId.L01].name}  anchor={bank5[LatticeId.L01].anchor()}")
    print(f"  L16: {bank5[LatticeId.L16].name}  n=7 -> {bank5[LatticeId.L16].at(7)}")
    print(f"  L32: {bank5[LatticeId.L32].name}  n=7 -> {bank5[LatticeId.L32].at(7)}")

    print("\n=== Each lattice transgresses independently (L01, n=0..5) ===\n")
    lat = bank5[LatticeId.L01]
    for n, c in lat.transgress(0, 5):
        print(f"  n={n}  regime={lat.regime_label(n)}  {c}")

    print("\n=== Velocity boundary on every lattice (p=3) ===\n")
    bank3 = LatticeBank32.single_prime(3)
    boundaries = {lat.id: lat.velocity_boundaries() for lat in bank3.lattices}
    assert all(b == [3] for b in boundaries.values())

    print("\n=== 3-bank vs 11-bank: cross-n meet on all 32 lattices ===\n")
    bank3 = LatticeBank32.single_prime(3)
    bank11 = LatticeBank32.single_prime(11)
    meets = 0
    for lid in LatticeId:
        c3 = bank3[lid].at(11)
        c11 = bank11[lid].at(3)
        if c3 == c11:
            meets += 1
            if lid in (LatticeId.L01, LatticeId.L16, LatticeId.L32):
                print(f"  {lid.name}: 3@n=11 & 11@n=3 -> {c3}")
    print(f"  All 32 meet at swap (3,11): {meets}/32")

    print("\n=== Prime pair a=3, p=541 — 32 lattices, case switches ===\n")
    bank541 = LatticeBank32.prime_pair(3, 541)
    lat = bank541[LatticeId.L01]
    for n in (2, 3, 4, 5, 540, 541, 542):
        print(f"  n={n}  {lat.regime_label(n)}  {lat.at(n)}")

    print("\n=== 3|541 bank vs 3|5 bank — natural meet on L01 ===\n")
    b3541 = LatticeBank32.prime_pair(3, 541)
    b35 = LatticeBank32.prime_pair(3, 5)
    c5 = b3541[LatticeId.L01].at(5)
    c541 = b35[LatticeId.L01].at(541)
    print(f"  3,541 @ n=5:   {c5}")
    print(f"  3,5   @ n=541: {c541}")
    print(f"  Match: {c5 == c541}")


if __name__ == "__main__":
    demo()
