"""
φ-Prime Lattice — geometric engine (VA1–VA4, 32 wings, k-anchor recursion).

Canonical spec implementation for the core package:
  - Anchor point (p,0,p) distinct from branch coordinates at transgressor n
  - Z plateau: interior segments use Z = sum(P) only when k > 2
  - swap_meet for cross-bank intersection witnesses
  - prime_factor_similarity for scale-invariant composite retrieval
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from operator import mul
from typing import Iterator, Literal, Sequence

from core.primes import chain_primes

# ---------------------------------------------------------------------------
# Vectors and coordinates
# ---------------------------------------------------------------------------

Coord = tuple[int, int, int]


@dataclass(frozen=True)
class Vector:
    index: int
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


class BranchKind(IntEnum):
    VA1 = 1
    VA2 = 2
    VA3 = 3
    VA4 = 4


class LatticeId(IntEnum):
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


PP_SUB = {
    BranchKind.VA1: "VA1A",
    BranchKind.VA2: "VA1B",
    BranchKind.VA3: "VA1C",
    BranchKind.VA4: "VA1D",
}


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


def lattice_id_parts(lid: LatticeId) -> tuple[BranchKind, Vector]:
    i = int(lid) - 1
    return BranchKind((i // 8) + 1), VECTORS[i % 8]


# ---------------------------------------------------------------------------
# Segment index and Z depth (composition lock)
# ---------------------------------------------------------------------------

def normalize_primes(primes: Sequence[int]) -> tuple[int, ...]:
    ps = tuple(sorted(set(primes)))
    if len(ps) != len(primes):
        raise ValueError("primes must be distinct")
    return ps


def segment_index(primes: tuple[int, ...], n: int) -> int:
    """0-based segment: n < p1 -> 0; pi <= n < p_{i+1} -> i; n >= pk -> k."""
    k = len(primes)
    if k == 0:
        return 0
    s = 0
    for i, p in enumerate(primes):
        if n < p:
            return s
        s = i + 1
    return k


def sum_primes(primes: tuple[int, ...]) -> int:
    return sum(primes)


def z_depth(primes: tuple[int, ...], n: int, seg: int) -> int:
    k = len(primes)
    if k <= 2:
        return sum_primes(primes) + n
    if 0 < seg < k:
        return sum_primes(primes)
    return sum_primes(primes) + n


# ---------------------------------------------------------------------------
# Single-prime: anchor vs branch
# ---------------------------------------------------------------------------

def prime_anchor_coord(p: int) -> Coord:
    """Base vector anchor for prime p before wing transform: (p, 0, p)."""
    return (p, 0, p)


def single_prime_branch(branch: BranchKind, p: int, n: int) -> Coord:
    """Branch coordinate at transgressor n (Regime A/B), not the anchor point."""
    b = n >= p
    if branch == BranchKind.VA1:
        return (p + n, p if b else n, p + n)
    if branch == BranchKind.VA2:
        return (p + n, -p if b else -n, p + n)
    if branch == BranchKind.VA3:
        return (n if b else p, 0, p + n)
    return (n + 2 * p if b else p + 2 * n, 0, p + n)


def prime_pair_case(a: int, p: int, n: int) -> int:
    """1-based case label for k=2 PDF tables."""
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


def canon_va1(primes: tuple[int, ...], n: int) -> Coord:
    k = len(primes)
    if k == 0:
        return (n, n, n)
    if k == 1:
        return single_prime_branch(BranchKind.VA1, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (p1 + pk, p1, z)
    if seg < k:
        return (pk + n, n, z)
    return (pk + n, pk, z)


def canon_va2(primes: tuple[int, ...], n: int) -> Coord:
    k = len(primes)
    if k == 1:
        return single_prime_branch(BranchKind.VA2, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (2 * n + p1 + pk, -p1, z)
    if seg < k:
        return (2 * p1 + n + pk, -n, z)
    return (2 * p1 + n + pk, -pk, z)


def canon_va3(primes: tuple[int, ...], n: int) -> Coord:
    k = len(primes)
    if k == 1:
        return single_prime_branch(BranchKind.VA3, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (n + pk, -n, z)
    if seg < k:
        return (p1 + pk, -p1, z)
    return (p1 + n, -p1, z)


def canon_va4(primes: tuple[int, ...], n: int) -> Coord:
    k = len(primes)
    if k == 1:
        return single_prime_branch(BranchKind.VA4, primes[0], n)
    p1, pk = primes[0], primes[-1]
    seg = segment_index(primes, n)
    z = z_depth(primes, n, seg)
    if seg == 0:
        return (2 * p1 + pk + n, n, z)
    if seg < k:
        return (2 * n + p1 + pk, p1, z)
    return (2 * pk + p1 + n, p1, z)


CANON = {
    BranchKind.VA1: canon_va1,
    BranchKind.VA2: canon_va2,
    BranchKind.VA3: canon_va3,
    BranchKind.VA4: canon_va4,
}


def canon_on_chain(branch: BranchKind, primes: Sequence[int], n: int) -> Coord:
    return CANON[branch](normalize_primes(primes), n)


# ---------------------------------------------------------------------------
# Species chains
# ---------------------------------------------------------------------------

def prime_chain(count: int) -> tuple[int, ...]:
    return chain_primes(count)


def even_chain(count: int, start: int = 2) -> tuple[int, ...]:
    return tuple(start + 2 * i for i in range(count))


def powers_of_two(count: int, start: int = 2) -> tuple[int, ...]:
    return tuple(start * (2**i) for i in range(count))


def fibonacci_chain(count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count == 1:
        return (1,)
    a, b = 1, 1
    out = [1, 1]
    for _ in range(count - 2):
        a, b = b, a + b
        out.append(b)
    return tuple(out)


def squares_chain(count: int) -> tuple[int, ...]:
    return tuple((i + 1) ** 2 for i in range(count))


# ---------------------------------------------------------------------------
# Coordinates API
# ---------------------------------------------------------------------------

def compute_coordinates(
    primes: Sequence[int],
    n: int,
    lattice_id: LatticeId | int = LatticeId.L01,
) -> Coord:
    """Full wing coordinate: canonical branch + vector transform."""
    branch, vector = lattice_id_parts(LatticeId(lattice_id))
    ps = normalize_primes(primes) if primes else ()
    if not ps:
        canon = (n, n, n)
    else:
        canon = CANON[branch](ps, n)
    return apply_vector(canon, vector)


def compute_anchor(
    p: int,
    lattice_id: LatticeId | int = LatticeId.L01,
) -> Coord:
    """Prime anchor (p,0,p) on the requested wing."""
    _, vector = lattice_id_parts(LatticeId(lattice_id))
    return apply_vector(prime_anchor_coord(p), vector)


def compute_all_wings(primes: Sequence[int], n: int) -> dict[LatticeId, Coord]:
    return {LatticeId(i): compute_coordinates(primes, n, i) for i in range(1, 33)}


# ---------------------------------------------------------------------------
# Swap meet
# ---------------------------------------------------------------------------

def swap_meet(
    p: int,
    q: int,
    *,
    lattice_id: LatticeId | int = LatticeId.L01,
    n_max: int = 2000,
) -> tuple[Coord, int, int] | None:
    """
    Witness where bank(p) at n=q meets bank(q) at n=p on one wing.
    Returns (coord, n_left, n_right) or None.
    """
    del n_max  # witness is at fixed transgressors n_p=q, n_q=p
    lid = LatticeId(lattice_id)
    c_left = compute_coordinates((p,), q, lid)
    c_right = compute_coordinates((q,), p, lid)
    if c_left == c_right:
        return (c_left, q, p)
    return None


def swap_meet_all_wings(p: int, q: int, n_max: int = 2000) -> dict[LatticeId, Coord]:
    """All 32 wings that satisfy the solo swap witness at (n_p=q, n_q=p)."""
    hits: dict[LatticeId, Coord] = {}
    for lid in LatticeId:
        w = swap_meet(p, q, lattice_id=lid, n_max=n_max)
        if w is not None:
            hits[lid] = w[0]
    return hits


def should_promote_intersection(
    existing_primes: Sequence[int],
    new_prime: int,
    *,
    lattice_id: LatticeId | int = LatticeId.L01,
    n_max: int = 800,
) -> bool:
    """
    True when extending the anchor chain by new_prime has a geometric swap/extension
    witness on at least one wing (3-way intersection promotion gate).
    """
    base = normalize_primes(existing_primes)
    if not base:
        return False
    extended = normalize_primes((*base, new_prime))
    if len(extended) <= len(base):
        return False
    lid = LatticeId(lattice_id)
    shallow = base
    p1, pk = shallow[0], shallow[-1]
    for na in range(1, n_max):
        c_sh = compute_coordinates(shallow, na, lid)
        c_deep = compute_coordinates(extended, na, lid)
        if c_sh == c_deep:
            return True
        if len(shallow) == 1 and swap_meet(shallow[0], new_prime, lattice_id=lid, n_max=n_max):
            return True
    if len(shallow) >= 2 and swap_meet(p1, pk, lattice_id=lid, n_max=min(n_max, pk + 10)):
        return True
    return False


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def prime_factors(n: int) -> frozenset[int]:
    if n < 2:
        return frozenset()
    factors: set[int] = set()
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            factors.add(d)
            x //= d
        d += 1 if d == 2 else 2
    if x > 1:
        factors.add(x)
    return frozenset(factors)


def prime_factor_similarity(c1: int, c2: int) -> float:
    """
    Scale-invariant Jaccard on prime factor sets (GCD-Jaccard).

    Use for cross-cluster composite comparison; raw Euclidean on large
    products is dominated by magnitude, not shared morphology.
    """
    f1 = prime_factors(c1)
    f2 = prime_factors(c2)
    if not f1 and not f2:
        return 1.0
    if not f1 or not f2:
        return 0.0
    inter = len(f1 & f2)
    union = len(f1 | f2)
    return inter / union if union else 0.0


def euclidean_distance(a: Coord, b: Coord) -> float:
    """Same-scale proximity within a neighborhood."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def composite_from_primes(primes: Sequence[int]) -> int:
    ps = normalize_primes(primes)
    if not ps:
        return 1
    return reduce(mul, ps, 1)


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_va_vb_swap() -> bool:
    """Y-over swap: (16,5,20) YXZ -> (5,16,20) before flips."""
    raw = (16, 5, 20)
    swapped = yxz_to_xyz(raw)
    return swapped == (5, 16, 20)


def verify_golden_k1_p5_n7() -> bool:
    return compute_coordinates((5,), 7, LatticeId.L01) == (12, 5, 12)


def verify_golden_k2_3_11_n5() -> bool:
    return canon_va1((3, 11), 5) == (16, 5, 19)


def verify_k3_z_plateau() -> bool:
    chain = (3, 5, 7, 11)
    z_vals = {z_depth(chain, n, segment_index(chain, n)) for n in range(3, 11)}
    return len(z_vals) == 1 and 26 in z_vals
