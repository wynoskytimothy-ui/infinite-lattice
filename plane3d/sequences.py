"""
AETHOS on arbitrary countable anchor sets (not only primes).

The same 4 branch formulas + 32 wings apply to any strictly increasing
sequence A = (a1, a2, ...). Transgressor n : 0 -> infinity crosses each
anchor; z accumulates sum(A) + n (with optional interior composition lock).

Examples: primes, evens, powers of two, Fibonacci, scaled irrationals, etc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from plane3d.lattice import BranchKind, Coord, LatticeBank32, LatticeId
from plane3d.recursive import (
    CANON,
    canon_recursive,
    segment_index,
    z_depth,
)


# ---------------------------------------------------------------------------
# Anchor chains (countable sets)
# ---------------------------------------------------------------------------

def normalize_chain(values: Sequence[int | float]) -> tuple[float, ...]:
    """Strictly increasing, distinct anchors."""
    raw = [float(v) for v in values]
    if len(raw) != len(set(raw)):
        raise ValueError("anchors must be distinct")
    chain = tuple(sorted(raw))
    for i in range(1, len(chain)):
        if chain[i] <= chain[i - 1]:
            raise ValueError("anchors must be strictly increasing")
    return chain


def segment_index_chain(chain: tuple[float, ...], n: float) -> int:
    s = 0
    for a in chain:
        if n < a:
            return s
        s += 1
    return s


def sum_chain(chain: tuple[float, ...]) -> float:
    return sum(chain)


def z_accumulator(chain: tuple[float, ...], n: float, seg: int, *, lock_interior: bool = True) -> float:
    s = sum_chain(chain)
    k = len(chain)
    if not lock_interior or k <= 2:
        return s + n
    if 0 < seg < k:
        return s
    return s + n


def canon_on_chain(branch: BranchKind, chain: Sequence[int | float], n: float, *, lock_interior: bool = True) -> Coord:
    """Same VA1–VA4 recursion; anchors replace primes."""
    c = normalize_chain(chain)
    k = len(c)
    if k == 0:
        return (n, n, n)
    if k == 1:
        # Single anchor: reuse integer path when integral
        a0 = c[0]
        if float(int(a0)) == a0:
            return CANON[branch]((int(a0),), int(n))
        return _single_anchor_float(branch, c, n)

    a1, ak = c[0], c[-1]
    seg = segment_index_chain(c, n)
    z = z_accumulator(c, n, seg, lock_interior=lock_interior)

    if branch == BranchKind.VA1:
        if seg == 0:
            return (a1 + ak, a1, z)
        if seg < k:
            return (ak + n, n, z)
        return (ak + n, ak, z)
    if branch == BranchKind.VA2:
        if seg == 0:
            return (2 * n + a1 + ak, -a1, z)
        if seg < k:
            return (2 * a1 + n + ak, -n, z)
        return (2 * a1 + n + ak, -ak, z)
    if branch == BranchKind.VA3:
        if seg == 0:
            return (n + ak, -n, z)
        if seg < k:
            return (a1 + ak, -a1, z)
        return (a1 + n, -a1, z)
    if seg == 0:
        return (2 * a1 + ak + n, n, z)
    if seg < k:
        return (2 * n + a1 + ak, a1, z)
    return (2 * ak + a1 + n, a1, z)


def _single_anchor_float(branch: BranchKind, chain: tuple[float, ...], n: float) -> Coord:
    p = chain[0]
    b = n >= p
    z = p + n
    if branch == BranchKind.VA1:
        return (p + n, p if b else n, z)
    if branch == BranchKind.VA2:
        return (p + n, -p if b else -n, z)
    if branch == BranchKind.VA3:
        return (n if b else p, 0, z)
    return (n + 2 * p if b else p + 2 * n, 0, z)


class SequenceKind(Enum):
    PRIMES = "primes"  # odd primes only: 3, 5, 7, 11, ... (2 skipped)
    EVENS = "evens"  # 2, 4, 6, 8, ...
    POWERS_OF_2 = "powers_of_2"  # 2, 4, 8, 16, ...
    FIBONACCI = "fibonacci"
    SQUARES = "squares"  # 1, 4, 9, 16, ...
    TRIPLES = "triples"  # 3, 6, 9, ...
    SQRT_SCALED = "sqrt_scaled"  # floor(scale * sqrt(k))
    CUSTOM = "custom"


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def chain_primes(count: int) -> tuple[int, ...]:
    """Odd primes only — anchor 2 is skipped (3, 5, 7, 11, ...)."""
    out: list[int] = []
    x = 3
    while len(out) < count:
        if _is_prime(x):
            out.append(x)
        x += 2
    return tuple(out)


def chain_evens(count: int, start: int = 2) -> tuple[int, ...]:
    return tuple(start + 2 * i for i in range(count))


def chain_powers_of_2(count: int, start: int = 2) -> tuple[int, ...]:
    return tuple(start * (2**i) for i in range(count))


def chain_fibonacci(count: int) -> tuple[int, ...]:
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


def chain_squares(count: int) -> tuple[int, ...]:
    return tuple((i + 1) ** 2 for i in range(count))


def chain_sqrt_scaled(count: int, scale: int = 100) -> tuple[int, ...]:
    """Integer lattice for sqrt(1), sqrt(2), ... via scaling."""
    return tuple(int(scale * math.sqrt(i + 1)) for i in range(count))


GENERATORS: dict[SequenceKind, Callable[[int], tuple[int, ...]]] = {
    SequenceKind.PRIMES: chain_primes,
    SequenceKind.EVENS: chain_evens,
    SequenceKind.POWERS_OF_2: chain_powers_of_2,
    SequenceKind.FIBONACCI: chain_fibonacci,
    SequenceKind.SQUARES: chain_squares,
    SequenceKind.TRIPLES: lambda k: tuple(3 * (i + 1) for i in range(k)),
    SequenceKind.SQRT_SCALED: chain_sqrt_scaled,
}


def make_chain(kind: SequenceKind, count: int, **kwargs) -> tuple[int, ...]:
    if kind == SequenceKind.CUSTOM:
        raise ValueError("use normalize_chain() for custom")
    gen = GENERATORS[kind]
    if kind == SequenceKind.SQRT_SCALED:
        scale = int(kwargs.get("scale", 100))
        return chain_sqrt_scaled(count, scale)
    return gen(count)


@dataclass(frozen=True)
class IntersectionType:
    """Named anchor-set species — same 32 wings, different collision geometry."""
    name: str
    kind: SequenceKind
    chain: tuple[int, ...]

    @classmethod
    def build(cls, name: str, kind: SequenceKind, count: int, **kwargs) -> IntersectionType:
        return cls(name=name, kind=kind, chain=make_chain(kind, count, **kwargs))


def bank_for_chain(chain: Sequence[int]) -> "LatticeBank32K":
    """32 wings, arbitrary integer anchor chain (primes are one instance)."""
    from plane3d.recursive import LatticeBank32K

    return LatticeBank32K(tuple(int(x) for x in chain))


def compare_types_at_n(
    types: list[IntersectionType],
    n: int,
    branch: BranchKind = BranchKind.VA1,
) -> dict[str, Coord]:
    return {t.name: canon_on_chain(branch, t.chain, n) for t in types}


def cross_type_meet(
    left: IntersectionType,
    right: IntersectionType,
    n_left: int,
    n_right: int,
    branch: BranchKind = BranchKind.VA1,
) -> bool:
    return canon_on_chain(branch, left.chain, n_left) == canon_on_chain(branch, right.chain, n_right)


def demo() -> None:
    types = [
        IntersectionType.build("primes", SequenceKind.PRIMES, 5),
        IntersectionType.build("evens", SequenceKind.EVENS, 5),
        IntersectionType.build("2-4-8-16", SequenceKind.POWERS_OF_2, 5),
        IntersectionType.build("sqrt~x100", SequenceKind.SQRT_SCALED, 5, scale=100),
    ]

    print("=== Same n, different countable anchor species (VA1) ===\n")
    print(f"  {'type':<12}  chain              n=10")
    for t in types:
        c = canon_on_chain(BranchKind.VA1, t.chain, 10)
        print(f"  {t.name:<12}  {str(t.chain):<18}  {c}")

    print("\n=== Single-anchor swap meet (first two elements as 1-chains) ===\n")
    # Evens: 2 and 4 — solo chains {2} and {4}
    e2 = (2,)
    e4 = (4,)
    c_a = canon_on_chain(BranchKind.VA1, e2, 4)
    c_b = canon_on_chain(BranchKind.VA1, e4, 2)
    print(f"  evens {{2}} @ n=4:  {c_a}")
    print(f"  evens {{4}} @ n=2:  {c_b}")
    print(f"  meet: {c_a == c_b}")

    print("\n=== Powers of 2: (2,4) @ n=8 meets (2,8) @ n=4 style ===\n")
    c24 = canon_on_chain(BranchKind.VA1, (2, 4), 8)
    c28 = canon_on_chain(BranchKind.VA1, (2, 8), 4)
    print(f"  (2,4)@8 = {c24}")
    print(f"  (2,8)@4 = {c28}")

    print("\n=== Primes still match PDF k=2 ===\n")
    from plane3d.recursive import verify_matches_spec_k2

    print(f"  spec k=2: {verify_matches_spec_k2()}")

    print("\n=== Infinite family: first k anchors, k = 1..4 (primes vs 2^n) ===\n")
    for k in range(1, 5):
        p = make_chain(SequenceKind.PRIMES, k)
        g = make_chain(SequenceKind.POWERS_OF_2, k)
        cp = canon_on_chain(BranchKind.VA1, p, 10)
        cg = canon_on_chain(BranchKind.VA1, g, 10)
        print(f"  k={k}  primes{p} z-component ~ {cp[2]}")
        print(f"       2^n   {g}     -> {cg}")


if __name__ == "__main__":
    demo()
