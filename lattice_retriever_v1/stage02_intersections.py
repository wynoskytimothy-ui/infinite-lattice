"""
Stage 02 — Intersections become subwords (32-lattice placement).

When two or three symbol-primes meet, placement is on the φ-Prime Lattice:
  4 branches × 8 vectors = 32 independent 3D lattices per prime configuration
  (AETHOS spec §2–§5; aethos_lattice.LatticeBank32).

The sum of primes (e.g. 73+23 = 69+27 = 96) is one anchor axis value — NOT
the node. The node is the full 32-coordinate signature at transgressor n.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Literal

from aethos_lattice import LatticeBank32, LatticeId
from aethos_recursive import LatticeBank32K
from aethos_words import letter_to_prime

from lattice_retriever_v1.stage01_symbols import SymbolPrimeSequence, symbols_to_primes

DEFAULT_TRANSGRESSOR_N = 7
NUM_LATTICES = 32


def intersection_sum(*primes: int) -> int:
    """Scalar anchor component (sum axis) — metadata only, not unique."""
    return sum(primes)


def pair_composite(p1: int, p2: int) -> int:
    """FTA-unique pair address (stage 05 preview)."""
    a, b = (p1, p2) if p1 <= p2 else (p2, p1)
    return a * b


def lattice_coords_32(
    primes: tuple[int, ...],
    *,
    n: int = DEFAULT_TRANSGRESSOR_N,
) -> tuple[tuple[int, int, int], ...]:
    """All 32 wing coordinates (X,Y,Z) for this prime meet at transgressor n."""
    if len(primes) == 2:
        a, b = (primes[0], primes[1]) if primes[0] <= primes[1] else (primes[1], primes[0])
        bank = LatticeBank32.prime_pair(a, b)
    elif len(set(primes)) == len(primes):
        bank = LatticeBank32K(tuple(sorted(primes)))
    else:
        # repeated symbol in subword (e.g. n-n-i) — pair-bank on distinct extremes
        a, b = min(primes), max(primes)
        bank = LatticeBank32.prime_pair(a, b)
    return tuple(bank[LatticeId(i)].at(n) for i in range(1, NUM_LATTICES + 1))


def lattice_signature(
    primes: tuple[int, ...],
    *,
    n: int = DEFAULT_TRANSGRESSOR_N,
) -> tuple[tuple[int, int, int], ...]:
    """Canonical node id — 32 coords; two meets collide iff all 32 match."""
    return lattice_coords_32(primes, n=n)


@dataclass(frozen=True)
class IntersectionAddress:
    """Subword meet: 32-lattice node + glass-box metadata."""

    way: Literal[2, 3]
    chars: tuple[str, ...]
    primes: tuple[int, ...]
    anchor_sum: int
    composite: int
    lattice_coords: tuple[tuple[int, int, int], ...]
    transgressor_n: int
    start_index: int

    @property
    def label(self) -> str:
        return "".join(self.chars)

    @property
    def address(self) -> int:
        """Legacy scalar — prefer composite + lattice_signature."""
        return self.composite if self.way == 2 else self.anchor_sum

    def explain(self) -> dict:
        return {
            "way": self.way,
            "label": self.label,
            "chars": list(self.chars),
            "primes": list(self.primes),
            "anchor_sum": self.anchor_sum,
            "composite": self.composite,
            "transgressor_n": self.transgressor_n,
            "lattice_L01": self.lattice_coords[0],
            "lattice_L32": self.lattice_coords[31],
            "n_lattices": len(self.lattice_coords),
            "start_index": self.start_index,
        }


def intersect_primes(
    chars: tuple[str, ...],
    primes: tuple[int, ...],
    *,
    start_index: int = 0,
    n: int = DEFAULT_TRANSGRESSOR_N,
) -> IntersectionAddress:
    """Build meet address with full 32-lattice placement."""
    if len(chars) not in (2, 3) or len(chars) != len(primes):
        raise ValueError(f"need 2 or 3 aligned symbols, got {len(chars)}")
    way: Literal[2, 3] = 2 if len(chars) == 2 else 3
    comp = (
        pair_composite(primes[0], primes[1])
        if way == 2
        else reduce(mul, sorted(set(primes)), 1)
    )
    return IntersectionAddress(
        way=way,
        chars=chars,
        primes=primes,
        anchor_sum=intersection_sum(*primes),
        composite=comp,
        lattice_coords=lattice_signature(primes, n=n),
        transgressor_n=n,
        start_index=start_index,
    )


def intersect_two(c1: str, c2: str, *, start_index: int = 0, n: int = DEFAULT_TRANSGRESSOR_N) -> IntersectionAddress:
    a, b = c1.lower(), c2.lower()
    return intersect_primes(
        (a, b),
        (letter_to_prime(a), letter_to_prime(b)),
        start_index=start_index,
        n=n,
    )


def intersect_three(
    c1: str, c2: str, c3: str, *, start_index: int = 0, n: int = DEFAULT_TRANSGRESSOR_N,
) -> IntersectionAddress:
    a, b, c = c1.lower(), c2.lower(), c3.lower()
    return intersect_primes(
        (a, b, c),
        (letter_to_prime(a), letter_to_prime(b), letter_to_prime(c)),
        start_index=start_index,
        n=n,
    )


def subword_intersections(
    seq: SymbolPrimeSequence,
    *,
    ways: tuple[int, ...] = (2, 3),
    n: int = DEFAULT_TRANSGRESSOR_N,
) -> tuple[IntersectionAddress, ...]:
    spans = seq.spans
    out: list[IntersectionAddress] = []
    for way in ways:
        if way not in (2, 3):
            continue
        for i in range(len(spans) - way + 1):
            chunk = spans[i : i + way]
            out.append(
                intersect_primes(
                    tuple(s.char for s in chunk),
                    tuple(s.prime for s in chunk),
                    start_index=chunk[0].index,
                    n=n,
                )
            )
    return tuple(out)


def subwords_from_text(text: str, *, n: int = DEFAULT_TRANSGRESSOR_N) -> tuple[IntersectionAddress, ...]:
    return subword_intersections(symbols_to_primes(text), n=n)


def find_intersection(text: str, label: str, *, n: int = DEFAULT_TRANSGRESSOR_N) -> IntersectionAddress | None:
    want = label.lower()
    for addr in subwords_from_text(text, n=n):
        if addr.label == want:
            return addr
    return None
