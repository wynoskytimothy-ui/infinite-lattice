"""
Stage 06 — Multi-way composites (Section 5 anchor-pair + transgressor).

Section 5 structure (locked):
  - Two anchors a ≤ p define the corridor (first two symbols, read-order).
  - Third prime n is the transgressor (third symbol read-order).
  - Case 1/2/3 = where n falls relative to a, p (prime_pair_case).

When a pool prime (Stage 04 promotion) is present in a triple, it is ALWAYS n —
the arriving promoted atom transgresses; the two letter anchors define the corridor.

Identity (Stage 05 inheritance):
  - meet_composite = product of k DISTINCT primes (order-free).
  - anchor_sum = a+p+n is placement metadata only — never identity.
  - quadrant + read_order carry orientation outside the product.

Repeated primes:
  - k-way identity requires k distinct factors; repeats raise RepeatedPrimeError.
  - Route doubles to Stage 02 pair meet on min/max distinct (explicit, not silent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Iterable

from aethos_lattice import prime_pair_canon, prime_pair_case
from aethos_words import letter_to_prime

from lattice_retriever_v1.stage01_symbols import symbols_to_primes
from lattice_retriever_v1.stage02_intersections import (
    DEFAULT_TRANSGRESSOR_N,
    IntersectionAddress,
    intersect_two,
    lattice_signature,
)
from lattice_retriever_v1.stage04_promote import MIN_POOL_PRIME, Stage04Registry
from lattice_retriever_v1.stage05_free_token import canonical_pair, is_prime

CorridorKey = tuple[int, int, int]  # (meet_composite, quadrant, transgressor_n)


class RepeatedPrimeError(ValueError):
    """Raised when a k-way meet would use a repeated prime factor."""


def _prime_factorization(n: int) -> dict[int, int]:
    if n < 2:
        raise ValueError(f"cannot factor {n}")
    out: dict[int, int] = {}
    x = n
    d = 2
    while d * d <= x:
        while x % d == 0:
            out[d] = out.get(d, 0) + 1
            x //= d
        d = 3 if d == 2 else d + 2
    if x > 1:
        out[x] = out.get(x, 0) + 1
    return out


def meet_composite_k(*primes: int) -> int:
    """Order-free k-way meet identity — product of distinct primes only."""
    if len(primes) < 2:
        raise ValueError(f"k-way meet needs at least 2 primes, got {len(primes)}")
    if len(set(primes)) != len(primes):
        raise RepeatedPrimeError(f"repeated prime in k-way meet: {primes}")
    for p in primes:
        if not is_prime(p):
            raise ValueError(f"non-prime factor {p}")
    return reduce(mul, sorted(primes), 1)


def factor_k_composite(composite: int, k: int) -> tuple[int, ...]:
    """
    Exact factor-back: recover exactly k distinct prime factors.

    Rejects squared factors and wrong factor count (generalizes Stage 05 semiprime gate).
    """
    factors = _prime_factorization(composite)
    if len(factors) != k or any(e != 1 for e in factors.values()):
        raise ValueError(f"composite is not a product of {k} distinct primes: {composite}")
    return tuple(sorted(factors))


def pool_prime_in(primes: Iterable[int]) -> int | None:
    """Return the pool-promoted prime in this set, if exactly one."""
    pools = [p for p in primes if p >= MIN_POOL_PRIME]
    if len(pools) == 1:
        return pools[0]
    if len(pools) > 1:
        raise ValueError(f"multiple pool primes in triple: {pools}")
    return None


def section5_triple_roles(
    read_order: tuple[int, int, int],
    *,
    pool_transgressor: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Assign Section 5 roles (a, p, n, case) from read-order triple.

    Rule (locked):
      - Default: first two symbols → anchors (a≤p), third → transgressor n.
      - If pool prime present: pool prime is ALWAYS n; remaining two → a≤p.
    """
    if len(read_order) != 3:
        raise ValueError(f"Section 5 triple needs 3 primes, got {read_order}")
    if len(set(read_order)) != 3:
        raise RepeatedPrimeError(f"triple roles require distinct primes: {read_order}")

    pool = pool_transgressor if pool_transgressor is not None else pool_prime_in(read_order)
    if pool is not None:
        n = pool
        rest = tuple(x for x in read_order if x != pool)
        if len(rest) != 2:
            raise ValueError(f"pool transgressor {pool} not in read_order {read_order}")
        a, p = canonical_pair(rest[0], rest[1])
    else:
        a, p = canonical_pair(read_order[0], read_order[1])
        n = read_order[2]
    return a, p, n, prime_pair_case(a, p, n)


def section5_va1a_coord(a: int, p: int, n: int) -> tuple[int, int, int]:
    """Section 5 VA1A placement row — coordinate only, not identity."""
    return prime_pair_canon("VA1A", a, p, n)


@dataclass(frozen=True)
class ThreeWayMeetAddress:
    """Section 5 anchor-pair + transgressor node (k=3)."""

    read_order: tuple[int, int, int]
    a: int
    p: int
    n: int
    case: int
    quadrant: int
    transgressor_n: int

    @property
    def meet_composite(self) -> int:
        return meet_composite_k(*self.read_order)

    @property
    def anchor_sum(self) -> int:
        """Placement metadata (a+p+n) — degenerate across distinct triples; not identity."""
        return self.a + self.p + self.n

    @property
    def lattice_signature(self) -> tuple[tuple[int, int, int], ...]:
        return lattice_signature(self.read_order, n=self.transgressor_n)

    @property
    def section5_coord(self) -> tuple[int, int, int]:
        return section5_va1a_coord(self.a, self.p, self.n)

    @property
    def corridor_key(self) -> CorridorKey:
        return (self.meet_composite, self.quadrant, self.transgressor_n)

    def explain(self) -> dict:
        return {
            "way": 3,
            "read_order": list(self.read_order),
            "section5": {"a": self.a, "p": self.p, "n": self.n, "case": self.case},
            "meet_composite": self.meet_composite,
            "anchor_sum": self.anchor_sum,
            "identity_is_product": True,
            "anchor_sum_is_metadata": True,
            "section5_VA1A": self.section5_coord,
            "quadrant": self.quadrant,
            "corridor_key": list(self.corridor_key),
            "lattice_L01": self.lattice_signature[0],
            "n_lattices": len(self.lattice_signature),
            "stored_row": False,
        }


def three_way_address(
    p1: int,
    p2: int,
    p3: int,
    *,
    quadrant: int = 1,
    transgressor_n: int = DEFAULT_TRANSGRESSOR_N,
    read_order: tuple[int, int, int] | None = None,
    pool_transgressor: int | None = None,
) -> ThreeWayMeetAddress:
    """Build 3-way node from three primes — computed, not stored."""
    order = read_order if read_order is not None else (p1, p2, p3)
    a, p, n, case = section5_triple_roles(order, pool_transgressor=pool_transgressor)
    qid = max(1, min(32, int(quadrant)))
    return ThreeWayMeetAddress(
        read_order=order,
        a=a,
        p=p,
        n=n,
        case=case,
        quadrant=qid,
        transgressor_n=transgressor_n,
    )


def regenerate_three_way_from_composite(
    composite: int,
    *,
    read_order: tuple[int, int, int],
    quadrant: int,
    transgressor_n: int = DEFAULT_TRANSGRESSOR_N,
    pool_transgressor: int | None = None,
) -> ThreeWayMeetAddress:
    """Factor composite → primes; rebuild Section 5 node (no registry row)."""
    recovered = factor_k_composite(composite, 3)
    if set(recovered) != set(read_order):
        raise ValueError(f"read_order {read_order} != factors {recovered}")
    return three_way_address(
        *recovered,
        quadrant=quadrant,
        transgressor_n=transgressor_n,
        read_order=read_order,
        pool_transgressor=pool_transgressor,
    )


def repeated_pair_fallback(text: str) -> IntersectionAddress:
    """Route repeated-letter chains to 2-way meet on distinct min/max primes."""
    seq = symbols_to_primes(text)
    primes = tuple(s.prime for s in seq.spans)
    if len(set(primes)) == len(primes):
        raise ValueError(f"no repeated prime in {text!r}")
    lo, hi = min(primes), max(primes)
    c_lo = next(s.char for s in seq.spans if s.prime == lo)
    c_hi = next(s.char for s in seq.spans if s.prime == hi)
    return intersect_two(c_lo, c_hi)


def letter_triple_from_text(text: str) -> tuple[int, int, int]:
    seq = symbols_to_primes(text)
    if len(seq.spans) != 3:
        raise ValueError(f"need exactly 3 symbols, got {len(seq.spans)}")
    return tuple(s.prime for s in seq.spans)


def _greedy_promoted_pieces(word: str, registry: Stage04Registry) -> list[tuple[str, int]]:
    """Left-to-right shortest-match promoted subwords (prefer th over thin at same start)."""
    w = word.lower()
    out: list[tuple[str, int]] = []
    i = 0
    min_len = registry.registry.subword_min_len
    max_len = registry.registry.subword_max_len
    while i < len(w):
        matched: tuple[str, int] | None = None
        for length in range(min_len, min(max_len, len(w) - i) + 1):
            piece = w[i : i + length]
            tok = registry.promoted_subword(piece)
            if tok is not None:
                matched = (piece, tok.prime)
                break
        if matched is None:
            i += 1
            continue
        out.append(matched)
        i += len(matched[0])
    return out


def decompose_word(
    word: str,
    registry: Stage04Registry | None = None,
) -> dict:
    """
    Glass-box word decomposition — promoted pieces or letter path.

    Returns meet_composite (product identity) and Section 5 triple when k=3.
    """
    w = word.lower()
    letters = symbols_to_primes(w)
    letter_primes = letters.primes

    promoted_pieces: list[tuple[str, int]] = []
    if registry is not None:
        promoted_pieces = _greedy_promoted_pieces(w, registry)

    if len(w) == 3 and len(set(letter_primes)) == 3:
        addr = three_way_address(*letter_primes, read_order=letter_primes)
        return {
            "word": w,
            "path": "letter_three_way",
            "constituents": list(letter_primes),
            "meet_composite": addr.meet_composite,
            "section5": addr.explain()["section5"],
        }

    if promoted_pieces:
        primes = tuple(p for _, p in promoted_pieces)
        if len(set(primes)) == len(primes):
            comp = meet_composite_k(*primes)
            return {
                "word": w,
                "path": "promoted_pieces",
                "pieces": [t for t, _ in promoted_pieces],
                "constituents": list(primes),
                "meet_composite": comp,
            }

    if len(set(letter_primes)) == len(letter_primes):
        comp = meet_composite_k(*letter_primes)
        return {
            "word": w,
            "path": "letter_product",
            "constituents": list(letter_primes),
            "meet_composite": comp,
        }

    return {
        "word": w,
        "path": "repeated_letter_pair_fallback",
        "constituents": list(letter_primes),
        "note": "repeated prime — use 2-way distinct pair, not k-way product identity",
    }


def ing_three_way_demo(*, quadrant: int = 1) -> ThreeWayMeetAddress:
    """
    Walk "ing" = {i,n,g} as Section 5 node — read-order anchors + transgressor.

    Letter primes (i=29, n=47, g=19): anchors a=29,p=47, transgressor n=19, Case 1.
    Product identity 19×29×47; anchor_sum 95 is metadata only.
    """
    read = (
        letter_to_prime("i"),
        letter_to_prime("n"),
        letter_to_prime("g"),
    )
    return three_way_address(*read, read_order=read, quadrant=quadrant)
