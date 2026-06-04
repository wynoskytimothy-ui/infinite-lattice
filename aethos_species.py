"""
Token species — NUM, CODE, URL anchors on alternate countable chains.

Section 5 / C6: observation pins a definite anchor on the lattice.
  WORD  -> odd-prime letter chain (L1)
  NUM   -> even-chain digit anchors (per-digit, arithmetic-friendly)
  CODE  -> reserved band (future)

Literature: per-digit tokenization improves numeric reasoning vs arbitrary
subword merges (Nogueira et al.; GPT cl100k 1–3 digit chunks). We use
explicit digit primes on SequenceKind.EVENS so place-value order is preserved
in parent_primes (left-to-right read order).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aethos_sequences import SequenceKind, make_chain

# Digits 0-9 -> first ten evens (2,4,6,...,20) — distinct from letter odd primes.
DIGIT_PRIMES: tuple[int, ...] = make_chain(SequenceKind.EVENS, 10)


class TokenSpecies(str, Enum):
    WORD = "WORD"
    NUM = "NUM"
    CODE = "CODE"
    URL = "URL"
    PUNCT = "PUNCT"


def digit_to_prime(d: str) -> int:
    if len(d) != 1 or not d.isdigit():
        raise ValueError(f"single digit required: {d!r}")
    return DIGIT_PRIMES[int(d)]


def prime_to_digit(p: int) -> str:
    idx = DIGIT_PRIMES.index(p)
    return str(idx)


def digit_chain(num: str) -> tuple[int, ...]:
    """Left-to-right digit anchors (preserves place-value order for formula_coord)."""
    return tuple(digit_to_prime(d) for d in num if d.isdigit())


def number_intersection(num: str) -> int:
    """Intersection anchor = sum of digit primes (parallel to letter intersection)."""
    return sum(digit_chain(num))


def is_numeric_token(text: str) -> bool:
    return bool(text) and text.isdigit()


@dataclass(frozen=True)
class SpeciesToken:
    text: str
    species: TokenSpecies
    prime: int
    parent_primes: tuple[int, ...]
    raw: str = ""

    @classmethod
    def from_number(cls, num: str, *, raw: str = "") -> SpeciesToken:
        chain = digit_chain(num)
        return cls(
            text=num,
            species=TokenSpecies.NUM,
            prime=number_intersection(num),
            parent_primes=chain,
            raw=raw or num,
        )
