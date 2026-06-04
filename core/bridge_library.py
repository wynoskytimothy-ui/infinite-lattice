"""
Adapter for aethos_library.PrimeAssigner (aethos13-ultrafast) when pasted.

Word-level freq-rank primes stay in the library; L2 morphology comes from core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.l2_subwords import SubwordPromoter, decompose, shared_l2_factors


@runtime_checkable
class PrimeAssignerLike(Protocol):
    """Minimal surface expected from aethos_library."""

    def get_word_prime(self, word: str) -> int: ...

    def get_pair_score(self, w1: str, w2: str) -> float: ...


@dataclass
class LibraryTokenEnrichment:
    word: str
    word_prime: int
    parent_primes: tuple[int, ...]
    shared_with: dict[str, tuple[int, ...]]


def l2_parent_primes_for_word(word: str, promoter: SubwordPromoter) -> tuple[int, ...]:
    """L2 pool primes found by n-gram scan (morphology under word prime)."""
    return decompose(word, promoter.l2_lookup)


def enrich_token_record(
    word: str,
    word_prime: int,
    parent_primes: tuple[int, ...],
) -> LibraryTokenEnrichment:
    return LibraryTokenEnrichment(
        word=word.lower(),
        word_prime=word_prime,
        parent_primes=parent_primes,
        shared_with={},
    )


def bridge_word_to_library(
    assigner: PrimeAssignerLike,
    word: str,
    promoter: SubwordPromoter,
) -> LibraryTokenEnrichment:
    """
    Combine library L3 prime with core L2 parent_primes for correlation scoring.
    """
    w = word.lower()
    wp = assigner.get_word_prime(w)
    parents = l2_parent_primes_for_word(w, promoter)
    if not parents:
        from core.l1_characters import word_letter_order

        parents = word_letter_order(w)
    return enrich_token_record(w, wp, parents)


def morphological_overlap(
    w1: str,
    w2: str,
    promoter: SubwordPromoter,
) -> tuple[int, ...]:
    return shared_l2_factors(w1, w2, promoter.l2_lookup)


def attach_parent_primes_to_token(token: Any, parent_primes: tuple[int, ...]) -> None:
    """Mutate a library token object if it supports parent_primes."""
    if hasattr(token, "parent_primes"):
        token.parent_primes = parent_primes
    elif isinstance(token, dict):
        token["parent_primes"] = list(parent_primes)


class StubPrimeAssigner:
    """Test double until aethos_library is pasted."""

    def __init__(self, word_primes: dict[str, int] | None = None) -> None:
        self._primes = word_primes or {}

    def get_word_prime(self, word: str) -> int:
        from core.l1_characters import intersection_prime

        return self._primes.get(word.lower(), intersection_prime(word))

    def get_pair_score(self, w1: str, w2: str) -> float:
        return float(len(morphological_overlap(w1, w2, self._promoter)) if hasattr(self, "_promoter") else 0.0)

    def bind_promoter(self, promoter: SubwordPromoter) -> None:
        self._promoter = promoter
