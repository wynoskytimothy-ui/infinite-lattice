"""
Stage 04 — Promote frequent meets to new L2 primes.

Frequent subword meets (stage 02 intersections) earn a dedicated pool prime.
Append-only: existing letter-prime meets still resolve; nothing relocates.

Reference primitive: aethos_promotion.PromotionRegistry (L2_SUBWORD tier).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from aethos_promotion import (
    PROMOTION_POOL,
    LatticeTier,
    PromotedToken,
    PromotionRegistry,
    intersection_prime,
)

from lattice_retriever_v1.stage02_intersections import (
    IntersectionAddress,
    find_intersection,
    intersect_three,
    intersect_two,
)

MIN_POOL_PRIME = PROMOTION_POOL[0]


@dataclass(frozen=True)
class PromotionRecord:
    """One L2 promotion event — glass-box unit for stage 04."""

    text: str
    prime: int
    parent_primes: tuple[int, ...]
    count: int
    parent_words: tuple[str, ...]
    letter_intersection_prime: int
    letter_meet: IntersectionAddress

    @property
    def is_pool_promoted(self) -> bool:
        return self.prime >= MIN_POOL_PRIME

    def explain(self) -> dict:
        return {
            "text": self.text,
            "tier": "L2_SUBWORD",
            "prime": self.prime,
            "parent_primes": list(self.parent_primes),
            "count": self.count,
            "parent_words": list(self.parent_words),
            "letter_intersection_prime": self.letter_intersection_prime,
            "letter_meet": self.letter_meet.explain(),
            "pool_promoted": self.is_pool_promoted,
            "append_only": True,
        }


@dataclass
class Stage04Registry:
    """
    Corpus ingest → L2 promotions on frequent subword meets.

    Wraps PromotionRegistry with v1 glass-box records. Letter-level meets
    from stage 02 remain valid after promotion (Hilbert hotel: nobody moves).

    L2 promotion gate (via PromotionRegistry._should_promote_l2):
      - raw subword count >= subword_promote_at, AND
      - distinct parent words >= subword_min_parents (default 2), AND
      - PMI / cohesion thresholds across those parents.
    One spammy document repeating the same word does not satisfy distinct parents.
    """

    registry: PromotionRegistry = field(default_factory=PromotionRegistry)
    promotions: tuple[PromotionRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.registry.defer_l2_promotion:
            # v1: flush after each observe_text call via observe_text below
            pass

    def _letter_meet(self, sw: str) -> IntersectionAddress:
        """Stage 02 witness for this subword (2- or 3-way; 4-char uses leading pair)."""
        sw = sw.lower()
        if len(sw) == 2:
            return intersect_two(sw[0], sw[1])
        if len(sw) == 3:
            return intersect_three(sw[0], sw[1], sw[2])
        if len(sw) == 4:
            # Stage 02 gate is 2/3-way only; pair witness on first two symbols
            return intersect_two(sw[0], sw[1])
        raise ValueError(f"subword length {len(sw)} outside stage 02 meet window")

    def _record_for(self, tok: PromotedToken) -> PromotionRecord:
        sw = tok.text.lower()
        parents = tuple(sorted(self.registry.subword_parent_words.get(sw, set())))
        return PromotionRecord(
            text=sw,
            prime=tok.prime,
            parent_primes=tok.parent_primes,
            count=self.registry.subword_counts.get(sw, 0),
            parent_words=parents,
            letter_intersection_prime=intersection_prime(sw),
            letter_meet=self._letter_meet(sw),
        )

    def observe_text(self, text: str) -> tuple[PromotionRecord, ...]:
        """Ingest one text span; return newly promoted L2 subwords (if any)."""
        before = set(self.registry.promoted.keys())
        self.registry.observe_text(text)
        new_keys = [
            k for k in self.registry.promoted
            if k not in before and k[0] == LatticeTier.L2_SUBWORD
        ]
        new_records = tuple(
            self._record_for(self.registry.promoted[k]) for k in sorted(new_keys)
        )
        if new_records:
            self.promotions = self.promotions + new_records
        return new_records

    def observe_stream(self, texts: Iterable[str]) -> tuple[PromotionRecord, ...]:
        """Ingest many documents; return all new promotions in order."""
        out: list[PromotionRecord] = []
        for text in texts:
            out.extend(self.observe_text(text))
        return tuple(out)

    def promoted_subword(self, sw: str) -> PromotedToken | None:
        key = (LatticeTier.L2_SUBWORD, sw.lower())
        return self.registry.promoted.get(key)

    def resolve_subword(self, sw: str) -> dict:
        """
        Glass-box resolution: promoted pool prime if available, else letter intersection.
        Stage 02 letter meet always included for audit.
        """
        sw = sw.lower()
        tok = self.promoted_subword(sw)
        letter = self._letter_meet(sw)
        return {
            "text": sw,
            "promoted": tok is not None,
            "prime": tok.prime if tok else intersection_prime(sw),
            "tier": "L2_SUBWORD" if tok else "L1_INTERSECTION",
            "parent_primes": list(tok.parent_primes if tok else letter.primes),
            "letter_meet": letter.explain(),
            "count": self.registry.subword_counts.get(sw, 0),
        }

    def letter_intersection_unchanged(self, sw: str, *, baseline: IntersectionAddress) -> bool:
        """After promotion, stage 02 letter meet signature is identical."""
        current = self._letter_meet(sw)
        return (
            current.primes == baseline.primes
            and current.lattice_coords == baseline.lattice_coords
            and current.composite == baseline.composite
        )

    def explain(self) -> dict:
        return {
            "n_promotions": len(self.promotions),
            "promotions": [p.explain() for p in self.promotions],
            "subword_counts_sample": dict(
                sorted(self.registry.subword_counts.items())[:20]
            ),
        }


def promote_from_stream(
    texts: Iterable[str],
    *,
    subword_promote_at: int = 2,
    subword_min_parents: int = 2,
) -> Stage04Registry:
    """Convenience: build registry from a text stream with v1 promotion defaults."""
    reg = PromotionRegistry(
        subword_promote_at=subword_promote_at,
        subword_min_parents=subword_min_parents,
        defer_l2_promotion=True,
    )
    stage = Stage04Registry(registry=reg)
    stage.observe_stream(texts)
    return stage


def find_letter_intersection_in_doc(text: str, label: str) -> IntersectionAddress | None:
    """Stage 02 lookup — still valid after stage 04 promotion."""
    return find_intersection(text, label)
