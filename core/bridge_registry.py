"""
Bridge core L2 morphology into PromotionRegistry (production path).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.l2_subwords import (
    SubwordConfig,
    SubwordStats,
    build_stats_from_vocab,
    should_promote_l2,
    max_subword_pmi,
)

if TYPE_CHECKING:
    from aethos_promotion import PromotionRegistry


def stats_from_registry(registry: PromotionRegistry) -> SubwordStats:
    """Copy registry subword fields into core SubwordStats."""
    cfg = SubwordConfig(
        subword_min_len=registry.subword_min_len,
        subword_max_len=registry.subword_max_len,
        subword_promote_at=registry.subword_promote_at,
        subword_min_parents=registry.subword_min_parents,
        subword_min_pmi=registry.subword_min_pmi,
        subword_min_z=registry.subword_min_z,
    )
    stats = SubwordStats(
        config=cfg,
        subword_counts=dict(registry.subword_counts),
        subword_parent_words={k: set(v) for k, v in registry.subword_parent_words.items()},
        subword_parent_pairs=dict(registry.subword_parent_pairs),
        word_counts=dict(registry.word_counts),
        word_observations=registry.word_observations,
    )
    return stats


def ranked_l2_candidates(registry: PromotionRegistry) -> list[tuple[float, str]]:
    stats = stats_from_registry(registry)
    candidates: list[tuple[float, str]] = []
    for sw in registry.subword_counts:
        if should_promote_l2(stats, sw):
            candidates.append((max_subword_pmi(stats, sw), sw))
    candidates.sort(key=lambda x: -x[0])
    return candidates


def sync_l2_to_registry(
    registry: PromotionRegistry,
    *,
    max_promote: int = 160,
    rebuild_stats: bool = True,
) -> int:
    """
    Promote top PMI L2 subwords using core eligibility; registry allocates primes.

    Returns number of newly promoted L2 entries.
    """
    from aethos_promotion import LatticeTier
    from aethos_subword_composite import rebuild_subword_stats_from_vocab

    if rebuild_stats:
        rebuild_subword_stats_from_vocab(registry)

    l2_before = sum(1 for k in registry.promoted if k[0] == LatticeTier.L2_SUBWORD)

    for _, sw in ranked_l2_candidates(registry)[:max_promote]:
        key = (LatticeTier.L2_SUBWORD, sw)
        if key in registry.promoted:
            continue
        try:
            registry._promote(LatticeTier.L2_SUBWORD, sw)
        except RuntimeError:
            break

    l2_after = sum(1 for k in registry.promoted if k[0] == LatticeTier.L2_SUBWORD)
    return l2_after - l2_before


def refresh_registry_l3_parents(registry: PromotionRegistry) -> int:
    from aethos_subword_composite import refresh_l3_parent_primes

    return refresh_l3_parent_primes(registry)


def run_core_l2_pass(
    registry: PromotionRegistry,
    *,
    max_promote: int = 160,
) -> tuple[int, int]:
    """Promote L2 via core rules + refresh L3 parent_primes. Returns (new_l2, refreshed_l3)."""
    new_l2 = sync_l2_to_registry(registry, max_promote=max_promote)
    refreshed = refresh_registry_l3_parents(registry)
    return new_l2, refreshed
