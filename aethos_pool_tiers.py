"""
Promotion pool tiering — L2 / L3 / SPECIES bands on PROMOTION_POOL.

Section 5: each allocated pool prime is a compression pin on the lattice.
Tier budgets stop L2 from exhausting L3/SPECIES capacity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PoolTier(str, Enum):
    L2_SUBWORD = "L2"
    L3_WORD = "L3"
    SPECIES = "SPECIES"


def pool_bands(pool_len: int) -> dict[PoolTier, tuple[int, int]]:
    """Split pool ~41% L2, ~41% L3, ~18% SPECIES (NUM/CODE/URL)."""
    l2_end = max(1, int(pool_len * 0.41))
    l3_end = max(l2_end + 1, int(pool_len * 0.82))
    return {
        PoolTier.L2_SUBWORD: (0, l2_end),
        PoolTier.L3_WORD: (l2_end, l3_end),
        PoolTier.SPECIES: (l3_end, pool_len),
    }


@dataclass(frozen=True)
class TierUsage:
    tier: PoolTier
    used: int
    capacity: int
    ratio: float
    warn: bool
    critical: bool

    def summary(self) -> str:
        flag = "OK"
        if self.critical:
            flag = "CRITICAL"
        elif self.warn:
            flag = "WARN"
        return f"{self.tier.value}: {self.used}/{self.capacity} ({self.ratio * 100:.0f}%) [{flag}]"


@dataclass
class PoolTierAllocator:
    """Per-tier cursor into PROMOTION_POOL slices."""

    pool_len: int
    cursors: dict[PoolTier, int] = field(default_factory=dict)
    _bands: dict[PoolTier, tuple[int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._bands:
            self._bands = pool_bands(self.pool_len)
        if not self.cursors:
            self.cursors = {t: 0 for t in PoolTier}

    def alloc(self, pool: tuple[int, ...], pool_tier: PoolTier) -> tuple[int, TierUsage]:
        start, end = self._bands[pool_tier]
        idx = self.cursors[pool_tier]
        if start + idx >= end:
            raise RuntimeError(f"promotion pool exhausted in tier {pool_tier.value}")
        prime = pool[start + idx]
        self.cursors[pool_tier] = idx + 1
        return prime, self.usage(pool_tier)

    def usage(self, pool_tier: PoolTier) -> TierUsage:
        start, end = self._bands[pool_tier]
        cap = end - start
        used = self.cursors[pool_tier]
        ratio = used / cap if cap else 1.0
        return TierUsage(
            tier=pool_tier,
            used=used,
            capacity=cap,
            ratio=ratio,
            warn=ratio >= 0.80,
            critical=ratio >= 0.95,
        )

    def all_usage(self) -> list[TierUsage]:
        return [self.usage(t) for t in PoolTier]

    def total_used(self) -> int:
        return sum(self.cursors.values())

    def warnings(self) -> list[str]:
        out: list[str] = []
        for u in self.all_usage():
            if u.critical:
                out.append(
                    f"{u.tier.value} pool {u.ratio * 100:.0f}% full — compression budget nearly exhausted"
                )
            elif u.warn:
                out.append(f"{u.tier.value} pool {u.ratio * 100:.0f}% full")
        return out
