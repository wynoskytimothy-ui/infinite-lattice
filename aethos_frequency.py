"""
Frequency & rarity — everything mapped by how often it appears.

Common -> promotes (L2 sub-word, L3 word), heavier lattice weights, stronger clusters.
Rare -> stays at symbol level, low weight, may never get its own promoted prime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from aethos_promotion import LatticeTier, PromotionRegistry

if TYPE_CHECKING:
    from aethos_natural import NaturalReader


class RarityBand(Enum):
    VERY_COMMON = "very_common"   # top tier by frequency
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    VERY_RARE = "very_rare"       # seen once or never promoted


@dataclass
class TokenFrequency:
    text: str
    count: int
    rank: int
    percentile: float          # 0..1, 1 = most common
    idf: float                 # inverse document frequency style
    band: RarityBand
    tier_reached: LatticeTier | None  # highest promotion tier
    promoted_prime: int | None


@dataclass
class FrequencyProfile:
    """
    Snapshot of common vs rare across the lattice after reading.
    """

    registry: PromotionRegistry
    word_counts: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    total_windows: int = 0

    @classmethod
    def from_reader(cls, reader: NaturalReader) -> FrequencyProfile:
        fp = cls(
            registry=reader.registry,
            word_counts=dict(reader.graph.word_count),
            total_tokens=sum(reader.graph.word_count.values()),
            total_windows=reader.graph.window_count,
        )
        return fp

    def _sorted_words(self) -> list[tuple[str, int]]:
        return sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))

    def _band(self, percentile: float, count: int) -> RarityBand:
        if count <= 1:
            return RarityBand.VERY_RARE
        if percentile >= 0.9:
            return RarityBand.VERY_COMMON
        if percentile >= 0.7:
            return RarityBand.COMMON
        if percentile >= 0.4:
            return RarityBand.UNCOMMON
        if percentile >= 0.15:
            return RarityBand.RARE
        return RarityBand.VERY_RARE

    def _tier_for(self, word: str) -> tuple[LatticeTier | None, int | None]:
        w = word.lower()
        if (LatticeTier.L3_WORD, w) in self.registry.promoted:
            t = self.registry.promoted[(LatticeTier.L3_WORD, w)]
            return LatticeTier.L3_WORD, t.prime
        if w in self.registry.intersections:
            t = self.registry.intersections[w]
            return LatticeTier.L3_WORD, t.prime  # intersection prime, not pool
        if (LatticeTier.L2_SUBWORD, w) in self.registry.promoted:
            t = self.registry.promoted[(LatticeTier.L2_SUBWORD, w)]
            return LatticeTier.L2_SUBWORD, t.prime
        if len(w) == 1 and w.isalpha():
            return LatticeTier.L1_SYMBOL, self.registry.resolve_prime(w, LatticeTier.L1_SYMBOL)
        return None, None

    def profile(self, word: str) -> TokenFrequency:
        w = word.lower()
        count = self.word_counts.get(w, self.registry.word_counts.get(w, 0))
        ranked = self._sorted_words()
        n = len(ranked) or 1
        rank = next((i + 1 for i, (t, _) in enumerate(ranked) if t == w), n)
        percentile = 1.0 - (rank - 1) / max(n - 1, 1)
        # IDF: rare in windows -> high idf
        df = count / max(self.total_windows, 1)
        idf = math.log((self.total_windows + 1) / (count + 1)) + 1.0
        tier, prime = self._tier_for(w)
        return TokenFrequency(
            text=w,
            count=count,
            rank=rank,
            percentile=percentile,
            idf=idf,
            band=self._band(percentile, count),
            tier_reached=tier,
            promoted_prime=prime,
        )

    def most_common(self, limit: int = 10) -> list[TokenFrequency]:
        ranked = self._sorted_words()[:limit]
        return [self.profile(w) for w, _ in ranked]

    def rarest(self, limit: int = 10, min_seen: int = 1) -> list[TokenFrequency]:
        candidates = [(w, c) for w, c in self._sorted_words() if c >= min_seen]
        tail = candidates[-limit:] if len(candidates) >= limit else candidates
        tail = list(reversed(tail))
        return [self.profile(w) for w, _ in tail]

    def by_band(self, band: RarityBand) -> list[str]:
        return [w for w in self.word_counts if self.profile(w).band == band]

    def promotion_eligible_not_yet(self) -> list[str]:
        """Seen but not promoted to L3 — still rare side of the lattice."""
        out = []
        for w, c in self.word_counts.items():
            if (LatticeTier.L3_WORD, w) not in self.registry.promoted:
                if c < self.registry.word_promote_at:
                    out.append(w)
        return sorted(out, key=lambda w: self.word_counts[w])

    def explain(self, word: str) -> str:
        p = self.profile(word)
        w = word.lower()
        if self.registry.is_intersection_only(w):
            tier_name = "intersection_only (L7-L9 sufficient)"
        elif p.tier_reached:
            tier_name = p.tier_reached.name
        else:
            tier_name = "not seen"
        split = "yes" if self.registry.contexts_differ(w) else "no"
        return (
            f"{word!r}: count={p.count}, rank={p.rank}/{len(self.word_counts)}, "
            f"percentile={p.percentile:.2f}, band={p.band.value}, "
            f"idf={p.idf:.2f}, tier={tier_name}, prime={p.promoted_prime}, "
            f"contexts_differ={split}"
        )

    def summary(self) -> str:
        bands = {b: 0 for b in RarityBand}
        for w in self.word_counts:
            bands[self.profile(w).band] += 1
        promoted_l3 = sum(1 for k in self.registry.promoted if k[0] == LatticeTier.L3_WORD)
        promoted_l2 = sum(1 for k in self.registry.promoted if k[0] == LatticeTier.L2_SUBWORD)
        lines = [
            "FREQUENCY MAP",
            f"  unique tokens:   {len(self.word_counts)}",
            f"  total token hits:{self.total_tokens}",
            f"  windows read:    {self.total_windows}",
            f"  promoted L3:     {promoted_l3} (common words)",
            f"  promoted L2:     {promoted_l2} (common sub-words)",
            "  rarity bands:",
        ]
        for b in RarityBand:
            lines.append(f"    {b.value:12s} {bands[b]}")
        return "\n".join(lines)


def demo() -> None:
    from aethos_natural import NaturalReader

    reader = NaturalReader(rebuild_every=2)
    reader.read(
        "phone phone phone technical chip software",
        "phone technical hardware network",
        "apple phone phone chip technical",
        "apple fruit pie pie orchard",
        "apple fruit fruit salad",
        "banana fruit once",
        "xylophone obscure word",
    )

    fp = FrequencyProfile.from_reader(reader)

    print("=" * 60)
    print("FREQUENCY -> COMMON vs RARE (natural after reading)")
    print("=" * 60)
    print(fp.summary())

    print("\n--- Most common (promoted, heavy lattice weight) ---")
    for p in fp.most_common(6):
        print(f"  {fp.explain(p.text)}")

    print("\n--- Rarest seen tokens ---")
    for p in fp.rarest(5):
        print(f"  {fp.explain(p.text)}")

    print("\n--- Same letters, different frequency: phone vs xylophone ---")
    print(f"  {fp.explain('phone')}")
    print(f"  {fp.explain('xylophone')}")

    print("\n--- apple: common bridge (high count, dual cluster) ---")
    print(f"  {fp.explain('apple')}")
    c1, _ = reader.infer_cluster("apple", ["phone"])
    c2, _ = reader.infer_cluster("apple", ["fruit"])
    print(f"  context phone -> {c1}")
    print(f"  context fruit -> {c2}")


if __name__ == "__main__":
    demo()
