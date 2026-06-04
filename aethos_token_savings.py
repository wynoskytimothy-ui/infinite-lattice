"""
Token & storage savings from intersection-only policy.

Every word NOT promoted to a new pool prime = tokens saved.
Meaning is carried by:
  - L1 letter primes (shared, fixed 26)
  - float correlations L4-L9 (no extra prime per edge)
  - intersection address (derived, not allocated from pool)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aethos_promotion import LatticeTier, PromotionRegistry

if TYPE_CHECKING:
    from aethos_crossmeaning import MarkovCrossLattice
    from aethos_natural import NaturalReader


# Rough storage units for comparison (not bytes-exact, but proportional)
BYTES_DEDICATED_L3_TOKEN = 32   # word + pool prime + parents metadata in a vocab table
BYTES_INTERSECTION_ENTRY = 12    # word key + derived prime id (letters only)
BYTES_CORRELATION_EDGE = 24      # pair keys + 4 floats (d4,d5,d6,strength) + strength int
BYTES_NAIVE_L3_PER_WORD = 32     # if we promoted every unique word once


@dataclass
class TokenSavingsReport:
    unique_words: int = 0
    intersection_only_words: int = 0
    dedicated_l3_words: int = 0
    l2_subwords_promoted: int = 0
    correlation_edges: int = 0
    pool_primes_allocated: int = 0
    pool_primes_saved: int = 0

    float_correlation_values: int = 0  # count of float dims stored
    intersection_prime_reuse: int = 0  # words using letter-sum, no pool slot

    @classmethod
    def from_registry(
        cls,
        registry: PromotionRegistry,
        cross: MarkovCrossLattice | None = None,
    ) -> TokenSavingsReport:
        r = cls()
        r.unique_words = len(registry.word_counts)
        r.intersection_only_words = len(registry.intersections)
        r.dedicated_l3_words = sum(
            1 for k in registry.promoted if k[0] == LatticeTier.L3_WORD
        )
        r.l2_subwords_promoted = sum(
            1 for k in registry.promoted if k[0] == LatticeTier.L2_SUBWORD
        )
        r.correlation_edges = len(registry.correlations)
        r.pool_primes_allocated = registry._next_promotion_idx

        # Naive policy: one new pool prime per unique word
        r.pool_primes_saved = max(0, r.unique_words - r.dedicated_l3_words)

        r.float_correlation_values = r.correlation_edges * 3  # d4, d5, d6
        if cross:
            r.float_correlation_values += len(cross.categories) * 3  # L7-L9 dims
            for cat in cross.categories.values():
                r.float_correlation_values += len(cat.prime_weights)

        r.intersection_prime_reuse = r.intersection_only_words
        return r

    @property
    def tokens_saved_estimate(self) -> int:
        """Vocab slots not allocated (intersection + avoided naive promote-all)."""
        return self.pool_primes_saved

    @property
    def storage_intersection_bytes(self) -> int:
        return self.intersection_only_words * BYTES_INTERSECTION_ENTRY

    @property
    def storage_dedicated_bytes(self) -> int:
        return self.dedicated_l3_words * BYTES_DEDICATED_L3_TOKEN

    @property
    def storage_correlation_bytes(self) -> int:
        return self.correlation_edges * BYTES_CORRELATION_EDGE

    @property
    def storage_naive_all_promote_bytes(self) -> int:
        """One new pool prime per unique word (worst case)."""
        return self.unique_words * BYTES_NAIVE_L3_PER_WORD

    @property
    def storage_actual_prime_bytes(self) -> int:
        """Pool primes + intersection table only (no float graph)."""
        return self.storage_intersection_bytes + self.storage_dedicated_bytes

    @property
    def storage_actual_bytes(self) -> int:
        return self.storage_actual_prime_bytes + self.storage_correlation_bytes

    @property
    def prime_slots_saved_ratio(self) -> float:
        """Share of vocabulary that did NOT take a dedicated pool prime."""
        if self.unique_words == 0:
            return 0.0
        return self.pool_primes_saved / self.unique_words

    @property
    def savings_ratio(self) -> float:
        """Prime-slot savings vs promote-every-word (ignores float graph)."""
        naive = self.storage_naive_all_promote_bytes
        if naive == 0:
            return 0.0
        return 1.0 - (self.storage_actual_prime_bytes / naive)

    @property
    def naive_prime_pairs_if_graph(self) -> int:
        """If each L4-L6 edge were two new primes instead of one float edge."""
        return self.correlation_edges * 2

    @property
    def correlation_vs_prime_pair_savings(self) -> float:
        """Fraction of hypothetical prime-pair slots replaced by floats."""
        n = self.naive_prime_pairs_if_graph
        if n == 0:
            return 0.0
        # one float edge vs two prime slots
        return 1.0 - (self.correlation_edges / n)

    def summary(self) -> str:
        return f"""
TOKEN SAVINGS (intersection policy vs promote-every-word)
=========================================================
  Unique words read:           {self.unique_words}
  Intersection-only (no pool): {self.intersection_only_words}  <- tokens saved
  Dedicated L3 (split meaning):{self.dedicated_l3_words}
  Pool primes allocated:       {self.pool_primes_allocated}
  Pool primes NOT allocated:   {self.pool_primes_saved}  (saved)

CORRELATIONS WITHOUT NEW PRIMES
  L4-L6 edges:                 {self.correlation_edges}
  Float values stored:         {self.float_correlation_values}  (dims 4-9, weights)
  => relationships are floats, not new primes per link

PRIME SLOTS (main token savings)
  Naive (1 pool prime / word):  {self.unique_words} slots
  Dedicated L3 only:            {self.dedicated_l3_words} slots
  Intersection (no pool):       {self.intersection_only_words} words
  Prime slots saved:            {self.pool_primes_saved} ({self.prime_slots_saved_ratio * 100:.0f}%)

STORAGE (relative units, primes only)
  Naive all-promote:            {self.storage_naive_all_promote_bytes}
  Actual prime footprint:       {self.storage_actual_prime_bytes}
  Prime storage saved:          {self.savings_ratio * 100:.1f}%

FLOAT GRAPH (correlations, not primes)
  L4-L6 edges as floats:        {self.correlation_edges} edges
  vs naive 2-primes per edge:   would be {self.naive_prime_pairs_if_graph} slots
  Correlation uses floats:      {self.correlation_vs_prime_pair_savings * 100:.0f}% fewer prime slots on graph

WHY IT HELPS WHEN BUILDING TOKENS
  - Rare / once words: 1 intersection entry, L7-L9 float correlations only
  - No wasted prime slot in vocabulary per hapax legomenon
  - Common words with 2 meanings: 1 dedicated prime only when contexts differ
  - Co-occurrence = cheap floats; prime pool reserved for true splits
""".strip()


def report_from_reader(reader: NaturalReader) -> TokenSavingsReport:
    return TokenSavingsReport.from_registry(reader.registry, reader.cross)


def demo() -> None:
    from aethos_natural import NaturalReader

    reader = NaturalReader(rebuild_every=2)
    reader.read(
        "phone phone phone technical chip software",
        "phone technical hardware network",
        "apple phone chip technical",
        "apple fruit pie orchard",
        "xylophone obscure rareword",
        "banana fruit once",
        "zebra unique animal",
    )

    rep = report_from_reader(reader)
    print("=" * 60)
    print(rep.summary())

    print("\n--- Per-word: intersection vs dedicated ---")
    for w in sorted(reader.registry.word_counts, key=lambda x: -reader.registry.word_counts[x]):
        if reader.registry.is_intersection_only(w):
            kind = "intersection (saved)"
        elif (LatticeTier.L3_WORD, w) in reader.registry.promoted:
            kind = "dedicated L3 prime"
        else:
            kind = "letters only"
        print(f"  {w:12s} count={reader.registry.word_counts[w]:3d}  {kind}")


if __name__ == "__main__":
    demo()
