"""
L2 subword layer (Step 4) — PMI-promoted n-grams share pool primes across related words.

Pure module: no imports from aethos_promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

from core.l1_characters import intersection_prime, word_letter_order
from core.primes import PrimePool, PoolTier, product_unique

# Mirror aethos_promotion.STOPWORDS (subset used for L2 gating)
STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "must", "not", "no", "so", "if", "then", "than", "that",
        "this", "these", "those", "it", "its", "into", "about", "called", "new",
        "runs", "released",
    }
)


def is_stopword(word: str) -> bool:
    return word.lower() in STOPWORDS


@dataclass
class SubwordConfig:
    subword_min_len: int = 2
    subword_max_len: int = 4
    subword_promote_at: int = 2
    subword_min_parents: int = 2
    subword_min_pmi: float = 0.25
    subword_min_z: float = 1.0


@dataclass
class SubwordStats:
    """Corpus subword statistics for PMI / z scoring."""

    config: SubwordConfig = field(default_factory=SubwordConfig)
    subword_counts: dict[str, int] = field(default_factory=dict)
    subword_parent_words: dict[str, set[str]] = field(default_factory=dict)
    subword_parent_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    word_counts: dict[str, int] = field(default_factory=dict)
    word_observations: int = 0

    def observe_word(self, word: str, *, count: int = 1) -> None:
        w = word.lower()
        if not w.isalpha() or len(w) < self.config.subword_min_len:
            return
        self.word_counts[w] = self.word_counts.get(w, 0) + count
        self.word_observations += 1
        min_len = self.config.subword_min_len
        max_len = self.config.subword_max_len
        seen: set[str] = set()
        for length in range(min_len, min(max_len + 1, len(w) + 1)):
            for i in range(len(w) - length + 1):
                sw = w[i : i + length]
                self.subword_counts[sw] = self.subword_counts.get(sw, 0) + 1
                seen.add(sw)
        for sw in seen:
            self.subword_parent_words.setdefault(sw, set()).add(w)
            key = (sw, w)
            self.subword_parent_pairs[key] = (
                self.subword_parent_pairs.get(key, 0) + count
            )

    def observe_corpus(self, words: list[str]) -> None:
        for w in words:
            self.observe_word(w, count=1)

    def observe_text_corpus(self, texts: list[str]) -> None:
        for text in texts:
            for raw in text.lower().split():
                w = "".join(c for c in raw if c.isalpha())
                if w:
                    self.observe_word(w, count=1)


def score_pmi(stats: SubwordStats, sw: str, parent: str) -> float:
    sw, parent = sw.lower(), parent.lower()
    n = max(stats.word_observations, 1)
    n_pair = stats.subword_parent_pairs.get((sw, parent), 0)
    n_sw = sum(
        stats.subword_parent_pairs.get((sw, p), 0)
        for p in stats.subword_parent_words.get(sw, ())
    )
    n_p = stats.word_counts.get(parent, 0)
    if n_pair == 0 or n_sw == 0 or n_p == 0:
        return 0.0
    p_pair = n_pair / n
    p_sw = n_sw / n
    p_p = n_p / n
    if p_pair * p_sw * p_p == 0:
        return 0.0
    return math.log2(p_pair / (p_sw * p_p))


def score_z(stats: SubwordStats, sw: str, parent: str) -> float:
    sw, parent = sw.lower(), parent.lower()
    n = max(stats.word_observations, 1)
    c_xy = stats.subword_parent_pairs.get((sw, parent), 0)
    c_x = sum(
        stats.subword_parent_pairs.get((sw, p), 0)
        for p in stats.subword_parent_words.get(sw, ())
    )
    c_y = stats.word_counts.get(parent, 0)
    if c_xy == 0 or c_x == 0 or c_y == 0:
        return 0.0
    expected = (c_x * c_y) / n
    if expected <= 0:
        return 0.0
    return (c_xy - expected) / math.sqrt(expected + 1e-9)


def max_subword_pmi(stats: SubwordStats, sw: str) -> float:
    parents = stats.subword_parent_words.get(sw.lower(), set())
    if not parents:
        return 0.0
    return max(score_pmi(stats, sw, p) for p in parents)


def max_subword_z(stats: SubwordStats, sw: str) -> float:
    parents = stats.subword_parent_words.get(sw.lower(), set())
    if not parents:
        return 0.0
    return max(score_z(stats, sw, p) for p in parents)


def should_promote_l2(stats: SubwordStats, sw: str, cfg: SubwordConfig | None = None) -> bool:
    cfg = cfg or stats.config
    sw = sw.lower()
    if is_stopword(sw):
        return False
    count = stats.subword_counts.get(sw, 0)
    if count < cfg.subword_promote_at:
        return False
    parents = stats.subword_parent_words.get(sw, set())
    if sw in parents and stats.word_counts.get(sw, 0) >= cfg.subword_promote_at:
        return True
    if len(parents) < cfg.subword_min_parents:
        return False
    pmis = [score_pmi(stats, sw, p) for p in parents]
    zs = [score_z(stats, sw, p) for p in parents]
    strong_pmi = sum(1 for x in pmis if x >= cfg.subword_min_pmi)
    strong_z = sum(1 for z in zs if z >= cfg.subword_min_z)
    if strong_z >= cfg.subword_min_parents:
        return True
    if max(zs, default=0.0) >= cfg.subword_min_z * 1.5:
        return True
    return (
        strong_pmi >= cfg.subword_min_parents
        or max(pmis, default=0.0) >= cfg.subword_min_pmi * 1.5
    )


def enumerate_ngrams(word: str, min_len: int, max_len: int) -> list[str]:
    w = word.lower()
    out: list[str] = []
    for length in range(min_len, min(max_len + 1, len(w) + 1)):
        for i in range(len(w) - length + 1):
            out.append(w[i : i + length])
    return out


def decompose(word: str, l2_lookup: dict[str, int], *, min_len: int = 2, max_len: int = 4) -> tuple[int, ...]:
    """N-gram scan: all promoted L2 pool primes present in word."""
    found: set[int] = set()
    for sw in enumerate_ngrams(word, min_len, max_len):
        if sw in l2_lookup:
            found.add(l2_lookup[sw])
    return tuple(sorted(found))


def shared_l2_factors(
    w1: str,
    w2: str,
    l2_lookup: dict[str, int],
    *,
    min_len: int = 2,
    max_len: int = 4,
) -> tuple[int, ...]:
    a = set(decompose(w1, l2_lookup, min_len=min_len, max_len=max_len))
    b = set(decompose(w2, l2_lookup, min_len=min_len, max_len=max_len))
    return tuple(sorted(a & b))


@dataclass
class PromotedL2:
    text: str
    prime: int
    parent_primes: tuple[int, ...]


@dataclass
class SubwordPromoter:
    stats: SubwordStats
    pool: PrimePool = field(default_factory=PrimePool)
    l2_lookup: dict[str, int] = field(default_factory=dict)
    promoted: dict[str, PromotedL2] = field(default_factory=dict)

    def parent_primes_for_subword(self, sw: str) -> tuple[int, ...]:
        return word_letter_order(sw)

    def promote_one(self, sw: str) -> PromotedL2 | None:
        sw = sw.lower()
        if sw in self.promoted:
            return self.promoted[sw]
        if not should_promote_l2(self.stats, sw):
            return None
        try:
            prime = self.pool.alloc(PoolTier.L2_SUBWORD)
        except RuntimeError:
            return None
        parents = self.parent_primes_for_subword(sw)
        tok = PromotedL2(text=sw, prime=prime, parent_primes=parents)
        self.promoted[sw] = tok
        self.l2_lookup[sw] = prime
        return tok

    def ranked_candidates(self) -> list[tuple[float, str]]:
        candidates: list[tuple[float, str]] = []
        for sw in self.stats.subword_counts:
            if should_promote_l2(self.stats, sw):
                candidates.append((max_subword_pmi(self.stats, sw), sw))
        candidates.sort(key=lambda x: -x[0])
        return candidates

    def promote_top(self, max_promote: int = 160) -> int:
        before = len(self.promoted)
        for _, sw in self.ranked_candidates()[:max_promote]:
            self.promote_one(sw)
            if self.pool.l2_remaining() <= 0:
                break
        return len(self.promoted) - before

    def l2_composite(self, word: str) -> int | None:
        """Product of two+ L2 primes in word (FTA unique composite)."""
        primes = decompose(word, self.l2_lookup)
        if len(primes) < 2:
            return None
        return reduce(mul, primes)


def build_stats_from_vocab(
    word_counts: dict[str, int],
    *,
    source_words: set[str] | None = None,
    config: SubwordConfig | None = None,
) -> SubwordStats:
    """Two-phase friendly: restrict to source_words when provided."""
    stats = SubwordStats(config=config or SubwordConfig())
    words = source_words if source_words else set(word_counts.keys())
    for w in words:
        c = word_counts.get(w, 1)
        stats.observe_word(w, count=1 if source_words else c)
    return stats
