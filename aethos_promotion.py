"""
AETHOS promotion lattice — symbols -> sub-words -> words -> correlation dimensions.

L1 (3D lattice): every symbol maps to a prime; 32 wings, 4 branches.
L2: common sub-words promoted to their own composite primes.
L3: real words promoted to word-primes.
L4-L6: correlation lattice linking sub-words and words (co-occurrence / similarity).

As tokens become common they promote up; higher tiers find more correlations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Iterator

from aethos_core import formula_coord
from aethos_lattice import LatticeId
from aethos_pool_tiers import PoolTier, PoolTierAllocator
from aethos_sequences import SequenceKind, make_chain
from aethos_species import TokenSpecies, digit_chain, number_intersection
from aethos_tokenize import tokenize_spans, tokenize_words
from aethos_words import LETTER_PRIMES, letter_to_prime, prime_to_letter, word_to_order

# Prime pool for promotions (after letter primes)
PROMOTION_POOL: tuple[int, ...] = make_chain(SequenceKind.PRIMES, 512)[26:]

# Function words — never allocate dedicated L3 pool primes (intersection is enough).
STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "into",
        "about",
        "called",
        "new",
        "runs",
        "released",
    }
)


def is_stopword(word: str) -> bool:
    return word.lower() in STOPWORDS

class LatticeTier(IntEnum):
    L1_SYMBOL = 1   # single symbol -> letter prime
    L2_SUBWORD = 2  # promoted sub-word
    L3_WORD = 3     # promoted full word
    L4_CORR = 4     # correlation axis 1
    L5_CORR = 5     # correlation axis 2
    L6_CORR = 6     # correlation axis 3


def intersection_prime(word: str) -> int:
    """No new pool prime — anchor is the intersection of letter primes (L1)."""
    return sum(letter_to_prime(c) for c in word.lower() if c.isalpha())


def letter_chain(word: str) -> tuple[int, ...]:
    return tuple(sorted(letter_to_prime(c) for c in word.lower() if c.isalpha()))


@dataclass(frozen=True)
class PromotedToken:
    text: str
    tier: LatticeTier
    prime: int
    parent_primes: tuple[int, ...]  # decomposition from lower tier
    intersection_only: bool = False  # True = no dedicated pool prime allocated


@dataclass
class CorrelationLink:
    """Edge in L4-L6 correlation space between two promoted tokens."""

    source: str
    target: str
    strength: int
    dim4: float
    dim5: float
    dim6: float

    @classmethod
    def from_pair(cls, a: PromotedToken, b: PromotedToken, strength: int = 1) -> CorrelationLink:
        # Deterministic correlation coordinates from token identities
        ha = hash((a.text, a.tier, a.prime))
        hb = hash((b.text, b.tier, b.prime))
        d4 = ((ha ^ hb) % 10_000) / 1000.0
        d5 = ((ha + hb) % 10_000) / 1000.0
        d6 = (abs(ha - hb) % 10_000) / 1000.0
        return cls(source=a.text, target=b.text, strength=strength, dim4=d4, dim5=d5, dim6=d6)


@dataclass
class PromotionRegistry:
    """
    Tracks symbol/subword/word frequency; assigns promoted primes;
    builds L4-L6 correlation links.

    Policy:
      - Seen once -> intersection only (letter primes); L7-L9 still correlate.
      - New L3 pool prime only if count >= 2 AND surrounding contexts differ
        (separate meanings — e.g. apple+phone vs apple+fruit).
    """

    subword_min_len: int = 2
    subword_max_len: int = 4
    subword_promote_at: int = 2
    subword_min_parents: int = 2  # L2: promote only when seen in 2+ distinct words
    subword_min_pmi: float = 0.25  # L2: PMI threshold (WordPiece-style cohesion)
    subword_min_z: float = 1.0  # L2: z-score under independence null (Significance-Gain BPE)
    word_promote_at: int = 2
    context_jaccard_max: float = 0.45  # below = truly different surroundings

    symbol_counts: dict[str, int] = field(default_factory=dict)
    subword_counts: dict[str, int] = field(default_factory=dict)
    subword_parent_words: dict[str, set[str]] = field(default_factory=dict)
    subword_parent_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    word_observations: int = 0  # parent-word slots used for subword PMI
    word_counts: dict[str, int] = field(default_factory=dict)
    number_counts: dict[str, int] = field(default_factory=dict)
    number_contexts: dict[str, list[frozenset[str]]] = field(default_factory=dict)
    word_contexts: dict[str, list[frozenset[str]]] = field(default_factory=dict)

    promoted: dict[tuple[LatticeTier, str], PromotedToken] = field(default_factory=dict)
    intersections: dict[str, PromotedToken] = field(default_factory=dict)
    number_intersections: dict[str, PromotedToken] = field(default_factory=dict)
    correlations: dict[tuple[str, str], CorrelationLink] = field(default_factory=dict)
    _pool: PoolTierAllocator = field(default_factory=lambda: PoolTierAllocator(pool_len=len(PROMOTION_POOL)))
    _next_promotion_idx: int = 0
    pool_warnings_log: list[str] = field(default_factory=list)
    max_window_tokens: int = 48
    max_corr_pairs: int = 256
    skip_stopword_pairs: bool = True
    defer_l2_promotion: bool = True
    fast_ingest: bool = False
    max_contexts_per_word: int = 12
    _l2_candidates: set[str] = field(default_factory=set)

    def _alloc_prime(self, lattice_tier: LatticeTier, *, species: bool = False) -> int:
        if lattice_tier == LatticeTier.L2_SUBWORD:
            tier = PoolTier.L2_SUBWORD
        elif lattice_tier == LatticeTier.L3_WORD:
            tier = PoolTier.L3_WORD
        else:
            tier = PoolTier.SPECIES if species else PoolTier.L3_WORD
        prime, usage = self._pool.alloc(PROMOTION_POOL, tier)
        self._next_promotion_idx = self._pool.total_used()
        for msg in self._pool.warnings():
            if msg not in self.pool_warnings_log:
                self.pool_warnings_log.append(msg)
        return prime

    def observe_symbol(self, char: str) -> None:
        c = char.lower()
        if c.isalpha():
            self.symbol_counts[c] = self.symbol_counts.get(c, 0) + 1

    def observe_subword(self, sw: str) -> None:
        """Increment subword frequency (pair stats recorded in observe_word)."""
        sw = sw.lower()
        if self.subword_min_len <= len(sw) <= self.subword_max_len:
            self.subword_counts[sw] = self.subword_counts.get(sw, 0) + 1

    def subword_pmi(self, sw: str, parent: str) -> float:
        """PMI between subword and host word (one pair count per host occurrence)."""
        sw, parent = sw.lower(), parent.lower()
        n = max(self.word_observations, 1)
        n_pair = self.subword_parent_pairs.get((sw, parent), 0)
        n_sw = sum(self.subword_parent_pairs.get((sw, p), 0) for p in self.subword_parent_words.get(sw, ()))
        n_p = self.word_counts.get(parent, 0)
        if n_pair == 0 or n_sw == 0 or n_p == 0:
            return 0.0
        p_pair = n_pair / n
        p_sw = n_sw / n
        p_p = n_p / n
        if p_pair * p_sw * p_p == 0:
            return 0.0
        return math.log2(p_pair / (p_sw * p_p))

    def subword_cohesion_z(self, sw: str, parent: str) -> float:
        """
        Z-score of subword–parent co-occurrence under independence null.

        Aligns with Significance-Gain BPE (arXiv:2603.19261): cohesion separate
        from raw frequency; sqrt(expected) penalizes rare false merges.
        """
        sw, parent = sw.lower(), parent.lower()
        n = max(self.word_observations, 1)
        c_xy = self.subword_parent_pairs.get((sw, parent), 0)
        c_x = sum(self.subword_parent_pairs.get((sw, p), 0) for p in self.subword_parent_words.get(sw, ()))
        c_y = self.word_counts.get(parent, 0)
        if c_xy == 0 or c_x == 0 or c_y == 0:
            return 0.0
        expected = (c_x * c_y) / n
        if expected <= 0:
            return 0.0
        return (c_xy - expected) / math.sqrt(expected + 1e-9)

    def max_subword_z(self, sw: str) -> float:
        parents = self.subword_parent_words.get(sw.lower(), set())
        if not parents:
            return 0.0
        return max(self.subword_cohesion_z(sw, p) for p in parents)

    def compression_strength(self, token: str) -> float:
        """
        Section 5 observation strength proxy: how strongly co-occurrence
        compresses this token toward definite lattice neighbors (PMI mass).
        """
        w = token.lower()
        corrs = self.correlations_for(w)
        if not corrs:
            return float(self.word_counts.get(w, 0) + self.number_counts.get(w, 0))
        return float(sum(c.strength for c in corrs))

    def max_subword_pmi(self, sw: str) -> float:
        parents = self.subword_parent_words.get(sw.lower(), set())
        if not parents:
            return 0.0
        return max(self.subword_pmi(sw, p) for p in parents)

    def pool_usage_report(self) -> str:
        return "\n".join(u.summary() for u in self._pool.all_usage())

    def _should_promote_l2(self, sw: str) -> bool:
        sw = sw.lower()
        if is_stopword(sw):
            return False
        count = self.subword_counts.get(sw, 0)
        if count < self.subword_promote_at:
            return False
        parents = self.subword_parent_words.get(sw, set())
        if sw in parents and self.word_counts.get(sw, 0) >= self.subword_promote_at:
            return True
        if len(parents) < self.subword_min_parents:
            return False
        pmis = [self.subword_pmi(sw, p) for p in parents]
        zs = [self.subword_cohesion_z(sw, p) for p in parents]
        strong_pmi = sum(1 for x in pmis if x >= self.subword_min_pmi)
        strong_z = sum(1 for z in zs if z >= self.subword_min_z)
        if strong_z >= self.subword_min_parents:
            return True
        if max(zs, default=0.0) >= self.subword_min_z * 1.5:
            return True
        return strong_pmi >= self.subword_min_parents or max(pmis, default=0.0) >= self.subword_min_pmi * 1.5

    def observe_word(self, word: str, context: frozenset[str] | None = None) -> None:
        w = word.lower()
        if not w.isalpha():
            return
        self.word_counts[w] = self.word_counts.get(w, 0) + 1
        self.word_observations += 1
        for i in range(len(w)):
            self.observe_symbol(w[i])
        if not self.fast_ingest:
            unique_subwords: set[str] = set()
            for ln in range(self.subword_min_len, min(self.subword_max_len + 1, len(w) + 1)):
                for i in range(len(w) - ln + 1):
                    sw = w[i : i + ln]
                    self.observe_subword(sw)
                    unique_subwords.add(sw)
            for sw in unique_subwords:
                self.subword_parent_words.setdefault(sw, set()).add(w)
                key = (sw, w)
                self.subword_parent_pairs[key] = self.subword_parent_pairs.get(key, 0) + 1
                if self.defer_l2_promotion:
                    self._l2_candidates.add(sw)
                elif self._should_promote_l2(sw):
                    self._promote(LatticeTier.L2_SUBWORD, sw)
        if context is not None:
            neighbors = frozenset(x for x in context if x != w)
            lst = self.word_contexts.setdefault(w, [])
            if neighbors not in lst:
                if len(lst) >= self.max_contexts_per_word:
                    lst.pop(0)
                lst.append(neighbors)
        if self._should_promote_l3(w):
            self._promote(LatticeTier.L3_WORD, w)
        else:
            self._ensure_intersection(w)

    def observe_number(self, num: str, context: frozenset[str] | None = None) -> None:
        """NUM species — per-digit even anchors (left-to-right), intersection sum."""
        if not num.isdigit():
            return
        self.number_counts[num] = self.number_counts.get(num, 0) + 1
        if num not in self.number_intersections:
            chain = digit_chain(num)
            self.number_intersections[num] = PromotedToken(
                text=num,
                tier=LatticeTier.L3_WORD,
                prime=number_intersection(num),
                parent_primes=chain,
                intersection_only=True,
            )
        if context is not None:
            self.number_contexts.setdefault(num, []).append(frozenset(x for x in context if x != num))

    def _flush_l2_candidates(self) -> None:
        pending = self._l2_candidates
        self._l2_candidates = set()
        for sw in pending:
            if self._should_promote_l2(sw):
                self._promote(LatticeTier.L2_SUBWORD, sw)

    def observe_text(self, text: str) -> None:
        spans = tokenize_spans(text)
        words = [s.text for s in spans]
        for span in spans:
            ctx = frozenset(words)
            if span.species == TokenSpecies.NUM:
                self.observe_number(span.text, ctx)
            else:
                self.observe_word(span.text, ctx)
        if self.defer_l2_promotion and not self.fast_ingest:
            self._flush_l2_candidates()
        bag = words
        if len(bag) > self.max_window_tokens:
            bag = bag[: self.max_window_tokens]
        if len(bag) > 1:
            self.observe_cooccurrence(bag)

    def contexts_differ(self, word: str) -> bool:
        """True when two+ occurrences have genuinely different neighbor sets."""
        ctxs = self.word_contexts.get(word.lower(), [])
        if len(ctxs) < 2:
            return False
        for i in range(len(ctxs)):
            for j in range(i + 1, len(ctxs)):
                a = frozenset(x for x in ctxs[i] if not is_stopword(x))
                b = frozenset(x for x in ctxs[j] if not is_stopword(x))
                if not a and not b:
                    continue
                union = len(a | b) or 1
                jacc = len(a & b) / union
                if jacc < self.context_jaccard_max:
                    return True
        return False

    def _should_promote_l3(self, word: str) -> bool:
        w = word.lower()
        if is_stopword(w) or len(w) <= 2:
            return False
        if self.word_counts.get(w, 0) < self.word_promote_at:
            return False
        return self.contexts_differ(w)

    def _ensure_intersection(self, word: str) -> PromotedToken:
        """Single occurrence (or same-meaning repeats): intersection is enough."""
        w = word.lower()
        if w in self.intersections:
            return self.intersections[w]
        parents = word_to_order(w)
        tok = PromotedToken(
            text=w,
            tier=LatticeTier.L3_WORD,
            prime=intersection_prime(w),
            parent_primes=parents,
            intersection_only=True,
        )
        self.intersections[w] = tok
        return tok

    def resolve_token(self, word: str) -> PromotedToken:
        """L3 promoted prime, intersection-only word, or NUM digit-chain record."""
        w = word.lower()
        if w.isdigit():
            if w in self.number_intersections:
                return self.number_intersections[w]
            chain = digit_chain(w)
            return PromotedToken(
                text=w,
                tier=LatticeTier.L3_WORD,
                prime=number_intersection(w),
                parent_primes=chain,
                intersection_only=True,
            )
        key = (LatticeTier.L3_WORD, w)
        if key in self.promoted:
            return self.promoted[key]
        return self._ensure_intersection(w)

    def is_intersection_only(self, word: str) -> bool:
        w = word.lower()
        if w.isdigit():
            return True
        return (LatticeTier.L3_WORD, w) not in self.promoted

    def _promote(self, tier: LatticeTier, text: str) -> PromotedToken:
        key = (tier, text)
        if key in self.promoted:
            return self.promoted[key]

        if tier == LatticeTier.L1_SYMBOL:
            prime = letter_to_prime(text)
            parents = (prime,)
        elif tier == LatticeTier.L2_SUBWORD:
            parents = word_to_order(text)
            try:
                prime = self._alloc_prime(LatticeTier.L2_SUBWORD)
            except RuntimeError:
                # Pool exhausted — fall back to intersection prime (Section 5: lighter pin)
                prime = intersection_prime(text)
        else:  # L3_WORD
            # decompose: use promoted subword primes where available, else letters
            parents = tuple(self.resolve_prime(t, LatticeTier.L2_SUBWORD) for t in _chunk_subwords(text))
            if not parents:
                parents = word_to_order(text)
            try:
                prime = self._alloc_prime(LatticeTier.L3_WORD)
            except RuntimeError:
                return self._ensure_intersection(text)

        tok = PromotedToken(text=text, tier=tier, prime=prime, parent_primes=parents, intersection_only=False)
        self.promoted[key] = tok
        if tier == LatticeTier.L3_WORD:
            self.intersections.pop(text, None)
        return tok

    def resolve_prime(self, text: str, tier: LatticeTier) -> int:
        key = (tier, text.lower())
        if key in self.promoted:
            return self.promoted[key].prime
        if tier == LatticeTier.L1_SYMBOL and len(text) == 1:
            return letter_to_prime(text)
        if tier == LatticeTier.L2_SUBWORD:
            sw = text.lower()
            key = (LatticeTier.L2_SUBWORD, sw)
            if key in self.promoted:
                return self.promoted[key].prime
            if self._should_promote_l2(sw):
                return self._promote(LatticeTier.L2_SUBWORD, sw).prime
            return intersection_prime(sw)
        if tier == LatticeTier.L3_WORD:
            return self.resolve_token(text).prime
        raise KeyError(f"not promoted: {tier} {text!r}")

    def link_correlation(self, a: PromotedToken, b: PromotedToken) -> CorrelationLink:
        """Record correlation in L4-L6 when tokens co-occur or relate."""
        pair = tuple(sorted((a.text, b.text)))
        if pair in self.correlations:
            link = self.correlations[pair]
            link.strength += 1
            return link
        link = CorrelationLink.from_pair(a, b)
        self.correlations[pair] = link
        return link

    def observe_cooccurrence(self, words: Iterable[str]) -> None:
        """Words in same window -> L4-L6 links (WORD + NUM tokens)."""
        word_list = [w.lower() for w in words if w and (str(w).isalpha() or str(w).isdigit())]
        tokens = [self.resolve_token(w) for w in word_list]
        pairs = 0
        for i, a in enumerate(tokens):
            for b in tokens[i + 1 :]:
                if (
                    self.skip_stopword_pairs
                    and is_stopword(a.text)
                    and is_stopword(b.text)
                ):
                    continue
                if pairs >= self.max_corr_pairs:
                    return
                self.link_correlation(a, b)
                pairs += 1

    def correlations_for(self, word: str) -> list[CorrelationLink]:
        w = word.lower()
        out = []
        for (a, b), link in self.correlations.items():
            if w in (a, b):
                out.append(link)
        return sorted(out, key=lambda x: -x.strength)

    def lattice_address(
        self,
        text: str,
        tier: LatticeTier,
        n: int = 7,
        lattice_id: LatticeId = LatticeId.L01,
    ) -> tuple[float, float, float]:
        """3D dot on tier's lattice — anchor chain from tokens, math from lattice core."""
        if tier == LatticeTier.L3_WORD:
            tok = self.resolve_token(text)
            if tok.intersection_only:
                chain = tuple(sorted(set(tok.parent_primes)))
            else:
                chain = tuple(sorted(set(tok.parent_primes + (tok.prime,))))
        else:
            tok = self.promoted.get((tier, text.lower()))
            if not tok:
                if tier == LatticeTier.L2_SUBWORD and not self._should_promote_l2(text.lower()):
                    return formula_coord(letter_chain(text), n, lattice_id)
                tok = self._promote(tier, text.lower())
            chain = tok.parent_primes if tier != LatticeTier.L1_SYMBOL else (tok.prime,)

        return formula_coord(chain, n, lattice_id)

    def correlation_point(self, link: CorrelationLink) -> tuple[float, float, float, float, float, float]:
        """Full 6D address: L1-L3 base + L4-L6 correlation offsets."""
        src = self.promoted.get((LatticeTier.L3_WORD, link.source)) or self.promoted.get(
            (LatticeTier.L2_SUBWORD, link.source)
        )
        if not src:
            return (0, 0, 0, link.dim4, link.dim5, link.dim6)
        base = self.lattice_address(src.text, src.tier)
        return (base[0], base[1], base[2], link.dim4, link.dim5, link.dim6)


def _chunk_subwords(word: str) -> list[str]:
    """Prefer known subword chunks (greedy from len 4 down)."""
    w = word.lower()
    found: list[str] = []
    i = 0
    while i < len(w):
        matched = False
        for ln in range(min(4, len(w) - i), 1, -1):
            piece = w[i : i + ln]
            found.append(piece)
            i += ln
            matched = True
            break
        if not matched:
            i += 1
    return found


@dataclass
class MultiLatticeStack:
    """Facade: ingest corpus, query promotions and correlations."""

    registry: PromotionRegistry = field(default_factory=PromotionRegistry)

    def train(self, *documents: str) -> None:
        for doc in documents:
            words = tokenize_words(doc)
            self.registry.observe_text(doc)
            self.registry.observe_cooccurrence(words)

    def explain(self, word: str) -> str:
        w = word.lower()
        lines = [f"Word {word!r}:"]
        if (LatticeTier.L1_SYMBOL, w[0] if w else "") in self.registry.promoted or w:
            letters = [letter_to_prime(c) for c in w]
            lines.append(f"  L1 symbols: {list(w)} -> primes {letters}")
        if (LatticeTier.L2_SUBWORD, w[:3]) in self.registry.promoted:
            t = self.registry.promoted[(LatticeTier.L2_SUBWORD, w[:3])]
            lines.append(f"  L2 subword {t.text!r} -> promoted prime {t.prime}")
        if (LatticeTier.L3_WORD, w) in self.registry.promoted:
            t = self.registry.promoted[(LatticeTier.L3_WORD, w)]
            lines.append(f"  L3 word -> dedicated prime {t.prime}, parents {t.parent_primes}")
            addr = self.registry.lattice_address(w, LatticeTier.L3_WORD)
            lines.append(f"  L3 lattice dot: {addr}")
        elif w in self.registry.intersections:
            t = self.registry.intersections[w]
            lines.append(
                f"  intersection-only (count={self.registry.word_counts.get(w,0)}): "
                f"prime={t.prime} from letters {t.parent_primes} — L7-L9 correlations only"
            )
        corrs = self.registry.correlations_for(w)
        if corrs:
            lines.append(f"  L4-L6 correlations ({len(corrs)}):")
            for c in corrs[:5]:
                pt = self.registry.correlation_point(c)
                lines.append(
                    f"    <-> {c.target if c.source == w else c.source} "
                    f"strength={c.strength} dims456=({pt[3]:.3f},{pt[4]:.3f},{pt[5]:.3f})"
                )
        return "\n".join(lines)


def demo() -> None:
    corpus = [
        "tab bat tab cat bat",
        "tab table bat battle",
        "bat bat tab tab",
        "cat bat hat mat",
    ]
    stack = MultiLatticeStack()
    stack.train(*corpus)

    print("=" * 60)
    print("PROMOTION LATTICE — symbols -> sub-words -> words -> L4-L6")
    print("=" * 60)

    print("\n--- Promoted sub-words (L2) ---")
    for (tier, text), tok in sorted(stack.registry.promoted.items()):
        if tier == LatticeTier.L2_SUBWORD:
            print(f"  {text!r} -> prime {tok.prime}  (from {tok.parent_primes})")

    print("\n--- Promoted words (L3) ---")
    for (tier, text), tok in sorted(stack.registry.promoted.items()):
        if tier == LatticeTier.L3_WORD:
            print(f"  {text!r} -> prime {tok.prime}  parents={tok.parent_primes}")

    print("\n--- tab vs bat (same letters, different L3 primes) ---")
    if (LatticeTier.L3_WORD, "tab") in stack.registry.promoted and (LatticeTier.L3_WORD, "bat") in stack.registry.promoted:
        tab_p = stack.registry.promoted[(LatticeTier.L3_WORD, "tab")].prime
        bat_p = stack.registry.promoted[(LatticeTier.L3_WORD, "bat")].prime
        tab_dot = stack.registry.lattice_address("tab", LatticeTier.L3_WORD)
        bat_dot = stack.registry.lattice_address("bat", LatticeTier.L3_WORD)
        print(f"  tab prime={tab_p}  dot={tab_dot}")
        print(f"  bat prime={bat_p}  dot={bat_dot}")

    print("\n--- Correlations (L4-L6) for 'tab' ---")
    print(stack.explain("tab"))

    print("\n--- Correlations (L4-L6) for 'bat' ---")
    print(stack.explain("bat"))


if __name__ == "__main__":
    demo()
