"""
Lightweight prime postings for BM25-class scoring on the two-lattice index.

One word-identity prime per term (Stage 04 / semantic._prime_for_term).
Append-only O(1) per token — no full AppendOnlyLatticeIndex required inside v1.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass
class PrimePostingIndex:
    """prime -> {doc_id: tf}; supports bounded-pool lattice_pure scoring."""

    postings: dict[int, dict[str, float]] = field(default_factory=dict)
    df: dict[int, int] = field(default_factory=dict)
    doc_card: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0
    _avg_card: float = 1.0

    def observe_doc(self, doc_id: str, words: tuple[str, ...], *, prime_fn) -> None:
        seen_primes: set[int] = set()
        for w in words:
            p = prime_fn(w)
            bucket = self.postings.setdefault(p, {})
            bucket[doc_id] = bucket.get(doc_id, 0.0) + 1.0
            if p not in seen_primes:
                self.df[p] = self.df.get(p, 0) + 1
                seen_primes.add(p)
        self.doc_card[doc_id] = len(seen_primes)
        self.n_docs += 1
        self._avg_card = sum(self.doc_card.values()) / max(self.n_docs, 1)

    def finalize(self) -> None:
        if self.n_docs:
            self._avg_card = sum(self.doc_card.values()) / self.n_docs

    def idf(self, prime: int) -> float:
        dfp = self.df.get(prime, 0)
        if dfp == 0:
            return 0.0
        n = max(self.n_docs, 1)
        return math.log(1 + (n - dfp + 0.5) / (dfp + 0.5))

    def score_lattice_pure(
        self,
        query: str,
        pool: set[str] | frozenset[str],
        *,
        prime_fn,
        sat_a: float = 1.0,
        lpow: float = 0.35,
    ) -> dict[str, float]:
        """BM25-inspired: IDF × geometric sat × κ-card length norm on bounded pool."""
        if not pool:
            return {}
        terms = [w for w in _TOKEN_RE.findall(query.lower()) if len(w) >= 2]
        scores: dict[str, float] = defaultdict(float)
        cand = list(pool)
        for term in terms:
            p = prime_fn(term)
            pl = self.postings.get(p)
            if not pl:
                continue
            idf = self.idf(p)
            for d in cand:
                tf = pl.get(d)
                if not tf:
                    continue
                sat = tf / (tf + sat_a)
                card = self.doc_card.get(d, 1)
                lennorm = (self._avg_card / card) ** lpow if card else 1.0
                scores[d] += idf * sat * lennorm
        return dict(scores)

    def route_pool_rarest(
        self,
        query: str,
        *,
        prime_fn,
        term_df: dict[str, int],
        max_pool: int = 1200,
    ) -> set[str]:
        """Rarest-term union postings — fast Phase A when shared index empty."""
        terms = [w for w in _TOKEN_RE.findall(query.lower()) if len(w) >= 2]
        if not terms:
            return set()
        ordered = sorted(terms, key=lambda t: (term_df.get(t, 10**9), t))
        pool: set[str] = set()
        for term in ordered:
            p = prime_fn(term)
            pl = self.postings.get(p, {})
            pool |= set(pl.keys())
            if len(pool) >= max_pool:
                break
        if len(pool) > max_pool:
            pool = set(list(pool)[:max_pool])
        return pool
