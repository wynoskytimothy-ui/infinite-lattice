"""
Two-lattice zero-shot retrieval — architecture map (MVP).

Vision → code
-------------
Layer 0 (doc lattice, append index):
  Each document occupies its own lattice cell: unique ``doc_prime`` from an
  append-only pool plus ``order_stream`` of Stage-04 token-identity primes on
  the transgressor rail.  ``SharedTermIndex`` is the inverted index on term
  primes — when query term *quantum* hits, every doc whose lattice carries
  that prime **lights up** together (shared witness, no qrels).

Layer 1 (correlation lattice, L4–L6):
  Per-doc ``TermCorrelationShell`` objects store CorrelationLink dim4/dim5/dim6
  weights from sliding 3-way windows (Stage 07 WingCage observe logic).
  Neighbor terms branch off the meet anchor with co-occurrence strength.

Scoring (ColBERT MaxSim analog):
  For each query term *t*:  witness(t, doc) = max shell touch for *t*
  Score = Σ_t  idf(t) × witness(t, doc)

Phase A routes via shared-term postings (rarest first); Phase B scores the
bounded pool only — never full-corpus scan.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal

from lattice_retriever_v1.stage07_semantic_light import HUB_WORDS, SemanticLightIndex

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class DocLatticePlacement:
    """One doc's Layer-0 lattice address on the transgressor rail."""

    doc_id: str
    doc_prime: int
    order_stream: tuple[int, ...]
    corridor_pins: frozenset[int]
    words: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShellNeighbor:
    """One correlated term on a doc's L4–L6 wing cage."""

    term: str
    prime: int
    strength: int
    dim4: float
    dim5: float
    dim6: float

    @property
    def drift_weight(self) -> float:
        return float(self.strength) * (self.dim4 + self.dim6 + 1.0)


@dataclass
class TermCorrelationShell:
    """
    Per-doc correlation shell keyed by anchor composite or bare term.

    dim4/dim5/dim6 come from CorrelationLink placement; neighbors carry
    co-occurrence strength from sliding 3-way observation windows.
    """

    key: str
    key_kind: Literal["term", "anchor"]
    anchor_composite: int
    anchor_primes: tuple[int, ...]
    dim4: float
    dim5: float
    dim6: float
    neighbors: dict[str, ShellNeighbor] = field(default_factory=dict)

    def witness_weight(self, term: str) -> float:
        """MaxSim-style witness: best touch of *term* on this shell."""
        t = term.lower()
        best = 0.0
        if self.key_kind == "term" and self.key == t:
            best = max(best, self.dim4 + self.dim6 + 1.0)
        if self.key_kind == "anchor" and t in self.key.split("|"):
            best = max(best, 2.0 * (self.dim4 + self.dim6 + 1.0))
        dot = self.neighbors.get(t)
        if dot is not None:
            best = max(best, dot.drift_weight)
        return best

    def explain(self) -> dict:
        return {
            "key": self.key,
            "key_kind": self.key_kind,
            "anchor_composite": self.anchor_composite,
            "dim4": round(self.dim4, 4),
            "dim5": round(self.dim5, 4),
            "dim6": round(self.dim6, 4),
            "neighbors": sorted(self.neighbors.keys()),
        }


@dataclass
class SharedTermIndex:
    """term_prime → posting list; shared rare terms light all member docs."""

    term_postings: dict[int, set[str]] = field(default_factory=dict)
    term_doc_freq: dict[str, int] = field(default_factory=dict)
    n_docs: int = 0

    def idf(self, term: str) -> float:
        df = self.term_doc_freq.get(term.lower(), 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def is_rare(self, term: str, *, max_df: int = 2) -> bool:
        t = term.lower()
        return t not in HUB_WORDS and self.term_doc_freq.get(t, 0) <= max_df

    def index_placement(
        self,
        placement: DocLatticePlacement,
        *,
        term_prime_fn: Callable[[str], int],
    ) -> None:
        seen: set[str] = set()
        for w in placement.words:
            if w not in seen:
                self.term_doc_freq[w] = self.term_doc_freq.get(w, 0) + 1
                seen.add(w)
        self.n_docs += 1
        for w in placement.words:
            prime = term_prime_fn(w)
            self.term_postings.setdefault(prime, set()).add(placement.doc_id)

    def route_pool(
        self,
        query_terms: list[str],
        *,
        term_prime_fn: Callable[[str], int],
    ) -> tuple[set[str], list[dict]]:
        """Phase A — rarest term first, intersect postings."""
        if not query_terms:
            all_docs: set[str] = set()
            for bucket in self.term_postings.values():
                all_docs |= bucket
            return all_docs, [{"step": "empty_query", "pool_size": len(all_docs)}]

        ordered = sorted(
            query_terms,
            key=lambda t: (self.term_doc_freq.get(t, 10**9), t),
        )
        steps: list[dict] = []
        pool: set[str] | None = None
        lit_terms: list[str] = []
        for term in ordered:
            prime = term_prime_fn(term)
            bucket = set(self.term_postings.get(prime, set()))
            lit_terms.append(term)
            pool = bucket if pool is None else pool & bucket
            steps.append(
                {
                    "step": "shared_term_light",
                    "term": term,
                    "term_prime": prime,
                    "doc_freq": self.term_doc_freq.get(term, 0),
                    "pool_size": len(pool),
                }
            )
            if not pool:
                break

        if not pool:
            rarest = ordered[0]
            prime = term_prime_fn(rarest)
            pool = set(self.term_postings.get(prime, set()))
            steps.append(
                {
                    "step": "widen_rarest",
                    "term": rarest,
                    "term_prime": prime,
                    "pool_size": len(pool),
                }
            )

        if len(ordered) > 1:
            union_pool: set[str] = set()
            for term in ordered:
                union_pool |= set(self.term_postings.get(term_prime_fn(term), set()))
            if union_pool:
                steps.append(
                    {
                        "step": "multiterm_union",
                        "intersect_size": len(pool or set()),
                        "pool_size": len(union_pool),
                    }
                )
                pool = union_pool

        return pool or set(), steps


@dataclass(frozen=True)
class Lattice2ScoreTrace:
    """Glass-box trace for one doc's MaxSim-on-witness score."""

    doc_id: str
    score: float
    term_witnesses: tuple[dict, ...]
    shells_touched: tuple[dict, ...]
    shared_terms: tuple[str, ...]

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "term_witnesses": list(self.term_witnesses),
            "shells_touched": list(self.shells_touched),
            "shared_terms": list(self.shared_terms),
        }


class Lattice2CorrelationPass:
    """Phase B — MaxSim-shaped score over per-doc L4–L6 correlation shells."""

    @staticmethod
    def witness(
        term: str,
        shells: Iterable[TermCorrelationShell],
        *,
        words: tuple[str, ...],
    ) -> tuple[float, list[dict]]:
        """max shell touch for *term* plus direct lexical presence."""
        t = term.lower()
        best = 0.0
        touches: list[dict] = []
        for shell in shells:
            w = shell.witness_weight(t)
            if w > 0:
                touches.append({"shell": shell.key, "witness": round(w, 4)})
                best = max(best, w)
        if t in words:
            identity = 2.0
            best = max(best, identity)
            touches.append({"shell": "identity", "witness": identity})
        return best, touches

    @classmethod
    def score(
        cls,
        query_terms: Iterable[str],
        doc_id: str,
        shell_index: dict[str, tuple[TermCorrelationShell, ...]],
        idf_fn: Callable[[str], float],
        *,
        placements: dict[str, DocLatticePlacement] | None = None,
    ) -> tuple[float, Lattice2ScoreTrace]:
        terms = [t.lower() for t in query_terms if len(t) >= 2]
        shells = shell_index.get(doc_id, ())
        words = placements[doc_id].words if placements and doc_id in placements else ()
        total = 0.0
        term_witnesses: list[dict] = []
        shells_hit: dict[str, dict] = {}
        shared: list[str] = []

        for term in terms:
            w, touches = cls.witness(term, shells, words=words)
            weighted = idf_fn(term) * w
            total += weighted
            if w > 0:
                shared.append(term)
            term_witnesses.append(
                {
                    "term": term,
                    "witness": round(w, 4),
                    "idf": round(idf_fn(term), 4),
                    "weighted": round(weighted, 4),
                    "touches": touches,
                }
            )
            for touch in touches:
                key = touch["shell"]
                if key not in shells_hit:
                    shells_hit[key] = touch

        trace = Lattice2ScoreTrace(
            doc_id=doc_id,
            score=total,
            term_witnesses=tuple(term_witnesses),
            shells_touched=tuple(shells_hit.values()),
            shared_terms=tuple(shared),
        )
        return total, trace


@dataclass(frozen=True)
class TwoLatticeHit:
    doc_id: str
    score: float
    trace: Lattice2ScoreTrace

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "trace": self.trace.explain(),
        }


@dataclass(frozen=True)
class TwoLatticeRetrieveTrace:
    query: str
    filter_steps: tuple[dict, ...]
    pool_size: int
    lit_docs: tuple[str, ...]
    hits: tuple[TwoLatticeHit, ...]

    def explain(self) -> dict:
        return {
            "query": self.query,
            "pool_size": self.pool_size,
            "filter_steps": list(self.filter_steps),
            "lit_docs": list(self.lit_docs),
            "hits": [h.explain() for h in self.hits],
        }


@dataclass
class ZeroShotTwoLatticeRetriever:
    """
    Zero-shot two-lattice retriever (MVP).

    Phase A: ``SharedTermIndex`` rare-term postings narrow the pool.
    Phase B: ``Lattice2CorrelationPass`` MaxSim-on-witness scores the pool.
    """

    semantic: SemanticLightIndex = field(default_factory=SemanticLightIndex)
    shared: SharedTermIndex = field(default_factory=SharedTermIndex)
    placements: dict[str, DocLatticePlacement] = field(default_factory=dict)
    shell_index: dict[str, tuple[TermCorrelationShell, ...]] = field(default_factory=dict)

    def _term_prime(self, term: str) -> int:
        return self.semantic._prime_for_term(term)

    def _query_terms(self, query: str) -> list[str]:
        return [w for w in _TOKEN_RE.findall(query.lower()) if len(w) >= 2]

    def index_placement(
        self,
        placement: DocLatticePlacement,
        shells: tuple[TermCorrelationShell, ...],
    ) -> None:
        self.placements[placement.doc_id] = placement
        self.shell_index[placement.doc_id] = shells
        self.shared.index_placement(placement, term_prime_fn=self._term_prime)

    def retrieve_with_trace(self, query: str, *, limit: int = 10) -> TwoLatticeRetrieveTrace:
        terms = self._query_terms(query)
        pool, steps = self.shared.route_pool(terms, term_prime_fn=self._term_prime)
        hits: list[TwoLatticeHit] = []
        for doc_id in sorted(pool):
            score, trace = Lattice2CorrelationPass.score(
                terms,
                doc_id,
                self.shell_index,
                self.shared.idf,
                placements=self.placements,
            )
            hits.append(TwoLatticeHit(doc_id=doc_id, score=score, trace=trace))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return TwoLatticeRetrieveTrace(
            query=query,
            filter_steps=tuple(steps),
            pool_size=len(pool),
            lit_docs=tuple(sorted(pool)),
            hits=tuple(hits[:limit]),
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[TwoLatticeHit]:
        return list(self.retrieve_with_trace(query, limit=limit).hits)
