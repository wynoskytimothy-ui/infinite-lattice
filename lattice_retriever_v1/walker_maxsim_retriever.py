"""
Walker MaxSim retriever — ColBERT late interaction on procedural pair walks.

Ingest (doc self cross-reference):
  - Adjacent word pairs place dots on oriented pair-origin rails (transgressor n=1,2,3…).
  - Sliding 3-way windows build L4–L6 correlation shells (doc_lattice_codec).

Query:
  - Same walk on query text.
  - Phase A: intersect rarest oriented-pair origins, then shared-term widen.
  - Phase B: MaxSim — for each query dot, max geometric witness over doc dots;
    plus per-term max shell witness (Lattice2CorrelationPass).

No learned embeddings — witness = same origin, same n, wing L01 proximity.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from lattice_retriever_v1.doc_lattice_codec import (
    DocPrimePool,
    build_doc_correlation_shells,
    encode_doc,
)
from lattice_retriever_v1.lattice2_correlation import (
    DocLatticePlacement,
    Lattice2CorrelationPass,
    Lattice2ScoreTrace,
    SharedTermIndex,
    TermCorrelationShell,
)
from lattice_retriever_v1.stage02_intersections import IntersectionAddress, intersect_primes
from lattice_retriever_v1.stage04_promote import Stage04Registry, promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex

_TOKEN_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class WordPairOrigin:
    """Oriented word bigram origin — one rail per (left→right)."""

    left: str
    right: str
    left_prime: int
    right_prime: int

    @property
    def key(self) -> tuple[str, str]:
        return (self.left, self.right)


@dataclass(frozen=True)
class WordWalkDot:
    """One dot on a word-pair origin rail."""

    origin: WordPairOrigin
    pair_n: int
    walk_index: int
    address: IntersectionAddress

    def explain(self) -> dict:
        return {
            "pair": f"{self.origin.left}->{self.origin.right}",
            "pair_n": self.pair_n,
            "walk_index": self.walk_index,
            "composite": self.address.composite,
            "L01": self.address.lattice_coords[0],
            "transgressor_n": self.address.transgressor_n,
        }


def _words(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(text.lower()))


def word_pair_walk(
    text: str,
    semantic: SemanticLightIndex,
) -> tuple[WordWalkDot, ...]:
    """Walk document/query — each adjacent pair transgresses its origin rail."""
    words = _words(text)
    if len(words) < 2:
        return ()
    counters: dict[tuple[str, str], int] = defaultdict(int)
    out: list[WordWalkDot] = []
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        counters[(w1, w2)] += 1
        pn = counters[(w1, w2)]
        lp = semantic._prime_for_term(w1)
        rp = semantic._prime_for_term(w2)
        origin = WordPairOrigin(left=w1, right=w2, left_prime=lp, right_prime=rp)
        addr = intersect_primes((w1, w2), (lp, rp), start_index=i, n=pn)
        out.append(WordWalkDot(origin=origin, pair_n=pn, walk_index=i, address=addr))
    return tuple(out)


def geometric_dot_witness(query_dot: WordWalkDot, doc_dot: WordWalkDot) -> float:
    """
    ColBERT-style token–token sim on geometry:
      same origin + exact n → strongest; nearby n + L01 proximity → partial.
    """
    if query_dot.origin.key != doc_dot.origin.key:
        return 0.0
    w = 2.0
    if query_dot.pair_n == doc_dot.pair_n:
        w += 3.0
    elif abs(query_dot.pair_n - doc_dot.pair_n) <= 1:
        w += 1.0
    qL = query_dot.address.lattice_coords[0]
    dL = doc_dot.address.lattice_coords[0]
    manhattan = sum(abs(a - b) for a, b in zip(qL, dL))
    w += max(0.0, 2.0 - manhattan / 50.0)
    return w


@dataclass(frozen=True)
class WalkWitness:
    query_dot: dict
    best_witness: float
    best_doc_dot: dict | None
    idf: float
    weighted: float


@dataclass(frozen=True)
class WalkerMaxSimTrace:
    doc_id: str
    walk_score: float
    shell_score: float
    total_score: float
    walk_witnesses: tuple[WalkWitness, ...]
    shell_trace: Lattice2ScoreTrace

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "walk_score": round(self.walk_score, 6),
            "shell_score": round(self.shell_score, 6),
            "total_score": round(self.total_score, 6),
            "walk_witnesses": [
                {
                    "query_dot": w.query_dot,
                    "best_witness": round(w.best_witness, 4),
                    "best_doc_dot": w.best_doc_dot,
                    "idf": round(w.idf, 4),
                    "weighted": round(w.weighted, 4),
                }
                for w in self.walk_witnesses
            ],
            "shell_trace": self.shell_trace.explain(),
        }


@dataclass
class OrientedPairIndex:
    """Phase A — inverted index on oriented word-pair origins."""

    pair_doc_freq: dict[tuple[str, str], int] = field(default_factory=dict)
    pair_postings: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    n_docs: int = 0

    def index_doc(self, doc_id: str, walk: tuple[WordWalkDot, ...]) -> None:
        self.n_docs += 1
        seen: set[tuple[str, str]] = set()
        for dot in walk:
            k = dot.origin.key
            if k in seen:
                continue
            seen.add(k)
            self.pair_doc_freq[k] = self.pair_doc_freq.get(k, 0) + 1
            self.pair_postings.setdefault(k, set()).add(doc_id)

    def idf_pair(self, left: str, right: str) -> float:
        df = self.pair_doc_freq.get((left, right), 0)
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def route_pool(
        self,
        query_walk: tuple[WordWalkDot, ...],
    ) -> tuple[set[str], list[dict]]:
        if not query_walk:
            return set(), [{"step": "empty_query_walk"}]
        keys = list(dict.fromkeys(d.origin.key for d in query_walk))
        ordered = sorted(keys, key=lambda k: (self.pair_doc_freq.get(k, 10**9), k))
        steps: list[dict] = []
        pool: set[str] | None = None
        for left, right in ordered:
            bucket = set(self.pair_postings.get((left, right), set()))
            pool = bucket if pool is None else pool & bucket
            steps.append(
                {
                    "step": "pair_origin_intersect",
                    "pair": [left, right],
                    "pair_df": self.pair_doc_freq.get((left, right), 0),
                    "pool_size": len(pool or set()),
                }
            )
            if not pool:
                break
        if not pool:
            rarest = ordered[0]
            pool = set(self.pair_postings.get(rarest, set()))
            steps.append(
                {
                    "step": "pair_widen_rarest",
                    "pair": list(rarest),
                    "pool_size": len(pool),
                }
            )
        return pool or set(), steps


class WalkerMaxSimPass:
    """Phase B — ColBERT MaxSim on query walk × doc walk + shell witness."""

    @staticmethod
    def _dot_idf(
        dot: WordWalkDot,
        idf_fn: Callable[[str], float],
        pair_idf_fn: Callable[[str, str], float] | None,
    ) -> float:
        li = idf_fn(dot.origin.left)
        ri = idf_fn(dot.origin.right)
        base = max(li, ri)
        if pair_idf_fn is not None:
            base = max(base, pair_idf_fn(dot.origin.left, dot.origin.right))
        return base

    @classmethod
    def walk_maxsim(
        cls,
        query_walk: tuple[WordWalkDot, ...],
        doc_walk: tuple[WordWalkDot, ...],
        idf_fn: Callable[[str], float],
        *,
        pair_idf_fn: Callable[[str, str], float] | None = None,
    ) -> tuple[float, tuple[WalkWitness, ...]]:
        witnesses: list[WalkWitness] = []
        total = 0.0
        for q in query_walk:
            best = 0.0
            best_d: WordWalkDot | None = None
            for d in doc_walk:
                gw = geometric_dot_witness(q, d)
                if gw > best:
                    best = gw
                    best_d = d
            idf = cls._dot_idf(q, idf_fn, pair_idf_fn)
            weighted = idf * best
            total += weighted
            witnesses.append(
                WalkWitness(
                    query_dot=q.explain(),
                    best_witness=best,
                    best_doc_dot=best_d.explain() if best_d and best > 0 else None,
                    idf=idf,
                    weighted=weighted,
                )
            )
        return total, tuple(witnesses)

    @classmethod
    def score(
        cls,
        query: str,
        doc_id: str,
        *,
        query_walk: tuple[WordWalkDot, ...],
        doc_walk: tuple[WordWalkDot, ...],
        shell_index: dict[str, tuple[TermCorrelationShell, ...]],
        placements: dict[str, DocLatticePlacement],
        idf_fn: Callable[[str], float],
        pair_idf_fn: Callable[[str, str], float] | None = None,
        shell_weight: float = 1.0,
    ) -> WalkerMaxSimTrace:
        walk_score, walk_witnesses = cls.walk_maxsim(
            query_walk, doc_walk, idf_fn, pair_idf_fn=pair_idf_fn
        )
        terms = _words(query)
        shell_score, shell_trace = Lattice2CorrelationPass.score(
            terms,
            doc_id,
            shell_index,
            idf_fn,
            placements=placements,
        )
        total = walk_score + shell_weight * shell_score
        return WalkerMaxSimTrace(
            doc_id=doc_id,
            walk_score=walk_score,
            shell_score=shell_score,
            total_score=total,
            walk_witnesses=walk_witnesses,
            shell_trace=shell_trace,
        )


@dataclass(frozen=True)
class WalkerMaxSimHit:
    doc_id: str
    score: float
    trace: WalkerMaxSimTrace

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "trace": self.trace.explain(),
        }


@dataclass(frozen=True)
class WalkerMaxSimRetrieveTrace:
    query: str
    filter_steps: tuple[dict, ...]
    pool_size: int
    lit_docs: tuple[str, ...]
    query_walk: tuple[dict, ...]
    hits: tuple[WalkerMaxSimHit, ...]

    def explain(self) -> dict:
        return {
            "query": self.query,
            "pool_size": self.pool_size,
            "lit_docs": list(self.lit_docs),
            "filter_steps": list(self.filter_steps),
            "query_walk": list(self.query_walk),
            "hits": [h.explain() for h in self.hits],
        }


@dataclass
class WalkerMaxSimRetriever:
    """
    Zero-shot walker retriever — pair-origin walk + MaxSim + correlation shells.

    Subsumes ``ZeroShotTwoLatticeRetriever`` routing/scoring with walk MaxSim layer.
    """

    semantic: SemanticLightIndex = field(default_factory=lambda: SemanticLightIndex())
    shared: SharedTermIndex = field(default_factory=SharedTermIndex)
    pairs: OrientedPairIndex = field(default_factory=OrientedPairIndex)
    placements: dict[str, DocLatticePlacement] = field(default_factory=dict)
    shell_index: dict[str, tuple[TermCorrelationShell, ...]] = field(default_factory=dict)
    doc_walks: dict[str, tuple[WordWalkDot, ...]] = field(default_factory=dict)
    shell_weight: float = 1.0

    def _term_prime(self, term: str) -> int:
        return self.semantic._prime_for_term(term)

    def index_doc(
        self,
        doc_id: str,
        text: str,
        *,
        placement: DocLatticePlacement,
        shells: tuple[TermCorrelationShell, ...],
    ) -> None:
        walk = word_pair_walk(text, self.semantic)
        self.doc_walks[doc_id] = walk
        self.placements[doc_id] = placement
        self.shell_index[doc_id] = shells
        self.pairs.index_doc(doc_id, walk)
        self.shared.index_placement(placement, term_prime_fn=self._term_prime)

    def _route_pool(
        self,
        query_terms: list[str],
        query_walk: tuple[WordWalkDot, ...],
    ) -> tuple[set[str], list[dict]]:
        pool, steps = self.pairs.route_pool(query_walk)
        if pool:
            return pool, steps
        return self.shared.route_pool(query_terms, term_prime_fn=self._term_prime)

    def retrieve_with_trace(self, query: str, *, limit: int = 10) -> WalkerMaxSimRetrieveTrace:
        terms = [w for w in _words(query) if len(w) >= 2]
        query_walk = word_pair_walk(query, self.semantic)
        pool, steps = self._route_pool(terms, query_walk)
        hits: list[WalkerMaxSimHit] = []
        for doc_id in sorted(pool):
            trace = WalkerMaxSimPass.score(
                query,
                doc_id,
                query_walk=query_walk,
                doc_walk=self.doc_walks.get(doc_id, ()),
                shell_index=self.shell_index,
                placements=self.placements,
                idf_fn=self.shared.idf,
                pair_idf_fn=self.pairs.idf_pair,
                shell_weight=self.shell_weight,
            )
            hits.append(WalkerMaxSimHit(doc_id=doc_id, score=trace.total_score, trace=trace))
        hits.sort(key=lambda h: (-h.score, h.doc_id))
        return WalkerMaxSimRetrieveTrace(
            query=query,
            filter_steps=tuple(steps),
            pool_size=len(pool),
            lit_docs=tuple(sorted(pool)),
            query_walk=tuple(d.explain() for d in query_walk),
            hits=tuple(hits[:limit]),
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[WalkerMaxSimHit]:
        return list(self.retrieve_with_trace(query, limit=limit).hits)


def build_walker_maxsim_retriever(
    corpus: dict[str, str],
    *,
    registry: Stage04Registry | None = None,
    shell_weight: float = 1.0,
) -> WalkerMaxSimRetriever:
    """Index corpus — bare placement + walk + per-doc correlation shells."""
    reg = registry or promote_from_stream(list(corpus.values()))
    pool = DocPrimePool()
    sem = SemanticLightIndex(registry=reg)
    retriever = WalkerMaxSimRetriever(semantic=sem, shell_weight=shell_weight)
    for doc_id, text in corpus.items():
        placement = encode_doc(doc_id, text, reg, pool, semantic=sem)
        shells = build_doc_correlation_shells(text, reg)
        retriever.index_doc(doc_id, text, placement=placement, shells=shells)
    return retriever
