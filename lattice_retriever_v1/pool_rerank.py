"""Pool-only zero-shot rerank passes — MNCR, CHMR, and Gravity-Weighted Cascade (GWCR)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from lattice_retriever_v1.lattice2_correlation import (
    DocLatticePlacement,
    Lattice2CorrelationPass,
    TermCorrelationShell,
)
from lattice_retriever_v1.stage07_semantic_light import HUB_WORDS

if TYPE_CHECKING:
    from lattice_retriever_v1.hybrid_retriever import HybridHit


@dataclass(frozen=True)
class MNCRBoostTrace:
    """Glass-box breakdown for one doc's missing-neighbor correlation boost."""

    doc_id: str
    raw_boost: float
    weighted_boost: float
    rare_terms: tuple[str, ...]
    hits: tuple[dict, ...]

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "raw_boost": round(self.raw_boost, 6),
            "weighted_boost": round(self.weighted_boost, 6),
            "rare_terms": list(self.rare_terms),
            "hits": list(self.hits),
        }


def _query_rare_terms(
    query_terms: list[str],
    idf_fn: Callable[[str], float],
    *,
    idf_gate: float,
    max_rare_terms: int,
    hub_words: frozenset[str],
) -> tuple[str, ...]:
    candidates = [
        t
        for t in query_terms
        if t not in hub_words and idf_fn(t) >= idf_gate
    ]
    candidates.sort(key=lambda t: (-idf_fn(t), t))
    return tuple(candidates[:max_rare_terms])


def _doc_missing_neighbor_boost(
    doc_id: str,
    rare_terms: tuple[str, ...],
    query_term_set: frozenset[str],
    shells: tuple[TermCorrelationShell, ...],
    words: tuple[str, ...],
    idf_fn: Callable[[str], float],
) -> MNCRBoostTrace:
    word_set = set(words)
    raw = 0.0
    hits: list[dict] = []
    for r in rare_terms:
        idf_r = idf_fn(r)
        for shell in shells:
            if shell.witness_weight(r) <= 0:
                continue
            for n, nb in shell.neighbors.items():
                if n in query_term_set or n not in word_set:
                    continue
                term_boost = idf_r * idf_fn(n) * nb.drift_weight
                if term_boost <= 0:
                    continue
                raw += term_boost
                hits.append(
                    {
                        "rare_term": r,
                        "neighbor": n,
                        "shell": shell.key,
                        "drift_weight": round(nb.drift_weight, 4),
                        "boost": round(term_boost, 4),
                    }
                )
    return MNCRBoostTrace(
        doc_id=doc_id,
        raw_boost=raw,
        weighted_boost=0.0,
        rare_terms=rare_terms,
        hits=tuple(hits),
    )


def missing_neighbor_rerank(
    hits: list[HybridHit],  # noqa: F821
    query_terms: list[str],
    shell_index: dict[str, tuple[TermCorrelationShell, ...]],
    placements: dict[str, DocLatticePlacement],
    idf_fn: Callable[[str], float],
    *,
    lambda_mn: float = 0.35,
    idf_gate: float = 0.0,
    max_rare_terms: int = 4,
    hub_words: frozenset[str] = HUB_WORDS,
) -> list[HybridHit]:  # noqa: F821
    """
    Zero-shot pool rerank: boost docs whose shell neighbors (not in query) correlate
    with rare query terms via L4–L6 witness shells.
    """
    from lattice_retriever_v1.hybrid_retriever import HybridHit

    if len(hits) < 2:
        return hits
    rare_terms = _query_rare_terms(
        query_terms,
        idf_fn,
        idf_gate=idf_gate,
        max_rare_terms=max_rare_terms,
        hub_words=hub_words,
    )
    if not rare_terms:
        return hits
    query_term_set = frozenset(query_terms)
    boosted: list[HybridHit] = []
    for h in hits:
        placement = placements.get(h.doc_id)
        shells = shell_index.get(h.doc_id, ())
        words = placement.words if placement is not None else ()
        trace = _doc_missing_neighbor_boost(
            h.doc_id,
            rare_terms,
            query_term_set,
            shells,
            words,
            idf_fn,
        )
        weighted = lambda_mn * trace.raw_boost
        if weighted <= 0:
            boosted.append(h)
            continue
        trace = MNCRBoostTrace(
            doc_id=trace.doc_id,
            raw_boost=trace.raw_boost,
            weighted_boost=weighted,
            rare_terms=trace.rare_terms,
            hits=trace.hits,
        )
        l2_trace = dict(h.l2_trace)
        l2_trace["missing_neighbor_rerank"] = trace.explain()
        boosted.append(
            HybridHit(
                doc_id=h.doc_id,
                score=h.score + weighted,
                lex_score=h.lex_score,
                l2_score=h.l2_score,
                l2_trace=l2_trace,
                walk_score=h.walk_score,
                walk_trace=h.walk_trace,
                rerank_boost=h.rerank_boost,
            )
        )
    boosted.sort(key=lambda x: (-x.score, x.doc_id))
    return boosted


@dataclass(frozen=True)
class CHMRBoostTrace:
    """Glass-box breakdown for one doc's cross-hit mutual reinforcement boost."""

    doc_id: str
    raw_boost: float
    weighted_boost: float
    consensus_terms: tuple[str, ...]
    hits: tuple[dict, ...]

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "raw_boost": round(self.raw_boost, 6),
            "weighted_boost": round(self.weighted_boost, 6),
            "consensus_terms": list(self.consensus_terms),
            "hits": list(self.hits),
        }


def _hit_rare_witness_terms(
    hit: HybridHit,  # noqa: F821
    idf_fn: Callable[[str], float],
    *,
    hub_words: frozenset[str],
    idf_gate: float,
) -> frozenset[str]:
    witnesses = hit.l2_trace.get("term_witnesses") or ()
    return frozenset(
        w["term"]
        for w in witnesses
        if w.get("witness", 0) > 0
        and w["term"] not in hub_words
        and idf_fn(w["term"]) >= idf_gate
    )


def cross_hit_mutual_rerank(
    hits: list[HybridHit],  # noqa: F821
    idf_fn: Callable[[str], float],
    *,
    lambda_ch: float = 0.25,
    consensus_k: int = 20,
    min_consensus: int = 2,
    hub_words: frozenset[str] = HUB_WORDS,
    idf_gate: float = 0.0,
) -> list[HybridHit]:  # noqa: F821
    """
    Zero-shot pool rerank: reinforce docs that share rare L2 witnesses with
    top-K preliminary hits via cross-hit consensus on term_witnesses.
    """
    from lattice_retriever_v1.hybrid_retriever import HybridHit

    if len(hits) < 2:
        return hits
    top_k = hits[:consensus_k]
    term_counts: dict[str, int] = {}
    for h in top_k:
        for t in _hit_rare_witness_terms(h, idf_fn, hub_words=hub_words, idf_gate=idf_gate):
            term_counts[t] = term_counts.get(t, 0) + 1
    consensus = {t: c for t, c in term_counts.items() if c >= min_consensus}
    if not consensus:
        return hits
    boosted: list[HybridHit] = []
    for h in hits:
        witness_terms = _hit_rare_witness_terms(h, idf_fn, hub_words=hub_words, idf_gate=idf_gate)
        hits_detail: list[dict] = []
        raw = 0.0
        for t, count in consensus.items():
            if t not in witness_terms:
                continue
            term_boost = idf_fn(t) * math.log1p(count)
            if term_boost <= 0:
                continue
            raw += term_boost
            hits_detail.append(
                {
                    "term": t,
                    "consensus_count": count,
                    "idf": round(idf_fn(t), 4),
                    "boost": round(term_boost, 4),
                }
            )
        weighted = lambda_ch * raw
        if weighted <= 0:
            boosted.append(h)
            continue
        trace = CHMRBoostTrace(
            doc_id=h.doc_id,
            raw_boost=raw,
            weighted_boost=weighted,
            consensus_terms=tuple(consensus),
            hits=tuple(hits_detail),
        )
        l2_trace = dict(h.l2_trace)
        l2_trace["cross_hit_mutual_rerank"] = trace.explain()
        boosted.append(
            HybridHit(
                doc_id=h.doc_id,
                score=h.score + weighted,
                lex_score=h.lex_score,
                l2_score=h.l2_score,
                l2_trace=l2_trace,
                walk_score=h.walk_score,
                walk_trace=h.walk_trace,
                rerank_boost=h.rerank_boost,
            )
        )
    boosted.sort(key=lambda x: (-x.score, x.doc_id))
    return boosted


@dataclass(frozen=True)
class GWCRBoostTrace:
    """Glass-box breakdown for one doc's gravity-weighted cascade rerank boost."""

    doc_id: str
    raw_boost: float
    weighted_boost: float
    satellites: tuple[str, ...]
    hits: tuple[dict, ...]

    def explain(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "raw_boost": round(self.raw_boost, 6),
            "weighted_boost": round(self.weighted_boost, 6),
            "satellites": list(self.satellites),
            "hits": list(self.hits),
        }


def _mncr_satellite_terms(hits: list[HybridHit]) -> frozenset[str]:  # noqa: F821
    """Satellites already boosted by MNCR — skip to reduce double-count risk."""
    terms: set[str] = set()
    for h in hits:
        mn = h.l2_trace.get("missing_neighbor_rerank")
        if not mn:
            continue
        for detail in mn.get("hits") or ():
            n = detail.get("neighbor")
            if n:
                terms.add(n)
    return frozenset(terms)


def _anchor_gravity_terms(
    query_terms: list[str],
    shells: tuple[TermCorrelationShell, ...],
    idf_fn: Callable[[str], float],
    *,
    hub_words: frozenset[str],
) -> tuple[tuple[str, float], ...]:
    scored: list[tuple[str, float]] = []
    for t in query_terms:
        if t in hub_words:
            continue
        witness, _ = Lattice2CorrelationPass.witness(t, shells, words=())
        if witness <= 0:
            continue
        scored.append((t, idf_fn(t) * witness))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return tuple(scored)


def _build_gravity_satellites(
    anchors: list[HybridHit],  # noqa: F821
    query_terms: list[str],
    shell_index: dict[str, tuple[TermCorrelationShell, ...]],
    idf_fn: Callable[[str], float],
    is_rare_fn: Callable[[str], bool],
    *,
    idf_gate: float,
    max_satellites: int,
    hub_words: frozenset[str],
    skip_terms: frozenset[str],
) -> dict[str, float]:
    query_term_set = frozenset(query_terms)
    satellites: dict[str, float] = {}
    for rank, anchor in enumerate(anchors, start=1):
        w_a = 1.0 / rank
        shells = shell_index.get(anchor.doc_id, ())
        if not shells:
            continue
        for g, _ in _anchor_gravity_terms(
            query_terms, shells, idf_fn, hub_words=hub_words
        ):
            idf_g = idf_fn(g)
            for shell in shells:
                if shell.witness_weight(g) <= 0:
                    continue
                for n, nb in shell.neighbors.items():
                    if n in query_term_set or n in skip_terms:
                        continue
                    if not is_rare_fn(n) or idf_fn(n) < idf_gate:
                        continue
                    satellites[n] = satellites.get(n, 0.0) + w_a * idf_g * nb.drift_weight
    if len(satellites) <= max_satellites:
        return satellites
    top = sorted(satellites.items(), key=lambda x: (-x[1], x[0]))[:max_satellites]
    return dict(top)


def gravity_cascade_rerank(
    hits: list[HybridHit],  # noqa: F821
    query_terms: list[str],
    shell_index: dict[str, tuple[TermCorrelationShell, ...]],
    idf_fn: Callable[[str], float],
    is_rare_fn: Callable[[str], bool],
    *,
    lambda_gw: float = 0.30,
    anchor_count: int = 3,
    max_satellites: int = 50,
    idf_gate: float = 0.0,
    hub_words: frozenset[str] = HUB_WORDS,
) -> list[HybridHit]:  # noqa: F821
    """
    Zero-shot pool rerank: propagate rare shell satellites from top anchor docs
    through a capped gravity table, then boost prelim hits by cascade witness.
    """
    from lattice_retriever_v1.hybrid_retriever import HybridHit

    if len(hits) < 2:
        return hits
    anchors = hits[: max(1, min(anchor_count, len(hits)))]
    skip_terms = _mncr_satellite_terms(hits)
    satellites = _build_gravity_satellites(
        anchors,
        query_terms,
        shell_index,
        idf_fn,
        is_rare_fn,
        idf_gate=idf_gate,
        max_satellites=max_satellites,
        hub_words=hub_words,
        skip_terms=skip_terms,
    )
    if not satellites:
        return hits
    boosted: list[HybridHit] = []
    for h in hits:
        shells = shell_index.get(h.doc_id, ())
        hits_detail: list[dict] = []
        raw = 0.0
        for sat, sat_weight in satellites.items():
            witness, _ = Lattice2CorrelationPass.witness(sat, shells, words=())
            if witness <= 0:
                continue
            term_cascade = sat_weight * witness
            raw += term_cascade
            hits_detail.append(
                {
                    "satellite": sat,
                    "satellite_weight": round(sat_weight, 4),
                    "witness": round(witness, 4),
                    "cascade": round(term_cascade, 4),
                }
            )
        weighted = lambda_gw * raw
        if weighted <= 0:
            boosted.append(h)
            continue
        trace = GWCRBoostTrace(
            doc_id=h.doc_id,
            raw_boost=raw,
            weighted_boost=weighted,
            satellites=tuple(satellites),
            hits=tuple(hits_detail),
        )
        l2_trace = dict(h.l2_trace)
        l2_trace["gravity_cascade_rerank"] = trace.explain()
        boosted.append(
            HybridHit(
                doc_id=h.doc_id,
                score=h.score + weighted,
                lex_score=h.lex_score,
                l2_score=h.l2_score,
                l2_trace=l2_trace,
                walk_score=h.walk_score,
                walk_trace=h.walk_trace,
                rerank_boost=h.rerank_boost,
            )
        )
    boosted.sort(key=lambda x: (-x.score, x.doc_id))
    return boosted
