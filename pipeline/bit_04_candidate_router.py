"""
BIT 4 — Attractor candidate router C(q)

Math:
  C(q) = ⋃_{w ∈ query} N(κ(cell(w)), r)
       ∪ ⋃_{nb ∈ L4-L6(w)} N(κ(cell(nb)), r)   [κ-neighbor expansion]

Pipeline:
  query → C(q) → rank_with_hub_signatures(C(q))

Fallback when |C(q)| < min_candidates: BIT 7 meet factors, then eval_beir cascade.
Full lexical doc union (union_lexical) is opt-in — default False.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from aethos_promotion import is_stopword
from aethos_tokenize import tokenize_words
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N, word_to_spacetime_cell
from pipeline.bit_02_attractor_key import (
    AttractorKey,
    attractor_neighbors,
    kappa_from_cell,
    kappa_pair_meet,
)
from pipeline.bit_03_doc_attractor_set import (
    DEFAULT_PAIR_RARE_GATE,
    CorpusAttractorIndex,
)

DEFAULT_HUB_IDF_GATE = 2.0
DEFAULT_PAIR_KEY_QUERY = True

DEFAULT_RADIUS = 1
DEFAULT_MIN_CANDIDATES = 8
DEFAULT_RECALL_TARGET = 0.90
DEFAULT_UNION_LEXICAL = False
DEFAULT_UNION_QUERY_LEXICAL = False  # opt-in; naive inv[w] union blows up |C| on SciFact
# Routing uses query-word κ only (plan BIT 4). L4–L6 κ expansion is for signal 8a (BIT 9/10).
DEFAULT_EXPAND_NEIGHBOR_KAPPA = False
DEFAULT_MAX_NEIGHBOR_WORDS = 8
DEFAULT_MEET_SUPPLEMENT = True
DEFAULT_MAX_ROUTE_CANDIDATES = 600
DEFAULT_LEXICAL_MAX_POSTINGS = 150
DEFAULT_LEXICAL_MAX_DF_FRAC = 0.12
DEFAULT_UNION_LEXICAL_ANCHORS = True


@dataclass(frozen=True)
class CandidateRouteResult:
    """Result of BIT 4 routing for one query."""

    doc_ids: list[str]
    tier: str
    n_attractor: int
    n_merged: int
    query_keys: frozenset[AttractorKey]
    n_query_keys: int = 0
    protected_doc_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Bit04GateReport:
    """BIT 4 gate metrics — in-corpus gold only."""

    recall_attr: float
    recall_attr_neighbor: float
    recall_merged: float
    n_queries: int
    n_gold_pairs: int
    n_gold_pairs_in_corpus: int
    n_gold_missing_corpus: int
    gold_missing_rate: float
    target: float
    passed: bool
    failures: list[tuple[str, str]] = field(default_factory=list)


def query_words_for_routing(words: Sequence[str], *, min_len: int = 3) -> list[str]:
    """Content words that map to lattice cells (matches hub/query profile)."""
    out: list[str] = []
    for w in words:
        if not w.isalpha() or len(w) < min_len or is_stopword(w):
            continue
        out.append(w.lower())
    return out


def _kappa_neighborhood_for_word(
    registry,
    word: str,
    *,
    n: int,
    radius: int,
) -> set[AttractorKey]:
    cell = word_to_spacetime_cell(registry, word, n=n)
    center = kappa_from_cell(cell, quantize=1.0)
    return attractor_neighbors(center, radius=radius)


def query_attractor_keys(
    registry,
    words: Sequence[str],
    *,
    n: int = DEFAULT_ANCHOR_N,
    radius: int = DEFAULT_RADIUS,
    min_len: int = 3,
    neighbor_map: dict[str, dict[str, float]] | None = None,
    expand_neighbors: bool = False,
    max_neighbor_words: int = DEFAULT_MAX_NEIGHBOR_WORDS,
    idf: Callable[[str], float] | None = None,
    hub_idf_gate: float = DEFAULT_HUB_IDF_GATE,
    pair_keys: bool = DEFAULT_PAIR_KEY_QUERY,
    pair_rare_gate: float = DEFAULT_PAIR_RARE_GATE,
    max_pair_keys: int = 15,
    hub_compound_keys: bool = True,
    max_hub_compound_keys: int = 24,
) -> set[AttractorKey]:
    """
    Keys for C(q): ⋃ N(κ(cell(w)), r) over query words.

    hub_compound_keys: high-df hubs (cell, cancer) route ONLY as hub+rare
    pair-meet keys matching ingest — not naked hub κ fan.
    """
    keys: set[AttractorKey] = set()
    all_routed = query_words_for_routing(words, min_len=min_len)
    rare_routed: list[str] = []
    hub_routed: list[str] = []

    if idf is not None:
        gate = hub_idf_gate if 0 < hub_idf_gate < 50 else 0.0
        for w in all_routed:
            iv = idf(w)
            if hub_compound_keys and iv < gate:
                hub_routed.append(w)
            elif iv >= gate:
                rare_routed.append(w)
    else:
        rare_routed = list(all_routed)

    word_key: dict[str, AttractorKey] = {}
    for w in rare_routed:
        try:
            nb = _kappa_neighborhood_for_word(registry, w, n=n, radius=radius)
            keys |= nb
            cell = word_to_spacetime_cell(registry, w, n=n)
            word_key[w] = kappa_from_cell(cell, quantize=1.0)
        except Exception:
            continue
        if expand_neighbors and neighbor_map:
            nbs = sorted(
                neighbor_map.get(w, {}).items(),
                key=lambda x: (-x[1], x[0]),
            )[:max_neighbor_words]
            for nb, _weight in nbs:
                if not nb.isalpha() or len(nb) < min_len or is_stopword(nb):
                    continue
                try:
                    keys |= _kappa_neighborhood_for_word(registry, nb, n=n, radius=radius)
                except Exception:
                    continue

    hub_keys: dict[str, AttractorKey] = {}
    for w in hub_routed:
        try:
            cell = word_to_spacetime_cell(registry, w, n=n)
            hub_keys[w] = kappa_from_cell(cell, quantize=1.0)
        except Exception:
            continue

    if hub_compound_keys and hub_keys and word_key:
        n_hc = 0
        rare_ws = sorted(word_key, key=lambda w: (-idf(w), w) if idf else (0, w))[:6]
        for hw in hub_keys:
            for rw in rare_ws:
                if n_hc >= max_hub_compound_keys:
                    break
                keys.add(kappa_pair_meet(hub_keys[hw], word_key[rw]))
                n_hc += 1
            if n_hc >= max_hub_compound_keys:
                break

    if pair_keys and idf is not None and len(word_key) >= 2:
        rare = sorted(
            [w for w in word_key if idf(w) >= pair_rare_gate],
            key=lambda w: (-idf(w), w),
        )[:6]
        n_pairs = 0
        for w1, w2 in itertools.combinations(rare, 2):
            if n_pairs >= max_pair_keys:
                break
            keys.add(kappa_pair_meet(word_key[w1], word_key[w2]))
            n_pairs += 1
    return keys


def candidates_from_attractor_keys(
    keys: set[AttractorKey] | frozenset[AttractorKey],
    index: CorpusAttractorIndex,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        for doc_id in index.by_key.get(key, []):
            if doc_id not in seen:
                seen.add(doc_id)
                out.append(doc_id)
    return out


def candidates_from_attractors(
    words: Sequence[str],
    registry,
    index: CorpusAttractorIndex,
    *,
    n: int = DEFAULT_ANCHOR_N,
    radius: int = DEFAULT_RADIUS,
    min_len: int = 3,
    neighbor_map: dict[str, dict[str, float]] | None = None,
    expand_neighbors: bool = DEFAULT_EXPAND_NEIGHBOR_KAPPA,
    max_neighbor_words: int = DEFAULT_MAX_NEIGHBOR_WORDS,
    idf: Callable[[str], float] | None = None,
    hub_idf_gate: float = DEFAULT_HUB_IDF_GATE,
    pair_keys: bool = DEFAULT_PAIR_KEY_QUERY,
    pair_rare_gate: float = DEFAULT_PAIR_RARE_GATE,
    hub_compound_keys: bool = True,
) -> tuple[list[str], frozenset[AttractorKey]]:
    """Route docs hitting any query κ bucket (hub-filtered + optional pair-meet keys)."""
    keys = frozenset(
        query_attractor_keys(
            registry,
            words,
            n=n,
            radius=radius,
            min_len=min_len,
            neighbor_map=neighbor_map,
            expand_neighbors=expand_neighbors,
            max_neighbor_words=max_neighbor_words,
            idf=idf,
            hub_idf_gate=hub_idf_gate,
            pair_keys=pair_keys,
            pair_rare_gate=pair_rare_gate,
            hub_compound_keys=hub_compound_keys,
        )
    )
    return candidates_from_attractor_keys(keys, index), keys


def gold_recall_in_candidates(
    gold_doc_ids: set[str],
    candidates: Sequence[str],
) -> float:
    """Fraction of gold docs present in candidate set."""
    if not gold_doc_ids:
        return 1.0
    hit = len(gold_doc_ids & set(candidates))
    return hit / len(gold_doc_ids)


def _meet_supplement_candidates(
    words: Sequence[str],
    registry,
    meet_index,
    *,
    min_factor_hits: int = 1,
) -> list[str]:
    """BIT 7 precision supplement when κ pool is thin."""
    if meet_index is None:
        return []
    try:
        from pipeline.bit_07_meet_witness import (
            MeetWitnessIndex,
            candidates_from_meet_witness,
        )

        if isinstance(meet_index, MeetWitnessIndex):
            return candidates_from_meet_witness(
                words,
                registry,
                meet_index,
                min_factor_hits=min_factor_hits,
            )
    except Exception:
        pass
    return []


def lexical_anchor_docs(
    words: Sequence[str],
    inv: dict[str, set[str]],
    doc_freq: dict[str, int],
    n_docs: int,
    *,
    max_postings: int = DEFAULT_LEXICAL_MAX_POSTINGS,
    max_df_frac: float = DEFAULT_LEXICAL_MAX_DF_FRAC,
    min_len: int = 3,
) -> set[str]:
    """
    BM25 backbone docs: inv[w] for selective routed query words.

    Skips ultra-common words (huge posting lists) to avoid |C| → full corpus.
    Protected from κ trim and rank cap.
    """
    out: set[str] = set()
    routed = query_words_for_routing(words, min_len=min_len)
    selective: list[str] = []
    for w in routed:
        postings = inv.get(w)
        if not postings:
            continue
        df = doc_freq.get(w, 0)
        if len(postings) <= max_postings:
            out |= postings
            selective.append(w)
        elif n_docs > 0 and df / n_docs <= max_df_frac:
            out |= postings
            selective.append(w)
    # Multi-word AND for high-df terms: docs matching 2+ routed words
    if len(selective) >= 2:
        for i, w1 in enumerate(selective):
            p1 = inv.get(w1, set())
            for w2 in selective[i + 1 :]:
                inter = p1 & inv.get(w2, set())
                if inter:
                    out |= inter
    return out


def trim_candidates_by_kappa_hits(
    doc_ids: Sequence[str],
    keys: frozenset[AttractorKey] | set[AttractorKey],
    index: CorpusAttractorIndex,
    *,
    cap: int,
    protect: Iterable[str] = (),
) -> list[str]:
    """Hard cap |C(q)| — keep protected docs, then top κ bucket hits."""
    if cap <= 0 or len(doc_ids) <= cap:
        return list(doc_ids)
    protect_set = set(protect) & set(doc_ids)
    protected = [d for d in doc_ids if d in protect_set]
    if len(protected) >= cap:
        return protected[:cap]
    remaining = cap - len(protected)
    pool = [d for d in doc_ids if d not in protect_set]
    if remaining <= 0 or not pool:
        return protected
    key_set = set(keys)
    if not key_set:
        return protected + list(pool)[:remaining]
    scored: dict[str, int] = {}
    for doc_id in pool:
        dk = index.doc_keys.get(doc_id, set())
        hit = len(key_set & dk)
        if hit > 0:
            scored[doc_id] = hit
    if not scored:
        return protected + list(pool)[:remaining]
    trimmed = sorted(scored, key=lambda d: (-scored[d], d))[:remaining]
    return protected + trimmed


def route_query_candidates(
    words: Sequence[str],
    registry,
    index: CorpusAttractorIndex,
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    all_ids: list[str],
    *,
    n: int = DEFAULT_ANCHOR_N,
    radius: int = DEFAULT_RADIUS,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    meet_index=None,
    union_lexical: bool = DEFAULT_UNION_LEXICAL,
    expand_neighbors: bool = DEFAULT_EXPAND_NEIGHBOR_KAPPA,
    max_neighbor_words: int = DEFAULT_MAX_NEIGHBOR_WORDS,
    meet_supplement: bool = DEFAULT_MEET_SUPPLEMENT,
    union_query_lexical: bool = DEFAULT_UNION_QUERY_LEXICAL,
    union_lexical_anchors: bool = DEFAULT_UNION_LEXICAL_ANCHORS,
    max_route_candidates: int = DEFAULT_MAX_ROUTE_CANDIDATES,
    doc_freq: dict[str, int] | None = None,
    n_docs: int = 0,
) -> CandidateRouteResult:
    """
    BIT 4 router: κ (+ optional neighbor κ) → lexical anchors → meet supplement.

    Lexical anchors (selective inv[w]) are protected from κ trim.
    """
    from eval_beir import candidate_ids

    c_attr, keys = candidates_from_attractors(
        words,
        registry,
        index,
        n=n,
        radius=radius,
        neighbor_map=neighbor_map,
        expand_neighbors=expand_neighbors,
        max_neighbor_words=max_neighbor_words,
    )
    merged: set[str] = set(c_attr)
    tier = "bit4_kappa_neighbor" if expand_neighbors else "bit4_attractor"

    protected: set[str] = set()
    if union_lexical_anchors and doc_freq is not None and n_docs > 0:
        protected = lexical_anchor_docs(words, inv, doc_freq, n_docs)
        if protected:
            merged |= protected
            tier = f"{tier}_lexical"

    if meet_supplement and len(merged) < min_candidates:
        for doc_id in _meet_supplement_candidates(words, registry, meet_index):
            merged.add(doc_id)
        if len(merged) >= min_candidates and meet_index is not None:
            tier = "bit4_meet_supplement"

    if union_query_lexical:
        for w in words:
            merged |= inv.get(w, set())
        if merged and tier.startswith("bit4_"):
            tier = "bit4_attractor_lexical" if tier == "bit4_attractor" else f"{tier}_lexical"

    if union_lexical:
        for w in words:
            merged |= inv.get(w, set())
            for nb in neighbor_map.get(w, {}):
                merged |= inv.get(nb, set())
        tier = "bit4_attractor_lexical"

    if len(merged) >= min_candidates:
        doc_ids = list(merged)
        protect = frozenset(protected)
        if max_route_candidates > 0 and len(doc_ids) > max_route_candidates:
            doc_ids = trim_candidates_by_kappa_hits(
                doc_ids, keys, index, cap=max_route_candidates, protect=protect,
            )
            tier = f"{tier}_trimmed"
        return CandidateRouteResult(
            doc_ids=doc_ids,
            tier=tier,
            n_attractor=len(c_attr),
            n_merged=len(doc_ids),
            query_keys=keys,
            n_query_keys=len(keys),
            protected_doc_ids=protect,
        )

    fallback = candidate_ids(
        list(words),
        inv,
        neighbor_map,
        all_ids,
        meet_index=meet_index.legacy_dict() if hasattr(meet_index, "legacy_dict") else meet_index,
        registry=registry,
    )
    protect = frozenset(protected)
    if max_route_candidates > 0 and len(fallback) > max_route_candidates and keys:
        fallback = trim_candidates_by_kappa_hits(
            fallback, keys, index, cap=max_route_candidates, protect=protect,
        )
        tier = "bit4_fallback_trimmed"
    else:
        tier = "bit4_fallback"
    return CandidateRouteResult(
        doc_ids=fallback,
        tier=tier,
        n_attractor=len(c_attr),
        n_merged=len(fallback),
        query_keys=keys,
        n_query_keys=len(keys),
        protected_doc_ids=protect,
    )


def candidate_set_sizes(
    results: Sequence[CandidateRouteResult],
) -> dict[str, float]:
    """p50 / p95 |C(q)| for pipeline ledger."""
    if not results:
        return {"p50": 0.0, "p95": 0.0, "n": 0.0}
    sizes = sorted(r.n_merged for r in results)
    n = len(sizes)

    def pct(p: float) -> float:
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return float(sizes[i])

    return {"p50": pct(0.5), "p95": pct(0.95), "n": float(n)}


def _gold_pairs_in_corpus(
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    loaded: set[str],
) -> tuple[int, int, int]:
    """Returns (total_gold_pairs, in_corpus_pairs, missing_corpus_pairs)."""
    total = in_corpus = 0
    for qid in queries:
        gold = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        for d in gold:
            total += 1
            if d in loaded:
                in_corpus += 1
    return total, in_corpus, total - in_corpus


def verify_bit04_gate(
    registry,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    hub_sigs: dict,
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    *,
    index: CorpusAttractorIndex | None = None,
    meet_index=None,
    radius: int = DEFAULT_RADIUS,
    min_candidates: int = DEFAULT_MIN_CANDIDATES,
    target: float = DEFAULT_RECALL_TARGET,
    union_lexical: bool = DEFAULT_UNION_LEXICAL,
    expand_neighbors: bool = DEFAULT_EXPAND_NEIGHBOR_KAPPA,
) -> Bit04GateReport:
    """
    BIT 4 gate on **in-corpus gold only**.

    Reports recall_attr (query κ only), recall_attr_neighbor (L4–L6 κ expansion),
    and recall_merged (full router with union_lexical setting).
    """
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures

    if index is None:
        index = build_attractor_index_from_hub_signatures(registry, hub_sigs)

    loaded = set(doc_ids)
    total_pairs, in_corpus_pairs, missing_pairs = _gold_pairs_in_corpus(
        queries, qrels, loaded
    )

    recalls_attr: list[float] = []
    recalls_neighbor: list[float] = []
    recalls_merged: list[float] = []
    failures: list[tuple[str, str]] = []

    for qid, query in queries.items():
        gold_all = {d for d, r in qrels.get(qid, {}).items() if r > 0}
        gold = gold_all & loaded
        if not gold:
            continue
        words = tokenize_words(query)

        c_pure, _ = candidates_from_attractors(
            words, registry, index, n=DEFAULT_ANCHOR_N, radius=radius,
            expand_neighbors=False,
        )
        recalls_attr.append(gold_recall_in_candidates(gold, c_pure))

        c_nb, _ = candidates_from_attractors(
            words, registry, index, n=DEFAULT_ANCHOR_N, radius=radius,
            neighbor_map=neighbor_map, expand_neighbors=True,
        )
        recalls_neighbor.append(gold_recall_in_candidates(gold, c_nb))

        route = route_query_candidates(
            words,
            registry,
            index,
            inv,
            neighbor_map,
            doc_ids,
            radius=radius,
            min_candidates=min_candidates,
            meet_index=meet_index,
            union_lexical=union_lexical,
            expand_neighbors=expand_neighbors,
        )
        rec_m = gold_recall_in_candidates(gold, route.doc_ids)
        recalls_merged.append(rec_m)

        if rec_m < target:
            failures.append(
                (
                    qid,
                    f"recall_merged={rec_m:.2f} tier={route.tier} "
                    f"|C|={route.n_merged} keys={route.n_query_keys} "
                    f"gold={len(gold)} hit={len(gold & set(route.doc_ids))}",
                )
            )

    recall_attr = sum(recalls_attr) / len(recalls_attr) if recalls_attr else 0.0
    recall_nb = sum(recalls_neighbor) / len(recalls_neighbor) if recalls_neighbor else 0.0
    recall_merged = sum(recalls_merged) / len(recalls_merged) if recalls_merged else 0.0
    missing_rate = missing_pairs / total_pairs if total_pairs else 0.0

    return Bit04GateReport(
        recall_attr=recall_attr,
        recall_attr_neighbor=recall_nb,
        recall_merged=recall_merged,
        n_queries=len(recalls_merged),
        n_gold_pairs=total_pairs,
        n_gold_pairs_in_corpus=in_corpus_pairs,
        n_gold_missing_corpus=missing_pairs,
        gold_missing_rate=missing_rate,
        target=target,
        passed=recall_merged >= target,
        failures=failures,
    )


def verify_bit04_gate_legacy_tuple(
    registry,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    hub_sigs: dict,
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    **kwargs,
) -> tuple[bool, float, list[tuple[str, str]]]:
    """Backward-compatible (passed, recall_merged, failures)."""
    report = verify_bit04_gate(
        registry, queries, qrels, doc_ids, doc_tokens, hub_sigs, inv, neighbor_map, **kwargs
    )
    return report.passed, report.recall_merged, report.failures
