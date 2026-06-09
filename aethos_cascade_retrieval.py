"""
Cascade retrieval — the real formula (zero-shot, no training).

Memory stays small: store intersections (2-way meets), not hub dots everywhere.
Query rotates on the 3D complex plane through rare primes/composites:

  1. Rare signal triggers (words + morph subwords) — hub/membrane skipped
  2. 2-way meet witnesses at deterministic κ positions (link strength)
  3. 3-way triple cascade — bridge completes when A∩B∩B∩C align; strengthens rank

Cross-correlations are built at ingest; 3-way bridges retrieval without qrel training.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Sequence

from aethos_query_oov import ephemeral_word_kappa_keys, word_needs_oov_build
from aethos_rare_rank import (
    _DocFreqCache,
    _pair_strength_from_adj,
    degree_map_from_plane,
    morph_trigger_pieces,
    rare_neighbors,
    rare_query_triggers,
)
from pipeline.bit_02_attractor_key import AttractorKey, kappa_branch_fan
from pipeline.bit_04_candidate_router import CandidateRouteResult
from pipeline.bit_12_symbol_plane_index import (
    correlation_meet_keys,
    query_symbol_plane_keys,
    symbol_word_chain,
    symbol_word_imag,
)

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex
    from pipeline.bit_12_symbol_plane_index import SymbolPlaneIndex

TWO_WAY_WEIGHT = 1.0
TRIPLE_BRIDGE_WEIGHT = 8.0
TRIPLE_FULL_WITNESS_WEIGHT = 12.0
SOLO_RARE_WEIGHT = 0.25
KAPPA_ROUTE_WEIGHT = 1.0


def _rail_from_imag(imag: int) -> float:
    from pipeline.bit_12_symbol_plane_index import _rail_from_imag as _r

    return _r(imag)


def triple_meet_keys(
    knowledge: SymbolKnowledgeIndex,
    a: str,
    b: str,
    c: str,
    *,
    quantize: float,
) -> frozenset[AttractorKey]:
    """3-way intersection witness: chain(A∪B∪C) @ imag(A)+imag(B)+imag(C), 4-wing fan."""
    chain = tuple(sorted(set(
        symbol_word_chain(knowledge, a)
        + symbol_word_chain(knowledge, b)
        + symbol_word_chain(knowledge, c)
    )))
    if not chain:
        return frozenset()
    imag = (
        symbol_word_imag(knowledge, a)
        + symbol_word_imag(knowledge, b)
        + symbol_word_imag(knowledge, c)
    )
    rail = _rail_from_imag(imag)
    return frozenset(kappa_branch_fan(chain, rail, quantize=quantize))


def _pair_meet_keys(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    left: str,
    right: str,
) -> frozenset[AttractorKey]:
    pair = tuple(sorted((left.lower(), right.lower())))
    cached = plane.pair_keys.get(pair)
    if cached:
        return cached
    return correlation_meet_keys(
        knowledge, left, right, quantize=plane.quantize,
    )


def _score_keys_on_docs(
    keys: frozenset[AttractorKey] | set[AttractorKey],
    plane: SymbolPlaneIndex,
    doc_scores: dict[str, float],
    weight: float,
) -> set[str]:
    """Add weight to docs touching witness keys; return doc ids hit."""
    hit: set[str] = set()
    for k in keys:
        for did in plane.by_key.get(k, ()):
            doc_scores[did] += weight
            hit.add(did)
    return hit


def search_docs_cascade(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    max_rare_neighbors: int = 8,
    max_candidates: int = 600,
    limit: int = 100,
    max_keys: int = 768,
    max_corr_neighbors: int = 4,
    expand_correlations: bool = True,
    kappa_route_weight: float = KAPPA_ROUTE_WEIGHT,
) -> tuple[CandidateRouteResult, list[tuple[str, float]]]:
    """
    κ plane routing + rare 2-way meets + 3-way bridge cascade.

    Route pool from 3D complex-plane keys (recall); rank strengthened by
    intersection witnesses built at ingest (precision). No training.
    """
    kappa_keys = query_symbol_plane_keys(
        knowledge,
        plane,
        words,
        expand_correlations=expand_correlations,
        max_keys=max_keys,
        max_corr_neighbors=max_corr_neighbors,
    )
    route_hits: Counter[str] = Counter()
    for k in kappa_keys:
        for did in plane.by_key.get(k, ()):
            route_hits[did] += 1

    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    rare_cache: dict[str, bool] = {}
    degrees = degree_map_from_plane(plane)
    triggers = rare_query_triggers(
        knowledge, words, df_cache=cache, rare_cache=rare_cache, degrees=degrees,
    )

    # Morph-only fallback when every query token is hub (e.g. short generic query).
    if not triggers:
        for token in words:
            for piece in morph_trigger_pieces(knowledge, token):
                if len(piece) >= 3:
                    triggers.append(piece.lower())
        triggers = list(dict.fromkeys(triggers))

    adj = plane.word_adjacency
    nbr_kw = dict(
        df_cache=cache,
        adjacency=adj,
        rare_cache=rare_cache,
        degrees=degrees,
    )

    cascade_scores: dict[str, float] = defaultdict(float)
    witness_keys: set[AttractorKey] = set(kappa_keys)
    pair_doc_sets: dict[tuple[str, str], set[str]] = {}

    for w in triggers:
        solo_keys: set[AttractorKey] = set()
        if word_needs_oov_build(knowledge, plane, w):
            knowledge.ensure_query_lattice(w, plane)
            solo_keys |= ephemeral_word_kappa_keys(
                knowledge, w, quantize=plane.quantize,
            )
        else:
            solo_keys |= plane.keys_for_word(w)
        witness_keys |= solo_keys
        _score_keys_on_docs(solo_keys, plane, cascade_scores, SOLO_RARE_WEIGHT)

        for other, nb_strength in rare_neighbors(
            knowledge, w, limit=max_rare_neighbors, **nbr_kw,
        ):
            strength = _pair_strength_from_adj(adj, w, other) or nb_strength
            meet = _pair_meet_keys(knowledge, plane, w, other)
            if not meet:
                continue
            witness_keys |= meet
            pair = tuple(sorted((w, other)))
            docs = _score_keys_on_docs(
                meet, plane, cascade_scores, strength * TWO_WAY_WEIGHT,
            )
            if docs:
                pair_doc_sets[pair] = pair_doc_sets.get(pair, set()) | docs

    trigger_list = sorted(set(triggers))

    # 2-way query triples (pairwise links among rare query triggers).
    for a, b in combinations(trigger_list, 2):
        ab_strength = _pair_strength_from_adj(adj, a, b)
        if ab_strength is None:
            continue
        meet = _pair_meet_keys(knowledge, plane, a, b)
        if not meet:
            continue
        witness_keys |= meet
        pair = tuple(sorted((a, b)))
        docs = _score_keys_on_docs(
            meet, plane, cascade_scores,
            (TRIPLE_BRIDGE_WEIGHT * 0.5) + ab_strength,
        )
        if docs:
            pair_doc_sets[pair] = pair_doc_sets.get(pair, set()) | docs

    # 3-way bridge cascade — strengthens when three rare witnesses align.
    for a, b, c in combinations(trigger_list, 3):
        pairs = (
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((a, c))),
        )
        strengths = [
            _pair_strength_from_adj(adj, x, y)
            for x, y in ((a, b), (b, c), (a, c))
        ]
        known = [s for s in strengths if s is not None]
        if len(known) < 2:
            continue

        tkeys = triple_meet_keys(knowledge, a, b, c, quantize=plane.quantize)
        if tkeys:
            witness_keys |= tkeys
            bridge = min(known)
            _score_keys_on_docs(
                tkeys, plane, cascade_scores,
                TRIPLE_BRIDGE_WEIGHT + bridge,
            )

        # Docs sharing all three 2-way meets = full triple witness (noise filter).
        doc_sets = [pair_doc_sets.get(p) for p in pairs]
        if all(ds for ds in doc_sets):
            full_witness = doc_sets[0] & doc_sets[1] & doc_sets[2]
            bonus = TRIPLE_FULL_WITNESS_WEIGHT + sum(known)
            for did in full_witness:
                cascade_scores[did] += bonus

    route_ids = [did for did, _ in route_hits.most_common(max_candidates)]

    combined: dict[str, float] = {}
    for did in route_ids:
        hits = route_hits[did]
        overlap = plane.score_overlap(kappa_keys, did) if kappa_keys else 0.0
        cascade = cascade_scores.get(did, 0.0)
        combined[did] = kappa_route_weight * (hits + overlap) + cascade

    ranked = sorted(combined.items(), key=lambda x: (-x[1], x[0]))

    route = CandidateRouteResult(
        doc_ids=route_ids,
        tier="cascade",
        n_attractor=len(route_ids),
        n_merged=len(route_ids),
        query_keys=frozenset(witness_keys),
        n_query_keys=len(witness_keys),
    )
    return route, ranked[:limit]
