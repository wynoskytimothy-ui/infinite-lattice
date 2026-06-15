"""
Rare-correlation query path — zero-shot retrieval through rare primes / composites.

Query triggers rare words + morph subwords → traverse only rare correlation links.
Doc scores are stored link strengths at intersection meet keys (not a separate rerank).
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import TYPE_CHECKING, Sequence

from aethos_symbol_cellular import CellularRole, _DEFAULT_MEMBRANE
from pipeline.bit_02_attractor_key import AttractorKey
from pipeline.bit_04_candidate_router import CandidateRouteResult, query_words_for_routing
from pipeline.bit_12_symbol_plane_index import (
    SymbolPlaneIndex,
    query_symbol_plane_keys,
)

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")

DEFAULT_RARE_DOC_FREQ = 50
HUB_DEGREE_LIMIT = 500
KAPPA_WEIGHT = 0.4
RARE_WEIGHT = 0.6
TRIPLE_BONUS = 3.0
# Ingest-time multipliers (stored link strength, not query re-rank)
INGEST_BOTH_RARE_FACTOR = 3.0
INGEST_ONE_RARE_FACTOR = 2.0
INGEST_TRIPLE_RARE_FACTOR = 2.0  # bridge via also rare — all three signal


def _tokens(text: str, *, min_len: int = 3) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _DEFAULT_MEMBRANE and len(t) >= min_len
    ]


def _unique_tokens(text: str, *, min_len: int = 3) -> list[str]:
    return list(dict.fromkeys(_tokens(text, min_len=min_len)))


class _DocFreqCache:
    def __init__(self, knowledge: SymbolKnowledgeIndex) -> None:
        self._knowledge = knowledge
        self._cache: dict[str, int] = {}
        self._corpus_warmed = False

    def warm_corpus(self) -> None:
        """Load doc-frequency from cellular profiles (built at index time)."""
        if self._corpus_warmed:
            return
        for w, prof in self._knowledge.cellular.profiles.items():
            self._cache[w] = prof.doc_count
        self._corpus_warmed = True

    def warm(self, words: set[str]) -> None:
        if not self._corpus_warmed:
            pending = {w.lower() for w in words} - set(self._cache)
            if not pending:
                return
            counts = dict.fromkeys(pending, 0)
            for text in self._knowledge.corpus.values():
                doc_toks = set(_tokens(text))
                for w in pending:
                    if w in doc_toks:
                        counts[w] += 1
            self._cache.update(counts)
            return
        for w in words:
            self._cache.setdefault(w.lower(), 0)

    def get(self, word: str) -> int:
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        prof = self._knowledge.cellular.profiles.get(w)
        if prof is not None:
            self._cache[w] = prof.doc_count
            return prof.doc_count
        # Unknown token: treat as common (avoid O(corpus) scan per token).
        self._cache[w] = len(self._knowledge.corpus) + 1
        return self._cache[w]


def degree_map_from_plane(plane: SymbolPlaneIndex) -> dict[str, int]:
    """O(|vocab|) adjacency sizes — avoids per-word bucket scans at query time."""
    return {w: len(nbrs) for w, nbrs in plane.word_adjacency.items()}


def word_degree(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    degrees: dict[str, int] | None = None,
) -> int:
    w = word.lower()
    if degrees is not None:
        return degrees.get(w, 0)
    return len(knowledge.neighbors(w))


def _link_degree_in_bucket(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    chamber: int | None = None,
) -> int:
    """Count edges touching word in an already-built chamber bucket (no lazy build)."""
    from aethos_symbol_subjects import MASTER_CHAMBER

    w = word.lower()
    ch = MASTER_CHAMBER if chamber is None else chamber
    bucket = knowledge.chamber_links.get(ch)
    if not bucket:
        return 0
    return sum(1 for lk in bucket.values() if lk.left == w or lk.right == w)


def is_hub_word(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    ingest_safe: bool = False,
    chamber: int | None = None,
    df_cache: _DocFreqCache | None = None,
    degrees: dict[str, int] | None = None,
) -> bool:
    """
    Hub if graph degree > HUB_DEGREE_LIMIT.

    ``ingest_safe=True`` avoids ``neighbors()`` (prevents recursion while links
    are still being built); uses membrane role + doc-fraction proxy, then any
    partial bucket degree already materialized.

    Pass ``degrees`` from ``degree_map_from_plane`` for fast query-time checks.
    """
    w = word.lower()
    if degrees is not None:
        return degrees.get(w, 0) > HUB_DEGREE_LIMIT
    if not ingest_safe:
        return word_degree(knowledge, word) > HUB_DEGREE_LIMIT

    prof = knowledge.cellular.profiles.get(w)
    if prof is not None and prof.role == CellularRole.MEMBRANE:
        return True

    cache = df_cache or _DocFreqCache(knowledge)
    n_docs = max(len(knowledge.corpus), 1)
    if cache.get(w) / n_docs >= 0.35:
        return True

    return _link_degree_in_bucket(knowledge, w, chamber=chamber) > HUB_DEGREE_LIMIT


def is_hub_word_query(knowledge: SymbolKnowledgeIndex, word: str) -> bool:
    """Query-time hub check (may lazy-build chamber links)."""
    return is_hub_word(knowledge, word, ingest_safe=False)


def ingest_rare_factor(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
    *,
    via: str | None = None,
    doc_freq_threshold: int = DEFAULT_RARE_DOC_FREQ,
    df_cache: _DocFreqCache | None = None,
    chamber: int | None = None,
) -> float:
    """
    Multiplicative ingest boost for direct/bridge links.

    Hub words (degree > HUB_DEGREE_LIMIT) never boost. Both rare → strongest;
    one rare → moderate; bridge with rare via + both endpoints rare → extra.
    """
    cache = df_cache or _DocFreqCache(knowledge)
    hub_kw = dict(ingest_safe=True, df_cache=cache, chamber=chamber)
    if is_hub_word(knowledge, left, **hub_kw) or is_hub_word(knowledge, right, **hub_kw):
        return 1.0
    left_r = is_rare_word(
        knowledge, left, doc_freq_threshold=doc_freq_threshold,
        df_cache=cache, ingest_safe=True, chamber=chamber,
    )
    right_r = is_rare_word(
        knowledge, right, doc_freq_threshold=doc_freq_threshold,
        df_cache=cache, ingest_safe=True, chamber=chamber,
    )
    if not left_r and not right_r:
        return 1.0
    factor = INGEST_BOTH_RARE_FACTOR if (left_r and right_r) else INGEST_ONE_RARE_FACTOR
    if via and left_r and right_r:
        if not is_hub_word(knowledge, via, **hub_kw) and is_rare_word(
            knowledge, via, doc_freq_threshold=doc_freq_threshold,
            df_cache=cache, ingest_safe=True, chamber=chamber,
        ):
            factor *= INGEST_TRIPLE_RARE_FACTOR
    return factor


def ingest_link_strength(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
    base_strength: float,
    *,
    via: str | None = None,
    doc_freq_threshold: int = DEFAULT_RARE_DOC_FREQ,
    df_cache: _DocFreqCache | None = None,
    chamber: int | None = None,
) -> float:
    """Apply ingest rare factor to raw co-occurrence / bridge strength."""
    factor = ingest_rare_factor(
        knowledge, left, right, via=via,
        doc_freq_threshold=doc_freq_threshold, df_cache=df_cache, chamber=chamber,
    )
    return base_strength * factor


def is_rare_word(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    doc_freq_threshold: int = DEFAULT_RARE_DOC_FREQ,
    df_cache: _DocFreqCache | None = None,
    ingest_safe: bool = False,
    chamber: int | None = None,
    degrees: dict[str, int] | None = None,
) -> bool:
    """Rare if cellular signal + low doc frequency (or profile marked rare)."""
    w = word.lower()
    if w in _DEFAULT_MEMBRANE or len(w) < 3:
        return False
    if is_hub_word(
        knowledge, w,
        ingest_safe=ingest_safe, df_cache=df_cache, chamber=chamber,
        degrees=degrees,
    ):
        return False

    role = knowledge.cellular.role_of(w)
    if role == CellularRole.MEMBRANE:
        return False

    prof = knowledge.cellular.profiles.get(w)
    if prof is not None and prof.rare:
        return True

    cache = df_cache or _DocFreqCache(knowledge)
    return cache.get(w) <= doc_freq_threshold


def rare_neighbors(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    limit: int = 12,
    df_cache: _DocFreqCache | None = None,
    adjacency: dict[str, list[tuple[str, float, str]]] | None = None,
    rare_cache: dict[str, bool] | None = None,
    degrees: dict[str, int] | None = None,
) -> list[tuple[str, float]]:
    """Neighbors filtered to rare-only, sorted by link strength."""
    cache = df_cache or _DocFreqCache(knowledge)
    rcache = rare_cache if rare_cache is not None else {}
    w = word.lower()
    out: list[tuple[str, float]] = []
    if adjacency is not None:
        for other, strength, _kind in adjacency.get(w, ()):
            if _rare_word_cached(
                knowledge, other, df_cache=cache, rare_cache=rcache, degrees=degrees,
            ):
                out.append((other, strength))
    else:
        for link in knowledge.neighbors(w):
            other = link.right if link.left == w else link.left
            if _rare_word_cached(
                knowledge, other, df_cache=cache, rare_cache=rcache, degrees=degrees,
            ):
                out.append((other, link.strength))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:limit]


def _pair_strength_from_adj(
    adjacency: dict[str, list[tuple[str, float, str]]],
    left: str,
    right: str,
) -> float | None:
    a, b = left.lower(), right.lower()
    for other, strength, _kind in adjacency.get(a, ()):
        if other == b:
            return strength
    for other, strength, _kind in adjacency.get(b, ()):
        if other == a:
            return strength
    return None


def _rare_word_cached(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    df_cache: _DocFreqCache,
    rare_cache: dict[str, bool],
    degrees: dict[str, int] | None = None,
) -> bool:
    w = word.lower()
    if w not in rare_cache:
        rare_cache[w] = is_rare_word(
            knowledge, w, df_cache=df_cache, degrees=degrees,
        )
    return rare_cache[w]


def score_doc_rare_correlations(
    knowledge: SymbolKnowledgeIndex,
    query_words: Sequence[str],
    doc_id: str,
    doc_text: str,
    *,
    df_cache: _DocFreqCache | None = None,
    rare_query: Sequence[str] | None = None,
    rare_cache: dict[str, bool] | None = None,
    degrees: dict[str, int] | None = None,
    rare_doc_tokens: set[str] | None = None,
) -> float:
    """
    Sum rare query→doc correlation strengths; bonus for rare 3-way triples.
    Hub words (degree > 500) are ignored.
    """
    cache = df_cache or _DocFreqCache(knowledge)
    rcache = rare_cache if rare_cache is not None else {}
    if rare_query is None:
        rare_query = [
            w.lower() for w in query_words
            if _rare_word_cached(
                knowledge, w, df_cache=cache, rare_cache=rcache, degrees=degrees,
            )
        ]
    if not rare_query:
        return 0.0

    if rare_doc_tokens is not None:
        rare_doc = set(rare_doc_tokens)
    else:
        doc_toks = _unique_tokens(doc_text)
        rare_doc = {
            t for t in doc_toks
            if _rare_word_cached(
                knowledge, t, df_cache=cache, rare_cache=rcache, degrees=degrees,
            )
        }
    if not rare_doc:
        return 0.0

    score = 0.0
    rare_qset = set(rare_query)

    for w in rare_query:
        for v in rare_doc:
            link = knowledge.correlates(w, v)
            if link is not None:
                score += link.strength

    for a, b in combinations(sorted(rare_qset), 2):
        ab = knowledge.correlates(a, b)
        if ab is None:
            continue
        for c in rare_doc:
            if c in rare_qset:
                continue
            ac = knowledge.correlates(a, c)
            bc = knowledge.correlates(b, c)
            if ac is not None and bc is not None:
                triple_strength = min(ab.strength, ac.strength, bc.strength)
                score += TRIPLE_BONUS + triple_strength

    return score


def rank_docs_rare_weighted(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    query_words: Sequence[str],
    candidate_doc_ids: Sequence[str],
    corpus: dict[str, str],
    *,
    kappa_weight: float = KAPPA_WEIGHT,
    rare_weight: float = RARE_WEIGHT,
    radius: int = 1,
    expand_correlations: bool = True,
) -> list[tuple[str, float]]:
    """Combine κ Jaccard + rare correlation score over route candidates."""
    keys = query_symbol_plane_keys(
        knowledge,
        plane,
        query_words,
        radius=radius,
        expand_correlations=expand_correlations,
    )
    if not candidate_doc_ids:
        return []

    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)
    rare_cache: dict[str, bool] = {}
    rare_query = [
        w.lower() for w in query_words
        if _rare_word_cached(
            knowledge, w, df_cache=cache, rare_cache=rare_cache, degrees=degrees,
        )
    ]

    kappa_scores: dict[str, float] = {}
    rare_scores: dict[str, float] = {}

    for did in candidate_doc_ids:
        kappa_scores[did] = plane.score_overlap(keys, did) if keys else 0.0
        rare_scores[did] = score_doc_rare_correlations(
            knowledge,
            query_words,
            did,
            corpus.get(did, ""),
            df_cache=cache,
            rare_query=rare_query,
            rare_cache=rare_cache,
            degrees=degrees,
        )

    max_rare = max(rare_scores.values()) if rare_scores else 0.0
    combined: list[tuple[str, float]] = []
    for did in candidate_doc_ids:
        k = kappa_scores[did]
        r = rare_scores[did] / max_rare if max_rare > 0 else 0.0
        total = kappa_weight * k + rare_weight * r
        if total > 0:
            combined.append((did, total))

    combined.sort(key=lambda x: -x[1])
    return combined


def morph_trigger_pieces(knowledge: SymbolKnowledgeIndex, token: str) -> list[str]:
    """Subwords / composite parts / rare morph word activated by a query token."""
    from aethos_symbol_morph_pieces import morph_pieces

    return morph_pieces(knowledge, token, mode="query")


def is_morph_signal_piece(knowledge: SymbolKnowledgeIndex, piece: str) -> bool:
    """Promoted L2 subword or rare morph composite — always eligible as a rare node."""
    w = piece.lower()
    morph = knowledge.morph
    if w in morph.subwords:
        return True
    comp = morph.composites.get(w)
    return comp is not None and comp.rare


def rare_query_triggers(
    knowledge: SymbolKnowledgeIndex,
    words: Sequence[str],
    *,
    df_cache: _DocFreqCache | None = None,
    rare_cache: dict[str, bool] | None = None,
    degrees: dict[str, int] | None = None,
) -> list[str]:
    """
    Rare query nodes: rare words plus morph subwords/composites/intersection parts.

    Hub tokens are skipped; morph L2 pieces (e.g. dimin, ed) fire even when the
    full token is common.
    """
    cache = df_cache or _DocFreqCache(knowledge)
    cache.warm_corpus()
    rcache = rare_cache if rare_cache is not None else {}
    triggers: list[str] = []

    for token in query_words_for_routing(words):
        if _rare_word_cached(
            knowledge, token, df_cache=cache, rare_cache=rcache, degrees=degrees,
        ):
            triggers.append(token.lower())
        for piece in morph_trigger_pieces(knowledge, token):
            pl = piece.lower()
            if is_morph_signal_piece(knowledge, pl):
                triggers.append(pl)
            elif _rare_word_cached(
                knowledge, pl, df_cache=cache, rare_cache=rcache, degrees=degrees,
            ):
                triggers.append(pl)

    return list(dict.fromkeys(triggers))


def search_docs_rare_correlations(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    max_neighbors: int = 12,
    max_candidates: int = 600,
    limit: int = 100,
) -> tuple[CandidateRouteResult, list[tuple[str, float]]]:
    """
    Primary zero-shot query: rare triggers → rare links only → strength-weighted docs.

    No κ Jaccard rerank — intersection meet keys inherit ``link.strength`` from ingest.
    """
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    rare_cache: dict[str, bool] = {}
    degrees = degree_map_from_plane(plane)
    triggers = rare_query_triggers(
        knowledge, words, df_cache=cache, rare_cache=rare_cache, degrees=degrees,
    )

    doc_scores: dict[str, float] = defaultdict(float)
    query_keys: set[AttractorKey] = set()
    adj = plane.word_adjacency
    nbr_kw = dict(
        df_cache=cache,
        adjacency=adj,
        rare_cache=rare_cache,
        degrees=degrees,
    )

    for w in triggers:
        for k in plane.keys_for_word(w):
            query_keys.add(k)
            for did in plane.by_key.get(k, ()):
                doc_scores[did] += 1.0

        for other, nb_strength in rare_neighbors(
            knowledge, w, limit=max_neighbors, **nbr_kw,
        ):
            strength = _pair_strength_from_adj(adj, w, other) or nb_strength
            pair = tuple(sorted((w, other)))
            for k in plane.pair_keys.get(pair, ()):
                query_keys.add(k)
                for did in plane.by_key.get(k, ()):
                    doc_scores[did] += strength

    trigger_set = set(triggers)
    for a, b in combinations(sorted(trigger_set), 2):
        ab_strength = _pair_strength_from_adj(adj, a, b)
        if ab_strength is None:
            continue
        for k in plane.pair_keys.get(tuple(sorted((a, b))), ()):
            query_keys.add(k)
            for did in plane.by_key.get(k, ()):
                doc_scores[did] += TRIPLE_BONUS + ab_strength

    ranked = sorted(doc_scores.items(), key=lambda x: (-x[1], x[0]))
    route_ids = [did for did, _ in ranked[:max_candidates]]

    route = CandidateRouteResult(
        doc_ids=route_ids,
        tier="rare_correlation",
        n_attractor=len(route_ids),
        n_merged=len(route_ids),
        query_keys=frozenset(query_keys),
        n_query_keys=len(query_keys),
    )
    return route, ranked[:limit]


def gold_rank(ranked: Sequence[str], gold_ids: Sequence[str]) -> int | None:
    """1-based rank of first gold doc, or None if absent."""
    gold = set(gold_ids)
    for i, did in enumerate(ranked):
        if did in gold:
            return i + 1
    return None
