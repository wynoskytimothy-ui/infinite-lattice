"""
BIT 12 — Symbol knowledge → 3D complex-plane κ index (fast search).

Links saved symbol correlations to SpacetimeCell / attractor keys:

  word  → chain(symbol primes) → Ψ(w) → κ(w)
  pair  → meet chain(A∪B) @ imag(A)+imag(B) → κ_meet(a,b)

Query routing (extends BIT 4):
  C(q) = ⋃ κ(w) ∪ N(κ, r) ∪ ⋃ κ_meet(w, neighbor) ∪ κ(neighbor)

Inverted index κ → doc_id gives O(|keys|) lookup instead of scanning 4M links.
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from aethos_rare_rank import _DocFreqCache

from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from aethos_symbol_map import text_icn_chain, text_intersection
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N, spacetime_cell_at_branch
from pipeline.bit_02_attractor_key import (
    AttractorKey,
    DEFAULT_QUANTIZE,
    attractor_neighbors,
    kappa_at_branch,
    kappa_branch_fan,
    kappa_from_cell,
)
from pipeline.bit_03_doc_attractor_set import CorpusAttractorIndex
from pipeline.bit_04_candidate_router import (
    DEFAULT_RADIUS,
    CandidateRouteResult,
    query_words_for_routing,
)

if TYPE_CHECKING:
    from aethos_symbol_knowledge import CrossLink, SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")
_MAX_RAIL = 4093


def _rail_from_imag(imag: int, *, default: int = DEFAULT_ANCHOR_N) -> float:
    """Map imaginary-line sum to transgressor n on the lattice."""
    if imag <= 0:
        return float(default)
    return float(min(max(imag % _MAX_RAIL, 1), _MAX_RAIL))


def symbol_word_chain(knowledge: SymbolKnowledgeIndex, word: str) -> tuple[int, ...]:
    """
    Prime chain for index-time tokens (doc ingest, pair_meets).

    composite → subword exact → L1 ICN (literal surface form).
    """
    w = word.lower()
    morph = knowledge.morph
    if w in morph.composites:
        return tuple(int(p) for p in morph.composites[w].meeting_primes)
    if w in morph.subwords:
        return (int(morph.subwords[w].prime),)
    chain = text_icn_chain(w)
    return tuple(int(p) for p in chain) if chain else ()


def symbol_word_chain_query(knowledge: SymbolKnowledgeIndex, word: str) -> tuple[int, ...]:
    """Query-time chain with P1 embedded-root decay (cellular → cell, not show → shows)."""
    from aethos_symbol_morph import canonical_morph_chain_and_imag

    chain, _imag = canonical_morph_chain_and_imag(knowledge.morph, word)
    if chain:
        return chain
    return symbol_word_chain(knowledge, word)


def symbol_word_imag(knowledge: SymbolKnowledgeIndex, word: str) -> int:
    w = word.lower()
    morph = knowledge.morph
    if w in morph.composites:
        return int(morph.composites[w].imaginary_position)
    if w in morph.subwords:
        return int(morph.subwords[w].imaginary_position)
    return text_intersection(w)


def symbol_word_imag_query(knowledge: SymbolKnowledgeIndex, word: str) -> int:
    from aethos_symbol_morph import canonical_morph_chain_and_imag

    _chain, imag = canonical_morph_chain_and_imag(knowledge.morph, word)
    if imag is not None:
        return imag
    return symbol_word_imag(knowledge, word)


def word_to_symbol_plane_cell(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    n: float | None = None,
    branch: BranchKind = BranchKind.VA1,
    wing: int = 1,
) -> SpacetimeCell:
    """Map symbol-knowledge token → SpacetimeCell on the 3D complex plane."""
    chain = symbol_word_chain(knowledge, word)
    if not chain:
        chain = (3,)  # fallback solo anchor
    rail = n if n is not None else _rail_from_imag(symbol_word_imag(knowledge, word))
    return spacetime_cell_at_branch(chain, rail, branch, wing)


def _canonical_surface_map(
    knowledge: SymbolKnowledgeIndex,
) -> dict[str, str]:
    """Vocab→morph-canonical surface (built once; avoids per-query family rebuild)."""
    cached = getattr(knowledge, "_canon_surface_map", None)
    if cached is not None:
        return cached
    knowledge._word_families()
    surf = {w: knowledge.morph_canonical_surface(w) for w in knowledge.vocab}
    knowledge._canon_surface_map = surf
    return surf


def canonical_pair_key(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
) -> tuple[str, str]:
    """P5 — one meet bucket per morph-canonical word pair."""
    surf = getattr(knowledge, "_canon_surface_map", None)
    if surf is not None:
        return tuple(sorted((
            surf.get(left.lower(), left.lower()),
            surf.get(right.lower(), right.lower()),
        )))
    return tuple(sorted((
        knowledge.morph_canonical_surface(left),
        knowledge.morph_canonical_surface(right),
    )))


def correlation_meet_keys(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
    *,
    link: CrossLink | None = None,
    quantize: float = DEFAULT_QUANTIZE,
    use_canonical: bool = True,
) -> frozenset[AttractorKey]:
    """
    κ buckets at the correlation meet: chain(A∪B) @ imag(A)+imag(B).

    Uses 4-way branch fan (BIT 1 rotation) for recall.
    """
    if use_canonical:
        la = knowledge.morph_canonical_surface(left)
        rb = knowledge.morph_canonical_surface(right)
    else:
        la, rb = left.lower(), right.lower()
    chain = tuple(sorted(set(symbol_word_chain(knowledge, la) + symbol_word_chain(knowledge, rb))))
    if not chain:
        return frozenset()
    imag = link.intersection_imag if link and link.intersection_imag else (
        symbol_word_imag(knowledge, la) + symbol_word_imag(knowledge, rb)
    )
    rail = _rail_from_imag(imag)
    return frozenset(kappa_branch_fan(chain, rail, quantize=quantize))


def _canonical_pair_link_cache(
    knowledge: SymbolKnowledgeIndex,
) -> dict[tuple[str, str], CrossLink]:
    """Lazy index: morph-canonical pair → strongest cross-link."""
    cached = getattr(knowledge, "_canon_pair_links", None)
    if cached is not None:
        return cached
    surf = _canonical_surface_map(knowledge)
    best: dict[tuple[str, str], CrossLink] = {}
    for (a, b), lk in knowledge.cross_links.items():
        cpk = tuple(sorted((surf.get(a, a), surf.get(b, b))))
        prev = best.get(cpk)
        if prev is None or lk.strength > prev.strength:
            best[cpk] = lk
    knowledge._canon_pair_links = best
    return best


def resolve_pair_link(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
) -> CrossLink | None:
    """Best cross-link for a pair (literal, truncated canonical, then family bucket)."""
    lk = knowledge.correlates(left, right)
    if lk is not None:
        return lk
    cpk = canonical_pair_key(knowledge, left, right)
    if cpk != tuple(sorted((left.lower(), right.lower()))):
        lk = knowledge.correlates(cpk[0], cpk[1])
        if lk is not None:
            return lk
    return _canonical_pair_link_cache(knowledge).get(cpk)


def get_pair_meet_keys(
    plane: SymbolPlaneIndex,
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
    *,
    link: CrossLink | None = None,
) -> frozenset[AttractorKey]:
    """Lookup or compute meet κ for a pair (canonical storage key)."""
    pair = canonical_pair_key(knowledge, left, right)
    stored = plane.pair_keys.get(pair)
    if stored:
        return stored
    lk = link if link is not None else resolve_pair_link(knowledge, left, right)
    meet = correlation_meet_keys(
        knowledge, left, right, link=lk, quantize=plane.quantize,
    )
    if meet:
        plane.pair_keys[pair] = meet
    return meet


@dataclass
class SymbolPlaneIndex:
    """
    κ inverted index built from symbol knowledge + 3D plane formulas.

    Mirrors CorpusAttractorIndex (BIT 3) but keyed from symbol primes.
    """

    quantize: float = DEFAULT_QUANTIZE
    anchor_n: int = DEFAULT_ANCHOR_N
    by_key: dict[AttractorKey, list[str]] = field(default_factory=dict)
    doc_keys: dict[str, set[AttractorKey]] = field(default_factory=dict)
    word_keys: dict[str, set[AttractorKey]] = field(default_factory=dict)
    pair_keys: dict[tuple[str, str], frozenset[AttractorKey]] = field(default_factory=dict)
    word_adjacency: dict[str, list[tuple[str, float, str]]] = field(default_factory=dict)
    build_ms: float = 0.0
    n_pair_keys: int = 0

    def add(self, doc_id: str, key: AttractorKey, word: str = "") -> None:
        bucket = self.by_key.setdefault(key, [])
        if doc_id not in bucket:
            bucket.append(doc_id)
        self.doc_keys.setdefault(doc_id, set()).add(key)
        if word:
            self.word_keys.setdefault(word.lower(), set()).add(key)

    def query_by_key(self, key: AttractorKey, *, radius: int = 0) -> list[str]:
        keys = attractor_neighbors(key, radius=radius) if radius > 0 else {key}
        seen: list[str] = []
        hit: set[str] = set()
        for k in keys:
            for did in self.by_key.get(k, []):
                if did not in hit:
                    hit.add(did)
                    seen.append(did)
        return seen

    def keys_for_word(self, word: str, *, radius: int = 0) -> set[AttractorKey]:
        base = set(self.word_keys.get(word.lower(), ()))
        if radius <= 0:
            return base
        out: set[AttractorKey] = set()
        for k in base:
            out |= attractor_neighbors(k, radius=radius)
        return out

    def score_overlap(
        self,
        query_keys: set[AttractorKey],
        doc_id: str,
    ) -> float:
        doc_k = self.doc_keys.get(doc_id, set())
        if not query_keys or not doc_k:
            return 0.0
        inter = len(query_keys & doc_k)
        union = len(query_keys | doc_k)
        return inter / union if union else 0.0

    def score_overlap_asymmetric(
        self,
        query_keys: set[AttractorKey],
        doc_id: str,
    ) -> float:
        """Query precision: fraction of query κ keys matched by doc."""
        doc_k = self.doc_keys.get(doc_id, set())
        if not query_keys or not doc_k:
            return 0.0
        inter = len(query_keys & doc_k)
        return inter / len(query_keys)


def _plane_rare_context(knowledge: SymbolKnowledgeIndex):
    """Doc-freq + degree maps for rare/hub filtering at plane build time."""
    from aethos_rare_rank import _DocFreqCache, is_hub_word, is_rare_word

    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees: dict[str, int] = {}
    for key_pair in knowledge.cross_links:
        a, b = key_pair
        degrees[a] = degrees.get(a, 0) + 1
        degrees[b] = degrees.get(b, 0) + 1

    def pair_eligible(left: str, right: str) -> bool:
        la, rb = left.lower(), right.lower()
        if is_hub_word(knowledge, la, degrees=degrees) and is_hub_word(
            knowledge, rb, degrees=degrees,
        ):
            return False
        return (
            is_rare_word(knowledge, la, df_cache=cache, degrees=degrees)
            or is_rare_word(knowledge, rb, df_cache=cache, degrees=degrees)
        )

    def neighbor_eligible(word: str) -> bool:
        return is_rare_word(
            knowledge, word, df_cache=cache, degrees=degrees,
        )

    return pair_eligible, neighbor_eligible


def _index_doc_meet_witnesses(
    idx: SymbolPlaneIndex,
    knowledge: SymbolKnowledgeIndex,
    doc_id: str,
    tokens: list[str],
    *,
    quantize: float,
    min_pair_strength: float,
    max_pairs: int,
    pair_eligible,
) -> int:
    """Attach correlation meet keys to doc when both tokens co-occur in text."""
    from itertools import combinations

    added = 0
    seen_pairs: set[tuple[str, str]] = set()
    for a, b in combinations(sorted(set(tokens)), 2):
        if added >= max_pairs:
            break
        if pair_eligible is not None and not pair_eligible(a, b):
            continue
        pair = canonical_pair_key(knowledge, a, b)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if pair not in idx.pair_keys:
            link = resolve_pair_link(knowledge, a, b)
            if link is None or (
                link.strength < min_pair_strength and link.kind == "direct"
            ):
                continue
        meet = get_pair_meet_keys(idx, knowledge, a, b)
        if not meet:
            continue
        for key in meet:
            idx.add(doc_id, key)
        added += 1
    return added


def build_symbol_plane_index(
    knowledge: SymbolKnowledgeIndex,
    *,
    quantize: float = DEFAULT_QUANTIZE,
    pair_key_limit: int = 500_000,
    min_pair_strength: float = 2.0,
    rare_pairs_only: bool = False,
    rare_adjacency_only: bool = False,
    max_adjacency_per_word: int | None = None,
    index_doc_meet_keys: bool = True,
    max_doc_meet_pairs: int = 48,
    dedupe_family_keys: bool = True,
    canonical_family_index: bool = True,
) -> SymbolPlaneIndex:
    """
    Build κ index from symbol knowledge corpus.

    Per doc: every token → κ branch fan.
    Global: strong correlation pairs → meet κ for query expansion.

    Slim mode (``rare_pairs_only`` / ``rare_adjacency_only`` / cap) indexes only
    rare-signal intersections and keeps top-N rare neighbors per word.
    """
    t0 = time.perf_counter()
    idx = SymbolPlaneIndex(quantize=quantize)
    pair_eligible = neighbor_eligible = None
    if rare_pairs_only or rare_adjacency_only:
        pair_eligible, neighbor_eligible = _plane_rare_context(knowledge)

    doc_tokens: dict[str, list[str]] = {}
    canon_cache: dict[str, str] = {}

    def _canonical(w: str) -> str:
        if w not in canon_cache:
            canon_cache[w] = knowledge.morph_canonical_surface(w)
        return canon_cache[w]

    for doc_id, text in knowledge.corpus.items():
        tokens = list(dict.fromkeys(
            t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2
        ))
        doc_tokens[doc_id] = tokens
        seen_family: set[str] = set()
        for w in tokens:
            canon = _canonical(w)
            if dedupe_family_keys:
                if canon in seen_family:
                    keys_for_w = idx.word_keys.get(canon, set())
                    if keys_for_w:
                        idx.word_keys.setdefault(w, set()).update(keys_for_w)
                    continue
                seen_family.add(canon)
            index_w = canon if canonical_family_index else w
            chain = symbol_word_chain(knowledge, index_w)
            if not chain:
                continue
            rail = _rail_from_imag(symbol_word_imag(knowledge, index_w))
            fan = kappa_branch_fan(chain, rail, quantize=quantize)
            for key in fan:
                idx.add(doc_id, key, word=w)
            if canon != w:
                idx.word_keys.setdefault(canon, set()).update(fan)
                idx.word_keys.setdefault(w, set()).update(fan)

    pair_count = 0
    for key_pair, link in knowledge.cross_links.items():
        if pair_count >= pair_key_limit:
            break
        if link.strength < min_pair_strength and link.kind == "direct":
            continue
        if rare_pairs_only and pair_eligible is not None:
            if not pair_eligible(key_pair[0], key_pair[1]):
                continue
        canon_pair = canonical_pair_key(knowledge, key_pair[0], key_pair[1])
        if canon_pair in idx.pair_keys:
            pair_count += 1
            continue
        meet_keys = correlation_meet_keys(
            knowledge, key_pair[0], key_pair[1], link=link, quantize=quantize,
        )
        if not meet_keys:
            continue
        idx.pair_keys[canon_pair] = meet_keys
        pair_count += 1

    idx.n_pair_keys = pair_count

    if index_doc_meet_keys:
        for doc_id, tokens in doc_tokens.items():
            _index_doc_meet_witnesses(
                idx,
                knowledge,
                doc_id,
                tokens,
                quantize=quantize,
                min_pair_strength=min_pair_strength,
                max_pairs=max_doc_meet_pairs,
                pair_eligible=pair_eligible,
            )

    adj: dict[str, list[tuple[str, float, str]]] = {}
    for key_pair, link in knowledge.cross_links.items():
        a, b = key_pair
        if rare_adjacency_only and neighbor_eligible is not None:
            if not neighbor_eligible(b):
                continue
            adj.setdefault(a, []).append((b, link.strength, link.kind))
            if not neighbor_eligible(a):
                continue
            adj.setdefault(b, []).append((a, link.strength, link.kind))
        else:
            adj.setdefault(a, []).append((b, link.strength, link.kind))
            adj.setdefault(b, []).append((a, link.strength, link.kind))
    for w in adj:
        adj[w].sort(key=lambda x: (-x[1], x[0]))
        if max_adjacency_per_word is not None:
            adj[w] = adj[w][:max_adjacency_per_word]
    idx.word_adjacency = adj

    idx.build_ms = (time.perf_counter() - t0) * 1000.0
    return idx


def query_symbol_plane_keys(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    expand_correlations: bool = True,
    max_corr_neighbors: int = 4,
    max_keys: int = 768,
    min_len: int = 3,
    chambers: frozenset[int] | None = None,
) -> set[AttractorKey]:
    """kappa keys for query: word cells + pre-indexed correlation meet expansion."""
    keys: set[AttractorKey] = set()
    routed = query_words_for_routing(words, min_len=min_len)
    active_chambers = (
        chambers
        if chambers is not None
        else knowledge.active_chambers_for_query(routed)
    )

    def _add_key(k: AttractorKey, *, expand_nb: bool = True) -> bool:
        if len(keys) >= max_keys:
            return False
        keys.add(k)
        if expand_nb and radius > 0:
            for nk in attractor_neighbors(k, radius=radius):
                keys.add(nk)
                if len(keys) >= max_keys:
                    return False
        return True

    from aethos_query_oov import expand_oov_query_word, word_needs_oov_build

    for w in routed:
        if word_needs_oov_build(knowledge, plane, w):
            expand_oov_query_word(
                knowledge,
                plane,
                w,
                _add_key,
                expand_correlations=expand_correlations,
                max_corr_neighbors=max_corr_neighbors,
                active_chambers=active_chambers,
                radius=radius,
                chamber_neighbors_fn=_chamber_adjacency_neighbors,
            )
            continue

        for k in plane.keys_for_word(w):
            if not _add_key(k, expand_nb=False):
                break
        if len(keys) < max_keys and radius > 0:
            for k in list(plane.keys_for_word(w)):
                for nk in attractor_neighbors(k, radius=radius):
                    if not _add_key(nk, expand_nb=False):
                        break

        # P1: query canonical κ + root word_keys (index stays literal)
        qchain = symbol_word_chain_query(knowledge, w)
        if qchain != symbol_word_chain(knowledge, w):
            rail = _rail_from_imag(symbol_word_imag_query(knowledge, w))
            for key in kappa_branch_fan(qchain, rail, quantize=plane.quantize):
                if not _add_key(key, expand_nb=False):
                    break
            from aethos_symbol_morph import longest_embedded_subword

            root = longest_embedded_subword(knowledge.morph, w)
            if root and root != w:
                for k in plane.keys_for_word(root):
                    if not _add_key(k, expand_nb=False):
                        break

        if expand_correlations:
            neighbors = _chamber_adjacency_neighbors(
                knowledge, w, active_chambers, max_corr_neighbors, plane=plane,
            )
            for other, _strength, _kind in neighbors:
                meet = plane.pair_keys.get(canonical_pair_key(knowledge, w, other))
                if meet:
                    for mk in meet:
                        if not _add_key(mk, expand_nb=False):
                            break

    return keys


def _chamber_adjacency_neighbors(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    chambers: frozenset[int],
    limit: int,
    *,
    plane: SymbolPlaneIndex | None = None,
) -> list[tuple[str, float, str]]:
    """Correlation neighbors from voted subject chambers (+ master)."""
    from aethos_symbol_subjects import MASTER_CHAMBER

    w = word.lower()
    # Fast path: plane.word_adjacency is built from master cross_links at index time.
    if plane is not None and plane.word_adjacency:
        return list(plane.word_adjacency.get(w, ()))[:limit]

    seen: set[str] = set()
    out: list[tuple[str, float, str]] = []
    for chamber in sorted(chambers):
        for lk in knowledge.neighbors(w, chamber=chamber)[:limit]:
            other = lk.right if lk.left == w else lk.left
            if other in seen:
                continue
            seen.add(other)
            out.append((other, lk.strength, lk.kind))
    out.sort(key=lambda x: (-x[1], x[0]))
    if not out and MASTER_CHAMBER not in chambers:
        for lk in knowledge.neighbors(w, chamber=MASTER_CHAMBER)[:limit]:
            other = lk.right if lk.left == w else lk.left
            out.append((other, lk.strength, lk.kind))
    return out[:limit]


def _word_idf(df_cache: _DocFreqCache, word: str, n_docs: int) -> float:
    df = df_cache.get(word)
    return math.log((n_docs + 1) / (df + 1)) + 1.0


def _key_bucket_idf(bucket_size: int, n_docs: int) -> float:
    return math.log((n_docs + 1) / (bucket_size + 1)) + 1.0


def _query_key_idf_weights(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    keys: set[AttractorKey],
    df_cache: _DocFreqCache,
    *,
    expand_correlations: bool = True,
    max_corr_neighbors: int = 4,
    min_len: int = 3,
) -> dict[AttractorKey, float]:
    """Map each query κ key → IDF weight (word-attributed, else bucket-rarity)."""
    from aethos_symbol_morph import longest_embedded_subword

    n_docs = len(knowledge.corpus)
    routed = query_words_for_routing(words, min_len=min_len)
    active_chambers = knowledge.active_chambers_for_query(routed)
    attributed: dict[AttractorKey, float] = {}

    def _note(key: AttractorKey, weight: float) -> None:
        attributed[key] = max(attributed.get(key, 0.0), weight)

    for w in routed:
        widf = _word_idf(df_cache, w, n_docs)
        word_keys: set[AttractorKey] = set(plane.keys_for_word(w))
        canon = knowledge.morph_canonical_surface(w)
        if canon != w:
            word_keys |= plane.keys_for_word(canon)
        root = longest_embedded_subword(knowledge.morph, w)
        if root and root != w:
            word_keys |= plane.keys_for_word(root)
        qchain = symbol_word_chain_query(knowledge, w)
        if qchain != symbol_word_chain(knowledge, w) and qchain:
            rail = _rail_from_imag(symbol_word_imag_query(knowledge, w))
            word_keys |= set(
                kappa_branch_fan(qchain, rail, quantize=plane.quantize),
            )
        for k in word_keys:
            if k in keys:
                _note(k, widf)

        if expand_correlations:
            for other, _strength, _kind in _chamber_adjacency_neighbors(
                knowledge,
                w,
                active_chambers,
                max_corr_neighbors,
                plane=plane,
            ):
                meet = plane.pair_keys.get(canonical_pair_key(knowledge, w, other))
                if not meet:
                    continue
                pair_idf = max(widf, _word_idf(df_cache, other, n_docs))
                for k in meet:
                    if k in keys:
                        _note(k, pair_idf)

    return attributed


def _query_key_attributed_idf_weights(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    keys: set[AttractorKey],
    df_cache: _DocFreqCache,
    *,
    expand_correlations: bool = True,
    max_corr_neighbors: int = 4,
    min_len: int = 3,
    rare_meet_boost: bool = True,
) -> dict[AttractorKey, float]:
    """
    Word-attributed IDF only — no bucket-rarity fallback on unattributed keys.

    Common terms get low IDF (~1); rare terms get high IDF. Hub dilution from
    anonymous meet/OOV buckets is blocked.
    """
    from aethos_rare_rank import degree_map_from_plane, is_rare_word

    weights = _query_key_idf_weights(
        knowledge,
        plane,
        words,
        keys,
        df_cache,
        expand_correlations=expand_correlations,
        max_corr_neighbors=max_corr_neighbors,
        min_len=min_len,
    )
    if not rare_meet_boost:
        return weights

    n_docs = len(knowledge.corpus)
    routed = query_words_for_routing(words, min_len=min_len)
    degrees = degree_map_from_plane(plane)
    active_chambers = knowledge.active_chambers_for_query(routed)
    attributed = dict(weights)

    def _rare(w: str) -> bool:
        return is_rare_word(
            knowledge, w, df_cache=df_cache, degrees=degrees,
        )

    for w in routed:
        widf = _word_idf(df_cache, w, n_docs)
        for other, _strength, _kind in _chamber_adjacency_neighbors(
            knowledge,
            w,
            active_chambers,
            max_corr_neighbors,
            plane=plane,
        ):
            if not (_rare(w) or _rare(other)):
                continue
            meet = plane.pair_keys.get(canonical_pair_key(knowledge, w, other))
            if not meet:
                continue
            pair_w = max(widf, _word_idf(df_cache, other, n_docs)) * 1.5
            for k in meet:
                if k in keys:
                    attributed[k] = max(attributed.get(k, 0.0), pair_w)

    return attributed


def _query_key_rare_weights(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    keys: set[AttractorKey],
    df_cache: _DocFreqCache,
    *,
    expand_correlations: bool = True,
    max_corr_neighbors: int = 4,
    min_len: int = 3,
) -> dict[AttractorKey, float]:
    """Score only keys from rare query tokens and rare-involved pair meets."""
    from aethos_rare_rank import degree_map_from_plane, is_rare_word
    from aethos_symbol_morph import longest_embedded_subword

    n_docs = len(knowledge.corpus)
    routed = query_words_for_routing(words, min_len=min_len)
    degrees = degree_map_from_plane(plane)
    active_chambers = knowledge.active_chambers_for_query(routed)
    attributed: dict[AttractorKey, float] = {}

    def _rare(w: str) -> bool:
        return is_rare_word(
            knowledge, w, df_cache=df_cache, degrees=degrees,
        )

    def _note(key: AttractorKey, weight: float) -> None:
        attributed[key] = max(attributed.get(key, 0.0), weight)

    for w in routed:
        if not _rare(w):
            continue
        widf = _word_idf(df_cache, w, n_docs)
        word_keys: set[AttractorKey] = set(plane.keys_for_word(w))
        canon = knowledge.morph_canonical_surface(w)
        if canon != w and _rare(canon):
            word_keys |= plane.keys_for_word(canon)
        root = longest_embedded_subword(knowledge.morph, w)
        if root and root != w and _rare(root):
            word_keys |= plane.keys_for_word(root)
        qchain = symbol_word_chain_query(knowledge, w)
        if qchain != symbol_word_chain(knowledge, w) and qchain:
            rail = _rail_from_imag(symbol_word_imag_query(knowledge, w))
            word_keys |= set(
                kappa_branch_fan(qchain, rail, quantize=plane.quantize),
            )
        for k in word_keys:
            if k in keys:
                _note(k, widf)

        if expand_correlations:
            for other, _strength, _kind in _chamber_adjacency_neighbors(
                knowledge,
                w,
                active_chambers,
                max_corr_neighbors,
                plane=plane,
            ):
                if not (_rare(w) or _rare(other)):
                    continue
                meet = plane.pair_keys.get(canonical_pair_key(knowledge, w, other))
                if not meet:
                    continue
                pair_w = widf
                if _rare(other):
                    pair_w = max(pair_w, _word_idf(df_cache, other, n_docs))
                for k in meet:
                    if k in keys:
                        _note(k, pair_w)

    return attributed


def _word_attributed_pool_scores(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    keys: set[AttractorKey],
    df_cache: _DocFreqCache,
    *,
    expand_correlations: bool = True,
    max_corr_neighbors: int = 4,
    min_len: int = 3,
    rare_meet_bonus: float = 0.5,
) -> dict[str, float]:
    """
    Pool score = sum of IDF(word) once per doc per query word (not per κ key).

    Stops hub docs from piling hits across hundreds of buckets for one common
    term. Meet keys add a fractional bonus when a rare word anchors the pair.
    """
    from aethos_rare_rank import degree_map_from_plane, is_rare_word
    from aethos_symbol_morph import longest_embedded_subword

    n_docs = len(knowledge.corpus)
    routed = query_words_for_routing(words, min_len=min_len)
    degrees = degree_map_from_plane(plane)
    active_chambers = knowledge.active_chambers_for_query(routed)
    hits: defaultdict[str, float] = defaultdict(float)

    def _rare(w: str) -> bool:
        return is_rare_word(
            knowledge, w, df_cache=df_cache, degrees=degrees,
        )

    def _word_keys(w: str) -> set[AttractorKey]:
        out: set[AttractorKey] = set(plane.keys_for_word(w))
        canon = knowledge.morph_canonical_surface(w)
        if canon != w:
            out |= plane.keys_for_word(canon)
        root = longest_embedded_subword(knowledge.morph, w)
        if root and root != w:
            out |= plane.keys_for_word(root)
        qchain = symbol_word_chain_query(knowledge, w)
        if qchain != symbol_word_chain(knowledge, w) and qchain:
            rail = _rail_from_imag(symbol_word_imag_query(knowledge, w))
            out |= set(kappa_branch_fan(qchain, rail, quantize=plane.quantize))
        return out & keys

    for w in routed:
        widf = _word_idf(df_cache, w, n_docs)
        matched: set[str] = set()
        for k in _word_keys(w):
            for did in plane.by_key.get(k, ()):
                if did not in matched:
                    matched.add(did)
                    hits[did] += widf

        if expand_correlations:
            for other, _strength, _kind in _chamber_adjacency_neighbors(
                knowledge,
                w,
                active_chambers,
                max_corr_neighbors,
                plane=plane,
            ):
                meet = plane.pair_keys.get(canonical_pair_key(knowledge, w, other))
                if not meet:
                    continue
                bonus = rare_meet_bonus
                if _rare(w) or _rare(other):
                    bonus += max(
                        _word_idf(df_cache, w, n_docs),
                        _word_idf(df_cache, other, n_docs),
                    ) * 0.25
                meet_matched: set[str] = set()
                for k in meet:
                    if k not in keys:
                        continue
                    for did in plane.by_key.get(k, ()):
                        if did in meet_matched:
                            continue
                        meet_matched.add(did)
                        hits[did] += bonus

    return dict(hits)


def route_symbol_plane_candidates(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    expand_correlations: bool = True,
    max_candidates: int = 1200,
    max_keys: int = 768,
    max_corr_neighbors: int = 4,
    query_keys: set[AttractorKey] | None = None,
    use_idf_weighting: bool = False,
    word_attributed_pool: bool = False,
    rare_boost_hits: bool = False,
    rare_only_hits: bool = False,
    attributed_keys_only: bool = False,
    rare_boost_scale: float = 1.0,
    df_cache: _DocFreqCache | None = None,
) -> CandidateRouteResult:
    """BIT 4-style router using symbol-plane κ index."""
    keys = query_keys or query_symbol_plane_keys(
        knowledge,
        plane,
        words,
        radius=radius,
        expand_correlations=expand_correlations,
        max_keys=max_keys,
        max_corr_neighbors=max_corr_neighbors,
    )

    hits: defaultdict[str, float] = defaultdict(float)
    from aethos_rare_rank import _DocFreqCache

    cache = df_cache if df_cache is not None else _DocFreqCache(knowledge)
    need_cache = word_attributed_pool or rare_boost_hits or use_idf_weighting or rare_only_hits
    if need_cache:
        cache.warm_corpus()

    if word_attributed_pool and not use_idf_weighting and not rare_only_hits:
        hits.update(_word_attributed_pool_scores(
            knowledge,
            plane,
            words,
            keys,
            cache,
            expand_correlations=expand_correlations,
            max_corr_neighbors=max_corr_neighbors,
        ))
    elif not (rare_boost_hits or use_idf_weighting or rare_only_hits):
        for k in keys:
            for did in plane.by_key.get(k, ()):
                hits[did] += 1.0

    if rare_boost_hits or use_idf_weighting or rare_only_hits:
        if rare_only_hits:
            key_weights = _query_key_rare_weights(
                knowledge,
                plane,
                words,
                keys,
                cache,
                expand_correlations=expand_correlations,
                max_corr_neighbors=max_corr_neighbors,
            )
            hits.clear()
            if not key_weights:
                for k in keys:
                    for did in plane.by_key.get(k, ()):
                        hits[did] += 1.0
            else:
                for k, weight in key_weights.items():
                    for did in plane.by_key.get(k, ()):
                        hits[did] += weight
        elif use_idf_weighting:
            if attributed_keys_only:
                key_weights = _query_key_attributed_idf_weights(
                    knowledge,
                    plane,
                    words,
                    keys,
                    cache,
                    expand_correlations=expand_correlations,
                    max_corr_neighbors=max_corr_neighbors,
                )
            else:
                key_weights = _query_key_idf_weights(
                    knowledge,
                    plane,
                    words,
                    keys,
                    cache,
                    expand_correlations=expand_correlations,
                    max_corr_neighbors=max_corr_neighbors,
                )
                for k in keys:
                    if k in key_weights:
                        continue
                    bucket_size = len(plane.by_key.get(k, ()))
                    key_weights[k] = _key_bucket_idf(
                        bucket_size, len(knowledge.corpus),
                    )
            hits.clear()
            for k, weight in key_weights.items():
                for did in plane.by_key.get(k, ()):
                    hits[did] += weight
        elif rare_boost_hits:
            key_weights = _query_key_rare_weights(
                knowledge,
                plane,
                words,
                keys,
                cache,
                expand_correlations=expand_correlations,
                max_corr_neighbors=max_corr_neighbors,
            )
            scale = rare_boost_scale
            for k, weight in key_weights.items():
                for did in plane.by_key.get(k, ()):
                    hits[did] += scale * weight

    seen = sorted(hits, key=lambda d: (-hits[d], d))[:max_candidates]

    return CandidateRouteResult(
        doc_ids=seen,
        tier="symbol_plane",
        n_attractor=len(seen),
        n_merged=len(seen),
        query_keys=frozenset(keys),
        n_query_keys=len(keys),
    )


def _query_pair_meets(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    canon_words: Sequence[str],
) -> list[tuple[frozenset[AttractorKey], float]]:
    """Precompute meet key sets + link strength for query word pair combinations."""
    from itertools import combinations

    pair_meets: list[tuple[frozenset[AttractorKey], float]] = []
    words = list(dict.fromkeys(w for w in canon_words if w))
    for a, b in combinations(words, 2):
        cpk = canonical_pair_key(knowledge, a, b)
        meet = plane.pair_keys.get(cpk)
        lk = knowledge.correlates(a, b)
        if lk is None and cpk != tuple(sorted((a.lower(), b.lower()))):
            lk = knowledge.correlates(cpk[0], cpk[1])
        if meet is None:
            lk = lk or resolve_pair_link(knowledge, a, b)
            meet = get_pair_meet_keys(plane, knowledge, a, b, link=lk)
        if meet:
            pair_meets.append((meet, lk.strength if lk else 1.0))
    return pair_meets


def _query_rare_pair_meets(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    canon_rare: Sequence[str],
) -> list[tuple[frozenset[AttractorKey], float]]:
    """Precompute meet key sets + link strength for rare query pair combinations."""
    return _query_pair_meets(knowledge, plane, canon_rare)


def score_doc_meet_witness(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    doc_id: str,
    *,
    query_keys: set[AttractorKey] | None = None,
    df_cache: _DocFreqCache | None = None,
    degrees: dict[str, int] | None = None,
    rare_cache: dict[str, bool] | None = None,
    rare_q: Sequence[str] | None = None,
    canon_rare: Sequence[str] | None = None,
    pair_meets: Sequence[tuple[frozenset[AttractorKey], float]] | None = None,
) -> float:
    """Rare pair meet key overlap between query witnesses and doc (indexed meets)."""
    from itertools import combinations

    from aethos_rare_rank import (
        _DocFreqCache,
        _rare_word_cached,
        degree_map_from_plane,
    )

    if pair_meets is not None:
        if not pair_meets:
            return 0.0
        doc_k = plane.doc_keys.get(doc_id, set())
        score = 0.0
        for meet, strength in pair_meets:
            hits = len(meet & doc_k)
            if hits:
                score += hits * strength
        return score

    cache = df_cache or _DocFreqCache(knowledge)
    if df_cache is None:
        cache.warm_corpus()
    deg = degrees if degrees is not None else degree_map_from_plane(plane)
    rc: dict[str, bool] = rare_cache if rare_cache is not None else {}
    rq = list(rare_q) if rare_q is not None else [
        w.lower() for w in words
        if _rare_word_cached(
            knowledge, w, df_cache=cache, rare_cache=rc, degrees=deg,
        )
    ]
    if not rq:
        return 0.0

    doc_k = plane.doc_keys.get(doc_id, set())

    cr = list(canon_rare) if canon_rare is not None else list(dict.fromkeys(
        knowledge.morph_canonical_surface(w) for w in rq
    ))
    score = 0.0
    for a, b in combinations(cr, 2):
        lk = resolve_pair_link(knowledge, a, b)
        meet = get_pair_meet_keys(plane, knowledge, a, b, link=lk)
        hits = len(meet & doc_k)
        if hits:
            score += hits * (lk.strength if lk else 1.0)
    return score


def rank_symbol_plane_witness(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    expand_correlations: bool = True,
    limit: int = 50,
    query_keys: set[AttractorKey] | None = None,
    candidate_doc_ids: Sequence[str] | None = None,
    rare_boost: float = 0.40,
    meet_boost: float = 0.25,
    witness_pool: int = 200,
    all_pair_meets: bool = True,
    asymmetric: bool = True,
) -> list[tuple[str, float]]:
    """
    Re-rank a κ-routed pool by corridor intersections (Step 11).

    Base: query-precision κ overlap (asymmetric by default). Normalized rare
    correlation + all-pair meet witnesses apply a multiplicative boost so
    lit corridor intersections reorder the routed pool.
    """
    from aethos_rare_rank import (
        _DocFreqCache,
        _rare_word_cached,
        degree_map_from_plane,
        score_doc_rare_correlations,
    )
    from pipeline.bit_04_candidate_router import query_words_for_routing

    _canonical_pair_link_cache(knowledge)
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)
    rare_cache: dict[str, bool] = {}
    rare_query = [
        w.lower() for w in words
        if _rare_word_cached(
            knowledge, w, df_cache=cache, rare_cache=rare_cache, degrees=degrees,
        )
    ]
    routed = query_words_for_routing(words)
    if all_pair_meets:
        canon_pairs = list(dict.fromkeys(
            knowledge.morph_canonical_surface(w) for w in routed
        ))
        pair_meets = _query_pair_meets(knowledge, plane, canon_pairs)
    else:
        canon_rare = list(dict.fromkeys(
            knowledge.morph_canonical_surface(w) for w in rare_query
        ))
        pair_meets = _query_pair_meets(knowledge, plane, canon_rare)

    keys = query_keys or query_symbol_plane_keys(
        knowledge, plane, words,
        radius=radius, expand_correlations=expand_correlations,
    )
    kappa_fn = (
        plane.score_overlap_asymmetric if asymmetric else plane.score_overlap
    )
    routed_pool = candidate_doc_ids is not None
    if routed_pool:
        candidates = list(candidate_doc_ids)
    else:
        candidates_set: set[str] = set()
        for k in keys:
            candidates_set.update(plane.by_key.get(k, []))
        candidates = sorted(candidates_set)
        if witness_pool and len(candidates) > witness_pool:
            candidates = sorted(
                candidates,
                key=lambda d: (-kappa_fn(keys, d), d),
            )[:witness_pool]

    if not candidates:
        return []

    doc_rare_cache: dict[str, set[str]] = {}

    def _rare_doc_tokens(did: str, text: str) -> set[str]:
        if did not in doc_rare_cache:
            doc_rare_cache[did] = {
                t for t in _TOKEN_RE.findall(text.lower())
                if len(t) >= 3 and _rare_word_cached(
                    knowledge, t, df_cache=cache,
                    rare_cache=rare_cache, degrees=degrees,
                )
            }
        return doc_rare_cache[did]

    kappa_scores: dict[str, float] = {}
    rare_scores: dict[str, float] = {}
    meet_scores: dict[str, float] = {}
    for did in candidates:
        text = knowledge.corpus.get(did, "")
        kappa_scores[did] = kappa_fn(keys, did) if keys else 0.0
        rare_scores[did] = score_doc_rare_correlations(
            knowledge,
            words,
            did,
            text,
            df_cache=cache,
            rare_query=rare_query,
            rare_cache=rare_cache,
            degrees=degrees,
            rare_doc_tokens=_rare_doc_tokens(did, text),
        )
        meet_scores[did] = score_doc_meet_witness(
            knowledge, plane, words, did,
            query_keys=keys,
            df_cache=cache,
            degrees=degrees,
            rare_cache=rare_cache,
            rare_q=rare_query,
            pair_meets=pair_meets,
        )

    max_rare = max(rare_scores.values()) if rare_scores else 0.0
    max_meet = max(meet_scores.values()) if meet_scores else 0.0

    scored: list[tuple[str, float]] = []
    for did in candidates:
        kappa = kappa_scores[did]
        if kappa <= 0:
            continue
        rare_norm = rare_scores[did] / max_rare if max_rare > 0 else 0.0
        meet_norm = meet_scores[did] / max_meet if max_meet > 0 else 0.0
        boost = 1.0 + rare_boost * rare_norm + meet_boost * meet_norm
        scored.append((did, kappa * boost))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:limit]


def rank_symbol_plane_docs(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    expand_correlations: bool = True,
    limit: int = 50,
    query_keys: set[AttractorKey] | None = None,
    candidate_doc_ids: Sequence[str] | None = None,
    asymmetric: bool = True,
) -> list[tuple[str, float]]:
    """Score routed docs by κ overlap (default: query-precision, not symmetric Jaccard)."""
    keys = query_keys or query_symbol_plane_keys(
        knowledge, plane, words,
        radius=radius, expand_correlations=expand_correlations,
    )
    if not keys:
        return []
    score_fn = (
        plane.score_overlap_asymmetric if asymmetric else plane.score_overlap
    )
    scores: list[tuple[str, float]] = []
    if candidate_doc_ids is not None:
        candidates = candidate_doc_ids
    else:
        candidates_set: set[str] = set()
        for k in keys:
            candidates_set.update(plane.by_key.get(k, []))
        candidates = candidates_set
    for did in candidates:
        s = score_fn(keys, did)
        if s > 0:
            scores.append((did, s))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:limit]


def merge_with_hub_index(
    symbol_plane: SymbolPlaneIndex,
    hub_index: CorpusAttractorIndex,
) -> CorpusAttractorIndex:
    """Union symbol-plane κ buckets into an existing BIT 3 index."""
    merged = CorpusAttractorIndex(
        quantize=hub_index.quantize,
        anchor_n=hub_index.anchor_n,
    )
    for key, docs in hub_index.by_key.items():
        for did in docs:
            w = hub_index.doc_witnesses.get(did, {}).get(key, "")
            merged.add(did, key, w)
    for key, docs in symbol_plane.by_key.items():
        for did in docs:
            merged.add(did, key)
    return merged


def verify_bit12_gate(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    probes: list[tuple[str, str]],
) -> tuple[bool, list[str]]:
    """
    BIT 12 gate: correlated pairs land in shared κ neighborhood on the plane.
    """
    failures: list[str] = []
    for a, b in probes:
        lk = resolve_pair_link(knowledge, a, b)
        if lk is None:
            failures.append(f"{a}+{b}: no knowledge link")
            continue
        meet = correlation_meet_keys(knowledge, a, b, link=lk)
        if not meet:
            failures.append(f"{a}+{b}: no meet κ")
            continue
        if not plane.keys_for_word(a) or not plane.keys_for_word(b):
            failures.append(f"{a}+{b}: missing word κ")
            continue
        stored = plane.pair_keys.get(canonical_pair_key(knowledge, a, b))
        if stored is None and lk.strength >= 2.0:
            failures.append(f"{a}+{b}: pair meet not indexed")

    route = route_symbol_plane_candidates(knowledge, plane, [a for a, _ in probes[:1]])
    if not route.doc_ids and plane.by_key:
        failures.append("router returned empty on probe query")

    return len(failures) == 0, failures

