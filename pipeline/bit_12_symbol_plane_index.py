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

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

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
    Prime chain for a corpus token from symbol / morph knowledge.

    morph composite → meeting_primes
    morph subword   → (prime,)
    else            → text_icn_chain (L1 symbol product factors)
    """
    w = word.lower()
    morph = knowledge.morph
    if w in morph.composites:
        return tuple(int(p) for p in morph.composites[w].meeting_primes)
    if w in morph.subwords:
        return (int(morph.subwords[w].prime),)
    chain = text_icn_chain(w)
    return tuple(int(p) for p in chain) if chain else ()


def symbol_word_imag(knowledge: SymbolKnowledgeIndex, word: str) -> int:
    w = word.lower()
    morph = knowledge.morph
    if w in morph.composites:
        return int(morph.composites[w].imaginary_position)
    if w in morph.subwords:
        return int(morph.subwords[w].imaginary_position)
    return text_intersection(w)


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


def correlation_meet_keys(
    knowledge: SymbolKnowledgeIndex,
    left: str,
    right: str,
    *,
    link: CrossLink | None = None,
    quantize: float = DEFAULT_QUANTIZE,
) -> frozenset[AttractorKey]:
    """
    κ buckets at the correlation meet: chain(A∪B) @ imag(A)+imag(B).

    Uses 4-way branch fan (BIT 1 rotation) for recall.
    """
    la, rb = left.lower(), right.lower()
    chain = tuple(sorted(set(symbol_word_chain(knowledge, la) + symbol_word_chain(knowledge, rb))))
    if not chain:
        return frozenset()
    imag = link.intersection_imag if link and link.intersection_imag else (
        symbol_word_imag(knowledge, la) + symbol_word_imag(knowledge, rb)
    )
    rail = _rail_from_imag(imag)
    return frozenset(kappa_branch_fan(chain, rail, quantize=quantize))


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


def build_symbol_plane_index(
    knowledge: SymbolKnowledgeIndex,
    *,
    quantize: float = DEFAULT_QUANTIZE,
    pair_key_limit: int = 500_000,
    min_pair_strength: float = 2.0,
    rare_pairs_only: bool = False,
    rare_adjacency_only: bool = False,
    max_adjacency_per_word: int | None = None,
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

    for doc_id, text in knowledge.corpus.items():
        tokens = list(dict.fromkeys(
            t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2
        ))
        for w in tokens:
            chain = symbol_word_chain(knowledge, w)
            if not chain:
                continue
            rail = _rail_from_imag(symbol_word_imag(knowledge, w))
            for key in kappa_branch_fan(chain, rail, quantize=quantize):
                idx.add(doc_id, key, word=w)

    pair_count = 0
    for key_pair, link in knowledge.cross_links.items():
        if pair_count >= pair_key_limit:
            break
        if link.strength < min_pair_strength and link.kind == "direct":
            continue
        if rare_pairs_only and pair_eligible is not None:
            if not pair_eligible(key_pair[0], key_pair[1]):
                continue
        meet_keys = correlation_meet_keys(
            knowledge, key_pair[0], key_pair[1], link=link, quantize=quantize,
        )
        if not meet_keys:
            continue
        idx.pair_keys[key_pair] = meet_keys
        pair_count += 1

    idx.n_pair_keys = pair_count

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

        if expand_correlations:
            neighbors = _chamber_adjacency_neighbors(
                knowledge, w, active_chambers, max_corr_neighbors, plane=plane,
            )
            for other, _strength, _kind in neighbors:
                meet = plane.pair_keys.get(tuple(sorted((w, other))))
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


def route_symbol_plane_candidates(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    words: Sequence[str],
    *,
    radius: int = DEFAULT_RADIUS,
    expand_correlations: bool = True,
    max_candidates: int = 600,
    max_keys: int = 768,
    max_corr_neighbors: int = 4,
    query_keys: set[AttractorKey] | None = None,
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
    from collections import Counter

    hits: Counter[str] = Counter()
    for k in keys:
        for did in plane.by_key.get(k, []):
            hits[did] += 1
    seen = [did for did, _ in hits.most_common(max_candidates)]

    return CandidateRouteResult(
        doc_ids=seen,
        tier="symbol_plane",
        n_attractor=len(seen),
        n_merged=len(seen),
        query_keys=frozenset(keys),
        n_query_keys=len(keys),
    )


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
) -> list[tuple[str, float]]:
    """Score docs by κ Jaccard overlap (BIT 9 / signal 8a on symbol plane)."""
    keys = query_keys or query_symbol_plane_keys(
        knowledge, plane, words,
        radius=radius, expand_correlations=expand_correlations,
    )
    if not keys:
        return []
    scores: list[tuple[str, float]] = []
    if candidate_doc_ids is not None:
        candidates = candidate_doc_ids
    else:
        candidates_set: set[str] = set()
        for k in keys:
            candidates_set.update(plane.by_key.get(k, []))
        candidates = candidates_set
    for did in candidates:
        s = plane.score_overlap(keys, did)
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
        lk = knowledge.correlates(a, b)
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
        stored = plane.pair_keys.get(tuple(sorted((a.lower(), b.lower()))))
        if stored is None and lk.strength >= 2.0:
            failures.append(f"{a}+{b}: pair meet not indexed")

    route = route_symbol_plane_candidates(knowledge, plane, [a for a, _ in probes[:1]])
    if not route.doc_ids and plane.by_key:
        failures.append("router returned empty on probe query")

    return len(failures) == 0, failures

