"""
Query-time OOV lattice build — words not in corpus still route via structure.

When a query token has no pre-indexed κ cell, build its lattice address from:
  • L1 symbol ICN chain (letter primes)
  • imaginary-line intersection sum
  • morph subword pieces already in the brain

Saved on the knowledge index for the session; routing expands through anchor
subwords and their stored correlations to pull more docs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from aethos_symbol_map import text_icn_chain, text_intersection
from aethos_symbol_morph import pick_root_suffix
from pipeline.bit_02_attractor_key import AttractorKey, attractor_neighbors, kappa_branch_fan
from pipeline.bit_12_symbol_plane_index import (
    DEFAULT_QUANTIZE,
    _rail_from_imag,
    correlation_meet_keys,
    symbol_word_chain,
    symbol_word_imag,
)

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex
    from pipeline.bit_12_symbol_plane_index import SymbolPlaneIndex

_SUFFIXES = ("tion", "ment", "ness", "ing", "ive", "ous", "ial", "ed", "es", "ly", "er")


@dataclass(frozen=True)
class QueryLatticeNode:
    """Lattice-built query word — cached on the brain for reuse."""

    word: str
    icn_chain: tuple[int, ...]
    intersection_imag: int
    subwords: tuple[str, ...]
    anchors: tuple[str, ...]
    in_vocab: bool
    built_ms: float = 0.0


def morph_subword_pieces(knowledge: SymbolKnowledgeIndex, token: str) -> list[str]:
    """Morph substrings of token that exist in the brain morph registry."""
    w = token.lower()
    morph = knowledge.morph
    out: list[str] = []
    if w in morph.composites:
        out.append(w)
        out.extend(morph.composites[w].parts)
    if w in morph.subwords:
        out.append(w)
    for sw in sorted(morph.subwords, key=len, reverse=True):
        if len(sw) >= 3 and sw in w:
            out.append(sw)
    split = pick_root_suffix(w, knowledge.vocab)
    if split:
        root, suffix = split
        out.extend([root, suffix])
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 2:
            root = w[: -len(suf)]
            if len(root) >= 3:
                out.append(root)
            if suf in morph.subwords or suf in knowledge.vocab:
                out.append(suf)
    return list(dict.fromkeys(p for p in out if len(p) >= 2))


def structural_anchor_words(
    knowledge: SymbolKnowledgeIndex,
    token: str,
    plane: SymbolPlaneIndex | None = None,
) -> tuple[str, ...]:
    """
    Known brain words to route through for an OOV or partial-match query token.

    Prefer pieces with plane adjacency (stored correlations).
    """
    w = token.lower()
    anchors: list[str] = []
    if w in knowledge.vocab:
        anchors.append(w)

    for piece in morph_subword_pieces(knowledge, w):
        if piece in knowledge.vocab:
            anchors.append(piece)
        if plane is not None and piece in plane.word_adjacency:
            anchors.append(piece)

    # Longest vocab substring hit (cap scans for speed on huge vocabs).
    if len(w) >= 5 and len(anchors) < 4:
        hits: list[str] = []
        for v in knowledge.vocab:
            if len(v) < 4:
                continue
            if v in w or w in v:
                hits.append(v)
        hits.sort(key=lambda x: (-len(x), x))
        for h in hits[:6]:
            anchors.append(h)

    return tuple(dict.fromkeys(anchors))


def build_query_lattice_node(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    plane: SymbolPlaneIndex | None = None,
) -> QueryLatticeNode:
    """Build + return lattice node for a query token (OOV-safe)."""
    t0 = time.perf_counter()
    w = word.lower()
    chain = symbol_word_chain(knowledge, w)
    if not chain:
        chain = text_icn_chain(w)
    imag = symbol_word_imag(knowledge, w)
    if not imag:
        imag = text_intersection(w)
    subwords = tuple(morph_subword_pieces(knowledge, w))
    anchors = structural_anchor_words(knowledge, w, plane)
    return QueryLatticeNode(
        word=w,
        icn_chain=tuple(int(p) for p in chain),
        intersection_imag=int(imag),
        subwords=subwords,
        anchors=anchors,
        in_vocab=w in knowledge.vocab,
        built_ms=(time.perf_counter() - t0) * 1000.0,
    )


def ephemeral_word_kappa_keys(
    knowledge: SymbolKnowledgeIndex,
    word: str,
    *,
    quantize: float = DEFAULT_QUANTIZE,
    radius: int = 0,
) -> set[AttractorKey]:
    """κ keys from lattice formula without a pre-built plane word cell."""
    chain = symbol_word_chain(knowledge, word)
    if not chain:
        chain = text_icn_chain(word.lower())
    if not chain:
        return set()
    rail = _rail_from_imag(symbol_word_imag(knowledge, word))
    keys: set[AttractorKey] = set(kappa_branch_fan(chain, rail, quantize=quantize))
    if radius > 0:
        for k in list(keys):
            keys |= attractor_neighbors(k, radius=radius)
    return keys


def word_needs_oov_build(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    word: str,
) -> bool:
    """True when the token has no indexed κ cell — needs lattice build."""
    w = word.lower()
    if plane.word_keys.get(w):
        return False
    if w in knowledge.vocab and plane.keys_for_word(w):
        return False
    return True


def expand_oov_query_word(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    word: str,
    add_key: Callable[..., bool],
    *,
    expand_correlations: bool,
    max_corr_neighbors: int,
    active_chambers: frozenset[int],
    radius: int,
    chamber_neighbors_fn: Callable[..., list[tuple[str, float, str]]],
) -> QueryLatticeNode:
    """
    OOV path: build lattice node, save on brain, route via subword anchors.
    """
    node = knowledge.ensure_query_lattice(word, plane)

    for k in ephemeral_word_kappa_keys(
        knowledge, word, quantize=plane.quantize, radius=0,
    ):
        if not add_key(k, expand_nb=False):
            return node

    for anchor in node.anchors:
        for k in plane.keys_for_word(anchor):
            if not add_key(k, expand_nb=False):
                return node
        if radius > 0:
            for k in list(plane.keys_for_word(anchor)):
                for nk in attractor_neighbors(k, radius=radius):
                    if not add_key(nk, expand_nb=False):
                        return node

        for mk in correlation_meet_keys(
            knowledge, word, anchor, quantize=plane.quantize,
        ):
            if not add_key(mk, expand_nb=False):
                return node

        if expand_correlations:
            neighbors = chamber_neighbors_fn(
                knowledge, anchor, active_chambers, max_corr_neighbors, plane=plane,
            )
            for other, _strength, _kind in neighbors:
                meet = plane.pair_keys.get(tuple(sorted((anchor, other))))
                if meet:
                    for mk in meet:
                        if not add_key(mk, expand_nb=False):
                            return node
                for mk in correlation_meet_keys(
                    knowledge, word, other, quantize=plane.quantize,
                ):
                    if not add_key(mk, expand_nb=False):
                        return node

    return node
