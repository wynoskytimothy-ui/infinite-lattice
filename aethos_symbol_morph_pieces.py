"""
Unified morph piece extraction — Concrete Plane P2.

Single source for ingest (cellular/entangle) and query (rare rank, OOV, cascade).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from aethos_symbol_morph import MorphRegistry, longest_embedded_subword, pick_root_suffix

if TYPE_CHECKING:
    from aethos_symbol_knowledge import SymbolKnowledgeIndex

_TOKEN_RE = re.compile(r"[a-z]+")
MorphPieceMode = Literal["ingest", "query"]

_QUERY_SUFFIXES = ("tion", "ment", "ness", "ing", "ive", "ous", "ial", "ed", "es", "ly", "er", "s")


def morph_pieces_for_token(
    morph: MorphRegistry,
    token: str,
    *,
    vocab: set[str] | None = None,
    mode: MorphPieceMode = "query",
    polarity_lexicon: dict[str, object] | None = None,
) -> list[str]:
    """
    Pieces activated by one token.

    ingest — catalog hits + polarity seeds (cellular entangle path).
    query  — catalog + embedded subwords + root/suffix splits (routing path).
    """
    w = token.lower()
    if not w:
        return []

    catalog = set(morph.composites) | set(morph.subwords)
    out: list[str] = []

    if w in morph.composites:
        out.append(w)
        out.extend(morph.composites[w].parts)
    if w in morph.subwords:
        out.append(w)

    if mode == "query":
        embedded = longest_embedded_subword(morph, w)
        if embedded:
            out.append(embedded)
        for sw in sorted(morph.subwords, key=len, reverse=True):
            if len(sw) >= 3 and sw in w:
                out.append(sw)
        if vocab is not None:
            split = pick_root_suffix(w, vocab)
            if split:
                root, suffix = split
                out.extend([root, suffix])
        for suf in _QUERY_SUFFIXES:
            if w.endswith(suf) and len(w) > len(suf) + 2:
                root = w[: -len(suf)]
                if len(root) >= 3:
                    out.append(root)
                if suf in morph.subwords or (vocab is not None and suf in vocab):
                    out.append(suf)
    elif w not in catalog and polarity_lexicon and w in polarity_lexicon:
        if w not in morph.subwords:
            morph.promote_morph_piece(w, parents=frozenset({w}))
        out.append(w)

    return list(dict.fromkeys(p for p in out if p and len(p) >= 2))


def morph_pieces_in_text(
    morph: MorphRegistry,
    text: str,
    *,
    mode: MorphPieceMode = "ingest",
    polarity_lexicon: dict[str, object] | None = None,
    vocab: set[str] | None = None,
) -> list[str]:
    """All morph pieces found across tokens in *text*."""
    found: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        found.extend(
            morph_pieces_for_token(
                morph,
                tok,
                vocab=vocab,
                mode=mode,
                polarity_lexicon=polarity_lexicon,
            )
        )
    return list(dict.fromkeys(found))


def morph_pieces(
    knowledge: SymbolKnowledgeIndex,
    token: str,
    *,
    mode: MorphPieceMode = "query",
) -> list[str]:
    """Knowledge-index wrapper (preferred call site)."""
    from aethos_symbol_entangle import _POLARITY_LEXICON

    return morph_pieces_for_token(
        knowledge.morph,
        token,
        vocab=knowledge.vocab,
        mode=mode,
        polarity_lexicon=_POLARITY_LEXICON if mode == "ingest" else None,
    )
