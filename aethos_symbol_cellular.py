"""
Cellular composite structure — membrane (frequent filler) vs signal (rare).

Frequent subwords (``the``, ``and``, ``ed``, ``in``, …) form the **membrane** —
they hold the sentence together like a cell wall but **must not** create false
correlations with rare terms.

**Rare** composites / subwords are **signal** units.  When building composite
relationships inside a document, only **rare ↔ rare** pairs entangle at their
intersection.  Filler co-occurrence is ignored for correlation (membrane passes
through without binding).

    [membrane: the, and, of, ed, es, …]  — structure only, no correlation edges
    [signal: diminished, hypothesis, raise, …]  — rare-rare entanglement only
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from aethos_symbol_entangle import (
    ContextScore,
    EntangledPair,
    EntanglementRegistry,
    _POLARITY_LEXICON,
    find_morph_pieces,
    polarity_of,
    score_context,
)
from aethos_symbol_morph import MorphRegistry, build_morph_registry

# Polarity-bearing terms are always signal (not membrane filler)
_POLARITY_SIGNAL: frozenset[str] = frozenset(_POLARITY_LEXICON.keys())

_TOKEN_RE = re.compile(r"[a-z]+")

# Default membrane — high-frequency English filler subwords
_DEFAULT_MEMBRANE: frozenset[str] = frozenset({
    "the", "and", "or", "of", "in", "on", "at", "to", "for", "a", "an",
    "is", "was", "are", "be", "ed", "es", "ing", "ly", "er", "th", "he",
    "it", "as", "by", "with", "from", "that", "this", "were",
    "we", "our", "they", "them", "their", "these", "those", "which",
    "who", "what", "when", "where", "how", "not", "no", "but", "if",
    "been", "has", "have", "had", "here", "there", "also", "than",
    "then", "into", "over", "such", "both", "between", "after", "before",
    "during", "through", "under", "more", "most", "other", "some", "any",
    "each", "all", "can", "could", "may", "might", "would", "should",
    "will", "shall", "do", "does", "did", "done", "being", "its",
})


class CellularRole(str, Enum):
    MEMBRANE = "membrane"  # frequent filler — structure, no correlation
    SIGNAL = "signal"      # rare — may entangle with other signal pieces


@dataclass(frozen=True)
class PieceProfile:
    text: str
    role: CellularRole
    frequency: int
    doc_count: int
    rare: bool


@dataclass
class CellularRegistry:
    """
    Classifies pieces from corpus frequency; gates rare-rare entanglement.
    """

    profiles: dict[str, PieceProfile] = field(default_factory=dict)
    filler_doc_fraction: float = 0.35
    rare_max_doc_count: int = 2
    membrane_lexicon: frozenset[str] = _DEFAULT_MEMBRANE
    rare_pairs: dict[tuple[str, str], EntangledPair] = field(default_factory=dict)
    blocked_pairs: int = 0

    def classify(
        self,
        piece: str,
        *,
        frequency: int,
        doc_count: int,
        n_docs: int,
        morph_rare: bool = False,
    ) -> PieceProfile:
        p = piece.lower()
        frac = doc_count / max(n_docs, 1)
        if p in _POLARITY_SIGNAL:
            is_membrane = False
            is_rare = True
        else:
            is_membrane = (
                p in self.membrane_lexicon
                or frac >= self.filler_doc_fraction
                or (frequency >= 10 and doc_count >= max(3, n_docs // 2))
            )
            is_rare = (
                morph_rare
                or (not is_membrane and doc_count <= self.rare_max_doc_count)
            )
        role = CellularRole.MEMBRANE if is_membrane else CellularRole.SIGNAL
        prof = PieceProfile(
            text=p,
            role=role,
            frequency=frequency,
            doc_count=doc_count,
            rare=is_rare,
        )
        self.profiles[p] = prof
        return prof

    def role_of(self, piece: str) -> CellularRole:
        p = piece.lower()
        if p in self.profiles:
            return self.profiles[p].role
        if p in self.membrane_lexicon:
            return CellularRole.MEMBRANE
        return CellularRole.SIGNAL

    def allows_correlation(self, left: str, right: str) -> bool:
        """
        Rare-rare only.  Membrane filler never correlates (blocks false edges).
        """
        rl, rr = self.role_of(left), self.role_of(right)
        if rl == CellularRole.MEMBRANE or rr == CellularRole.MEMBRANE:
            return False
        # both signal
        lp = self.profiles.get(left.lower())
        rp = self.profiles.get(right.lower())
        if lp and rp:
            return lp.rare and rp.rare
        return rl == CellularRole.SIGNAL and rr == CellularRole.SIGNAL


def _ensure_membrane_tokens(text: str, morph: MorphRegistry) -> None:
    """Promote stopword membrane tokens when present (structure only)."""
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _DEFAULT_MEMBRANE and tok not in morph.subwords:
            morph.promote_morph_piece(tok, parents=frozenset({tok}))


def build_cellular_profiles(
    corpus: dict[str, str],
    morph: MorphRegistry,
    *,
    filler_doc_fraction: float = 0.35,
    rare_max_doc_count: int = 2,
    knowledge_mode: bool = False,
) -> CellularRegistry:
    """Count piece frequencies across corpus; label membrane vs signal."""
    if knowledge_mode:
        filler_doc_fraction = 1.1  # never auto-membrane by doc fraction
        rare_max_doc_count = max(rare_max_doc_count, len(corpus))
    cell = CellularRegistry(
        filler_doc_fraction=filler_doc_fraction,
        rare_max_doc_count=rare_max_doc_count,
    )
    n_docs = len(corpus)
    freq: dict[str, int] = {}
    doc_hits: dict[str, set[str]] = {}

    for doc_id, text in corpus.items():
        _ensure_membrane_tokens(text, morph)
        pieces = find_morph_pieces(text, morph)
        for tok in _TOKEN_RE.findall(text.lower()):
            if tok in _DEFAULT_MEMBRANE and tok not in pieces:
                pieces.append(tok)
        for p in pieces:
            freq[p] = freq.get(p, 0) + 1
            doc_hits.setdefault(p, set()).add(doc_id)

    for piece, count in freq.items():
        morph_rare = (
            piece in morph.composites and morph.composites[piece].rare
        )
        cell.classify(
            piece,
            frequency=count,
            doc_count=len(doc_hits.get(piece, set())),
            n_docs=n_docs,
            morph_rare=morph_rare,
        )
    return cell


@dataclass
class CellularContextScore(ContextScore):
    """Context score with membrane/signal split."""

    membrane_pieces: tuple[str, ...] = ()
    signal_pieces: tuple[str, ...] = ()
    rare_correlations: tuple[EntangledPair, ...] = ()
    blocked_correlations: int = 0


def score_context_cellular(
    text: str,
    morph: MorphRegistry,
    entangle: EntanglementRegistry,
    cellular: CellularRegistry,
    *,
    piece_weight: float = 1.0,
    entangle_weight: float = 0.5,
) -> CellularContextScore:
    """
    Score doc: membrane pieces contribute polarity only;
    entanglement only on rare-rare signal pairs.
    """
    _ensure_membrane_tokens(text, morph)
    pieces = find_morph_pieces(text, morph)
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _DEFAULT_MEMBRANE and tok not in pieces:
            pieces.append(tok)
    membrane = tuple(p for p in pieces if cellular.role_of(p) == CellularRole.MEMBRANE)
    signal = tuple(p for p in pieces if cellular.role_of(p) == CellularRole.SIGNAL)

    base = sum(polarity_of(p).value * piece_weight for p in pieces)

    entangled_found: list[EntangledPair] = []
    rare_found: list[EntangledPair] = []
    blocked = 0
    bonus = 0.0

    for i, a in enumerate(pieces):
        for b in pieces[i + 1 :]:
            if not cellular.allows_correlation(a, b):
                blocked += 1
                cellular.blocked_pairs += 1
                continue
            pair = entangle.bind_pair(a, b)
            if pair:
                entangled_found.append(pair)
                key = tuple(sorted((a.lower(), b.lower())))
                cellular.rare_pairs[key] = pair
                rare_found.append(pair)
                if pair.opposite:
                    bonus += entangle_weight
                else:
                    bonus += entangle_weight * 0.5

    return CellularContextScore(
        text=text,
        pieces_found=tuple(pieces),
        base_score=base,
        entanglement_bonus=bonus,
        total=base + bonus,
        entangled=tuple(entangled_found),
        membrane_pieces=membrane,
        signal_pieces=signal,
        rare_correlations=tuple(rare_found),
        blocked_correlations=blocked,
    )


def build_cellular_entanglement(
    corpus: dict[str, str],
    morph: MorphRegistry | None = None,
    *,
    knowledge_mode: bool = False,
) -> tuple[EntanglementRegistry, CellularRegistry]:
    """Full pipeline: morph + cellular profiles + rare-rare entanglement only."""
    vocab: set[str] = set()
    for text in corpus.values():
        vocab.update(_TOKEN_RE.findall(text.lower()))
    morph = morph or build_morph_registry(vocab)
    cellular = build_cellular_profiles(corpus, morph, knowledge_mode=knowledge_mode)
    entangle = EntanglementRegistry(morph=morph)

    for doc_id, text in corpus.items():
        sc = score_context_cellular(text, morph, entangle, cellular)
        entangle.doc_scores[doc_id] = sc

    return entangle, cellular


def demo() -> None:
    corpus = {
        "d1": "the diminished score was lower after treatment in the cell",
        "d2": "the raise improves scores in the membrane",
        "d3": "diminis and raise together in a rare pathway",
        "d4": "the and of in ed es filler words everywhere",
        "d5": "hypothesis diminished when rare terms co-occur",
    }
    ent, cell = build_cellular_entanglement(corpus)

    print("=" * 60)
    print("CELLULAR COMPOSITES — membrane filler vs rare-rare signal")
    print("=" * 60)

    print("\n  piece roles:")
    for p, prof in sorted(cell.profiles.items(), key=lambda x: (-x[1].frequency, x[0])):
        print(
            f"    {p!r:14}  {prof.role.value:8}  rare={prof.rare}  "
            f"freq={prof.frequency}  docs={prof.doc_count}"
        )

    print(f"\n  blocked membrane correlations (total): {cell.blocked_pairs}")

    for doc_id, sc in ent.doc_scores.items():
        if not isinstance(sc, CellularContextScore):
            continue
        print(f"\n  [{doc_id}] signal={sc.signal_pieces}  membrane={sc.membrane_pieces}")
        print(f"    rare correlations: {len(sc.rare_correlations)}  blocked: {sc.blocked_correlations}")
        for ep in sc.rare_correlations:
            tag = "OPPOSITE" if ep.opposite else "aligned"
            print(f"      {ep.left!r}+{ep.right!r}  imag={ep.intersection_imag}  {tag}")

    print(f"\n  rare-rare pairs in index: {len(cell.rare_pairs)}")
    for key, ep in list(cell.rare_pairs.items())[:6]:
        print(f"    {key[0]!r}+{key[1]!r}  opposite={ep.opposite}")


if __name__ == "__main__":
    demo()
