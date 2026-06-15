"""
Composite entanglement — co-occurring composites bind at intersection; opposites.

When composites appear **together** in a document (or query), they **entangle**
at their composite intersection on the imaginary line:

    intersection_imag = imag(A) + imag(B)   (2-way meet sum)

Polarity
--------
  **raise** family  →  positive  (+1)  improves score
  **diminish** family →  negative  (−1)  lowers score

When ``diminis`` and ``raise`` are found together they **entangle** (linked for
retrieval) but are **opposites** (signs fight).  Same for ``diminished`` + ``raise``.

``lower`` binds with diminish (same negative pole).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from aethos_complex_plane import imaginary_start
from aethos_symbol_morph import MorphComposite, MorphRegistry, build_morph_registry

import re

_TOKEN_RE = re.compile(r"[a-z]+")


class Polarity(IntEnum):
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1


# Morph pieces and word families → score sign
_POLARITY_LEXICON: dict[str, Polarity] = {
    "dimin": Polarity.NEGATIVE,
    "diminis": Polarity.NEGATIVE,
    "diminish": Polarity.NEGATIVE,
    "diminished": Polarity.NEGATIVE,
    "diminishes": Polarity.NEGATIVE,
    "lower": Polarity.NEGATIVE,
    "reduce": Polarity.NEGATIVE,
    "reduced": Polarity.NEGATIVE,
    "raise": Polarity.POSITIVE,
    "raised": Polarity.POSITIVE,
    "raises": Polarity.POSITIVE,
    "increase": Polarity.POSITIVE,
    "improve": Polarity.POSITIVE,
    "improves": Polarity.POSITIVE,
}


def polarity_of(text: str) -> Polarity:
    t = text.lower()
    if t in _POLARITY_LEXICON:
        return _POLARITY_LEXICON[t]
    for key, pol in _POLARITY_LEXICON.items():
        if key in t:
            return pol
    return Polarity.NEUTRAL


def are_opposites(a: str, b: str) -> bool:
    pa, pb = polarity_of(a), polarity_of(b)
    return pa != Polarity.NEUTRAL and pb != Polarity.NEUTRAL and pa == -pb


@dataclass(frozen=True)
class EntangledPair:
    """Two composites/subwords found together — bound at intersection."""

    left: str
    right: str
    left_prime: int
    right_prime: int
    intersection_imag: int
    opposite: bool
    co_occurrences: int = 1

    @property
    def intersection_z(self) -> complex:
        return imaginary_start(self.intersection_imag).z


@dataclass
class ContextScore:
    """Score breakdown for one document or query."""

    text: str
    pieces_found: tuple[str, ...]
    base_score: float
    entanglement_bonus: float
    total: float
    entangled: tuple[EntangledPair, ...]


@dataclass
class EntanglementRegistry:
    morph: MorphRegistry
    pairs: dict[tuple[str, str], EntangledPair] = field(default_factory=dict)
    doc_scores: dict[str, ContextScore] = field(default_factory=dict)

    def _prime_for(self, piece: str) -> int | None:
        if piece in self.morph.composites:
            return self.morph.composites[piece].composite_prime
        if piece in self.morph.subwords:
            return self.morph.subwords[piece].prime
        return None

    def _imag_for(self, piece: str) -> int | None:
        if piece in self.morph.composites:
            return self.morph.composites[piece].imaginary_position
        if piece in self.morph.subwords:
            return self.morph.subwords[piece].imaginary_position
        return None

    def bind_pair(self, left: str, right: str, *, count: int = 1) -> EntangledPair | None:
        """Entangle two pieces at composite intersection (imag sum)."""
        key = tuple(sorted((left.lower(), right.lower())))
        imag_l = self._imag_for(key[0])
        imag_r = self._imag_for(key[1])
        prime_l = self._prime_for(key[0])
        prime_r = self._prime_for(key[1])
        if imag_l is None or imag_r is None or prime_l is None or prime_r is None:
            return None
        intersection = imag_l + imag_r
        opp = are_opposites(key[0], key[1])
        if key in self.pairs:
            existing = self.pairs[key]
            self.pairs[key] = EntangledPair(
                left=existing.left,
                right=existing.right,
                left_prime=existing.left_prime,
                right_prime=existing.right_prime,
                intersection_imag=existing.intersection_imag,
                opposite=existing.opposite,
                co_occurrences=existing.co_occurrences + count,
            )
            return self.pairs[key]
        pair = EntangledPair(
            left=key[0],
            right=key[1],
            left_prime=prime_l,
            right_prime=prime_r,
            intersection_imag=intersection,
            opposite=opp,
            co_occurrences=count,
        )
        self.pairs[key] = pair
        return pair


def find_morph_pieces(text: str, morph: MorphRegistry) -> list[str]:
    """Token-level morph composites, subwords, and polarity lexicon pieces."""
    from aethos_symbol_morph_pieces import morph_pieces_in_text

    return morph_pieces_in_text(
        morph,
        text,
        mode="ingest",
        polarity_lexicon=_POLARITY_LEXICON,
    )


def score_context(
    text: str,
    morph: MorphRegistry,
    entangle: EntanglementRegistry,
    *,
    piece_weight: float = 1.0,
    entangle_weight: float = 0.5,
) -> ContextScore:
    """
    Score a document/query.

    raise → positive contribution; diminish/lower → negative.
    Co-occurring entangled opposites bind at intersection (+ entangle_weight).
    """
    pieces = find_morph_pieces(text, morph)
    base = sum(polarity_of(p).value * piece_weight for p in pieces)

    entangled_found: list[EntangledPair] = []
    bonus = 0.0
    for i, a in enumerate(pieces):
        for b in pieces[i + 1 :]:
            pair = entangle.bind_pair(a, b)
            if pair:
                entangled_found.append(pair)
                if pair.opposite:
                    # opposites entangle — link strength, signs already in base
                    bonus += entangle_weight * pair.co_occurrences
                else:
                    bonus += entangle_weight * 0.5

    total = base + bonus
    return ContextScore(
        text=text,
        pieces_found=tuple(pieces),
        base_score=base,
        entanglement_bonus=bonus,
        total=total,
        entangled=tuple(entangled_found),
    )


def build_entanglement_index(
    corpus: dict[str, str],
    morph: MorphRegistry | None = None,
) -> EntanglementRegistry:
    """Ingest corpus; entangle composites found together in each doc."""
    morph = morph or build_morph_registry(_default_vocab(corpus))
    reg = EntanglementRegistry(morph=morph)
    for doc_id, text in corpus.items():
        sc = score_context(text, morph, reg)
        reg.doc_scores[doc_id] = sc
    return reg


def _default_vocab(corpus: dict[str, str]) -> set[str]:
    words: set[str] = set()
    for text in corpus.values():
        words.update(_TOKEN_RE.findall(text.lower()))
    return words


def demo() -> None:
    corpus = {
        "d1": "the diminished score was lower after treatment",
        "d2": "raise improves scores",
        "d3": "diminis and raise appear together but oppose each other",
        "d4": "scores lower when diminish pathway activates",
    }
    morph = build_morph_registry(_default_vocab(corpus))
    reg = build_entanglement_index(corpus, morph)

    print("=" * 60)
    print("COMPOSITE ENTANGLEMENT — opposites at intersection")
    print("=" * 60)

    for doc_id, sc in reg.doc_scores.items():
        print(f"\n  [{doc_id}] total={sc.total:+.1f}  base={sc.base_score:+.1f}  "
              f"entangle={sc.entanglement_bonus:+.1f}")
        print(f"    pieces: {sc.pieces_found}")
        for ep in sc.entangled:
            opp = "OPPOSITE" if ep.opposite else "aligned"
            print(
                f"    entangled {ep.left!r}+{ep.right!r}  "
                f"intersection imag={ep.intersection_imag}  z={ep.intersection_z}  {opp}"
            )

    key = tuple(sorted(("diminis", "raise")))
    if key in reg.pairs:
        p = reg.pairs[key]
        print(f"\n  diminis + raise: opposite={p.opposite}  intersection={p.intersection_imag}")


if __name__ == "__main__":
    demo()
