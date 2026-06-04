"""
LatticeToken — structured token records for downstream models.

Each word in a document becomes one LatticeToken carrying promotion tier,
prime decomposition, lattice address, and optional cluster routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from aethos_lattice import LatticeId
from aethos_promotion import (
    LatticeTier,
    PromotionRegistry,
    intersection_prime,
    letter_chain,
)
from aethos_species import TokenSpecies, digit_chain, number_intersection
from aethos_tokenize import tokenize_spans, tokenize_words


def tokenize_words(text: str) -> list[str]:
    """Re-export shared tokenizer (NFKC + apostrophe + NUM)."""
    from aethos_tokenize import tokenize_words as _tw

    return _tw(text)


def anchor_chain(text: str) -> tuple[int, ...]:
    """L1 anchors: sorted letter primes (WORD) or ordered digit evens (NUM)."""
    t = text.lower()
    if t.isdigit():
        return digit_chain(t)
    return letter_chain(t)


@dataclass(frozen=True)
class LatticeToken:
    """One promoted token slot in document order."""

    text: str
    species: str  # WORD | NUM
    tier: str
    prime: int
    parent_primes: tuple[int, ...]
    letter_chain: tuple[int, ...]
    intersection_only: bool
    lattice_local: tuple[float, float, float]
    cluster_id: str
    cluster_score: float
    doc_index: int
    position: int

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "species": self.species,
            "tier": self.tier,
            "prime": self.prime,
            "parent_primes": self.parent_primes,
            "anchor_chain": self.letter_chain,
            "intersection_only": self.intersection_only,
            "lattice_local": self.lattice_local,
            "cluster_id": self.cluster_id,
            "cluster_score": self.cluster_score,
            "doc_index": self.doc_index,
            "position": self.position,
        }


def _tier_label(registry: PromotionRegistry, word: str) -> str:
    w = word.lower()
    if w.isdigit():
        return "num_intersection"
    if (LatticeTier.L3_WORD, w) in registry.promoted:
        return "dedicated_l3"
    if w in registry.intersections:
        return "intersection_only"
    return "letters_only"


def encode_word_token(
    registry: PromotionRegistry,
    word: str,
    *,
    doc_index: int = 0,
    position: int = 0,
    context: Iterable[str] | None = None,
    infer_cluster=None,
    n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
    species: str = "WORD",
) -> LatticeToken:
    """Build one LatticeToken for a WORD or NUM token."""
    w = word.lower()
    tok = registry.resolve_token(w)
    tier = _tier_label(registry, w)
    cid, score = "", 0.0
    if infer_cluster is not None:
        ctx = list(context) if context is not None else None
        cid, score = infer_cluster(w, ctx)
    local = registry.lattice_address(w, LatticeTier.L3_WORD, n=n, lattice_id=lattice_id)
    chain = anchor_chain(w)
    return LatticeToken(
        text=w,
        species=species,
        tier=tier,
        prime=tok.prime,
        parent_primes=tok.parent_primes,
        letter_chain=chain,
        intersection_only=registry.is_intersection_only(w),
        lattice_local=local,
        cluster_id=cid,
        cluster_score=score,
        doc_index=doc_index,
        position=position,
    )


def encode_document(
    text: str,
    registry: PromotionRegistry,
    *,
    doc_index: int = 0,
    infer_cluster=None,
    n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
) -> list[LatticeToken]:
    """Encode every token in one document as LatticeToken records."""
    spans = tokenize_spans(text)
    out: list[LatticeToken] = []
    words = [s.text for s in spans]
    ctx = frozenset(words)
    for span in spans:
        out.append(
            encode_word_token(
                registry,
                span.text,
                doc_index=doc_index,
                position=span.position,
                context=ctx,
                infer_cluster=infer_cluster,
                n=n,
                lattice_id=lattice_id,
                species=span.species.value,
            )
        )
    return out


def encode_corpus(
    documents: Iterable[str],
    registry: PromotionRegistry,
    *,
    infer_cluster=None,
    n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
) -> list[LatticeToken]:
    """Flatten many documents into one token stream."""
    tokens: list[LatticeToken] = []
    for doc_index, doc in enumerate(documents):
        tokens.extend(
            encode_document(
                doc,
                registry,
                doc_index=doc_index,
                infer_cluster=infer_cluster,
                n=n,
                lattice_id=lattice_id,
            )
        )
    return tokens


def verify_l1_consistency(word: str) -> tuple[bool, str]:
    """L1 check: intersection prime equals sum of letter primes."""
    w = word.lower()
    if not w.isalpha():
        return False, "non-alpha word"
    expected = sum(letter_chain(w))
    actual = intersection_prime(w)
    if expected != actual:
        return False, f"intersection mismatch {actual} != {expected}"
    return True, "ok"
