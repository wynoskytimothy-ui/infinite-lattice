"""
Corpus-scoped pool primes — append-only, deterministic per corpus name.

Each corpus name receives one SPECIES-tier pool prime from the shared
PromotionRegistry (Hilbert hotel: letter meets and L2 promotions never relocate).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from aethos_promotion import LatticeTier

from lattice_retriever_v1.stage04_promote import Stage04Registry

_CORPUS_BY_REGISTRY: dict[int, dict[str, int]] = defaultdict(dict)


def _corpus_map(registry: Stage04Registry) -> dict[str, int]:
    return _CORPUS_BY_REGISTRY[id(registry.registry)]


def allocate_corpus_prime(name: str, registry: Stage04Registry) -> int:
    """Append-only SPECIES pool prime for *name*; replay returns the same prime."""
    pool = _corpus_map(registry)
    if name in pool:
        return pool[name]
    prime = registry.registry._alloc_prime(LatticeTier.L4_CORR, species=True)
    pool[name] = prime
    return prime


@dataclass(frozen=True)
class CorpusScope:
    """One corpus slice: name, dedicated pool prime, doc-id tag prefix."""

    corpus_name: str
    corpus_prime: int
    doc_id_prefix: str


def corpus_scope(name: str, registry: Stage04Registry) -> CorpusScope:
    """Build a scope record after allocating (or replaying) the corpus prime."""
    prime = allocate_corpus_prime(name, registry)
    return CorpusScope(corpus_name=name, corpus_prime=prime, doc_id_prefix=name)


def tag_doc_id(corpus_name: str, doc_id: str) -> str:
    """Scoped doc id — e.g. ``scifact:doc1``."""
    return f"{corpus_name}:{doc_id}"
