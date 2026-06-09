"""
BIT 3 — Doc attractor set K(doc)

Math:
  K(doc) = { κ(cell(w)) : w ∈ hubs(doc), strength(w) ≥ τ }

Inverted index: κ → [doc_id]
Uses BIT 1 word_to_spacetime_cell (hub-aligned chain), not coord-only rebuild.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aethos_hub_signature import LatticeHubSignature, build_all_hub_signatures
from aethos_physics import SpacetimeCell
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N, word_to_spacetime_cell
from pipeline.bit_02_attractor_key import (
    AttractorKey,
    DEFAULT_QUANTIZE,
    attractor_neighbors,
    kappa_from_cell,
)


@dataclass(frozen=True)
class DocAttractorSet:
    """Unique κ buckets for one document with strongest hub witness per key."""

    doc_id: str
    keys: frozenset[AttractorKey]
    witnesses: dict[AttractorKey, str]  # key → hub word (max strength)
    strengths: dict[AttractorKey, float] = field(default_factory=dict)


@dataclass
class CorpusAttractorIndex:
    """
    Inverted index: attractor bucket → doc_ids and hub words.

    Multiple hubs per doc may land in the same bucket (meet collapse).
    """

    quantize: float = DEFAULT_QUANTIZE
    anchor_n: int = DEFAULT_ANCHOR_N
    by_key: dict[AttractorKey, list[str]] = field(default_factory=dict)
    doc_keys: dict[str, set[AttractorKey]] = field(default_factory=dict)
    key_words: dict[AttractorKey, set[str]] = field(default_factory=dict)
    doc_witnesses: dict[str, dict[AttractorKey, str]] = field(default_factory=dict)

    def add(
        self,
        doc_id: str,
        key: AttractorKey,
        word: str = "",
        *,
        strength: float = 0.0,
    ) -> None:
        bucket = self.by_key.setdefault(key, [])
        if doc_id not in bucket:
            bucket.append(doc_id)
        self.doc_keys.setdefault(doc_id, set()).add(key)
        if word:
            self.key_words.setdefault(key, set()).add(word)
            witnesses = self.doc_witnesses.setdefault(doc_id, {})
            prev = witnesses.get(key)
            if prev is None or strength > 0:
                witnesses[key] = word

    def doc_set(self, doc_id: str) -> DocAttractorSet | None:
        keys = self.doc_keys.get(doc_id)
        if not keys:
            return None
        witnesses = self.doc_witnesses.get(doc_id, {})
        return DocAttractorSet(
            doc_id=doc_id,
            keys=frozenset(keys),
            witnesses={k: witnesses.get(k, "") for k in keys},
        )

    def neighbors(
        self,
        key: AttractorKey,
        *,
        radius: int = 1,
    ) -> set[AttractorKey]:
        return attractor_neighbors(key, radius=radius)

    def query_by_key(
        self,
        key: AttractorKey,
        *,
        radius: int = 0,
    ) -> list[str]:
        keys = self.neighbors(key, radius=radius) if radius > 0 else {key}
        seen: list[str] = []
        hit: set[str] = set()
        for k in keys:
            for doc_id in self.by_key.get(k, []):
                if doc_id not in hit:
                    hit.add(doc_id)
                    seen.append(doc_id)
        return seen

    def query_by_cell(
        self,
        cell: SpacetimeCell,
        *,
        radius: int = 1,
    ) -> list[str]:
        key = kappa_from_cell(cell, quantize=self.quantize)
        return self.query_by_key(key, radius=radius)

    def score_doc_overlap(
        self,
        query_keys: Iterable[AttractorKey],
        doc_id: str,
    ) -> float:
        q = set(query_keys)
        d = self.doc_keys.get(doc_id, set())
        if not q or not d:
            return 0.0
        inter = len(q & d)
        union = len(q | d)
        return inter / union if union else 0.0

    def rank_docs_by_overlap(
        self,
        query_keys: Iterable[AttractorKey],
        *,
        candidate_doc_ids: Sequence[str] | None = None,
    ) -> list[tuple[float, str]]:
        q = set(query_keys)
        pool = candidate_doc_ids if candidate_doc_ids is not None else list(self.doc_keys)
        scored: list[tuple[float, str]] = []
        for doc_id in pool:
            s = self.score_doc_overlap(q, doc_id)
            if s > 0:
                scored.append((s, doc_id))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return scored

    def z_modulus_band(self, key: AttractorKey) -> float:
        x, y, _ = key
        return math.sqrt(x * x + y * y) * self.quantize

    def summary(self) -> dict[str, int | float]:
        return {
            "buckets": len(self.by_key),
            "docs": len(self.doc_keys),
            "avg_keys_per_doc": (
                sum(len(v) for v in self.doc_keys.values()) / len(self.doc_keys)
                if self.doc_keys
                else 0.0
            ),
        }


def doc_attractor_set_from_signature(
    registry,
    sig: LatticeHubSignature,
    *,
    n: int = DEFAULT_ANCHOR_N,
    quantize: float = DEFAULT_QUANTIZE,
    strength_tau: float = 0.0,
) -> DocAttractorSet:
    """
    K(doc) from hub signature using BIT 1 cells.

    Duplicate κ in one doc collapse to the max-strength hub witness.
    """
    best: dict[AttractorKey, tuple[float, str]] = {}
    for word, entry in sig.hubs.items():
        if entry.strength < strength_tau:
            continue
        cell = word_to_spacetime_cell(registry, word, n=n)
        key = kappa_from_cell(cell, quantize=quantize)
        prev = best.get(key)
        if prev is None or entry.strength > prev[0]:
            best[key] = (entry.strength, word)
    return DocAttractorSet(
        doc_id=sig.doc_id,
        keys=frozenset(best.keys()),
        witnesses={k: w for k, (_, w) in best.items()},
        strengths={k: s for k, (s, _) in best.items()},
    )


def build_attractor_index_from_hub_signatures(
    registry,
    signatures: dict[str, LatticeHubSignature],
    *,
    n: int = DEFAULT_ANCHOR_N,
    quantize: float = DEFAULT_QUANTIZE,
    strength_tau: float = 0.0,
) -> CorpusAttractorIndex:
    """Build inverted κ index from hub signatures (BIT 3 deliverable)."""
    idx = CorpusAttractorIndex(quantize=quantize, anchor_n=n)
    for doc_id, sig in signatures.items():
        doc_set = doc_attractor_set_from_signature(
            registry,
            sig,
            n=n,
            quantize=quantize,
            strength_tau=strength_tau,
        )
        for key in doc_set.keys:
            word = doc_set.witnesses.get(key, "")
            strength = doc_set.strengths.get(key, 0.0)
            idx.add(doc_id, key, word, strength=strength)
    return idx


def build_attractor_index_from_corpus(
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    registry,
    *,
    top_k: int = 12,
    n: int = DEFAULT_ANCHOR_N,
    quantize: float = DEFAULT_QUANTIZE,
) -> CorpusAttractorIndex:
    """Convenience: hub signatures → attractor index."""
    sigs = build_all_hub_signatures(
        doc_ids,
        doc_tokens,
        registry,
        top_k=top_k,
        anchor_n=n,
    )
    return build_attractor_index_from_hub_signatures(
        registry,
        sigs,
        n=n,
        quantize=quantize,
    )


def top_hub_key_for_doc(
    registry,
    sig: LatticeHubSignature,
    *,
    n: int = DEFAULT_ANCHOR_N,
    quantize: float = DEFAULT_QUANTIZE,
) -> tuple[AttractorKey, str] | None:
    """Strongest hub word → κ (for gate retrieval checks)."""
    if not sig.hubs:
        return None
    word = max(sig.hubs.items(), key=lambda x: (-x[1].strength, x[0]))[0]
    cell = word_to_spacetime_cell(registry, word, n=n)
    return kappa_from_cell(cell, quantize=quantize), word


def verify_bit03_gate(
    registry,
    signatures: dict[str, LatticeHubSignature],
    *,
    n: int = DEFAULT_ANCHOR_N,
    quantize: float = DEFAULT_QUANTIZE,
    sample_size: int = 50,
    seed: int = 0,
) -> tuple[int, int, list[tuple[str, str]]]:
    """
    BIT 3 gate: each sampled doc retrievable by its top-hub κ at radius 0.

    Returns (passed, total, failures).
    """
    idx = build_attractor_index_from_hub_signatures(
        registry,
        signatures,
        n=n,
        quantize=quantize,
    )
    doc_ids = list(signatures.keys())
    if not doc_ids:
        return 0, 0, [("", "no signatures")]
    rng = random.Random(seed)
    if len(doc_ids) > sample_size:
        doc_ids = rng.sample(doc_ids, sample_size)
    passed = 0
    failures: list[tuple[str, str]] = []
    for doc_id in doc_ids:
        sig = signatures[doc_id]
        top = top_hub_key_for_doc(registry, sig, n=n, quantize=quantize)
        if top is None:
            failures.append((doc_id, "no hubs"))
            continue
        key, word = top
        hits = idx.query_by_key(key, radius=0)
        if doc_id not in hits:
            failures.append((doc_id, f"top hub {word!r} κ={key} missed doc"))
            continue
        passed += 1
    return passed, len(doc_ids), failures
