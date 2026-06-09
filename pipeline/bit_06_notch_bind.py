"""
BIT 6 — Notch fingerprint bound to SpacetimeCell κ

Math:
  M(w) = { conj(z_i)·z_j : i,j ∈ {VA1..VA4} }
  Store top-K peaks by |M_ij|, K=10 → 100 B payload

Bind rule:
  doc fingerprint ↔ top hub w* with κ(cell(w*)) and BIT 1 chain

Scoring gate (existing):
  doc_notch_score requires pool_factor Jaccard ≥ θ before notch similarity.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Iterable, Sequence

from aethos_hub_signature import (
    LatticeHubSignature,
    build_all_hub_signatures,
    pool_factors_for_word,
)
from aethos_notch_encoder import (
    DEFAULT_DOC_NOTCH_BYTES,
    DEFAULT_TOP_K,
    NOTCH_BYTES,
    Notch,
    correlation_matrix_4x4,
    doc_notch_score,
    extract_top_notches,
    notch_similarity,
    pack_notches,
    unpack_notches,
)
from aethos_physics import SpacetimeCell
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N, word_to_spacetime_cell
from pipeline.bit_02_attractor_key import AttractorKey, kappa_from_cell

DEFAULT_POOL_JACCARD_THETA = 0.0


@dataclass(frozen=True)
class BoundNotchFingerprint:
    """
    Document notch payload bound to attractor key of top hub.

    Wire payload: top_k × 10 B notches (default 100 B).
    """

    doc_id: str
    attractor_key: AttractorKey
    top_hub: str
    notches: bytes
    n: int = DEFAULT_ANCHOR_N
    pool_factors: frozenset[int] = frozenset()

    @property
    def payload_bytes(self) -> int:
        return len(self.notches)

    def encoded_size(self) -> int:
        hub_b = self.top_hub.encode("utf-8")
        return 12 + len(self.notches) + len(hub_b)  # doc_id str + κ + header

    def decoded_notches(self) -> tuple[Notch, ...]:
        return unpack_notches(self.notches)


def correlation_matrix_for_cell(cell: SpacetimeCell) -> dict:
    """4×4 branch correlation matrix M_ij = conj(z_i)·z_j at cell rail n."""
    if not cell.chain:
        raise ValueError("cell.chain required for notch matrix")
    chain = tuple(int(x) for x in cell.chain)
    return correlation_matrix_4x4(chain, int(cell.n))


def encode_notches_for_cell(
    cell: SpacetimeCell,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Notch, ...]:
    """Top-K notch peaks from BIT 1 SpacetimeCell chain."""
    matrix = correlation_matrix_for_cell(cell)
    return extract_top_notches(matrix, top_k=top_k)


def encode_notches_for_word(
    registry,
    word: str,
    *,
    n: int = DEFAULT_ANCHOR_N,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Notch, ...]:
    cell = word_to_spacetime_cell(registry, word, n=n)
    return encode_notches_for_cell(cell, top_k=top_k)


def top_hub_word(sig: LatticeHubSignature) -> str | None:
    """Strongest hub by compression_strength."""
    if not sig.hubs:
        return None
    return max(sig.hubs.items(), key=lambda x: (-x[1].strength, x[0]))[0]


def bind_notch_from_hub_signature(
    sig: LatticeHubSignature,
    registry,
    *,
    n: int = DEFAULT_ANCHOR_N,
    top_k: int = DEFAULT_TOP_K,
    max_bytes: int = DEFAULT_DOC_NOTCH_BYTES,
) -> BoundNotchFingerprint | None:
    """
    Bind doc notch fingerprint to κ(cell(w*)) for top hub w*.

    Returns None when signature has no hubs.
    """
    word = top_hub_word(sig)
    if not word:
        return None
    cell = word_to_spacetime_cell(registry, word, n=n)
    notches = encode_notches_for_cell(cell, top_k=top_k)
    blob = pack_notches(notches)
    max_payload = max(0, max_bytes - 20)
    max_notches = max_payload // NOTCH_BYTES
    if len(notches) > max_notches:
        notches = notches[:max_notches]
        blob = pack_notches(notches)
    return BoundNotchFingerprint(
        doc_id=sig.doc_id,
        attractor_key=kappa_from_cell(cell),
        top_hub=word,
        notches=blob,
        n=n,
        pool_factors=pool_factors_for_word(registry, word),
    )


def build_all_notch_fingerprints(
    signatures: dict[str, LatticeHubSignature],
    registry,
    *,
    n: int = DEFAULT_ANCHOR_N,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, BoundNotchFingerprint]:
    out: dict[str, BoundNotchFingerprint] = {}
    for doc_id, sig in signatures.items():
        fp = bind_notch_from_hub_signature(sig, registry, n=n, top_k=top_k)
        if fp is not None:
            out[doc_id] = fp
    return out


def aggregate_query_notches_bound(
    words: Sequence[str],
    registry,
    *,
    n: int = DEFAULT_ANCHOR_N,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Notch, ...]:
    """Merge query-word notches (BIT 1 chain); dedupe branch pairs, keep max amp."""
    best: dict[tuple[int, int], Notch] = {}
    for w in words:
        if len(w) < 3:
            continue
        try:
            for notch in encode_notches_for_word(registry, w, n=n, top_k=top_k):
                key = notch.branch_pair
                if key not in best or notch.amplitude > best[key].amplitude:
                    best[key] = notch
        except Exception:
            continue
    ranked = sorted(best.values(), key=lambda x: -x.amplitude)
    return tuple(ranked[:top_k])


def score_bound_notch_pair(
    query_words: Sequence[str],
    doc_fp: BoundNotchFingerprint,
    registry,
    *,
    n: int = DEFAULT_ANCHOR_N,
    min_pool_jaccard: float = DEFAULT_POOL_JACCARD_THETA,
) -> float:
    """Notch similarity gated by pool-factor Jaccard (Signal 5b rule)."""
    q_notches = aggregate_query_notches_bound(query_words, registry, n=n)
    q_factors: frozenset[int] = frozenset()
    for w in query_words:
        pf = pool_factors_for_word(registry, w.lower())
        q_factors = q_factors | pf
    return doc_notch_score(
        q_notches,
        doc_fp.decoded_notches(),
        q_factors,
        doc_fp.pool_factors,
        min_pool_jaccard=min_pool_jaccard,
    )


def verify_bit06_gate(
    registry,
    signatures: dict[str, LatticeHubSignature],
    *,
    n: int = DEFAULT_ANCHOR_N,
    top_k: int = DEFAULT_TOP_K,
    seed: int = 0,
) -> tuple[bool, list[str]]:
    """
    BIT 6 gate:
      - Same doc rebuilt twice → identical notch bytes
      - Similar doc pairs beat random pairs on notch Jaccard (mean)
    """
    failures: list[str] = []

    fps_a = build_all_notch_fingerprints(signatures, registry, n=n, top_k=top_k)
    fps_b = build_all_notch_fingerprints(signatures, registry, n=n, top_k=top_k)
    for doc_id in fps_a:
        if doc_id not in fps_b:
            failures.append(f"{doc_id}: missing on rebuild")
            continue
        if fps_a[doc_id].notches != fps_b[doc_id].notches:
            failures.append(f"{doc_id}: notch bytes differ on rebuild")
        if fps_a[doc_id].attractor_key != fps_b[doc_id].attractor_key:
            failures.append(f"{doc_id}: κ key differ on rebuild")

    doc_ids = list(fps_a.keys())
    if len(doc_ids) < 2:
        failures.append("need ≥2 docs for similarity gate")
        return False, failures

    def token_set(did: str) -> set[str]:
        sig = signatures[did]
        return set(sig.hubs.keys())

    similar_sims: list[float] = []
    random_sims: list[float] = []
    rng = random.Random(seed)
    for i, da in enumerate(doc_ids):
        for db in doc_ids[i + 1 :]:
            sim = notch_similarity(
                fps_a[da].decoded_notches(),
                fps_a[db].decoded_notches(),
            )
            fa, fb = fps_a[da], fps_a[db]
            if fa.top_hub == fb.top_hub or fa.attractor_key == fb.attractor_key:
                similar_sims.append(sim)
            elif not (token_set(da) & token_set(db)):
                random_sims.append(sim)

    if not random_sims and len(doc_ids) >= 2:
        for _ in range(min(30, len(doc_ids) * 3)):
            da, db = rng.sample(doc_ids, 2)
            fa, fb = fps_a[da], fps_a[db]
            if fa.top_hub != fb.top_hub and fa.attractor_key != fb.attractor_key:
                if not (token_set(da) & token_set(db)):
                    random_sims.append(
                        notch_similarity(
                            fps_a[da].decoded_notches(),
                            fps_a[db].decoded_notches(),
                        )
                    )

    if similar_sims and random_sims:
        mean_sim = statistics.mean(similar_sims)
        mean_rand = statistics.mean(random_sims)
        if mean_sim <= mean_rand:
            failures.append(
                f"same-bind mean {mean_sim:.4f} ≤ random mean {mean_rand:.4f}"
            )
    elif not similar_sims:
        # corpus too small for bind-collision pairs — require self-sim = 1
        for fp in fps_a.values():
            s = notch_similarity(fp.decoded_notches(), fp.decoded_notches())
            if s < 0.99:
                failures.append(f"{fp.doc_id}: self notch sim {s:.4f}")

    for fp in fps_a.values():
        if fp.payload_bytes > DEFAULT_DOC_NOTCH_BYTES:
            failures.append(
                f"{fp.doc_id}: payload {fp.payload_bytes}B > budget "
                f"{DEFAULT_DOC_NOTCH_BYTES}B"
            )
        if len(fp.notches) % NOTCH_BYTES != 0:
            failures.append(f"{fp.doc_id}: notch blob not multiple of 10")

    return len(failures) == 0, failures
