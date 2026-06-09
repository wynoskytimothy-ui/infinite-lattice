"""
BIT 10 — Scoring fusion: signal 8a (κ Jaccard) + optional |C| cap.

Wires QueryCellProfile.kappa_neighbor_q into rank_with_hub_signatures via
aethos_hub_signature (BM25 gate preserved; cap disabled by default).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pipeline.bit_02_attractor_key import AttractorKey

# Kept in sync with aethos_hub_signature.LAMBDA_KAPPA
DEFAULT_LAMBDA_KAPPA = 0.0
# Gate verifies mechanism at non-zero λ; production default stays 0 until train-tuned.
GATE_LAMBDA_KAPPA = 0.25


def cap_candidates_by_kappa_overlap(
    candidate_ids: Sequence[str],
    attractor_index,
    query_kappa_keys: Iterable[AttractorKey],
    *,
    cap: int,
    protect: Iterable[str] = (),
) -> list[str]:
    """
    Pre-filter candidates before hub scoring: top ``cap`` by κ Jaccard.

    ``protect`` docs (lexical anchors) are always kept; remaining slots filled
    by overlap rank, then original order.
    """
    if cap <= 0 or attractor_index is None or not query_kappa_keys:
        return list(candidate_ids)

    original = list(candidate_ids)
    protect_set = set(protect) & set(original)
    protected = [d for d in original if d in protect_set]
    if len(protected) >= cap:
        return protected[:cap]
    remaining = cap - len(protected)

    pool = [d for d in original if d not in protect_set]
    pre_ranked = attractor_index.rank_docs_by_overlap(
        query_kappa_keys,
        candidate_doc_ids=pool,
    )
    capped = [doc_id for _, doc_id in pre_ranked[:remaining]]
    capped_set = set(capped)
    fill = remaining - len(capped)
    if fill > 0:
        remainder = [d for d in pool if d not in capped_set]
        capped.extend(remainder[:fill])
    return protected + capped


def signal_8a_kappa_jaccard(
    profile,
    doc_id: str,
    attractor_index,
    query_kappa_keys: Iterable[AttractorKey] | None,
    *,
    lambda_kappa: float = DEFAULT_LAMBDA_KAPPA,
) -> float:
    """
    Signal 8a: IDF_max × Jaccard(κ_q, K(doc)) × λ_κ.

    Same units as other lattice bonuses in score_document / rank loop.
    """
    if attractor_index is None or not query_kappa_keys:
        return 0.0
    jaccard = attractor_index.score_doc_overlap(query_kappa_keys, doc_id)
    if jaccard <= 0.0:
        return 0.0
    idf_max = max(profile.idf.values(), default=1.0)
    return idf_max * jaccard * lambda_kappa


def verify_bit10_gate(
    registry,
    profile,
    cell_profile,
    attractor_index,
    hub_sigs: dict,
    candidate_ids: list[str],
    all_ids: list[str],
    *,
    doc_tokens: dict[str, frozenset[str]] | None = None,
) -> tuple[bool, list[str]]:
    """
    BIT 10 gate on small corpus:
      - 8a measurable on overlapping doc
      - cap shrinks pool when enabled
      - gated doc does not receive 8a bonus in rank loop
    """
    from aethos_hub_signature import rank_with_hub_signatures

    failures: list[str] = []
    keys = cell_profile.kappa_neighbor_q

    overlap_doc = "d0"
    weak_doc = "d3"
    s8_overlap = signal_8a_kappa_jaccard(
        profile, overlap_doc, attractor_index, keys,
        lambda_kappa=GATE_LAMBDA_KAPPA,
    )
    s8_weak = signal_8a_kappa_jaccard(
        profile, weak_doc, attractor_index, keys,
        lambda_kappa=GATE_LAMBDA_KAPPA,
    )
    if s8_overlap <= 0.0:
        failures.append(f"signal 8a zero on related doc {overlap_doc!r}")
    if s8_overlap <= s8_weak:
        failures.append(
            f"signal 8a related {s8_overlap:.4f} <= unrelated {s8_weak:.4f}"
        )

    capped = cap_candidates_by_kappa_overlap(
        candidate_ids,
        attractor_index,
        keys,
        cap=2,
    )
    if len(capped) > 2:
        failures.append(f"cap=2 produced {len(capped)} candidates")

    ranked = rank_with_hub_signatures(
        profile,
        [overlap_doc, weak_doc],
        hub_sigs,
        all_ids,
        doc_tokens=doc_tokens,
        attractor_index=attractor_index,
        query_kappa_keys=keys,
        top_k=2,
    )
    if ranked and ranked[0] != overlap_doc:
        failures.append(f"8a rank expected {overlap_doc!r} first, got {ranked[0]!r}")

    return len(failures) == 0, failures
