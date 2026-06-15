"""
Composite-only training — query→gold rare-word bridges, no bulk anchor scan.

Skips: build_heavy_anchor_index, train_on_qrels, convergence, λ calibration.
Runs: discover_discriminating_intersections, train_negative_anchors, discover_meta_intersections.
"""

from __future__ import annotations

import time
from pathlib import Path

from aethos_discriminative import (
    HeavyAnchorIndex,
    discover_discriminating_intersections,
    discover_meta_intersections,
    train_negative_anchors,
)
from aethos_persist import brain_path_for_dataset, save_brain
from core.learning_engine import BadCorrelationStore, bad_correlation_path
from eval_beir import build_neighbor_weights, load_qrels


def retrain_composites_on_bundle(
    bundle,
    *,
    dataset: str,
    mode: str = "quality",
    max_new_anchors: int = 2000,
    max_new_meta: int = 500,
    max_new_negatives: int = 500,
    clear_bad_correlation: bool = True,
    verbose: bool = True,
) -> int:
    """
    Replace anchor index with composite-only bridges discovered from train qrels.

    Returns total anchor count after training.
    """
    import aethos_hub_signature as hs

    hs.LAMBDA_COORD = 0.5
    hs.LAMBDA_NEIGHBOR = 0.15

    qrels_train_path = _train_qrels_path(bundle, dataset)
    qrels_train = load_qrels(qrels_train_path) if qrels_train_path else {}
    if not qrels_train:
        qrels_train = {
            qid: rel
            for qid, rel in bundle.qrels.items()
            if qid in bundle.queries
        }

    anchor_idx = HeavyAnchorIndex()
    bundle.anchor_idx = anchor_idx

    if clear_bad_correlation:
        bad_path = bad_correlation_path(dataset, mode)
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        BadCorrelationStore().save(bad_path)
        if verbose:
            print("  bad-correlation queue: cleared", flush=True)

    neighbor_map = bundle.neighbor_map or build_neighbor_weights(bundle.pipe.registry)
    bundle.neighbor_map = neighbor_map

    t0 = time.perf_counter()
    n_disc = discover_discriminating_intersections(
        anchor_idx,
        bundle.pipe.registry,
        bundle.queries,
        qrels_train,
        bundle.cidx.doc_ids,
        bundle.cidx.doc_tokens,
        bundle.cidx.doc_freq,
        len(bundle.cidx.doc_ids),
        bundle.hub_sigs,
        neighbor_map,
        bundle.cidx.doc_tf,
        bundle.cidx.doc_len,
        bundle.cidx.avg_dl,
        bundle.sub_comp_idx,
        bundle.phrase_idx,
        max_new_anchors=max_new_anchors,
        verbose=verbose,
    )
    n_neg = train_negative_anchors(
        anchor_idx,
        bundle.pipe.registry,
        bundle.queries,
        qrels_train,
        bundle.cidx.doc_ids,
        bundle.cidx.doc_tokens,
        bundle.cidx.doc_freq,
        len(bundle.cidx.doc_ids),
        bundle.hub_sigs,
        neighbor_map,
        bundle.cidx.doc_tf,
        bundle.cidx.doc_len,
        bundle.cidx.avg_dl,
        bundle.sub_comp_idx,
        bundle.phrase_idx,
        max_new_negatives=max_new_negatives,
        verbose=verbose,
    )
    n_meta = discover_meta_intersections(
        anchor_idx,
        bundle.cidx.doc_tokens,
        bundle.cidx.doc_freq,
        len(bundle.cidx.doc_ids),
        max_new=max_new_meta,
        verbose=verbose,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    b_path = brain_path_for_dataset(dataset, mode)
    b_path.parent.mkdir(parents=True, exist_ok=True)
    save_brain(anchor_idx, hs.LAMBDA_COORD, hs.LAMBDA_NEIGHBOR, b_path)

    n_active = sum(
        1 for a in anchor_idx.anchors.values() if a.learned_weight >= 0.05
    )
    if verbose:
        print(
            f"  composite-only train: {elapsed_ms:.0f} ms  "
            f"+{n_disc} discriminators  +{n_neg} negatives  +{n_meta} meta  "
            f"total={anchor_idx.n_anchors} anchors  ({n_active} w>=0.05)",
            flush=True,
        )
        print(
            f"  brain saved: {n_active} active -> {b_path.name}  "
            f"(λ_coord={hs.LAMBDA_COORD}, λ_neighbor={hs.LAMBDA_NEIGHBOR})",
            flush=True,
        )
    return anchor_idx.n_anchors


def _train_qrels_path(bundle, dataset: str) -> Path | None:
    """Resolve train qrels next to corpus if bundle was built from BEIR paths."""
    for base in (
        Path(__file__).resolve().parent.parent / "beir_datasets" / dataset / "qrels" / "train.tsv",
    ):
        if base.is_file():
            return base
    from beir_data_root import resolve_beir_root

    p = Path(resolve_beir_root()) / dataset / "qrels" / "train.tsv"
    return p if p.is_file() else None
