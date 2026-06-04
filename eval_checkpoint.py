"""
Pickle checkpoint for BEIR eval — build once, score many times (A/B λ runs).

Stores indices + hub signatures + trained anchors after the expensive build phase.
Does not store raw co-occurrence counts; distilled registry is saved separately.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvalBundle:
    """Everything needed for the query-scoring loop."""

    dataset: str
    mode: str
    qids: list[str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    cidx: Any
    hub_sigs: dict
    neighbor_map: dict
    meet_index: dict
    sub_comp_idx: Any
    comp_idx: Any
    phrase_idx: Any
    anchor_idx: Any
    pipe: Any
    hub_bytes_per_doc: float
    p50_ingest_ms: float
    p99_ingest_ms: float
    bytes_per_doc: float
    n_docs: int


def checkpoint_path(dataset: str, mode: str = "quality") -> Path:
    root = Path(__file__).resolve().parent / "brains"
    return root / f"{dataset}_{mode}.eval.pkl"


def save_checkpoint(bundle: EvalBundle, path: str | Path | None = None) -> Path:
    p = Path(path) if path is not None else checkpoint_path(bundle.dataset, bundle.mode)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_checkpoint(path: str | Path) -> EvalBundle:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, EvalBundle):
        raise TypeError(f"expected EvalBundle, got {type(obj)}")
    return obj
