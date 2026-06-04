"""
Serialize / restore promotion registry, natural-reading state, and trained
retrieval brain (anchor weights + calibrated signal λ values).

JSON on disk — enough to resume corpus training without re-reading text,
and to accumulate anchor knowledge across runs (compound learning).

Brain file layout
-----------------
  {
    "version": 2,
    "lambda_coord": 0.5,
    "lambda_neighbor": 0.1,
    "anchors": {
      "<composite_int>": {
        "prime_a": int, "prime_b": int,
        "word_a": str, "word_b": str,
        "correct_count": int, "wrong_count": int,
        "doc_ids": [str, ...],
        "n_docs": int
      },
      ...
    }
  }

On next run the saved counts are loaded as starting values; new training
ADDS to them, so accuracy compounds across multiple runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aethos_natural import CooccurrenceGraph, NaturalReader
from aethos_promotion import (
    CorrelationLink,
    LatticeTier,
    PromotedToken,
    PromotionRegistry,
)


def _token_to_dict(tok: PromotedToken) -> dict[str, Any]:
    return {
        "text": tok.text,
        "tier": int(tok.tier),
        "prime": tok.prime,
        "parent_primes": list(tok.parent_primes),
        "intersection_only": tok.intersection_only,
    }


def _token_from_dict(d: dict[str, Any]) -> PromotedToken:
    return PromotedToken(
        text=d["text"],
        tier=LatticeTier(int(d["tier"])),
        prime=int(d["prime"]),
        parent_primes=tuple(d["parent_primes"]),
        intersection_only=bool(d.get("intersection_only", False)),
    )


def _link_to_dict(link: CorrelationLink) -> dict[str, Any]:
    return asdict(link)


def _link_from_dict(d: dict[str, Any]) -> CorrelationLink:
    return CorrelationLink(**d)


def registry_to_dict(reg: PromotionRegistry) -> dict[str, Any]:
    promoted = {}
    for (tier, text), tok in reg.promoted.items():
        promoted[f"{int(tier)}:{text}"] = _token_to_dict(tok)
    intersections = {w: _token_to_dict(t) for w, t in reg.intersections.items()}
    correlations = {
        f"{a}|{b}": _link_to_dict(link) for (a, b), link in reg.correlations.items()
    }
    contexts = {
        w: [list(sorted(c)) for c in ctxs] for w, ctxs in reg.word_contexts.items()
    }
    return {
        "version": 1,
        "symbol_counts": reg.symbol_counts,
        "subword_counts": reg.subword_counts,
        "word_counts": reg.word_counts,
        "word_contexts": contexts,
        "promoted": promoted,
        "intersections": intersections,
        "correlations": correlations,
        "next_promotion_idx": reg._next_promotion_idx,
    }


def registry_from_dict(doc: dict[str, Any]) -> PromotionRegistry:
    reg = PromotionRegistry()
    reg.symbol_counts = dict(doc.get("symbol_counts", {}))
    reg.subword_counts = dict(doc.get("subword_counts", {}))
    reg.word_counts = dict(doc.get("word_counts", {}))
    for w, ctxs in doc.get("word_contexts", {}).items():
        reg.word_contexts[w] = [frozenset(c) for c in ctxs]
    for key, d in doc.get("promoted", {}).items():
        tier_s, text = key.split(":", 1)
        reg.promoted[(LatticeTier(int(tier_s)), text)] = _token_from_dict(d)
    for w, d in doc.get("intersections", {}).items():
        reg.intersections[w] = _token_from_dict(d)
    for key, d in doc.get("correlations", {}).items():
        a, b = key.split("|", 1)
        reg.correlations[(a, b)] = _link_from_dict(d)
    reg._next_promotion_idx = int(doc.get("next_promotion_idx", 0))
    return reg


def graph_to_dict(g: CooccurrenceGraph) -> dict[str, Any]:
    pairs = {f"{a}|{b}": c for (a, b), c in g.pair_count.items()}
    return {
        "pair_count": pairs,
        "word_count": dict(g.word_count),
        "window_count": g.window_count,
    }


def graph_from_dict(doc: dict[str, Any]) -> CooccurrenceGraph:
    g = CooccurrenceGraph()
    for key, c in doc.get("pair_count", {}).items():
        a, b = key.split("|", 1)
        g.pair_count[(a, b)] = int(c)
    g.word_count = dict(doc.get("word_count", {}))
    g.window_count = int(doc.get("window_count", 0))
    return g


def reader_to_dict(reader: NaturalReader) -> dict[str, Any]:
    cats = {}
    for cid, cat in reader.cross.categories.items():
        cats[cid] = {
            "name": cat.name,
            "dim7": cat.dim7,
            "dim8": cat.dim8,
            "dim9": cat.dim9,
            "prime_weights": {str(k): v for k, v in cat.prime_weights.items()},
            "word_weights": dict(cat.word_weights),
        }
    return {
        "version": 1,
        "registry": registry_to_dict(reader.registry),
        "graph": graph_to_dict(reader.graph),
        "word_to_cluster": dict(reader.word_to_cluster),
        "cluster_hubs": dict(reader.cluster_hubs),
        "documents_read": reader.documents_read,
        "categories": cats,
    }


def reader_from_dict(doc: dict[str, Any], *, rebuild_every: int = 3) -> NaturalReader:
    reader = NaturalReader(rebuild_every=rebuild_every)
    reader.registry = registry_from_dict(doc["registry"])
    reader.cross.registry = reader.registry
    reader.graph = graph_from_dict(doc.get("graph", {}))
    reader.word_to_cluster = dict(doc.get("word_to_cluster", {}))
    reader.cluster_hubs = dict(doc.get("cluster_hubs", {}))
    reader.documents_read = int(doc.get("documents_read", 0))
    for cid, cdoc in doc.get("categories", {}).items():
        cat = reader.cross.ensure_category(cdoc["name"])
        cat.dim7 = float(cdoc.get("dim7", 0))
        cat.dim8 = float(cdoc.get("dim8", 0))
        cat.dim9 = float(cdoc.get("dim9", 0))
        cat.prime_weights = {int(k): float(v) for k, v in cdoc.get("prime_weights", {}).items()}
        cat.word_weights = dict(cdoc.get("word_weights", {}))
        reader.cross.categories[cid] = cat
    return reader


def save_reader(reader: NaturalReader, path: str | Path) -> Path:
    p = Path(path)
    p.write_text(json.dumps(reader_to_dict(reader), indent=2), encoding="utf-8")
    return p


def load_reader(path: str | Path, *, rebuild_every: int = 3) -> NaturalReader:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return reader_from_dict(doc, rebuild_every=rebuild_every)


# ---------------------------------------------------------------------------
# Brain save / load  — anchor weights + calibrated signal λ values
# ---------------------------------------------------------------------------

BRAIN_VERSION = 2


def save_brain(anchor_idx, lambda_coord: float, lambda_neighbor: float, path: str | Path) -> None:
    """
    Persist trained anchor weights and calibrated signal λ values.

    Anchor ``correct_count`` / ``wrong_count`` are saved so the next run
    starts with accumulated knowledge and adds to it (compound learning).

    Parameters
    ----------
    anchor_idx       : HeavyAnchorIndex from aethos_discriminative
    lambda_coord     : calibrated LAMBDA_COORD (from calibrate_signal_weights)
    lambda_neighbor  : calibrated LAMBDA_NEIGHBOR
    path             : file path for the JSON brain file
    """
    anchors: dict[str, dict] = {}
    for comp, anchor in anchor_idx.anchors.items():
        if anchor.correct_count > 0 or anchor.wrong_count > 0:
            anchors[str(comp)] = {
                "prime_a": anchor.prime_a,
                "prime_b": anchor.prime_b,
                "word_a": anchor.word_a,
                "word_b": anchor.word_b,
                "correct_count": anchor.correct_count,
                "wrong_count": anchor.wrong_count,
                "doc_ids": sorted(anchor.doc_ids),
                "n_docs": len(anchor.doc_ids),
            }

    doc = {
        "version": BRAIN_VERSION,
        "lambda_coord": lambda_coord,
        "lambda_neighbor": lambda_neighbor,
        "anchors": anchors,
    }
    Path(path).write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")


def load_brain(path: str | Path, anchor_idx, verbose: bool = True) -> tuple[float, float]:
    """
    Load saved anchor weights into an existing HeavyAnchorIndex.

    Adds saved ``correct_count`` / ``wrong_count`` to current counts so
    training in this run compounds the knowledge from previous runs.

    Returns (lambda_coord, lambda_neighbor) — the last calibrated values.

    Parameters
    ----------
    path       : brain file written by save_brain()
    anchor_idx : HeavyAnchorIndex to update in-place
    verbose    : print a summary line
    """
    brain_path = Path(path)
    if not brain_path.exists():
        return 0.5, 0.1   # defaults if no brain file yet

    doc = json.loads(brain_path.read_text(encoding="utf-8"))
    if doc.get("version", 1) < BRAIN_VERSION:
        if verbose:
            print(f"  brain file version {doc.get('version')} < {BRAIN_VERSION}, skipping load", flush=True)
        return 0.5, 0.1

    loaded = 0
    for comp_str, a in doc.get("anchors", {}).items():
        comp = int(comp_str)
        if comp in anchor_idx.anchors:
            anchor = anchor_idx.anchors[comp]
            anchor.correct_count += int(a.get("correct_count", 0))
            anchor.wrong_count += int(a.get("wrong_count", 0))
            loaded += 1

    lc = float(doc.get("lambda_coord", 0.5))
    ln = float(doc.get("lambda_neighbor", 0.1))

    if verbose:
        print(
            f"  brain loaded: {loaded} anchors updated, "
            f"λ_coord={lc}, λ_neighbor={ln} (from {brain_path.name})",
            flush=True,
        )
    return lc, ln


def brain_path_for_dataset(dataset: str, mode: str = "scale") -> Path:
    """Return the canonical brain file path for a dataset+mode combination."""
    return Path(__file__).resolve().parent / "brains" / f"{dataset}_{mode}.brain.json"


def demo() -> None:
    from aethos_pipeline import smoke_corpus

    path = Path(__file__).resolve().parent / "fixtures" / "reader_state.json"
    r1 = NaturalReader(rebuild_every=2)
    r1.read(*smoke_corpus())
    save_reader(r1, path)
    r2 = load_reader(path, rebuild_every=2)
    assert r1.registry.word_counts == r2.registry.word_counts
    assert r1.word_to_cluster == r2.word_to_cluster
    print(f"Saved and loaded {path}")
    print(f"  words: {len(r2.registry.word_counts)}, clusters: {len(r2.cross.categories)}")


if __name__ == "__main__":
    demo()
