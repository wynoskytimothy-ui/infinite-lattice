"""Load and evaluate release gates for the clean pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_GATES_PATH = Path(__file__).resolve().parent / "gates.json"


@dataclass
class CorpusGates:
    name: str
    ndcg10_min: float
    ndcg10_target: float
    recall10_min: float
    recall100_min: float
    p50_query_ms_max: float
    p99_query_ms_max: float
    hot_bytes_per_doc_max: float
    bm25_ref: float | None = None


@dataclass
class PresetGates:
    name: str
    description: str
    pool_cap: int
    kappa_scoring: bool
    lambda_kappa: float
    mode: str
    train_mode: str = "full"
    max_composite_anchors: int = 2000
    max_composite_meta: int = 500
    max_composite_negatives: int = 500
    clear_bad_correlation: bool = False


@dataclass
class GateReport:
    dataset: str
    preset: str
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"{self.dataset} [{self.preset}]  {status}"]
        for key, ok in self.checks.items():
            val = self.values.get(key)
            mark = "ok" if ok else "FAIL"
            if val is not None:
                lines.append(f"  {key}: {val:.4f}  [{mark}]")
            else:
                lines.append(f"  {key}: —  [{mark}]")
        for msg in self.failures:
            lines.append(f"  ! {msg}")
        return "\n".join(lines)


def load_gates(path: Path | None = None) -> dict[str, Any]:
    p = path or _GATES_PATH
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get_preset(name: str = "lean", path: Path | None = None) -> PresetGates:
    data = load_gates(path)
    presets = data.get("presets") or {}
    raw = presets.get(name)
    if not raw:
        raise KeyError(f"unknown preset {name!r}; have {list(presets)}")
    return PresetGates(
        name=name,
        description=str(raw.get("description", "")),
        pool_cap=int(raw.get("pool_cap", 350)),
        kappa_scoring=bool(raw.get("kappa_scoring", False)),
        lambda_kappa=float(raw.get("lambda_kappa", 0.0)),
        mode=str(raw.get("mode", "quality")),
        train_mode=str(raw.get("train_mode", "full")),
        max_composite_anchors=int(raw.get("max_composite_anchors", 2000)),
        max_composite_meta=int(raw.get("max_composite_meta", 500)),
        max_composite_negatives=int(raw.get("max_composite_negatives", 500)),
        clear_bad_correlation=bool(raw.get("clear_bad_correlation", False)),
    )


def get_corpus_gates(dataset: str, path: Path | None = None) -> CorpusGates:
    data = load_gates(path)
    corpora = data.get("corpora") or {}
    raw = corpora.get(dataset)
    if not raw:
        raise KeyError(f"no gates for dataset {dataset!r}; have {list(corpora)}")
    return CorpusGates(
        name=dataset,
        ndcg10_min=float(raw["ndcg10_min"]),
        ndcg10_target=float(raw.get("ndcg10_target", raw["ndcg10_min"])),
        recall10_min=float(raw.get("recall10_min", 0.0)),
        recall100_min=float(raw.get("recall100_min", 0.0)),
        p50_query_ms_max=float(raw["p50_query_ms_max"]),
        p99_query_ms_max=float(raw.get("p99_query_ms_max", 9999.0)),
        hot_bytes_per_doc_max=float(raw["hot_bytes_per_doc_max"]),
        bm25_ref=float(raw["bm25_ref"]) if raw.get("bm25_ref") is not None else None,
    )


def evaluate_gates(
    *,
    dataset: str,
    preset: str = "lean",
    ndcg10: float,
    recall10: float,
    recall100: float,
    p50_query_ms: float,
    p99_query_ms: float,
    hot_bytes_per_doc: float,
    path: Path | None = None,
) -> GateReport:
    cg = get_corpus_gates(dataset, path)
    checks = {
        "ndcg10_min": ndcg10 >= cg.ndcg10_min,
        "recall10_min": recall10 >= cg.recall10_min,
        "recall100_min": recall100 >= cg.recall100_min,
        "p50_query_ms_max": p50_query_ms <= cg.p50_query_ms_max,
        "p99_query_ms_max": p99_query_ms <= cg.p99_query_ms_max,
        "hot_bytes_per_doc_max": hot_bytes_per_doc <= cg.hot_bytes_per_doc_max,
    }
    values = {
        "ndcg10_min": ndcg10,
        "recall10_min": recall10,
        "recall100_min": recall100,
        "p50_query_ms_max": p50_query_ms,
        "p99_query_ms_max": p99_query_ms,
        "hot_bytes_per_doc_max": hot_bytes_per_doc,
    }
    failures: list[str] = []
    if ndcg10 < cg.ndcg10_target:
        failures.append(
            f"below target NDCG@10 {cg.ndcg10_target:.3f} (have {ndcg10:.4f})"
        )
    if cg.bm25_ref is not None and ndcg10 < cg.bm25_ref:
        failures.append(f"below BM25 ref {cg.bm25_ref:.3f}")
    passed = all(checks.values())
    return GateReport(
        dataset=dataset,
        preset=preset,
        passed=passed,
        checks=checks,
        values=values,
        failures=failures,
    )
