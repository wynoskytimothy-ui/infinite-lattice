"""Smoke tests for aethos_clean package."""

from __future__ import annotations

from pathlib import Path

import pytest

from aethos_clean import CleanPipeline, evaluate_gates, get_corpus_gates, get_preset
from aethos_clean.gates import load_gates


def _beir_available() -> bool:
    try:
        from beir_data_root import resolve_beir_root

        root = Path(resolve_beir_root())
        return (root / "scifact" / "corpus.jsonl").is_file()
    except Exception:
        return False


def test_gates_json_loads():
    data = load_gates()
    assert "presets" in data
    assert "lean" in data["presets"]
    assert "scifact" in data["corpora"]


def test_lean_preset_defaults():
    p = get_preset("lean")
    assert p.pool_cap == 350
    assert p.kappa_scoring is False
    assert p.lambda_kappa == 0.0
    assert p.mode == "quality"
    assert p.train_mode == "full"


def test_lean_composite_preset():
    p = get_preset("lean_composite")
    assert p.train_mode == "composite_only"
    assert p.max_composite_anchors == 2000
    assert p.clear_bad_correlation is True


def test_scifact_gate_thresholds():
    g = get_corpus_gates("scifact")
    assert g.ndcg10_target == 0.78
    assert g.bm25_ref == 0.643


def test_evaluate_gates_pass():
    report = evaluate_gates(
        dataset="scifact",
        preset="lean",
        ndcg10=0.70,
        recall10=0.75,
        recall100=0.88,
        p50_query_ms=12.0,
        p99_query_ms=45.0,
        hot_bytes_per_doc=180.0,
    )
    assert report.passed is True
    assert report.checks["ndcg10_min"] is True


def test_evaluate_gates_fail_latency():
    report = evaluate_gates(
        dataset="scifact",
        preset="lean",
        ndcg10=0.70,
        recall10=0.75,
        recall100=0.88,
        p50_query_ms=120.0,
        p99_query_ms=45.0,
        hot_bytes_per_doc=180.0,
    )
    assert report.passed is False
    assert report.checks["p50_query_ms_max"] is False


@pytest.mark.skipif(
    not Path(__file__).resolve().parent.parent.joinpath("beir_datasets", "scifact").exists()
    and not _beir_available(),
    reason="BEIR scifact not on disk",
)
def test_clean_pipeline_micro_eval():
    pipe = CleanPipeline.from_beir(
        "scifact",
        preset="lean",
        max_docs=40,
        max_queries=5,
    )
    pipe.index(rebuild=True, skip_training=True, save=False)
    assert pipe.indexed
    result = pipe.query("vitamin D immune system", top_k=5)
    assert len(result.ranked_ids) <= 5
    assert result.latency_ms >= 0
    ev = pipe.evaluate(check_gates=False)
    assert ev.n_queries == 5
    assert 0.0 <= ev.ndcg10 <= 1.0
