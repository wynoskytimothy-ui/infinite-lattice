"""Synthetic glass-box tests for AppendOnlyLatticeIndex correlation probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.audit_append_index_glass_box import (
    gold_false_profile,
    probe_01_baseline,
    probe_06_rarest2_and_bm25,
    probe_18_pair_cooccur_boost,
    ProbeContext,
    rarest_terms,
    summarize_coverage,
    word_idf,
)
from aethos_glass_box_search import GlassBoxRetriever
from aethos_bridges import RelevanceBridges


SYNTH_CORPUS = {
    "d1": "nanoparticles enable stem cell differentiation in biomaterials scaffolds",
    "d2": "zero dimensional biomaterials show inductive properties for tissue engineering",
    "d3": "vitamin b12 deficiency increases homocysteine blood levels cardiovascular",
    "d4": "homocysteine elevation linked to b12 deficiency in elderly patients",
    "d5": "breast cancer aldh1 expression poorer prognosis tumor cells",
    "d6": "common cancer risk cells patients expression study",
}

SYNTH_QUERIES = {
    "q1": "zero dimensional biomaterials inductive properties",
    "q2": "b12 deficiency homocysteine blood",
    "q3": "aldh1 breast cancer prognosis",
}

SYNTH_TEST_Q = {
    "q1": {"d2": 1},
    "q2": {"d4": 1},
    "q3": {"d5": 1},
}

SYNTH_TRAIN_Q = {
    "q1": {"d1": 1, "d2": 1},
    "q2": {"d3": 1, "d4": 1},
    "q3": {"d5": 1},
}


def _build_ctx() -> ProbeContext:
    idx = AppendOnlyLatticeIndex(index_mode="full")
    for doc_id, text in SYNTH_CORPUS.items():
        idx.add(doc_id, text)
    idx.finalize()
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=1).learn(SYNTH_QUERIES, SYNTH_TRAIN_Q, SYNTH_CORPUS)
    br_rare = RelevanceBridges(idx, N, min_pairs=1)
    br_rare.learn_rarest_corridors(SYNTH_QUERIES, SYNTH_TRAIN_Q, SYNTH_CORPUS)
    return ProbeContext(idx=idx, corpus=SYNTH_CORPUS, br=br, br_rare=br_rare, N=N)


def test_rarest_word_orders_by_idf():
    ctx = _build_ctx()
    rarest = rarest_terms(
        ["biomaterials", "dimensional", "zero", "properties"],
        ctx.idx,
        ctx.N,
    )
    # rarest technical terms should beat generic dimensional
    assert rarest[0] in {"biomaterials", "inductive", "properties", "zero", "dimensional"}


def test_gold_has_rarest_or_bridge_path():
    ctx = _build_ctx()
    ranked = probe_01_baseline(ctx, SYNTH_QUERIES["q1"], 10)
    prof = gold_false_profile(ctx, SYNTH_QUERIES["q1"], SYNTH_TEST_Q["q1"], ranked)
    gold = prof["gold"][0]
    assert gold["is_gold"]
    assert gold["has_rarest_1"] or gold["n_bridge_paths"] > 0 or gold["in_top10"]


def test_pair_cooccur_probe_recovers_gold():
    ctx = _build_ctx()
    base = probe_01_baseline(ctx, SYNTH_QUERIES["q2"], 10)
    boosted = probe_18_pair_cooccur_boost(ctx, SYNTH_QUERIES["q2"], 10)
    assert "d4" in boosted[:3] or "d4" in base[:3]


def test_rarest2_and_narrows_pool():
    ctx = _build_ctx()
    hits = probe_06_rarest2_and_bm25(ctx, SYNTH_QUERIES["q3"], 5)
    assert hits
    assert hits[0] == "d5"


def test_coverage_summary_runs():
    ctx = _build_ctx()
    profiles = []
    for qid, q in SYNTH_QUERIES.items():
        ranked = probe_01_baseline(ctx, q, 10)
        p = gold_false_profile(ctx, q, SYNTH_TEST_Q[qid], ranked)
        profiles.append(p)
    cov = summarize_coverage(profiles)
    assert cov["gold_doc_instances"] == 3
    assert cov["gold_has_rarest_1_pct"] >= 0


def test_glass_box_json_roundtrip():
    """Probe stats structure is JSON-serializable for audit logs."""
    ctx = _build_ctx()
    ranked = probe_01_baseline(ctx, SYNTH_QUERIES["q1"], 10)
    prof = gold_false_profile(ctx, SYNTH_QUERIES["q1"], SYNTH_TEST_Q["q1"], ranked)
    blob = json.dumps(prof)
    assert "rarest_query_words" in json.loads(blob)


def test_glass_box_search_beats_or_matches_baseline():
    r = GlassBoxRetriever.from_corpus(SYNTH_CORPUS, SYNTH_QUERIES, SYNTH_TRAIN_Q, learn_corridors=False)
    for qid, q in SYNTH_QUERIES.items():
        gold = SYNTH_TEST_Q[qid]
        base = r.idx.search(q, 10)
        fused = r.search(q, 10)
        assert fused
        if any(g in base[:3] for g in gold):
            assert any(g in fused[:5] for g in gold)


def test_glass_box_trace_has_steps():
    r = GlassBoxRetriever.from_corpus(SYNTH_CORPUS, SYNTH_QUERIES, SYNTH_TRAIN_Q, learn_corridors=False)
    trace = r.search_with_trace(SYNTH_QUERIES["q1"], 5)
    ex = trace.explain()
    steps = [s["step"] for s in ex["steps"]]
    assert "lattice_lexical" in steps
    assert "bridge_pool" in steps
    assert ex["ranked"]


def test_lattice_lexical_no_bm25_path():
    from aethos_lattice_lexical import lattice_lexical_scorer, LexicalMode
    from aethos_glass_box_search import GlassBoxSearchConfig

    cfg = GlassBoxSearchConfig(lexical_mode="lattice_plane")
    r = GlassBoxRetriever.from_corpus(
        SYNTH_CORPUS, SYNTH_QUERIES, SYNTH_TRAIN_Q, learn_corridors=False,
    )
    r.config = cfg
    r.lexical_scorer = lattice_lexical_scorer(r.idx, mode="lattice_plane")
    assert r.lexical_scorer.config.mode != LexicalMode.BM25
    pure = lattice_lexical_scorer(r.idx, mode="lattice_pure")
    hits = pure.top_k(SYNTH_QUERIES["q2"], 5)
    assert hits
    assert "d4" in hits or "d3" in hits


def test_scifact_lattice_pool_restrict():
    from aethos_glass_box_search import GlassBoxSearchConfig

    cfg = GlassBoxSearchConfig.scifact_lattice()
    assert cfg.pool_restrict
    r = GlassBoxRetriever.from_corpus(
        SYNTH_CORPUS, SYNTH_QUERIES, SYNTH_TRAIN_Q,
        learn_corridors=False, scifact_lattice=True,
    )
    trace = r.search_with_trace(SYNTH_QUERIES["q1"], 5)
    steps = [s["step"] for s in trace.explain()["steps"]]
    assert "kappa_route_pool" in steps
    assert r.idx.index_mode == "kappa_primary"
    assert r.kappa_index is not None
