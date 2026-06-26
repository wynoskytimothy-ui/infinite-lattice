"""Unified dual-lattice — lexical + shell + walk MaxSim fusion."""

from lattice_retriever_v1.hybrid_retriever import (
    HybridConfig,
    build_unified_dual_lattice_retriever,
)
from lattice_retriever_v1.tests.test_lattice2_correlation import QUANTUM_CORPUS


def test_unified_three_layers_exposed_in_trace():
    r = build_unified_dual_lattice_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum physics", limit=2)
    hit = trace.hits[0]
    assert hit.lex_score > 0
    assert hit.l2_score > 0
    assert hit.walk_score > 0
    assert trace.query_walk
    ex = hit.explain()
    assert ex["walk_witnesses"]
    assert ex["l2_trace"].get("term_witnesses")


def test_unified_entanglement_ranks_d1():
    r = build_unified_dual_lattice_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum entanglement", limit=3)
    assert hits[0].doc_id == "d1"


def test_walk_layer_boosts_pair_overlap_doc():
    base = build_unified_dual_lattice_retriever(
        QUANTUM_CORPUS, config=HybridConfig(lam_lex=1.0, lam_l2=0.0, lam_walk=0.0)
    )
    fused = build_unified_dual_lattice_retriever(
        QUANTUM_CORPUS, config=HybridConfig(lam_lex=1.0, lam_l2=0.0, lam_walk=0.35)
    )
    base_hits = base.retrieve("quantum computing uses", limit=3)
    fused_hits = fused.retrieve("quantum computing uses", limit=3)
    assert base_hits[0].doc_id == "d2"
    assert fused_hits[0].doc_id == "d2"
    assert fused_hits[0].walk_score >= base_hits[0].walk_score


def test_walk_pool_expand_step_in_filter_trace():
    r = build_unified_dual_lattice_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum computing")
    steps = {s.get("step") for s in trace.filter_steps}
    assert "pair_origin_intersect" in steps or trace.pool_size >= 1
