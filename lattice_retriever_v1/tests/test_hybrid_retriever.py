"""Hybrid BM25-class + ColBERT-style MaxSim fusion."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.tests.test_lattice2_correlation import QUANTUM_CORPUS


def test_hybrid_quantum_ranks_shared_docs():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum", limit=3)
    assert len(hits) >= 2
    top_ids = {h.doc_id for h in hits[:2]}
    assert top_ids == {"d1", "d2"}
    assert hits[0].lex_score > 0
    assert hits[0].l2_score > 0


def test_hybrid_fusion_exposes_both_layers():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum entanglement", limit=2)
    assert trace.pool_size >= 1
    assert trace.route_mode == "primary"
    hit = trace.hits[0]
    assert hit.doc_id == "d1"
    assert hit.lex_score >= 0
    assert hit.l2_score >= 0
    assert hit.l2_trace.get("term_witnesses")


def test_hybrid_entanglement_prefers_d1():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum entanglement", limit=3)
    assert hits[0].doc_id == "d1"


def test_postings_idf_positive_for_rare_term():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    assert r.router.semantic.idf("entanglement") > 0


def test_hybrid_uses_stage08_route_mode():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum")
    assert trace.route_mode in ("primary", "widen_rarest", "lift_pin_widen", "fta_letter_fallback", "empty")
    assert trace.pool_size >= 2
