"""Append-index BM25 pool scoring as hybrid L0 lexical floor."""

from dataclasses import replace

import pytest

from aethos_lattice_lexical import lattice_lexical_scorer
from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.tests.test_lattice2_correlation import QUANTUM_CORPUS


def test_dense_pool_scores_match_dict_bm25_pool():
    """Dense cached path must match dict-loop score_bm25_pool (regression)."""
    r = build_hybrid_retriever(
        QUANTUM_CORPUS, config=HybridConfig(lexical_mode="append_index")
    )
    q = "quantum entanglement"
    pool = frozenset(QUANTUM_CORPUS)
    dict_scores = r.lexical.score_bm25_pool(q, pool)
    r.lexical.cache_dense_scores(q)
    dense_scores = r.lexical.score_bm25_pool(q, pool)
    assert set(dict_scores) == set(dense_scores)
    for d, s in dict_scores.items():
        assert dense_scores[d] == pytest.approx(s, rel=1e-3, abs=1e-4)


def test_append_index_scores_pool_via_bm25():
    r = build_hybrid_retriever(
        QUANTUM_CORPUS, config=HybridConfig(lexical_mode="append_index")
    )
    pool = frozenset(QUANTUM_CORPUS)
    append_scores = r.lexical.score_pool("quantum entanglement", pool)
    plane = build_hybrid_retriever(
        QUANTUM_CORPUS, config=HybridConfig(lexical_mode="lattice_plane")
    )
    plane_scores = plane.lexical.score_pool("quantum entanglement", pool)
    assert append_scores["d1"] > 0
    assert append_scores["d1"] > plane_scores.get("d1", 0.0)
    assert append_scores["d1"] >= append_scores.get("d2", 0.0)


def test_hybrid_append_index_ranks_entanglement_doc():
    r = build_hybrid_retriever(
        QUANTUM_CORPUS,
        config=HybridConfig(
            lexical_mode="append_index",
            lam_lex=1.0,
            lam_l2=0.0,
            lam_walk=0.0,
            enable_stage08_rerank=False,
        ),
    )
    hits = r.retrieve("quantum entanglement", limit=3)
    assert hits[0].doc_id == "d1"
    assert hits[0].lex_score > 0


def test_append_index_rerank_on_preserves_lex_order():
    """Default Stage08 rerank must not scramble append_index BM25 ordering."""
    base = HybridConfig(
        lexical_mode="append_index",
        lam_lex=1.0,
        lam_l2=0.0,
        lam_walk=0.0,
    )
    r_off = build_hybrid_retriever(
        QUANTUM_CORPUS, config=replace(base, enable_stage08_rerank=False)
    )
    r_on = build_hybrid_retriever(
        QUANTUM_CORPUS, config=replace(base, enable_stage08_rerank=True)
    )
    q = "quantum entanglement"
    hits_off = r_off.retrieve(q, limit=3)
    hits_on = r_on.retrieve(q, limit=3)
    assert hits_on[0].doc_id == hits_off[0].doc_id == "d1"
    assert [h.doc_id for h in hits_on] == [h.doc_id for h in hits_off]
