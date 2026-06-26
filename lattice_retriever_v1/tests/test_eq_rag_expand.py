"""Soft EQ-RAG complement term expansion — Phase A pool routing only."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever


def _eq_rag_corpus() -> dict[str, str]:
    """Gold shares rare terms; query omits complement rarebridge."""
    return {
        "gold": "xyzzyalpha mitochondria atp energy cells rarebridge",
        "noise_a": "xyzzyalpha common filler text " * 6,
        "noise_b": "mitochondria common filler text " * 6,
        "noise_c": "unrelated gamma delta epsilon zeta " * 4,
    }


def _eq_rag_hybrid(*, enable_expand: bool):
    corpus = _eq_rag_corpus()
    cfg = HybridConfig(
        lexical_mode="append_index",
        enable_pair_meet=False,
        enable_walk_pool_expand=False,
        enable_append_pool_union=False,
        enable_stage08_rerank=False,
        enable_corpus_lattice=True,
        enable_eq_rag_expand=enable_expand,
        eq_rag_idf_gate=0.0,
        pair_idf_gate=0.0,
        eq_rag_expand_cap=4,
    )
    return build_hybrid_retriever(corpus, config=cfg, corpus_name="eq_rag_probe")


def test_eq_rag_expand_recovers_complement_terms():
    r = _eq_rag_hybrid(enable_expand=True)
    query = "xyzzyalpha mitochondria"
    expanded, recovered, steps = r._eq_rag_soft_expand_terms(r._query_terms(query))
    assert recovered, "expected complement recovery from rare pair"
    assert "rarebridge" in recovered
    assert len(expanded) > len(r._query_terms(query))
    assert steps and steps[0]["step"] == "eq_rag_expanded_terms"


def test_eq_rag_expand_trace_and_pool_routing():
    r_off = _eq_rag_hybrid(enable_expand=False)
    r_on = _eq_rag_hybrid(enable_expand=True)
    query = "xyzzyalpha mitochondria"
    trace_off = r_off.retrieve_with_trace(query, limit=5)
    trace_on = r_on.retrieve_with_trace(query, limit=5)

    assert not any(
        s.get("step") == "eq_rag_expanded_terms" for s in trace_off.filter_steps
    )
    eq_steps = [s for s in trace_on.filter_steps if s.get("step") == "eq_rag_expanded_terms"]
    assert eq_steps
    assert "rarebridge" in eq_steps[0].get("recovered_terms", [])
    assert "gold" in trace_on.pool_docs
