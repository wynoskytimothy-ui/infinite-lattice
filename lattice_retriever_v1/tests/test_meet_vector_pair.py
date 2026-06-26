"""Meet-vector pair routing — global_3way docs union into pair_meet expand."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever


def _meet_corpus() -> dict[str, str]:
    """Two docs share rare triple meet geometry; query hits pair via meet keys."""
    return {
        "gold": "xyzzyalpha mitochondria atp energy cells rarebridge marker",
        "peer": "xyzzyalpha mitochondria rarebridge collateral witness",
        "noise": "common filler mitochondria energy " * 8,
    }


def _hybrid(*, meet_vector: bool, corpus_sidecar: bool):
    cfg = HybridConfig(
        lexical_mode="append_index",
        enable_walk_pool_expand=False,
        enable_append_pool_union=False,
        enable_stage08_rerank=False,
        enable_meet_vector_pair=meet_vector,
        enable_corpus_lattice=corpus_sidecar,
        pair_idf_gate=0.0,
    )
    return build_hybrid_retriever(
        _meet_corpus(), config=cfg, corpus_name="meet_vector_probe"
    )


def test_meet_vector_pair_adds_global_meet_docs():
    r = _hybrid(meet_vector=True, corpus_sidecar=False)
    query = "xyzzyalpha mitochondria"
    trace = r.retrieve_with_trace(query, limit=5)
    steps = [s for s in trace.filter_steps if s.get("step") == "pair_meet_expand"]
    assert steps, "expected pair_meet_expand step"
    assert "gold" in trace.pool_docs or "peer" in trace.pool_docs


def test_meet_vector_off_skips_meet_key_docs():
    r = _hybrid(meet_vector=False, corpus_sidecar=False)
    query = "xyzzyalpha mitochondria"
    expanded, steps = r._pair_meet_expand(query, set())
    assert steps == () or steps[0].get("meet_key_docs", 0) == 0
