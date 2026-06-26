"""Two-lattice zero-shot retrieval — synthetic quantum fixture."""

from lattice_retriever_v1.doc_lattice_codec import build_two_lattice_retriever


QUANTUM_CORPUS = {
    "d1": "quantum physics explores entanglement deeply",
    "d2": "quantum computing uses superposition states",
    "d3": "classical mechanics describes motion clearly",
}


def test_quantum_query_ranks_shared_docs_above_other():
    r = build_two_lattice_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum", limit=3)
    assert len(hits) >= 2
    top_ids = {h.doc_id for h in hits[:2]}
    assert top_ids == {"d1", "d2"}
    assert hits[0].doc_id in ("d1", "d2")
    assert hits[-1].doc_id == "d3" or hits[-1].score <= hits[0].score
    assert hits[0].score > hits[-1].score


def test_two_term_query_ranks_doc_sharing_both_higher():
    r = build_two_lattice_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum entanglement", limit=3)
    assert len(hits) >= 2
    assert hits[0].doc_id == "d1"
    assert hits[0].score > hits[1].score
    d1_trace = hits[0].trace
    assert "quantum" in d1_trace.shared_terms
    assert "entanglement" in d1_trace.shared_terms


def test_shared_term_index_lights_both_quantum_docs():
    r = build_two_lattice_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum")
    assert trace.pool_size == 2
    assert set(trace.lit_docs) == {"d1", "d2"}
    assert any(s.get("step") == "shared_term_light" for s in trace.filter_steps)


def test_glass_box_trace_has_cage_dots():
    r = build_two_lattice_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum computing")
    ex = trace.explain()
    assert ex["hits"]
    hit = ex["hits"][0]
    assert hit["trace"]["term_witnesses"]
    assert hit["trace"]["shared_terms"]


def test_doc_lattice_placement_unique_primes():
    r = build_two_lattice_retriever(QUANTUM_CORPUS)
    primes = {p.doc_prime for p in r.placements.values()}
    assert len(primes) == 3
    for p in r.placements.values():
        assert len(p.order_stream) == len(p.words)
