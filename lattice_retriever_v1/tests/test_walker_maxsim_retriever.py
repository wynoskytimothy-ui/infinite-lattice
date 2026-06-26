"""Walker MaxSim — ColBERT-shaped late interaction on pair walks."""

from lattice_retriever_v1.walker_maxsim_retriever import (
    build_walker_maxsim_retriever,
    geometric_dot_witness,
    word_pair_walk,
)
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex

QUANTUM_CORPUS = {
    "d1": "quantum physics explores entanglement deeply",
    "d2": "quantum computing uses superposition states",
    "d3": "classical mechanics describes motion clearly",
}


def test_quantum_query_ranks_shared_docs():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum", limit=3)
    assert len(hits) >= 2
    top_ids = {h.doc_id for h in hits[:2]}
    assert top_ids == {"d1", "d2"}
    assert hits[0].score > hits[-1].score


def test_two_term_query_ranks_entanglement_doc_first():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum entanglement", limit=3)
    assert hits[0].doc_id == "d1"
    assert hits[0].score > hits[1].score


def test_pair_origin_routes_exact_bigram():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum computing")
    assert trace.pool_size == 1
    assert trace.hits[0].doc_id == "d2"
    assert any(s.get("step") == "pair_origin_intersect" for s in trace.filter_steps)


def test_single_term_uses_shared_term_widen():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum")
    assert trace.pool_size == 2
    assert set(trace.lit_docs) == {"d1", "d2"}


def test_glass_box_walk_witness_in_trace():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    trace = r.retrieve_with_trace("quantum physics")
    assert trace.query_walk
    hit = trace.hits[0]
    ex = hit.explain()
    assert ex["trace"]["walk_witnesses"]
    assert ex["trace"]["shell_trace"]["term_witnesses"]


def test_geometric_witness_exact_n_beats_near_n():
    reg = promote_from_stream(["quantum physics quantum physics"])
    sem = SemanticLightIndex(registry=reg)
    walk = word_pair_walk("quantum physics quantum physics", sem)
    q = walk[0]
    d_exact = walk[0]
    d_near = walk[1]
    assert geometric_dot_witness(q, d_exact) > geometric_dot_witness(q, d_near)


def test_walk_maxsim_beats_shell_only_on_pair_overlap():
    r = build_walker_maxsim_retriever(QUANTUM_CORPUS)
    hits = r.retrieve("quantum computing uses", limit=3)
    assert hits[0].doc_id == "d2"
    walk_part = hits[0].trace.walk_score
    assert walk_part > 0
