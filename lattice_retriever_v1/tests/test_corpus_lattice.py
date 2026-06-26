"""Corpus lattice skeleton — global 3-way meets across shared rare terms."""

from lattice_retriever_v1.corpus_lattice import CorpusLatticeBuilder
from lattice_retriever_v1.corpus_prime import corpus_scope
from lattice_retriever_v1.doc_lattice_codec import DocPrimePool
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex


def test_two_docs_share_rare_term_global_3way():
    corpus = {
        "d1": "mitochondria produces atp energy in cells",
        "d2": "atp powers muscle contraction during exercise",
        "d3": "plants convert sunlight into glucose sugar",
    }
    reg = promote_from_stream(list(corpus.values()))
    scope = corpus_scope("bio_probe", reg)
    sem = SemanticLightIndex(registry=reg)
    pool = DocPrimePool()
    builder = CorpusLatticeBuilder(
        scope.corpus_prime,
        reg,
        sem,
        pool,
        k_rare=8,
        max_df_frac=0.5,
    )
    for doc_id, text in corpus.items():
        builder.observe_doc(doc_id, text)
        sem.observe_doc(text, mode="positional")
    lattice = builder.finalize()

    assert lattice.corpus_prime == scope.corpus_prime
    assert len(lattice.doc_registry) == 3
    assert "d1" in lattice.doc_registry
    assert "atp" in lattice.doc_registry["d1"].rare_terms
    assert "atp" in lattice.doc_registry["d2"].rare_terms
    assert len(lattice.global_3way) >= 1
    shared = [
        rec
        for rec in lattice.global_3way.values()
        if "atp" in rec.correlated_terms and {"d1", "d2"} <= rec.doc_ids
    ]
    assert shared, "expected global_3way entry linking d1 and d2 via atp"


def test_route_pool_finds_shared_rare_docs():
    corpus = {
        "d1": "mitochondria produces atp energy",
        "d2": "atp powers muscle cells",
        "d3": "sunlight feeds plant growth",
    }
    reg = promote_from_stream(list(corpus.values()))
    scope = corpus_scope("route_probe", reg)
    sem = SemanticLightIndex(registry=reg)
    pool = DocPrimePool()
    builder = CorpusLatticeBuilder(scope.corpus_prime, reg, sem, pool, max_df_frac=0.5)
    for doc_id, text in corpus.items():
        builder.observe_doc(doc_id, text)
        sem.observe_doc(text, mode="positional")
    lattice = builder.finalize()
    pool_docs, _ = lattice.route_pool(["atp"], semantic=sem)
    assert "d1" in pool_docs
    assert "d2" in pool_docs


def test_route_pool_no_mod_heuristic_false_positive():
    """route_pool must not widen via meet_key % term_prime (removed heuristic)."""
    corpus = {
        "d1": "xyzzy alpha rareterm one",
        "d2": "xyzzy beta rareterm two",
        "d3": "unrelated gamma delta epsilon",
    }
    reg = promote_from_stream(list(corpus.values()))
    scope = corpus_scope("mod_probe", reg)
    sem = SemanticLightIndex(registry=reg)
    pool = DocPrimePool()
    builder = CorpusLatticeBuilder(scope.corpus_prime, reg, sem, pool, max_df_frac=0.5)
    for doc_id, text in corpus.items():
        builder.observe_doc(doc_id, text)
        sem.observe_doc(text, mode="positional")
    lattice = builder.finalize()
    routed, _ = lattice.route_pool(["rareterm"], semantic=sem)
    assert "d1" in routed
    assert "d2" in routed
    assert "d3" not in routed


def test_hybrid_corpus_lattice_pool_trace():
    from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever

    corpus = {
        "d1": "mitochondria produces atp energy",
        "d2": "atp powers muscle cells",
        "d3": "sunlight feeds plant growth",
    }
    r = build_hybrid_retriever(
        corpus,
        corpus_name="cl_trace",
        config=HybridConfig(enable_corpus_lattice=True),
    )
    trace = r.retrieve_with_trace("atp energy mitochondria", limit=3)
    assert any(s.get("step") == "corpus_lattice_pool" for s in trace.filter_steps)


def test_hybrid_wires_corpus_prime():
    from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever

    corpus = {"a": "quantum physics explores entanglement"}
    r = build_hybrid_retriever(
        corpus,
        corpus_name="quantum_probe",
        config=HybridConfig(enable_corpus_lattice=True),
    )
    assert r.corpus_prime is not None
    assert r.corpus_lattice is not None
    assert r.corpus_lattice.corpus_prime == r.corpus_prime
