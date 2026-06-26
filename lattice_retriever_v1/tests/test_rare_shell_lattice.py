"""Rare-shell lattice ingest sidecar — anchor postings + global invert."""

from lattice_retriever_v1.doc_lattice_codec import (
    build_rare_correlation_shells,
    select_rare_in_doc,
)
from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.rare_shell_lattice import RareShellLatticeIndex
from lattice_retriever_v1.stage04_promote import promote_from_stream
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.tests.test_lattice2_correlation import QUANTUM_CORPUS


def _build_rare_index(corpus: dict[str, str]) -> RareShellLatticeIndex:
    reg = promote_from_stream(list(corpus.values()))
    semantic = SemanticLightIndex(registry=reg)
    idx = RareShellLatticeIndex()
    for doc_id, text in corpus.items():
        semantic.observe_doc(text)
        shells = build_rare_correlation_shells(text, reg, semantic)
        idx.observe_doc(doc_id, shells)
    return idx


def test_select_rare_in_doc_filters_hub_and_common():
    reg = promote_from_stream(list(QUANTUM_CORPUS.values()))
    semantic = SemanticLightIndex(registry=reg)
    for text in QUANTUM_CORPUS.values():
        semantic.observe_doc(text)
    words = tuple("quantum physics explores entanglement deeply".split())
    rare = select_rare_in_doc(words, semantic, k=8, max_df_frac=0.05)
    assert "entanglement" in rare
    assert "quantum" not in rare  # df=2 in 3-doc corpus, max_df=1


def test_rare_anchor_postings_light_correct_docs():
    idx = _build_rare_index(QUANTUM_CORPUS)
    reg = promote_from_stream(list(QUANTUM_CORPUS.values()))
    semantic = SemanticLightIndex(registry=reg)
    for text in QUANTUM_CORPUS.values():
        semantic.observe_doc(text)

    pool, steps = idx.route_pool(["entanglement"], semantic=semantic)
    assert "d1" in pool
    assert "d2" not in pool
    assert any(s.get("step") == "rare_anchor_light" for s in steps)

    pool_q, _ = idx.route_pool(["quantum"], semantic=semantic)
    assert pool_q <= {"d1", "d2"}


def test_route_pool_widen_unions_rarest_anchors_no_intersect():
    idx = _build_rare_index(QUANTUM_CORPUS)
    reg = promote_from_stream(list(QUANTUM_CORPUS.values()))
    semantic = SemanticLightIndex(registry=reg)
    for text in QUANTUM_CORPUS.values():
        semantic.observe_doc(text)

    pool, steps = idx.route_pool_widen(
        ["quantum", "entanglement"], semantic=semantic
    )
    assert "d1" in pool
    assert any(s.get("step") == "rare_anchor_widen" for s in steps)
    assert steps[0]["term"] == "entanglement"


def test_global_invert_anchor_postings():
    idx = _build_rare_index(QUANTUM_CORPUS)
    assert idx.n_docs == 3
    assert idx.anchor_postings
    for doc_id, shells in idx.doc_shells.items():
        for shell in shells:
            assert doc_id in idx.anchor_postings[shell.anchor_composite]
    ex = idx.explain()
    assert ex["n_docs"] == 3
    assert ex["n_anchors"] >= 1


def test_hybrid_rare_shell_default_off_no_regression():
    r_off = build_hybrid_retriever(QUANTUM_CORPUS)
    r_on = build_hybrid_retriever(
        QUANTUM_CORPUS,
        config=HybridConfig(enable_rare_shell_lattice=True, lam_rare=0.25),
    )
    assert r_off.rare_lattice is None
    assert r_on.rare_lattice is not None
    assert r_on.rare_lattice.n_docs == 3

    hits_off = r_off.retrieve("quantum", limit=3)
    assert {h.doc_id for h in hits_off[:2]} == {"d1", "d2"}

    trace_on = r_on.retrieve_with_trace("entanglement", limit=3)
    assert trace_on.hits[0].doc_id == "d1"
    assert any(s.get("step") == "rare_shell_widen" for s in trace_on.filter_steps)


def test_hybrid_rare_shell_widen_never_shrinks_pool():
    pool_off = {
        "enable_append_pool_union": False,
        "enable_pair_meet": False,
        "enable_walk_pool_expand": False,
    }
    r_base = build_hybrid_retriever(
        QUANTUM_CORPUS,
        config=HybridConfig(enable_rare_shell_lattice=False, **pool_off),
    )
    r_widen = build_hybrid_retriever(
        QUANTUM_CORPUS,
        config=HybridConfig(
            enable_rare_shell_lattice=True,
            rare_shell_pool_mode="widen",
            **pool_off,
        ),
    )
    base = r_base.retrieve_with_trace("quantum entanglement", limit=3)
    widen = r_widen.retrieve_with_trace("quantum entanglement", limit=3)
    assert widen.pool_size >= base.pool_size


def test_hybrid_rare_shell_intersect_mode_still_available():
    r = build_hybrid_retriever(
        QUANTUM_CORPUS,
        config=HybridConfig(
            enable_rare_shell_lattice=True,
            rare_shell_pool_mode="intersect",
            lam_rare=0.25,
        ),
    )
    trace = r.retrieve_with_trace("entanglement", limit=3)
    assert any(s.get("step") == "rare_shell_intersect" for s in trace.filter_steps)
