"""Self-teach audit and demotion tests."""

from lattice_retriever_v1.glass_box_demote import apply_lexical_demotion, scifact_polluter_docs
from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.self_teach_audit import (
    audit_and_self_teach,
    synthetic_query_from_doc,
)
from lattice_retriever_v1.tests.test_lattice2_correlation import QUANTUM_CORPUS


def test_synthetic_query_uses_rare_terms():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    text = QUANTUM_CORPUS["d1"]
    q = synthetic_query_from_doc(text, r.append_idx, len(r.append_idx.alive))
    assert len(q.split()) >= 2
    assert set(q.split()) <= set(text.split())


def test_self_teach_audit_runs_on_corpus():
    r = build_hybrid_retriever(QUANTUM_CORPUS)
    teach, stats = audit_and_self_teach(r, QUANTUM_CORPUS)
    assert stats.docs_audited == 3
    assert stats.explain()["docs_audited"] == 3
    r.teach = teach
    hits = r.retrieve("quantum entanglement", limit=3)
    assert hits


def test_polluter_demotion_lowers_score():
    polluters = scifact_polluter_docs()
    assert polluters
    scores = {"good": 10.0, next(iter(polluters)): 10.0}
    pool = set(scores)
    corpus = {d: "word " * 5 for d in pool}
    idx = build_hybrid_retriever({"good": "quantum physics"}).append_idx
    out, n = apply_lexical_demotion(
        scores,
        "quantum",
        pool,
        idx,
        corpus,
        polluter_docs=polluters,
        polluter_penalty=0.5,
    )
    pid = next(iter(polluters))
    assert out["good"] == 10.0
    assert out[pid] < 10.0
    assert n >= 1
