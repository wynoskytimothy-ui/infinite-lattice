"""Gate: L35 cage anchor bridge — boost docs sharing wing-cage anchors with top doc."""

from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, RetrieveHit


def _build_cage_fixture() -> LatticeRetriever:
    r = LatticeRetriever()
    r.index_doc("gold", "alpha beta gamma")
    r.index_doc(
        "decoy",
        "alpha beta gamma expression cells tissue the and is",
    )
    return r


def test_cage_anchor_rerank_unit():
    r = _build_cage_fixture()
    r.cage_anchor_rerank_lambda = 2.0
    r.cage_anchor_rerank_tiebreak_frac = 0.0
    base = [
        RetrieveHit("decoy", 10.0, ({"kind": "identity_overlap"},)),
        RetrieveHit("gold", 9.0, ({"kind": "identity_overlap"},)),
    ]
    out = r._cage_anchor_rerank(base, ["alpha", "beta", "gamma"])
    assert out[0].doc_id == "gold"
    bridge = [x for x in out[0].reasons if x.get("kind") == "cage_anchor_bridge"]
    assert bridge
    assert bridge[0]["anchor_doc"] == "decoy"


def test_cage_anchor_tiebreak_skips_large_gap():
    r = _build_cage_fixture()
    r.cage_anchor_rerank_lambda = 2.0
    r.cage_anchor_rerank_tiebreak_frac = 0.05
    base = [
        RetrieveHit("decoy", 10.0, ({"kind": "identity_overlap"},)),
        RetrieveHit("gold", 9.0, ({"kind": "identity_overlap"},)),
    ]
    out = r._cage_anchor_rerank(base, ["alpha", "beta", "gamma"])
    assert out[0].doc_id == "decoy"


def test_cage_anchor_shared_anchors_non_empty():
    r = _build_cage_fixture()
    shared = r._shared_cage_anchors("decoy", "gold")
    assert shared
