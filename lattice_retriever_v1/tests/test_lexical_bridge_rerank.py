"""Gate: L40 blind lexical bridge — boost docs sharing rare terms with top anchor."""

from lattice_retriever_v1.glass_box_mine import mine_query
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, RetrieveHit


def _build_bridge_fixture() -> LatticeRetriever:
    r = LatticeRetriever()
    r.index_doc("gold", "mutation zzbridge")
    r.index_doc(
        "decoy",
        "mutation zzbridge protein expression cells tissue the and is",
    )
    assert r.semantic.is_rare("zzbridge")
    return r


def test_bridge_rerank_promotes_gold_over_decoy():
    r = _build_bridge_fixture()
    r.bridge_rerank_lambda = 3.0
    r_off = _build_bridge_fixture()
    r_off.enable_lexical_bridge_rerank = False

    hits_off = r_off.retrieve("mutation zzbridge", limit=3)
    hits_on = r.retrieve("mutation zzbridge", limit=3)

    assert hits_off[0].doc_id == "decoy"
    assert hits_on[0].doc_id == "gold"
    bridge = [x for x in hits_on[0].reasons if x.get("kind") == "lexical_bridge"]
    assert bridge
    assert "zzbridge" in bridge[0]["shared_rare_terms"]


def test_lexical_bridge_rerank_unit():
    r = _build_bridge_fixture()
    r.bridge_rerank_lambda = 2.0
    r.bridge_rerank_tiebreak_frac = 0.0
    base = [
        RetrieveHit("decoy", 12.0, ({"kind": "identity_overlap"},)),
        RetrieveHit("gold", 10.0, ({"kind": "identity_overlap"},)),
    ]
    out = r._lexical_bridge_rerank(base)
    assert out[0].doc_id == "gold"


def test_lexical_bridge_tiebreak_skips_large_gap():
    r = _build_bridge_fixture()
    r.bridge_rerank_lambda = 3.0
    r.bridge_rerank_tiebreak_frac = 0.05
    base = [
        RetrieveHit("decoy", 12.0, ({"kind": "identity_overlap"},)),
        RetrieveHit("gold", 10.0, ({"kind": "identity_overlap"},)),
    ]
    out = r._lexical_bridge_rerank(base)
    assert out[0].doc_id == "decoy"
    assert not any(x.get("kind") == "lexical_bridge" for x in out[1].reasons)


def test_lexical_bridge_tiebreak_applies_within_delta():
    r = _build_bridge_fixture()
    r.bridge_rerank_lambda = 3.0
    r.bridge_rerank_tiebreak_frac = 0.05
    base = [
        RetrieveHit("decoy", 10.0, ({"kind": "identity_overlap"},)),
        RetrieveHit("gold", 9.7, ({"kind": "identity_overlap"},)),
    ]
    out = r._lexical_bridge_rerank(base)
    assert out[0].doc_id == "gold"
    bridge = [x for x in out[0].reasons if x.get("kind") == "lexical_bridge"]
    assert bridge


def test_mine_rank1_gained_with_bridge():
    r_off = _build_bridge_fixture()
    r_off.enable_lexical_bridge_rerank = False
    r_on = _build_bridge_fixture()
    r_on.bridge_rerank_lambda = 3.0
    q = "mutation zzbridge"
    before = mine_query(r_off, "q1", q, ["gold"])
    after = mine_query(r_on, "q1", q, ["gold"])
    assert before.gold_ranks["gold"] != 1
    assert after.gold_ranks["gold"] == 1
