"""Gate: MNCR — boost docs with shell-correlated neighbors missing from query."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, HybridHit, build_hybrid_retriever
from lattice_retriever_v1.pool_rerank import missing_neighbor_rerank

_MNCR_CORPUS = {
    "gold": "alpha zzanchor zzneighbor",
    "decoy": "alpha zzanchor",
}


def _build_mncr_fixture():
    r = build_hybrid_retriever(_MNCR_CORPUS)
    assert r.router.semantic.is_rare("zzanchor")
    assert r.router.semantic.is_rare("zzneighbor")
    return r


def test_missing_neighbor_rerank_unit():
    r = _build_mncr_fixture()
    idf_fn = r.router.semantic.idf
    base = [
        HybridHit("decoy", 12.0, 12.0, 0.0, {}, 0.0, ()),
        HybridHit("gold", 10.0, 10.0, 0.0, {}, 0.0, ()),
    ]
    out = missing_neighbor_rerank(
        base,
        ["alpha", "zzanchor"],
        r.shell_index,
        r.placements,
        idf_fn,
        lambda_mn=2.0,
    )
    assert out[0].doc_id == "gold"
    mn = out[0].l2_trace.get("missing_neighbor_rerank")
    assert mn
    assert any(h["neighbor"] == "zzneighbor" for h in mn["hits"])


def test_mncr_promotes_gold_over_decoy():
    r_off = build_hybrid_retriever(
        _MNCR_CORPUS,
        config=HybridConfig(enable_missing_neighbor_rerank=False),
    )
    r_on = build_hybrid_retriever(
        _MNCR_CORPUS,
        config=HybridConfig(
            enable_missing_neighbor_rerank=True,
            mn_rerank_lambda=0.35,
        ),
    )
    for retriever in (r_off, r_on):
        retriever.router.enable_lexical_bridge_rerank = False
        retriever.router.enable_cage_anchor_rerank = False

    q = "alpha zzanchor"
    hits_off = r_off.retrieve(q, limit=2)
    hits_on = r_on.retrieve(q, limit=2)

    assert hits_off[0].doc_id == "decoy"
    assert hits_on[0].doc_id == "gold"
    mn = hits_on[0].l2_trace.get("missing_neighbor_rerank")
    assert mn
    assert any(h["neighbor"] == "zzneighbor" for h in mn["hits"])
