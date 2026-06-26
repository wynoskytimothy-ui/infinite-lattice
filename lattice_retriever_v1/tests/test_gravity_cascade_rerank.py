"""Gate: GWCR — boost docs via shell satellites from top anchor docs."""

from lattice_retriever_v1.hybrid_retriever import HybridHit, build_hybrid_retriever
from lattice_retriever_v1.pool_rerank import gravity_cascade_rerank

_GWCR_CORPUS = {
    "anchor": "alpha zzmutation zzgene anchorboost zzmutation anchorboost",
    "decoy": "alpha zzmutation",
    "gold": "alpha zzgene",
}


def _build_gwcr_fixture():
    r = build_hybrid_retriever(_GWCR_CORPUS)
    assert r.router.semantic.is_rare("zzmutation")
    assert r.router.semantic.is_rare("zzgene")
    return r


def test_gravity_cascade_rerank_unit():
    r = _build_gwcr_fixture()
    idf_fn = r.router.semantic.idf
    is_rare_fn = r.router.semantic.is_rare
    base = [
        HybridHit("anchor", 15.0, 15.0, 0.0, {}, 0.0, ()),
        HybridHit("decoy", 12.0, 12.0, 0.0, {}, 0.0, ()),
        HybridHit("gold", 10.0, 10.0, 0.0, {}, 0.0, ()),
    ]
    out = gravity_cascade_rerank(
        base,
        ["alpha", "zzmutation"],
        r.shell_index,
        idf_fn,
        is_rare_fn,
        lambda_gw=2.0,
        anchor_count=1,
    )
    gold = next(h for h in out if h.doc_id == "gold")
    decoy = next(h for h in out if h.doc_id == "decoy")
    assert gold.score > decoy.score
    gw = gold.l2_trace.get("gravity_cascade_rerank")
    assert gw
    assert "zzgene" in gw["satellites"]
    assert any(h["satellite"] == "zzgene" for h in gw["hits"])


def test_gwcr_promotes_gold_over_decoy():
    r = _build_gwcr_fixture()
    prelim = [
        HybridHit("anchor", 15.0, 15.0, 0.0, {}, 0.0, ()),
        HybridHit("decoy", 12.0, 12.0, 0.0, {}, 0.0, ()),
        HybridHit("gold", 10.0, 10.0, 0.0, {}, 0.0, ()),
    ]
    q = "alpha zzmutation"
    idf_fn = r.router.semantic.idf
    is_rare_fn = r.router.semantic.is_rare

    out = gravity_cascade_rerank(
        prelim,
        r._query_terms(q),
        r.shell_index,
        idf_fn,
        is_rare_fn,
        lambda_gw=2.0,
        anchor_count=1,
    )
    gold = next(h for h in out if h.doc_id == "gold")
    decoy = next(h for h in out if h.doc_id == "decoy")
    assert gold.score > decoy.score

    r_on = r.with_config(
        enable_gravity_cascade_rerank=True,
        gw_rerank_lambda=2.0,
        gw_anchor_count=1,
    )
    r_on.router.enable_lexical_bridge_rerank = False
    r_on.router.enable_cage_anchor_rerank = False
    wired = r_on._apply_stage08_rerank(prelim, q)
    gold_w = next(h for h in wired if h.doc_id == "gold")
    decoy_w = next(h for h in wired if h.doc_id == "decoy")
    assert gold_w.score > decoy_w.score
    gw = gold_w.l2_trace.get("gravity_cascade_rerank")
    assert gw
    assert any(h["satellite"] == "zzgene" for h in gw["hits"])
