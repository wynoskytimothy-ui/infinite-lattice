"""Gate: CHMR — boost docs sharing rare L2 witnesses across top-K prelim hits."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, HybridHit, build_hybrid_retriever
from lattice_retriever_v1.lattice2_correlation import Lattice2CorrelationPass
from lattice_retriever_v1.pool_rerank import cross_hit_mutual_rerank

_CHMR_CORPUS = {
    "decoy": "alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha alpha zzconsensus",
    "cluster_a": "zzconsensus zzwitness",
    "cluster_b": "zzconsensus zzwitness extra_b",
    "cluster_c": "zzconsensus zzwitness extra_c",
}
_CHMR_QUERY = "alpha zzconsensus zzwitness"


def _build_chmr_fixture():
    return build_hybrid_retriever(
        _CHMR_CORPUS,
        config=HybridConfig(lexical_mode="lattice_plane", lam_l2=0.0, lam_walk=0.0),
    )


def _prelim_hits(retriever, query: str) -> list[HybridHit]:
    terms = retriever._query_terms(query)
    pool = frozenset(_CHMR_CORPUS)
    lex = retriever.lexical.score_pool(query, pool)
    hits: list[HybridHit] = []
    idf_fn = retriever.router.semantic.idf
    for doc_id in _CHMR_CORPUS:
        l2_score, trace = Lattice2CorrelationPass.score(
            terms,
            doc_id,
            retriever.shell_index,
            idf_fn,
            placements=retriever.placements,
        )
        hits.append(
            HybridHit(
                doc_id=doc_id,
                score=lex[doc_id],
                lex_score=lex[doc_id],
                l2_score=l2_score,
                l2_trace=trace.explain(),
            )
        )
    hits.sort(key=lambda h: (-h.score, h.doc_id))
    return hits


def test_cross_hit_mutual_rerank_unit():
    idf_fn = lambda t: 2.5 if t.startswith("zz") else 1.0
    base = [
        HybridHit(
            "decoy",
            12.0,
            12.0,
            0.0,
            {
                "term_witnesses": [
                    {"term": "alpha", "witness": 2.0},
                    {"term": "zzconsensus", "witness": 0.0},
                ]
            },
        ),
        HybridHit(
            "cluster_a",
            10.0,
            10.0,
            0.0,
            {
                "term_witnesses": [
                    {"term": "alpha", "witness": 0.0},
                    {"term": "zzconsensus", "witness": 2.0},
                ]
            },
        ),
        HybridHit(
            "cluster_b",
            9.5,
            9.5,
            0.0,
            {
                "term_witnesses": [
                    {"term": "alpha", "witness": 0.0},
                    {"term": "zzconsensus", "witness": 2.0},
                ]
            },
        ),
        HybridHit(
            "cluster_c",
            9.0,
            9.0,
            0.0,
            {
                "term_witnesses": [
                    {"term": "alpha", "witness": 0.0},
                    {"term": "zzconsensus", "witness": 2.0},
                ]
            },
        ),
    ]
    out = cross_hit_mutual_rerank(base, idf_fn, lambda_ch=2.0, consensus_k=4)
    assert out[0].doc_id != "decoy"
    ch = out[0].l2_trace.get("cross_hit_mutual_rerank")
    assert ch
    assert "zzconsensus" in ch["consensus_terms"]
    assert any(h["term"] == "zzconsensus" for h in ch["hits"])


def test_chmr_promotes_consensus_cluster_over_decoy():
    r = _build_chmr_fixture()
    prelim = _prelim_hits(r, _CHMR_QUERY)
    assert prelim[0].doc_id == "decoy"
    assert max(h.lex_score for h in prelim) == prelim[0].lex_score

    out = cross_hit_mutual_rerank(
        prelim,
        r.router.semantic.idf,
        lambda_ch=2.0,
        consensus_k=4,
    )
    assert out[0].doc_id in {"cluster_a", "cluster_b", "cluster_c"}
    ch = out[0].l2_trace.get("cross_hit_mutual_rerank")
    assert ch
    assert "zzconsensus" in ch["consensus_terms"]

    r_on = _build_chmr_fixture().with_config(
        enable_cross_hit_rerank=True,
        ch_rerank_lambda=2.0,
        ch_consensus_k=4,
    )
    r_on.router.enable_lexical_bridge_rerank = False
    r_on.router.enable_cage_anchor_rerank = False
    wired = r_on._apply_stage08_rerank(prelim, _CHMR_QUERY)
    assert wired[0].doc_id in {"cluster_a", "cluster_b", "cluster_c"}
    assert wired[0].l2_trace.get("cross_hit_mutual_rerank")
