"""Append-index BM25 top-K union into Stage08 Phase A pool."""

from aethos_promotion import PromotionRegistry

from lattice_retriever_v1.hybrid_retriever import HybridConfig, build_hybrid_retriever
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, MissPolicy


def _morph_corpus() -> dict[str, str]:
    corpus: dict[str, str] = {"gold": "run experiment only"}
    for i in range(24):
        corpus[f"n{i}"] = "running quickly today " * 4
    return corpus


def _build_morph_hybrid(*, enable_union: bool) -> object:
    inner = PromotionRegistry(defer_l2_promotion=True, fast_ingest=False)
    stage = Stage04Registry(registry=inner)
    semantic = SemanticLightIndex(registry=stage)
    router = LatticeRetriever(semantic=semantic)
    router.enable_lift_pin_fallback = False
    router.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    corpus = _morph_corpus()
    for doc_id, text in corpus.items():
        stage.observe_text(text)
        router.index_doc(doc_id, text)
    assert semantic.registry.promoted_subword("run") is not None
    cfg = HybridConfig(
        lexical_mode="append_index",
        lam_lex=1.0,
        lam_l2=0.0,
        lam_walk=0.0,
        enable_pair_meet=False,
        enable_walk_pool_expand=False,
        enable_stage08_rerank=False,
        enable_append_pool_union=enable_union,
        append_pool_k=50,
    )
    from lattice_retriever_v1.hybrid_retriever import HybridZeroShotRetriever
    from aethos_append_index import AppendOnlyLatticeIndex
    from aethos_lattice_lexical import lattice_lexical_scorer

    append_idx = AppendOnlyLatticeIndex()
    for doc_id, text in corpus.items():
        append_idx.add(doc_id, text)
    append_idx.finalize()
    lexical = lattice_lexical_scorer(append_idx, mode=cfg.lexical_mode)
    return HybridZeroShotRetriever(
        router=router,
        append_idx=append_idx,
        lexical=lexical,
        corpus=dict(corpus),
        config=cfg,
    )


def test_append_pool_union_recovers_gold_corridor_misses():
    """Gold on lift pin only — Stage08 miss; append BM25 top-K pulls it in."""
    q = "running experiment zzunknown"
    r_off = _build_morph_hybrid(enable_union=False)
    r_on = _build_morph_hybrid(enable_union=True)
    stage_pool, _, _, _, _ = r_off.router.route_pool(q)
    assert "gold" not in stage_pool
    trace_off = r_off.retrieve_with_trace(q, limit=5)
    trace_on = r_on.retrieve_with_trace(q, limit=5)
    assert "gold" not in trace_off.pool_docs
    assert "gold" in trace_on.pool_docs
    assert any(s.get("step") == "append_pool_union" for s in trace_on.filter_steps)


def test_append_top_survives_pool_cap():
    corpus = {f"d{i}": f"topic{i} filler text " * 8 for i in range(40)}
    corpus["target"] = "quantum entanglement special marker"
    cfg = HybridConfig(
        lexical_mode="append_index",
        lam_lex=1.0,
        lam_l2=0.0,
        lam_walk=0.0,
        enable_pair_meet=False,
        enable_walk_pool_expand=False,
        enable_stage08_rerank=False,
        enable_append_pool_union=True,
        append_pool_k=15,
        max_pool=10,
    )
    r = build_hybrid_retriever(corpus, config=cfg)
    q = "quantum entanglement special marker"
    append_top = frozenset(r.append_idx.search(q, k=cfg.append_pool_k))
    trace = r.retrieve_with_trace(q, limit=5)
    assert append_top <= trace.pool_docs
    assert len(trace.pool_docs) <= cfg.max_pool or len(append_top) > cfg.max_pool


def test_append_pool_union_disabled_skips_step():
    r = build_hybrid_retriever(
        _morph_corpus(),
        config=HybridConfig(
            enable_append_pool_union=False,
            enable_pair_meet=False,
            enable_walk_pool_expand=False,
        ),
    )
    trace = r.retrieve_with_trace("running experiment", limit=3)
    assert not any(s.get("step") == "append_pool_union" for s in trace.filter_steps)
