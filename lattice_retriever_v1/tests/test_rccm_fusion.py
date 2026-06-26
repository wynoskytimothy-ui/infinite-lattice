"""RCCM Phase 1 — multiplicative rare-L2 fusion (MRL2)."""

from lattice_retriever_v1.hybrid_retriever import HybridConfig, HybridZeroShotRetriever


def test_lex_mult_rare_l2_no_reorder_when_l2_zero():
    pool = {"a", "b", "c"}
    lex = {"a": 3.0, "b": 2.0, "c": 1.0}
    l2 = {"a": 5.0, "b": 4.0, "c": 3.0}
    l2_rare = {d: 0.0 for d in pool}
    fused = HybridZeroShotRetriever._fuse(
        lex,
        l2,
        {},
        pool,
        lam_lex=1.0,
        lam_l2=0.25,
        lam_walk=0.0,
        fuse_mode="lex_mult_rare_l2",
        l2_rare=l2_rare,
        rccm_eps=0.05,
    )
    order_lex = sorted(pool, key=lambda d: (-lex[d], d))
    order_fused = sorted(pool, key=lambda d: (-fused[d], d))
    assert order_fused == order_lex


def test_lex_mult_rare_l2_boosts_high_rare_l2():
    pool = {"a", "b"}
    lex = {"a": 2.0, "b": 2.0}
    l2_rare = {"a": 0.0, "b": 10.0}
    fused = HybridZeroShotRetriever._fuse(
        lex,
        {},
        {},
        pool,
        lam_lex=1.0,
        lam_l2=0.0,
        lam_walk=0.0,
        fuse_mode="lex_mult_rare_l2",
        l2_rare=l2_rare,
        rccm_eps=0.5,
    )
    assert fused["b"] > fused["a"]


def test_resolve_rccm_config_preset():
    from lattice_retriever_v1.hybrid_retriever import resolve_rccm_config

    cfg = resolve_rccm_config(HybridConfig(enable_rccm=True))
    assert cfg.cage_ingest_mode == "rare_combo"
    assert cfg.enable_corpus_lattice is True
    assert cfg.enable_rare_shell_lattice is True
    assert cfg.rare_shell_pool_mode == "widen"
    assert cfg.enable_append_pool_union is True
    assert cfg.fuse_mode == "lex_mult_rare_l2"
