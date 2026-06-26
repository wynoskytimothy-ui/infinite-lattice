"""
Gate: pin selectivity — rarest-filter narrows on min-df selective pin, not pin union.

A df=1 term must not widen to the whole corpus because promiscuous L2 subword pins
share df with thousands of unrelated docs. Whole-word identity always narrows.
"""

from aethos_promotion import PromotionRegistry

from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever

ING_CORPUS = [
    "running quickly",
    "walking slowly",
    "thinking deeply",
    "building houses",
]


def _build_selectivity_index() -> LatticeRetriever:
    inner = PromotionRegistry(defer_l2_promotion=True, fast_ingest=False)
    stage = Stage04Registry(registry=inner)
    r = LatticeRetriever(semantic=SemanticLightIndex(registry=stage))
    for i in range(40):
        text = f"{ING_CORPUS[i % len(ING_CORPUS)]} batch{i}"
        stage.observe_text(text)
        r.index_doc(f"noise{i}", text)
    stage.observe_text("zzunique marker only")
    r.index_doc("gold", "zzunique marker only")
    assert r.semantic.registry.promoted_subword("ing") is not None
    return r


def test_promiscuous_subword_pin_excluded_from_narrowing():
    r = _build_selectivity_index()
    w = "zzunique"
    identity, selective, lift = r._split_pins(w)
    assert identity in selective
    ing = r.semantic.registry.promoted_subword("ing")
    if ing is not None and ing.prime in r._corridor_pins(w):
        assert ing.prime in lift or ing.prime not in selective


def test_df1_query_pool_stays_small():
    r = _build_selectivity_index()
    pool, mode, steps, _, _ = r.route_pool("zzunique marker")
    assert len(pool) <= 2, f"pool={len(pool)} mode={mode} steps={steps}"
    assert "gold" in pool


def test_routing_pin_is_identity_for_rare_term():
    r = _build_selectivity_index()
    _, _, steps, _, _ = r.route_pool("zzunique")
    narrow = [s for s in steps if s.get("step") == "rarest_filter"]
    zz = next(s for s in narrow if s.get("term") == "zzunique")
    assert zz["routing_pin"] == r._identity_pin("zzunique")
    assert zz["pin_doc_freq"] == 1
