"""
Gate: lift-pin fallback — selective intersect empty -> widen into lift pins.

Fixture: gold has "run", query "running quickly". Selective identity for
"running" does not post to gold; promoted stem "run" lift pin does.
"""

from aethos_promotion import PromotionRegistry

from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, MissPolicy

MORPH_PROMOTE_CORPUS = [
    "run fast",
    "run slow",
    "running quickly",
    "walking slowly",
    "runner wins",
    "building houses",
]


def _build_lift_fixture() -> LatticeRetriever:
    inner = PromotionRegistry(defer_l2_promotion=True, fast_ingest=False)
    stage = Stage04Registry(registry=inner)
    r = LatticeRetriever(semantic=SemanticLightIndex(registry=stage))
    for i, line in enumerate(MORPH_PROMOTE_CORPUS * 8):
        stage.observe_text(line)
        r.index_doc(f"noise{i}", line)
    stage.observe_text("run experiment only")
    r.index_doc("gold", "run experiment only")
    assert r.semantic.registry.promoted_subword("run") is not None
    return r


def test_selective_intersect_misses_gold_without_stem():
    r = _build_lift_fixture()
    r.enable_lift_pin_fallback = False
    r.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    pool, _, steps, _, _ = r.route_pool("running zzunknown")
    assert "gold" not in pool
    assert not any(s.get("step") == "lift_pin_widen" for s in steps)


def test_lift_pin_widen_recovers_gold():
    r = _build_lift_fixture()
    r.enable_lift_pin_fallback = True
    r.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    pool, mode, steps, _, _ = r.route_pool("running zzunknown")
    assert mode == "lift_pin_widen", f"mode={mode} steps={steps}"
    assert "gold" in pool
    lift = next(s for s in steps if s["step"] == "lift_pin_widen")
    assert lift["lift_pins"]


def test_mine_bucket_L26_drains_with_lift_fallback():
    from lattice_retriever_v1.glass_box_mine import bucket_delta, mine_query

    r_off = _build_lift_fixture()
    r_off.enable_lift_pin_fallback = False
    r_on = _build_lift_fixture()
    r_on.enable_lift_pin_fallback = True
    r_off.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    r_on.miss_policy = MissPolicy.WIDEN_RAREST_ONLY
    q = "running zzunknown"
    before = mine_query(r_off, "q1", q, ["gold"])
    after = mine_query(r_on, "q1", q, ["gold"])
    delta = bucket_delta(before, after)
    assert after.gold_in_pool
    assert not before.gold_in_pool
    assert delta.get("L01_gold_in_pool") == "gained"
