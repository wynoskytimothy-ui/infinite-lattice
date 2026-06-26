"""
Gate: morphological surface-form mismatch — doc and query must share corridor pins.

The test that matters is not "promotions exist" but "run and running resolve to
overlapping posting keys." Green here → wiring works. SciFact pool_recall move
is the integration confirmation.
"""

from aethos_promotion import PromotionRegistry

from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, MissPolicy

# Distinct parent words so "run" and "ing" promote (Stage 04 gate).
MORPH_PROMOTE_CORPUS = [
    "run fast",
    "run slow",
    "running quickly",
    "walking slowly",
    "runner wins",
    "building houses",
]


def _build_morph_retriever(doc_text: str, *, doc_id: str = "d_morph") -> LatticeRetriever:
    inner = PromotionRegistry(defer_l2_promotion=True, fast_ingest=False)
    stage = Stage04Registry(registry=inner)
    for line in MORPH_PROMOTE_CORPUS:
        stage.observe_text(line)
    stage.observe_text(doc_text)
    retriever = LatticeRetriever(semantic=SemanticLightIndex(registry=stage))
    retriever.index_doc(doc_id, doc_text)
    return retriever


def test_run_promoted_in_fixture_corpus():
    r = _build_morph_retriever("the run was fast")
    assert r.semantic.registry.promoted_subword("run") is not None


def test_run_and_running_share_corridor_pin():
    """Index run, factor running — overlapping promoted-atom pins."""
    r = _build_morph_retriever("the run was fast")
    run_pins = r.semantic.corridor_pins_for_term("run")
    running_pins = r.semantic.corridor_pins_for_term("running")
    shared = run_pins & running_pins
    assert shared, f"run pins {sorted(run_pins)} vs running {sorted(running_pins)}"


def test_index_run_query_running_enters_pool():
    r = _build_morph_retriever("the run was fast")
    pool, mode, steps, _, _ = r.route_pool("running")
    assert "d_morph" in pool, f"mode={mode} steps={steps}"


def test_index_running_query_run_enters_pool():
    """Inverse: doc has running, query has run."""
    r = _build_morph_retriever("running quickly today")
    pool, _, _, _, _ = r.route_pool("run today")
    assert "d_morph" in pool


def test_no_shared_pin_without_promotion():
    """Whole-word-only identities must not meet across run/running."""
    inner = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
    stage = Stage04Registry(registry=inner)
    r = LatticeRetriever(semantic=SemanticLightIndex(registry=stage))
    r.miss_policy = MissPolicy.EMPTY
    r.index_doc("d", "the run was fast")
    run_pins = r.semantic.corridor_pins_for_term("run")
    running_pins = r.semantic.corridor_pins_for_term("running")
    assert run_pins.isdisjoint(running_pins)
    pool, _, _, _, _ = r.route_pool("running")
    assert "d" not in pool
