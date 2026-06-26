"""Stage 07 gate — wing cages, L4–L6 correlations, subword disambiguation."""

from lattice_retriever_v1.stage07_semantic_light import (
    SemanticLightIndex,
    WingCage,
    build_demo_registry,
    word_path_identities,
)


def test_th_ing_hing_promoted_separately():
    reg = build_demo_registry()
    th, ing, hing = reg.promoted_subword("th"), reg.promoted_subword("ing"), reg.promoted_subword("hing")
    assert th and ing and hing
    assert len({th.prime, ing.prime, hing.prime}) == 3


def test_thing_paths_th_ing_vs_t_hing_vs_letters():
    """th+ing ≠ t+hing ≠ letter product — same word, different free-token vectors."""
    reg = build_demo_registry()
    paths = word_path_identities("thing", reg)
    assert paths["structural_paths_distinct"] is True
    c = paths["composites"]
    assert c["letter_product"] != c["th_plus_ing"]
    assert c["t_plus_hing"] != c["th_plus_ing"]
    assert c["t_plus_hing"] != c["letter_product"]


def test_cat_pet_lift_via_rare_purr():
    idx = SemanticLightIndex(registry=build_demo_registry())
    idx.observe_corpus([
        "cat purrs loudly",
        "pet purrs softly",
        "cat and pet purr",
    ])
    # cage anchored on cat-pet-purr window
    cage = idx._cage_for_triple("cat", "pet", "purr")
    score_cat = idx.touch_weight(["cat", "purr"], cage)
    score_the = idx.touch_weight(["the"], cage)
    assert score_cat > 0
    assert score_the == 0


def test_hub_contribution_under_five_percent():
    idx = SemanticLightIndex(registry=build_demo_registry())
    idx.observe_corpus(["cat purrs the pet purrs"])
    cage = idx._cage_for_triple("cat", "purrs", "the")
    total = sum(
        idx.touch_weight([t], cage) for t in cage.correlations
    ) + idx.touch_weight(["cat"], cage)
    hub = idx.touch_weight(["the"], cage)
    assert total > 0
    assert hub / total < 0.05


def test_wing_cage_lazy_no_base_address_change():
    reg = build_demo_registry()
    addr_before = reg.promoted_subword("ing").prime
    cage = WingCage(
        anchor_label="ing",
        anchor_composite=19 * 29 * 47,
        anchor_primes=(19, 29, 47),
    )
    cage.add_correlation("running", 999, source_prime=19, strength=3)
    assert reg.promoted_subword("ing").prime == addr_before
    assert cage.correlations["running"].strength == 3
    assert 1 <= cage.correlations["running"].rotation_quadrant <= 4


def test_six_word_window_three_way_sliding():
    idx = SemanticLightIndex(registry=build_demo_registry())
    idx.observe_doc("apple phone sells iphone tablet watch case", max_window=6)
    assert len(idx.cages) >= 4  # sliding triples in 6-word window


def test_apple_phone_correlation_cage():
    idx = SemanticLightIndex(registry=build_demo_registry())
    idx.observe_corpus([
        "apple phone sells well",
        "iphone apple store opens",
        "apple phone iphone bundle",
    ])
    cage = idx._cage_for_triple("apple", "phone", "iphone")
    ex = cage.explain()
    assert ex["anchor_composite"] > 0
    assert ex["n_correlations"] >= 0


def test_glass_box_lift_score():
    idx = SemanticLightIndex(registry=build_demo_registry())
    idx.observe_corpus(["cat purrs", "pet purrs"])
    cage = idx._cage_for_triple("cat", "pet", "purr")
    out = idx.lift_score(["cat", "purr"], cage.anchor_composite)
    assert "score" in out
    assert out["cage"] is not None
