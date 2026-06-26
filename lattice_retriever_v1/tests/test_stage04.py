"""Stage 04 gate — frequent meets → L2 pool primes, append-only."""

from aethos_promotion import intersection_prime

from lattice_retriever_v1.stage02_intersections import find_intersection, intersect_three
from lattice_retriever_v1.stage04_promote import MIN_POOL_PRIME, promote_from_stream

# Distinct parent words + enough …ing occurrences for L2 promotion (PMI/cohesion).
ING_CORPUS = [
    "running quickly",
    "walking slowly",
    "thinking deeply",
    "building houses",
]


def test_ing_promoted_after_frequent_occurrences():
    """Enough …ing in distinct parent words → dedicated L2 pool prime."""
    stage = promote_from_stream(ING_CORPUS)
    ing = stage.promoted_subword("ing")
    assert ing is not None, stage.registry.subword_counts
    assert ing.prime >= MIN_POOL_PRIME
    assert ing.tier.name == "L2_SUBWORD"
    assert ing.text == "ing"


def test_old_letter_intersections_still_resolve():
    """Promotion does not relocate letter meets — stage 02 unchanged."""
    baseline = intersect_three("i", "n", "g")
    stage = promote_from_stream(ING_CORPUS)
    assert stage.promoted_subword("ing") is not None
    assert stage.letter_intersection_unchanged("ing", baseline=baseline)
    addr = find_intersection("running", "ing")
    assert addr is not None
    assert addr.label == "ing"
    assert addr.lattice_coords == baseline.lattice_coords


def test_promoted_prime_differs_from_letter_intersection_sum():
    stage = promote_from_stream(ING_CORPUS)
    ing = stage.promoted_subword("ing")
    assert ing is not None
    assert ing.prime != intersection_prime("ing")
    assert intersection_prime("ing") == intersect_three("i", "n", "g").anchor_sum


def test_append_only_idempotent():
    stage = promote_from_stream(ING_CORPUS)
    first_prime = stage.promoted_subword("ing")
    assert first_prime is not None
    stage.observe_text("running again")
    second = stage.promoted_subword("ing")
    assert second.prime == first_prime.prime


def test_glass_box_explain():
    stage = promote_from_stream(ING_CORPUS)
    rec = next(p for p in stage.promotions if p.text == "ing")
    ex = rec.explain()
    assert ex["tier"] == "L2_SUBWORD"
    assert ex["append_only"] is True
    assert ex["letter_meet"]["label"] == "ing"
    assert ex["letter_meet"]["n_lattices"] == 32
    assert len(ex["parent_words"]) >= 2

    res = stage.resolve_subword("ing")
    assert res["promoted"] is True
    assert res["letter_meet"]["label"] == "ing"


def test_th_promoted_in_corpus():
    texts = ["math scores", "path taken", "bath ready", "myth ancient"]
    stage = promote_from_stream(texts)
    th = stage.promoted_subword("th")
    assert th is not None
    assert th.prime >= MIN_POOL_PRIME
    assert find_intersection("math", "th") is not None


def _promotion_snapshot(stage):
    """Comparable promotion state for determinism / replay gates."""
    return tuple(
        (r.text, r.prime, r.parent_primes, r.parent_words, r.count)
        for r in stage.promotions
    )


def test_distinct_parents_not_raw_count():
    """High raw frequency in one parent must not promote — gate uses distinct parents."""
    spam_corpus = ["running running running running running running"]
    stage = promote_from_stream(spam_corpus)
    ing = stage.promoted_subword("ing")
    parents = stage.registry.subword_parent_words.get("ing", set())
    raw = stage.registry.subword_counts.get("ing", 0)

    assert raw >= stage.registry.subword_promote_at
    assert len(parents) == 1
    assert ing is None

    # Positive control: same raw total spread across distinct parent words → promotes
    good = promote_from_stream(ING_CORPUS)
    assert good.promoted_subword("ing") is not None
    good_parents = good.registry.subword_parent_words["ing"]
    assert len(good_parents) >= good.registry.subword_min_parents


def test_thethethe_repetition_does_not_promote():
    """One spam token repeated must not burn pool primes on its subwords."""
    stage = promote_from_stream(["thethethe thethethe thethethe"])
    assert stage.registry.subword_counts.get("th", 0) >= 2
    assert len(stage.registry.subword_parent_words.get("th", set())) == 1
    assert stage.promoted_subword("th") is None
    assert stage.promoted_subword("he") is None


def test_promote_from_stream_deterministic_replay():
    """Same corpus → identical promotion records (pool primes + order). Stage 05 depends on this."""
    first = promote_from_stream(ING_CORPUS)
    second = promote_from_stream(ING_CORPUS)
    assert _promotion_snapshot(first) == _promotion_snapshot(second)
    assert first.promoted_subword("ing").prime == second.promoted_subword("ing").prime


def test_corpus_replay_on_same_registry_idempotent():
    """Re-ingesting the same corpus must not reallocate or reorder promotions."""
    stage = promote_from_stream(ING_CORPUS)
    before = _promotion_snapshot(stage)
    stage.observe_stream(ING_CORPUS)
    after = _promotion_snapshot(stage)
    assert before == after
