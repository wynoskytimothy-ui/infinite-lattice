"""Stage 06 gate — Section 5 triple, k-way factor-back, transgressor rule, repeats."""

import pytest

from aethos_words import letter_to_prime

from lattice_retriever_v1.stage04_promote import MIN_POOL_PRIME, promote_from_stream
from lattice_retriever_v1.tests.test_stage04 import ING_CORPUS
from lattice_retriever_v1.stage06_composites import (
    RepeatedPrimeError,
    decompose_word,
    factor_k_composite,
    ing_three_way_demo,
    meet_composite_k,
    regenerate_three_way_from_composite,
    repeated_pair_fallback,
    section5_triple_roles,
    three_way_address,
)

ING_READ = (
    letter_to_prime("i"),
    letter_to_prime("n"),
    letter_to_prime("g"),
)


def test_ing_three_way_section5_read_order():
    """ing = {i,n,g}: first two anchor corridor, third transgresses — Case 1."""
    addr = ing_three_way_demo()
    assert addr.read_order == ING_READ
    assert addr.a == letter_to_prime("i")
    assert addr.p == letter_to_prime("n")
    assert addr.n == letter_to_prime("g")
    assert addr.case == 1  # n < a < p  →  19 < 29 < 47
    assert addr.section5_coord == (76, 29, 95)
    assert addr.anchor_sum == 95
    assert addr.meet_composite == 19 * 29 * 47


def test_product_identity_not_anchor_sum():
    """Distinct triples can share anchor_sum; product stays unique."""
    addr = ing_three_way_demo()
    # sorted-role triple would differ in placement but same sum
    alt_a, alt_p, alt_n = 19, 29, 47
    assert alt_a + alt_p + alt_n == addr.anchor_sum
    assert meet_composite_k(alt_a, alt_p, alt_n) == addr.meet_composite


def test_factor_k_composite_exact():
    comp = meet_composite_k(*ING_READ)
    assert factor_k_composite(comp, 3) == tuple(sorted(ING_READ))


def test_factor_k_rejects_squared_prime():
    with pytest.raises(ValueError):
        factor_k_composite(12, 3)  # 2*2*3
    with pytest.raises(RepeatedPrimeError):
        meet_composite_k(19, 19, 29)


def test_pool_prime_is_always_transgressor():
    pool = MIN_POOL_PRIME + 2  # stand-in pool prime
    read = (letter_to_prime("i"), letter_to_prime("n"), pool)
    a, p, n, case = section5_triple_roles(read, pool_transgressor=pool)
    assert n == pool
    assert (a, p) == (letter_to_prime("i"), letter_to_prime("n"))
    assert case == 3  # a < p < n


def test_corridor_regeneration_three_way():
    original = ing_three_way_demo(quadrant=6)
    replay = regenerate_three_way_from_composite(
        original.meet_composite,
        read_order=original.read_order,
        quadrant=6,
    )
    assert original.meet_composite == replay.meet_composite
    assert original.section5_coord == replay.section5_coord
    assert original.lattice_signature == replay.lattice_signature
    assert original.corridor_key == replay.corridor_key


def test_repeated_letter_routed_to_pair_not_kway():
    with pytest.raises(RepeatedPrimeError):
        three_way_address(letter_to_prime("l"), letter_to_prime("l"), letter_to_prime("o"))
    addr = repeated_pair_fallback("all")
    assert addr.way == 2


def test_thing_letter_path_decomposes():
    dec = decompose_word("thing")
    assert dec["path"] == "letter_product"
    assert len(dec["constituents"]) == 5
    assert dec["meet_composite"] == meet_composite_k(*dec["constituents"])


def test_thing_letter_path_and_promoted_pool_as_transgressor():
    """thing = letter product; promoted ing pool prime enters Section 5 as n."""
    corpus = ING_CORPUS + ["math path", "bath myth"]
    reg = promote_from_stream(corpus)
    ing_tok = reg.promoted_subword("ing")
    assert ing_tok is not None

    dec = decompose_word("thing")
    assert dec["path"] == "letter_product"
    assert len(dec["constituents"]) == 5

    read = (letter_to_prime("i"), letter_to_prime("n"), ing_tok.prime)
    addr = three_way_address(*read, read_order=read)
    assert addr.n == ing_tok.prime
    assert addr.case == 3  # a < p < pool n
    assert addr.meet_composite == meet_composite_k(*read)


def test_glass_box_explain():
    ex = ing_three_way_demo().explain()
    assert ex["identity_is_product"] is True
    assert ex["anchor_sum_is_metadata"] is True
    assert ex["section5"]["case"] == 1
    assert ex["n_lattices"] == 32
