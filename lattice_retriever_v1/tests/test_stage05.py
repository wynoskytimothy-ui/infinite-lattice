"""Stage 05 gate — free P×Q address, factor-back, order policy, corridor regeneration."""

from aethos_words import letter_to_prime

from lattice_retriever_v1.stage03_rotation import rotate_token
from lattice_retriever_v1.stage05_free_token import (
    addresses_bit_identical,
    factor_pair_composite,
    free_token_address,
    meet_composite,
    regenerate_from_composite,
)


def test_meet_composite_order_free():
    p, q = letter_to_prime("t"), letter_to_prime("h")
    assert meet_composite(p, q) == meet_composite(q, p)


def test_distinct_pairs_distinct_composite():
    a = letter_to_prime("t")
    b = letter_to_prime("h")
    c = letter_to_prime("e")
    assert meet_composite(a, b) != meet_composite(a, c)


def test_factor_back_exact():
    p, q = letter_to_prime("i"), letter_to_prime("n")
    composite = meet_composite(p, q)
    got = factor_pair_composite(composite)
    assert got == (min(p, q), max(p, q))
    assert set(got) == {p, q}


def test_factor_back_rejects_non_semiprime():
    try:
        factor_pair_composite(12)  # 2*2*3
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_corridor_regeneration_from_primes_no_cache():
    """Address computed, not stored — rebuild from primes alone, bit-identical."""
    p, q = 73, 23
    original = free_token_address(p, q, quadrant=9, transgressor_n=7, invoke_order=(p, q))
    # discard original object; only pass canonical factors + rotation
    replay = free_token_address(p, q, quadrant=9, transgressor_n=7, invoke_order=(p, q))
    assert addresses_bit_identical(original, replay)
    assert original.lattice_signature == replay.lattice_signature


def test_corridor_regeneration_from_composite_only():
    """Throw away primes — factor composite, re-open corridor with same rotation."""
    p, q = letter_to_prime("t"), letter_to_prime("h")
    original = free_token_address(p, q, quadrant=14, transgressor_n=7)
    composite = original.meet_composite
    replay = regenerate_from_composite(composite, quadrant=14, transgressor_n=7)
    assert addresses_bit_identical(original, replay)


def test_order_reenters_outside_product():
    """
    P×Q is order-blind; invoke_order + quadrant carry orientation.
    Same meet, different corridor keys when rotation differs.
    """
    p, q = letter_to_prime("a"), letter_to_prime("t")
    addr_pq = free_token_address(p, q, quadrant=4, invoke_order=(p, q))
    addr_qp = free_token_address(q, p, quadrant=11, invoke_order=(q, p))
    assert addr_pq.meet_composite == addr_qp.meet_composite
    assert addr_pq.lattice_signature == addr_qp.lattice_signature
    assert addr_pq.invoke_order != addr_qp.invoke_order
    assert addr_pq.corridor_key != addr_qp.corridor_key


def test_anagram_words_same_pair_product_different_quadrant_when_order_differs():
    """tas vs sat: pair product order-free; invoke_order + quadrant carry orientation."""
    tas = rotate_token("tas", {"t": 500, "a": 200, "s": 50})
    sat = rotate_token("sat", {"s": 50, "a": 200, "t": 500})
    assert tas.quadrant != sat.quadrant
    tp, ap = letter_to_prime("t"), letter_to_prime("a")
    assert meet_composite(tp, ap) == meet_composite(ap, tp)
    c1 = free_token_address(tp, ap, quadrant=tas.quadrant, invoke_order=(tp, ap))
    c2 = free_token_address(ap, tp, quadrant=sat.quadrant, invoke_order=(ap, tp))
    assert c1.meet_composite == c2.meet_composite
    assert c1.corridor_key != c2.corridor_key


def test_glass_box_explain():
    p, q = letter_to_prime("t"), letter_to_prime("h")
    ex = free_token_address(p, q, quadrant=2).explain()
    assert ex["stored_row"] is False
    assert ex["order_in_product"] is False
    assert ex["order_in_quadrant_and_invoke_order"] is True
    assert ex["n_lattices"] == 32
