"""Oriented compound pins — reversed bigrams route to separate posting buckets."""

from lattice_retriever_v1.stage05_free_token import (
    free_token_address,
    meet_composite,
    oriented_corridor_pin,
)
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever
from aethos_words import letter_to_prime
from lattice_retriever_v1.stage03_rotation import rotate_token


def test_oriented_pin_differs_when_invoke_order_differs():
    p, q = letter_to_prime("a"), letter_to_prime("t")
    tas = rotate_token("tas", {"t": 500, "a": 200, "s": 50})
    sat = rotate_token("sat", {"s": 50, "a": 200, "t": 500})
    addr_ta = free_token_address(p, q, quadrant=tas.quadrant, invoke_order=(p, q))
    addr_at = free_token_address(p, q, quadrant=sat.quadrant, invoke_order=(q, p))
    assert addr_ta.meet_composite == addr_at.meet_composite == meet_composite(p, q)
    assert oriented_corridor_pin(addr_ta) != oriented_corridor_pin(addr_at)


def test_reversed_word_pairs_index_separate_postings():
    r = LatticeRetriever()
    r.index_doc("ap", "apple phone sells well")
    r.index_doc("pa", "phone apple store opens")
    pin_ap = r._compound_pin("apple", "phone")
    pin_pa = r._compound_pin("phone", "apple")
    assert pin_ap != pin_pa
    assert r.postings[pin_ap] == {"ap"}
    assert r.postings[pin_pa] == {"pa"}


def test_reversed_word_pairs_route_separate_pools():
    r = LatticeRetriever()
    r.index_doc("ap", "apple phone sells well")
    r.index_doc("pa", "phone apple store opens")
    for _ in range(20):
        r.index_doc(f"noise{_}", "common words only here")

    pool_ap, mode_ap, steps_ap, _, _ = r.route_pool("apple phone")
    pool_pa, mode_pa, steps_pa, _, _ = r.route_pool("phone apple")

    assert "ap" in pool_ap
    assert "pa" not in pool_ap
    assert "pa" in pool_pa
    assert "ap" not in pool_pa
    compound_ap = next(s for s in steps_ap if s.get("kind") == "compound")
    compound_pa = next(s for s in steps_pa if s.get("kind") == "compound")
    assert compound_ap["terms"] == ["apple", "phone"]
    assert compound_pa["terms"] == ["phone", "apple"]
    assert compound_ap["routing_pin"] != compound_pa["routing_pin"]
    assert "oriented_key" in compound_ap


def test_furry_cat_vs_cat_fur_separate_oriented_pins():
    r = LatticeRetriever()
    r.index_doc("fc", "furry cat naps")
    r.index_doc("cf", "cat fur sheds")
    assert r._compound_pin("furry", "cat") != r._compound_pin("cat", "fur")
    pool_fc, _, _, _, _ = r.route_pool("furry cat")
    pool_cf, _, _, _, _ = r.route_pool("cat fur")
    assert pool_fc == ["fc"]
    assert pool_cf == ["cf"]
