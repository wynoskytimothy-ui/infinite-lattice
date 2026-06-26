"""Gate tests — k-way meet algebra derived from spec §4–§5 (k=2..6)."""

from __future__ import annotations

from lattice_retriever_v1.k_meet import (
    branch_compose,
    compose_k,
    deep_segments,
    full_sunflower_unified,
    pair_branch_compose,
    slide_meet,
    sub_sunflowers,
    swap_meet_primes,
    velocity_meet,
)


def test_k2_swap_meet():
    w = swap_meet_primes(3, 11)
    assert w.unified
    assert w.coord == (14.0, 3.0, 14.0)
    r = compose_k(3, 11)
    assert r.k == 2
    assert r.swap is not None and r.swap.unified
    assert r.velocity is None


def test_k3_pair_compose_creates_triple():
    """§5: bank(3,5)@7 = bank(3,7)@5 = bank(3,5,7)@5."""
    hit = pair_branch_compose(3, 5, 7)
    assert hit is not None
    coord, n_deep = hit
    assert coord == (12, 5, 15)
    assert n_deep == 5


def test_k3_velocity_slide_sunflower():
    r = compose_k(3, 5, 7)
    assert r.velocity is not None and r.velocity.unified
    assert r.velocity.coord == (12, 5, 15)
    assert r.velocity.n_shallow == 7
    assert r.velocity.n_deep == 5
    assert r.slide is not None and r.slide.unified
    assert r.full_sunflower_unified


def test_k3_user_example_3_7_541():
    r = compose_k(3, 7, 541)
    assert r.velocity is not None and r.velocity.unified
    assert r.velocity.coord == (548, 7, 551)
    assert r.velocity.n_shallow == 541
    assert r.velocity.n_deep == 7
    assert r.slide is not None and r.slide.unified
    assert r.slide.n_right == 3


def test_k4_velocity_extension_not_sunflower():
    """4-way: triple facets lock; full sunflower splits; velocity extension unifies."""
    r = compose_k(3, 5, 7, 11)
    assert not r.full_sunflower_unified
    assert r.velocity is not None and r.velocity.unified
    assert r.velocity.coord == (18, 7, 26)
    assert r.velocity.n_shallow == 11
    assert r.velocity.n_deep == 7
    for triple in ((3, 5, 7), (3, 5, 11), (3, 7, 11), (5, 7, 11)):
        assert full_sunflower_unified(sub_sunflowers(*triple))


def test_k4_branch_compose_two_triples_meet_at_k_node():
    """Two 3-way vectors share prefix; meet k-node at prior anchor, not each other."""
    bc = branch_compose(3, 5, 7, 11)
    assert bc is not None
    assert not bc.pair_vectors_unified
    assert bc.left_meets_k_node
    assert bc.k_node_unified
    assert bc.coord_k_node == (18, 7, 26)
    assert bc.n_k_node == 7


def test_k5_k6_velocity_extension():
    for primes in ((3, 5, 7, 11, 13), (3, 5, 7, 11, 13, 17)):
        v = velocity_meet(*primes)
        assert v is not None and v.unified
        r = compose_k(*primes)
        assert not r.full_sunflower_unified
        assert r.velocity is not None and r.velocity.unified
        segs = deep_segments(*primes)
        assert len(segs) == len(primes)


def test_compose_steps_use_velocity_rule():
    r = compose_k(3, 5, 7, 11)
    last = r.compose_steps[-1]
    assert last.unified
    assert last.n_shallow == 11
    assert last.n_deep == 7
    assert last.coord == (18, 7, 26)


def test_glass_box_explain():
    ex = compose_k(3, 5, 7, 11).explain()
    assert ex["k"] == 4
    assert ex["velocity"]["unified"] is True
    assert ex["branch_compose"]["k_node_unified"] is True
    assert ex["full_sunflower_unified"] is False
