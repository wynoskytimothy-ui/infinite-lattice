"""Stage 02 gate — intersections → subword addresses."""

from lattice_retriever_v1.stage02_intersections import (
    find_intersection,
    intersect_two,
    intersect_three,
    subwords_from_text,
)


def test_intersect_th_stable():
    a = intersect_two("t", "h")
    b = intersect_two("t", "h")
    assert a.address == b.address
    assert a.explain() == b.explain()
    assert a.label == "th"
    assert a.way == 2


def test_ing_three_way_distinct_from_in_two_way():
    ing = intersect_three("i", "n", "g")
    inn = intersect_two("i", "n")
    assert ing.way == 3
    assert inn.way == 2
    assert ing.address != inn.address
    assert ing.label == "ing"
    assert inn.label == "in"


def test_subwords_from_text_running():
    addrs = subwords_from_text("running")
    labels = {a.label for a in addrs}
    assert "ru" in labels
    assert "un" in labels
    assert "nn" in labels
    assert "ni" in labels
    assert "ing" in labels
    assert "run" in labels


def test_find_intersection_glass_box():
    addr = find_intersection("thing", "th")
    assert addr is not None
    assert addr.way == 2
    ex = addr.explain()
    assert ex["label"] == "th"
    assert ex["n_lattices"] == 32
    assert "lattice_L01" in ex


def test_same_sum_different_32_lattice_nodes():
    """73+23 and 69+27 both sum to 96 — different nodes on all 32 lattices."""
    from lattice_retriever_v1.stage02_intersections import lattice_signature

    sig_a = lattice_signature((73, 23))
    sig_b = lattice_signature((69, 27))
    assert 73 + 23 == 69 + 27 == 96
    assert sig_a != sig_b
    assert sum(1 for i in range(32) if sig_a[i] == sig_b[i]) == 0
