"""Tests — pair-origin vectors: each 2-way meet has its own n rail (1,2,3…)."""

from __future__ import annotations

from lattice_retriever_v1.intersection_dot_codec import (
    ProceduralLatticeIndex,
    decode_pair_vectors,
    document_pair_walk,
    encode_pair_vectors,
    pair_origin_vectors,
    procedural_index_ledger,
    read_document_from_walk,
    regenerate_dot_from_formula,
    SymbolAlphabet,
)


def test_each_pair_has_own_origin_n_starts_at_one():
    data = b"the"
    walk = document_pair_walk(data)
    assert len(walk) == 2
    th, he = walk[0], walk[1]
    assert th.pair_n == 1
    assert he.pair_n == 1
    assert th.origin.left_byte == ord("t")
    assert he.origin.left_byte == ord("h")
    assert read_document_from_walk(walk) == data


def test_repeat_pair_transgresses_on_same_origin():
    data = b"the the"
    walk = document_pair_walk(data)
    vectors = pair_origin_vectors(walk)
    th_vec = next(v for v in vectors if v.origin.left_byte == ord("t"))
    he_vec = next(v for v in vectors if v.origin.left_byte == ord("h"))
    assert th_vec.n_sequence == (1, 2)
    assert he_vec.n_sequence == (1, 2)


def test_formula_regenerates_from_pair_n():
    data = b"ab"
    alpha = SymbolAlphabet.from_bytes(data)
    d = document_pair_walk(data, alpha)[0]
    d2 = regenerate_dot_from_formula(d.origin, pair_n=d.pair_n, walk_index=d.walk_index)
    assert d2.address.lattice_coords == d.address.lattice_coords


def test_encode_decode_roundtrip():
    for data in (b"hello", b"the quick brown", b"the the", b"a", b""):
        payload, _, _, _ = encode_pair_vectors(data)
        assert decode_pair_vectors(payload) == data


def test_reversed_pair_different_origins():
    payload_ab, _, _, _ = encode_pair_vectors(b"ab")
    payload_ba, _, _, _ = encode_pair_vectors(b"ba")
    assert payload_ab != payload_ba


def test_same_sum_different_meets():
    from lattice_retriever_v1.stage02_intersections import intersect_primes

    a = intersect_primes(("a", "b"), (73, 23), n=1, start_index=0)
    b = intersect_primes(("c", "d"), (69, 27), n=1, start_index=0)
    assert a.anchor_sum == b.anchor_sum == 96
    assert a.lattice_coords != b.lattice_coords


def test_procedural_index_same_size_any_corpus():
    """Index = alphabet only; any walk length — same stored bytes."""
    import random

    alpha = SymbolAlphabet(symbols=tuple(range(100)))
    idx = ProceduralLatticeIndex(alpha)
    rng = random.Random(0)
    hot = list(range(10))

    def walk(n):
        data = bytes(rng.choice(hot) if rng.random() < 0.85 else rng.randint(0, 99) for _ in range(n))
        dots = idx.place_walk(data)
        return len(dots)

    n1, n2 = walk(1_000), walk(50_000)
    assert idx.stored_bytes == 102
    assert idx.n_pair_origins == 10_000
    assert n1 != n2


def test_branch_one_symbol_opens_new_origins():
    idx = ProceduralLatticeIndex(SymbolAlphabet(symbols=tuple(range(100))))
    assert idx.n_pair_origins == 10_000
    _, new_origins = idx.branch_symbol(200)
    assert idx.n_symbols == 101
    assert idx.n_pair_origins == 101 * 101
    assert new_origins == 101 * 101 - 10_000
    assert idx.stored_bytes == 101 + 2
