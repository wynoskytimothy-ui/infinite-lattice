"""Prime corridor — main spine + P* rails, 3-way branch case witness."""

from __future__ import annotations

import os

from lattice_retriever_v1.prime_corridor_codec import (
    corridor_triplet,
    decode_prime_corridor,
    encode_prime_corridor,
    spine_triplet,
)


def test_spine_and_corridor_tables():
    assert spine_triplet(0) == (0, 0, 0)
    assert spine_triplet(3) == (3, 0, 3)
    assert corridor_triplet(2, 1) == (3, 1, 3)
    assert corridor_triplet(2, 2) == (4, 2, 4)
    assert corridor_triplet(3, 3) == (6, 3, 6)


def test_single_symbol_pure():
    data = bytes([42]) * 2000
    wire, meta = encode_prime_corridor(data)
    assert decode_prime_corridor(wire) == data
    assert meta["mode"] == "prime_corridor_pure"
    assert len(wire) < 24


def test_ten_digit_cycle():
    data = bytes(range(10)) * 100
    wire, meta = encode_prime_corridor(data)
    assert decode_prime_corridor(wire) == data
    assert meta["coords_stored"] == 0
    assert meta["origin_stored"] == 0


def test_random_roundtrip():
    data = bytes(os.urandom(512))
    wire, meta = encode_prime_corridor(data)
    assert decode_prime_corridor(wire) == data
    assert meta["walker_stored"] == 0
