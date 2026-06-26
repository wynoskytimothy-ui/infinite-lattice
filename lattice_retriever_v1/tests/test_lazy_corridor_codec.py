"""Lazy corridor — prime function, no stored steps."""

from __future__ import annotations

import zlib

from lattice_retriever_v1.lazy_corridor_codec import (
    decode_lazy_corridor,
    encode_lazy_corridor,
    lazy_read_corridor,
)
from lattice_retriever_v1.unit_lattice_codec import SymbolAlphabet


def test_constant_symbol_lazy():
    data = bytes([7]) * 10000
    wire, meta = encode_lazy_corridor(data)
    assert meta["walker_steps_stored"] == 0
    assert meta["mode"] == "constant"
    assert decode_lazy_corridor(wire) == data
    assert len(wire) < len(zlib.compress(data, 9))


def test_single_rail_lazy():
    data = bytes([3]) * 5000
    wire, meta = encode_lazy_corridor(data)
    assert decode_lazy_corridor(wire) == data
    assert meta["total_wire_bytes"] < 30


def test_lazy_read_no_stored_coords():
    alpha = SymbolAlphabet(symbols=(5,))
    steps = list(
        lazy_read_corridor(
            alpha, mode=0, walker_prime=2, seed_byte=5, n_end=3, origin_id=0
        )
    )
    assert all(s.explain()["coords_stored"] == 0 for s in steps)
