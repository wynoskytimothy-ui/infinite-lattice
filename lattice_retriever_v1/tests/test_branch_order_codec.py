"""Branch order — lumber + count only."""

from __future__ import annotations

import zlib

from lattice_retriever_v1.branch_order_codec import (
    decode_branch_order,
    encode_branch_order,
)


def test_one_symbol_count_only():
    data = bytes([4]) * 50000
    wire, meta = encode_branch_order(data)
    assert meta["mode"] == "one_symbol_branch"
    assert meta["walker_stored"] == 0
    assert meta["stored_extra_bytes"] == 4
    assert decode_branch_order(wire) == data
    assert len(wire) < len(zlib.compress(data, 9))


def test_wire_is_lumber_plus_count():
    data = bytes([9]) * 1000
    wire, meta = encode_branch_order(data)
    assert meta["total_wire_bytes"] == meta["bare_lumber_bytes"] + 4
