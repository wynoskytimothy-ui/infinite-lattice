"""Walker codec — 2-way walks, 3-way locks symbols."""

from __future__ import annotations

from lattice_retriever_v1.walker_codec import (
    decode_walker,
    encode_walker,
    verify_3way_from_walker,
    walk_2way,
    detect_walker_span,
)


def test_2way_walker_roundtrip():
    data = b"the quick brown"
    wire, meta = encode_walker(data)
    assert decode_walker(wire) == data
    assert meta["witness_stored_bytes"] == 0


def test_3way_locks_symbols_from_walker():
    data = b"hello world"
    v = verify_3way_from_walker(data)
    assert v["roundtrip_2way"]
    assert v["symbols_match_3way"]


def test_single_rail_span_repetitive():
    # same oriented pair every step: one 2-way origin, n=1..EOF
    data = bytes([5] * 500)
    steps = walk_2way(data)
    span = detect_walker_span(steps)
    assert span is not None
    wire, meta = encode_walker(data)
    assert meta["mode"] == "single_rail_span"
    assert meta["walker_body_bytes"] == 6
    assert decode_walker(wire) == data
