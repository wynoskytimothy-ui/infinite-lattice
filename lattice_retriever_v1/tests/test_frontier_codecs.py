"""Wing channel + deep branch + session frontier codecs."""

from __future__ import annotations

from lattice_retriever_v1.deep_branch_codec import decode_deep_branch, encode_deep_branch
from lattice_retriever_v1.lattice_compressor import LatticeCompressor
from lattice_retriever_v1.wing_channel_codec import decode_wing_channel, encode_wing_channel


def test_wing_channel_single_symbol():
    data = bytes([7]) * 500
    enc = encode_wing_channel(data)
    assert enc is not None
    wire, meta = enc
    assert decode_wing_channel(wire) == data
    assert meta["walker_stored"] == 0


def test_deep_branch_two_symbol():
    data = bytes([1, 2]) * 200
    enc = encode_deep_branch(data)
    assert enc is not None
    wire, meta = enc
    assert decode_deep_branch(wire) == data
    assert meta["n_cases"] == len(data) - 2


def test_session_recompress_count_only():
    c = LatticeCompressor()
    vocab = bytes([4]) * 1000
    c.compress(vocab)
    r = c.recompress(bytes([4]) * 5000)
    assert c.decompress(r.wire) == bytes([4]) * 5000
    assert r.mode == "session_count"
    assert r.wire_bytes == 9
