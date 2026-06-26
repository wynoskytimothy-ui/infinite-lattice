"""Full potential codec — portable, any data, cold decode."""

from __future__ import annotations

import os
import zlib

from lattice_retriever_v1.full_potential_codec import decode_full_potential, encode_full_potential
from lattice_retriever_v1.lattice_compressor import LatticeCompressor


def test_random_cold_decode():
    data = bytes(os.urandom(4096))
    wire, meta = encode_full_potential(data)
    assert decode_full_potential(wire) == data
    assert meta["coords_stored"] == 0


def test_english_like_low_vocab():
    data = b"the cat sat on the mat " * 200
    wire, meta = encode_full_potential(data)
    assert decode_full_potential(wire) == data
    # promotion + formula should beat raw; zlib also strong on repetition
    assert len(wire) < len(data)


def test_compress_fast_portable():
    data = bytes(range(10)) * 100
    c = LatticeCompressor()
    r = c.compress_fast(data)
    assert LatticeCompressor().decompress(r.wire) == data
