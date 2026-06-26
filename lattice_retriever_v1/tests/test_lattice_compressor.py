"""Frontier lattice compressor."""

from __future__ import annotations

from lattice_retriever_v1.lattice_compressor import LatticeCompressor, frontier_report


def test_single_symbol_frontier():
    data = bytes([3]) * 5000
    c = LatticeCompressor()
    r = c.compress(data)
    assert c.decompress(r.wire) == data
    assert r.ratio > 100


def test_picks_smallest_path():
    data = bytes([2]) * 8000
    r = LatticeCompressor().compress(data)
    assert r.wire_bytes < len(data)


def test_recompress_uses_session():
    c = LatticeCompressor()
    d1 = bytes([1]) * 1000
    c.compress(d1)
    r2 = c.recompress(bytes([1]) * 2000)
    assert c.decompress(r2.wire) == bytes([1]) * 2000
