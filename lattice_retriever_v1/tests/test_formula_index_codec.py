"""Formula corridor + inverted index tallies (no coordinates)."""

from __future__ import annotations

from lattice_retriever_v1.formula_corridor_read import formula_can_lossless_read
from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex, encode_formula_index, decode_formula_index
from lattice_retriever_v1.lattice_compressor import LatticeCompressor


def test_single_symbol_formula_only():
    data = bytes([7]) * 500
    idx = FormulaWalkIndex()
    wire, meta = encode_formula_index(data, idx)
    assert meta["coords_stored"] == 0
    assert meta["walker_on_wire"] == 0
    assert decode_formula_index(wire, idx) == data


def test_ten_digit_lossless_via_index_tallies():
    data = bytes(range(10)) * 100
    idx = FormulaWalkIndex()
    wire, meta = encode_formula_index(data, idx)
    assert len(wire) < len(data)
    assert meta["formula_pure"] is False
    assert decode_formula_index(wire, idx) == data


def test_compress_fast_no_coords_on_wire():
    data = bytes(range(10)) * 50
    c = LatticeCompressor()
    r = c.compress_fast(data)
    assert c.decompress(r.wire) == data
    assert r.walker_stored == 0
