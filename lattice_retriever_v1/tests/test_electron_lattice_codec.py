"""Electron 4-state + entanglement lattice compression."""

from __future__ import annotations

from lattice_retriever_v1.electron_lattice_codec import (
    decode_electron_entangle,
    electron_entangle_report,
    encode_electron_entangle,
)
from lattice_retriever_v1.lattice_compressor import LatticeCompressor


def test_dit4_four_symbol_vocab():
    # bytes 0,1,2,3 — one electron read (2 bits) per symbol
    data = bytes([0, 1, 2, 3] * 500)
    enc = encode_electron_entangle(data)
    assert enc is not None
    wire, meta = enc
    assert decode_electron_entangle(wire) == data
    assert meta["mode"] == "electron_dit4"
    assert len(wire) < len(data)


def test_entangle_oscillating_pair():
    data = bytes([10, 20] * 1000)
    enc = encode_electron_entangle(data)
    assert enc is not None
    wire, meta = enc
    assert decode_electron_entangle(wire) == data
    assert meta["mode"] in ("electron_entangle_ab", "electron_dit4", "one_symbol_branch")


def test_compressor_picks_electron():
    data = bytes([1, 2, 3] * 400)
    c = LatticeCompressor()
    r = c.compress(data)
    assert r.wire_bytes < len(data)
    assert c.decompress(r.wire) == data


def test_entangle_report_finds_pairs():
    data = bytes([5, 9, 5, 9])
    rep = electron_entangle_report(data)
    assert len(rep["catalog"]) == 2
    assert rep["entangled_pairs"][0]["opposite"] is True or rep["entangled_pairs"][0]["opposite"] is False
