"""Trigger formula — 2-way n + 3-way symbol lock, no coords."""

from __future__ import annotations

import os

from lattice_retriever_v1.trigger_formula_codec import decode_trigger_formula, encode_trigger_formula


def test_random_roundtrip():
    data = bytes(os.urandom(256))
    wire, meta = encode_trigger_formula(data)
    assert decode_trigger_formula(wire) == data
    assert meta["coords_stored"] == 0
    assert meta["origin_stored"] == 0


def test_single_symbol_pure():
    data = bytes([99]) * 5000
    wire, meta = encode_trigger_formula(data)
    assert decode_trigger_formula(wire) == data
    assert meta["mode"] == "trigger_formula_pure"
    assert len(wire) < 20


def test_ten_digit_cycle():
    data = bytes(range(10)) * 50
    wire, meta = encode_trigger_formula(data)
    assert decode_trigger_formula(wire) == data
    assert meta["coords_stored"] == 0
