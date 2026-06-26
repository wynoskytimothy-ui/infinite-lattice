"""Unit lattice — bare lumber + procedural infinite space."""

from __future__ import annotations

from lattice_retriever_v1.unit_lattice_codec import (
    LatticeUnit,
    encode_bare_lumber,
    live_roundtrip,
    walk_new_land,
)


def test_digits_unit_space():
    unit = LatticeUnit.digits()
    assert unit.n_tokens == 10
    assert unit.n_origins_procedural == 100
    assert unit.explain()["stored_scaffold_bytes"] == 0


def test_bare_lumber_only_storage():
    data = bytes(list(range(10)) * 1000)
    lumber, wire, fp = encode_bare_lumber(data)
    assert len(wire) == lumber.stored_bytes == 20
    assert fp.formula_stored_bytes == 0
    assert fp.walk_trace_stored_bytes == 0
    assert fp.coord_bytes_if_materialized > 0
    assert fp.explain()["coord_bytes_stored"] == 0


def test_live_walk_roundtrip():
    data = b"the quick brown fox 0123456789"
    assert live_roundtrip(data) == data


def test_walk_discovers_dots_without_storing_coords():
    data = b"ab"
    steps = list(walk_new_land(data))
    assert len(steps) == 1
    assert steps[0].explain()["coords_stored"] == 0
    assert steps[0].explain()["L01"] is not None
