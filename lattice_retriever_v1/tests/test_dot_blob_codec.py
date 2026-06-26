"""Dot-blob storage — token catalog + dots; formula reconstructs."""

from __future__ import annotations

import random

from lattice_retriever_v1.dot_blob_codec import (
    bare_lumber_report,
    compress_to_dot_blob,
    reconstruct_from_dot_blob,
    formula_regenerate_dot,
)


def test_roundtrip():
    for data in (b"hello", b"the quick brown", b"the the", b"a", b""):
        _, _, wire = compress_to_dot_blob(data)
        assert reconstruct_from_dot_blob(wire) == data


def test_formula_not_stored():
    data = b"test data here"
    _, ledger, wire = compress_to_dot_blob(data)
    assert ledger.formula_stored_bytes == 0
    assert ledger.coord_bytes_if_stored > 0
    assert b"prime_pair" not in wire
    assert b"lattice" not in wire


def test_random_vs_repetitive_same_bare_lumber():
    rng = random.Random(0)
    hot = list(range(10))

    def rep(n):
        return bytes(rng.choice(hot) for _ in range(n))

    def rnd(n):
        return bytes(rng.randint(0, 9) for _ in range(n))

    r_rep = bare_lumber_report(rep(50_000))
    r_rnd = bare_lumber_report(rnd(50_000))
    assert r_rep["bare_lumber_bytes"] == r_rnd["bare_lumber_bytes"]
    assert r_rep["n_tokens"] == r_rnd["n_tokens"] == 10
    assert r_rep["lumber_ratio_x"] > 1000
    assert r_rnd["lumber_ratio_x"] > 1000


def test_formula_reconstructs_dot_from_blob():
    data = b"ab"
    blob, _, _ = compress_to_dot_blob(data)
    assert blob.dots
    oid, pn = blob.dots[0]
    info = formula_regenerate_dot(blob, oid, pn)
    assert "lattice_L01" in info
    assert info["pair_n"] == pn
