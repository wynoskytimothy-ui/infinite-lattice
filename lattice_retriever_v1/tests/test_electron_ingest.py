"""Electron ingest profiler — coin states and wire vs bare lumber."""

from __future__ import annotations

from lattice_retriever_v1.brain_loop import BrainLoop
from lattice_retriever_v1.electron_ingest import (
    ELECTRON_MAX_ALPHABET,
    electron_ingest_profile,
    should_electron_ingest,
    wire_electron_if_eligible,
)
from lattice_retriever_v1.electron_lattice_codec import decode_electron_entangle


def test_dit4_alphabet_eligible_and_compresses():
    data = bytes([0, 1, 2, 3] * 200)
    assert should_electron_ingest(data)
    prof = electron_ingest_profile(data)
    assert prof["eligible"] is True
    assert prof["n_symbols"] <= ELECTRON_MAX_ALPHABET
    assert len(prof["coin_states"]) == 4
    assert prof["electron_wire_bytes"] is not None
    assert prof["electron_wire_bytes"] < prof["raw_bytes"]
    assert prof["bare_lumber_bytes"] < prof["raw_bytes"]


def test_large_alphabet_not_eligible():
    data = bytes(range(20)) * 10
    assert not should_electron_ingest(data)
    prof = electron_ingest_profile(data)
    assert prof["eligible"] is False
    assert prof["electron_wire_bytes"] is None
    assert prof["small_vocab"] is False


def test_wire_round_trip():
    data = bytes([10, 20, 10, 20] * 50)
    enc = wire_electron_if_eligible(data)
    assert enc is not None
    wire, _meta = enc
    assert decode_electron_entangle(wire) == data


def test_brain_loop_byte_corpus_hook():
    loop = BrainLoop()
    byte_corpus = {"doc_a": bytes([0, 1, 2, 3] * 20)}
    loop.index_corpus({}, byte_corpus=byte_corpus)
    profiles = loop.electron_ingest_profiles()
    assert "doc_a" in profiles
    assert profiles["doc_a"]["eligible"] is True
    assert profiles["doc_a"]["coin_states"]
