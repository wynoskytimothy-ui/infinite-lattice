"""
Electron ingest profiler — wire electron tokenizer when alphabet is tiny.

When unique symbols ≤ 4 (dit4 path), profile coin states and compare
electron wire size against bare lumber. Standalone API or optional hook
from brain_loop.index_corpus(byte_corpus=...).
"""

from __future__ import annotations

from lattice_retriever_v1.electron_lattice_codec import (
    build_electron_alphabet,
    encode_electron_entangle,
)
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit

ELECTRON_MAX_ALPHABET = 4
SMALL_VOCAB_THRESHOLD = 16


def unique_symbol_count(data: bytes) -> int:
    return len(set(data))


def should_electron_ingest(data: bytes) -> bool:
    """True when dit4 electron path applies (≤4 unique byte symbols)."""
    return 0 < unique_symbol_count(data) <= ELECTRON_MAX_ALPHABET


def wire_electron_if_eligible(data: bytes) -> tuple[bytes, dict] | None:
    """Encode with electron_lattice_codec when alphabet is tiny; else None."""
    if not should_electron_ingest(data):
        return None
    return encode_electron_entangle(data)


def electron_ingest_profile(data: bytes) -> dict:
    """
    Glass-box ingest profile: coin states, bare lumber vs electron wire.

    Reports compression only when encode_electron_entangle succeeds.
    """
    if not data:
        return {
            "empty": True,
            "eligible": False,
            "n_symbols": 0,
            "raw_bytes": 0,
            "bare_lumber_bytes": 0,
            "coin_states": [],
        }

    unit = LatticeUnit.from_data(data)
    lumber = BareLumber(unit=unit, raw_len=len(data))
    lumber_wire = lumber.to_wire()
    catalog = build_electron_alphabet(data)
    coin_states = [
        {"byte": s.byte, "coin": s.coin.name, "bits": list(s.bits)}
        for s in catalog
    ]

    eligible = should_electron_ingest(data)
    enc = wire_electron_if_eligible(data)

    out: dict = {
        "n_symbols": unit.n_tokens,
        "raw_bytes": len(data),
        "bare_lumber_bytes": len(lumber_wire),
        "eligible": eligible,
        "small_vocab": unit.n_tokens <= SMALL_VOCAB_THRESHOLD,
        "coin_states": coin_states,
    }

    if enc is not None:
        wire, meta = enc
        out["electron_wire_bytes"] = len(wire)
        out["electron_mode"] = meta["mode"]
        out["ratio_vs_raw"] = meta["ratio_x"]
        out["ratio_vs_lumber"] = (
            round(len(lumber_wire) / len(wire), 3) if len(wire) else None
        )
        out["compressed"] = len(wire) < len(data)
    else:
        out["electron_wire_bytes"] = None
        out["electron_mode"] = None
        out["compressed"] = False

    return out
