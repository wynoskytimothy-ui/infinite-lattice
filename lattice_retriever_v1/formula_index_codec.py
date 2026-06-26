"""
Symbol lumber session — inverted index = rail tallies only (never coordinates).

Wire: lumber + count.
Session: which (pair-origin, n) fired — formula postings, not coords.
Decode: formula corridor read first; else replay tallies through formula.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from lattice_retriever_v1.formula_corridor_read import formula_can_lossless_read, formula_corridor_read
from lattice_retriever_v1.intersection_dot_codec import (
    _oriented_pair_catalog,
    document_pair_walk,
    read_document_from_walk,
    regenerate_dot_from_formula,
)
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit

MAGIC = b"IDX1"


@dataclass
class FormulaWalkIndex:
    """
    Inverted index — pair-origin rail tallies at ingest (0 coordinate bytes).

    Dots regenerate from formula on read; tallies say which n fired per step.
    """

    unit: LatticeUnit | None = None
    _tallies: tuple[tuple[int, int], ...] | None = None
    _symbols: tuple[int, ...] | None = None
    _count: int = 0

    def ingest(self, data: bytes) -> None:
        self.unit = LatticeUnit.from_data(data)
        self._symbols = self.unit.symbols
        self._count = len(data)
        if len(data) < 2:
            self._tallies = ()
            return
        alpha = self.unit.alphabet
        catalog = _oriented_pair_catalog(alpha)
        cat_index = {k: i for i, k in enumerate(catalog)}
        walk = document_pair_walk(data, alpha)
        self._tallies = tuple((cat_index[d.origin], d.pair_n) for d in walk)

    def replay_tallies(self, lumber: BareLumber, count: int) -> bytes:
        """Formula regenerates vectors; tallies name which nodes triggered."""
        alpha = lumber.unit.alphabet
        if count <= 0:
            return b""
        if count == 1:
            return bytes([alpha.symbols[0]])
        if self._tallies is None or self._symbols != alpha.symbols or self._count != count:
            raise ValueError("inverted index tally mismatch — re-ingest")
        catalog = _oriented_pair_catalog(alpha)
        dots = tuple(
            regenerate_dot_from_formula(catalog[oid], pair_n=pn, walk_index=wi)
            for wi, (oid, pn) in enumerate(self._tallies)
        )
        return read_document_from_walk(dots)


def encode_formula_index(data: bytes, index: FormulaWalkIndex) -> tuple[bytes, dict]:
    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    count = len(data)
    pure = formula_can_lossless_read(alpha, data)
    if not pure:
        index.ingest(data)
    else:
        index.ingest(data)
        index._tallies = None
    wire_body = lumber.to_wire()[len(b"LUM1") :] + struct.pack("<I", count)
    wire = MAGIC + wire_body
    back = decode_formula_index(wire, index)
    if back != data:
        raise ValueError("formula roundtrip failed")
    mode = "formula_pure_count" if pure else "formula_index_count"
    return wire, {
        "mode": mode,
        "raw_bytes": len(data),
        "symbol_count": count,
        "n_symbols": unit.n_tokens,
        "bare_lumber_bytes": lumber.stored_bytes,
        "stored_extra_bytes": 4,
        "total_wire_bytes": len(wire),
        "walker_stored": 0,
        "coords_stored": 0,
        "walker_on_wire": 0,
        "formula_pure": pure,
        "ratio_x": round(len(data) / len(wire), 3),
    }


def decode_formula_index(wire: bytes, index: FormulaWalkIndex | None = None) -> bytes:
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    (count,) = struct.unpack_from("<I", wire, off)
    unit = LatticeUnit(symbols=symbols)
    alpha = unit.alphabet
    trial = formula_corridor_read(alpha, count)
    if trial is not None and len(trial) == raw_len:
        return trial
    if index is None:
        raise ValueError("need inverted index session for lossless read")
    lumber = BareLumber(unit=unit, raw_len=raw_len)
    return index.replay_tallies(lumber, count)
