"""
Branch-order codec — store lumber + symbol count only; formula recomputes the walk.

User model:
  - Two primes branch; each branch is a symbol.
  - 2-way vector holds ALL order — no walker stored.
  - Decompress: formula branches, reads which symbol at each n.
  - Store only: bare lumber + count (symbols till EOF).

When vocabulary collapses to one symbol (or branch-determined order),
the entire file is lumber + 4 bytes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from aethos_lattice import prime_pair_case

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginKey,
    SymbolAlphabet,
    _oriented_pair_catalog,
    document_pair_walk,
    dot_on_origin,
    read_document_from_walk,
)
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit

MAGIC = b"BRN1"
FORMULA_INDEX_MAGIC = b"IDX1"


@dataclass(frozen=True)
class BranchOrderWire:
    symbol_count: int
    lumber: BareLumber

    @property
    def stored_bytes(self) -> int:
        return self.lumber.stored_bytes + 4


def _two_branch_symbols(alpha: SymbolAlphabet) -> tuple[int, int]:
    """First two symbols in catalog — the two primes that branch."""
    syms = alpha.symbols
    if len(syms) < 2:
        return syms[0], syms[0]
    return syms[0], syms[1]


def branch_read_symbol_at_n(
    alpha: SymbolAlphabet,
    sym_a: int,
    sym_b: int,
    n: int,
) -> int:
    """
    2-way branch between two symbol-primes at transgressor n.
    Case on rail picks which branch (symbol) fired — order from formula.
    """
    pa, pb = alpha.prime_for(sym_a), alpha.prime_for(sym_b)
    case = prime_pair_case(pa, pb, n)
    if case == 1:
        return sym_a
    if case == 2:
        return sym_b
    return sym_b if n % 2 == 0 else sym_a


def lazy_reconstruct_branch_order(alpha: SymbolAlphabet, count: int) -> bytes:
    """
    Reconstruct symbol stream from lumber + count only — formula branch walk.

    - 1 symbol in lumber: repeat count times.
    - 2+ symbols: 2-way vector between branch pair + n=1..count reads order.
    """
    syms = alpha.symbols
    if count <= 0:
        return b""
    if len(syms) == 1:
        return bytes([syms[0]]) * count

    sym_a, sym_b = _two_branch_symbols(alpha)
    if count == 1:
        return bytes([sym_a])

    # branch 1 fires first; branch 2 walks forward on the 2-way rail
    out = bytearray([sym_a])
    current = sym_a
    counters: dict[tuple[int, int], int] = {}
    for wi in range(count - 1):
        nxt = sym_b if current == sym_a else sym_a
        # refine: read from 2-way case at rail position wi+1
        rail_n = wi + 1
        branch_pick = branch_read_symbol_at_n(alpha, sym_a, sym_b, rail_n)
        if current == sym_a:
            nxt = branch_pick if branch_pick != sym_a else sym_b
        else:
            nxt = branch_pick if branch_pick != sym_b else sym_a
        key = (current, nxt)
        counters[key] = counters.get(key, 0) + 1
        out.append(nxt)
        current = nxt
    return bytes(out)


def reconstruct_from_2way_vector(alpha: SymbolAlphabet, data: bytes) -> bytes:
    """
    Full lossless read: 2-way vector (origin, pair_n) at each step IS the order.
    Used to VERIFY branch-only storage — not stored on wire.
    """
    if len(data) < 2:
        return data
    walk = document_pair_walk(data, alpha)
    dots = tuple(
        dot_on_origin(d.origin, pair_n=d.pair_n, walk_index=d.walk_index) for d in walk
    )
    return read_document_from_walk(dots)


def encode_branch_order(
    data: bytes,
    *,
    walk_index: "FormulaWalkIndex | None" = None,
) -> tuple[bytes, dict]:
    """
    Store lumber + symbol count only when formula branch reproduces order.
    Multi-symbol: walk goes to inverted index (walk_index), not wire.
    """
    from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex, encode_formula_index

    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    count = len(data)

    # single-symbol: always branch-only
    if len(alpha.symbols) == 1:
        wire = _pack(lumber, count)
        return wire, _meta(data, lumber, count, mode="one_symbol_branch")

    # two-symbol: formula branch when order is branch-determined
    if len(alpha.symbols) == 2:
        trial = lazy_reconstruct_branch_order(alpha, count)
        if trial == data:
            wire = _pack(lumber, count)
            return wire, _meta(data, lumber, count, mode="two_branch_formula")

    # multi-symbol: lumber + count; tallies in inverted index session if needed
    idx = walk_index if walk_index is not None else FormulaWalkIndex()
    return encode_formula_index(data, idx)


def decode_branch_order(
    wire: bytes,
    *,
    walk_index: "FormulaWalkIndex | None" = None,
) -> bytes:
    if wire.startswith(FORMULA_INDEX_MAGIC):
        from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex, decode_formula_index

        idx = walk_index if walk_index is not None else FormulaWalkIndex()
        return decode_formula_index(wire, idx)
    if not wire.startswith(MAGIC):
        from lattice_retriever_v1.lazy_corridor_codec import decode_lazy_corridor

        return decode_lazy_corridor(wire)
    lumber, count = _unpack(wire)
    alpha = lumber.unit.alphabet
    if len(alpha.symbols) == 1:
        return bytes([alpha.symbols[0]]) * count
    if len(alpha.symbols) == 2:
        return lazy_reconstruct_branch_order(alpha, count)
    from lattice_retriever_v1.formula_corridor_read import formula_corridor_read

    out = formula_corridor_read(alpha, count)
    if out is not None:
        return out
    if walk_index is not None:
        return decode_formula_index(FORMULA_INDEX_MAGIC + wire[len(MAGIC) :], walk_index)
    raise ValueError("formula corridor read failed — no coords or walk on wire")


def _pack(lumber: BareLumber, count: int) -> bytes:
    return MAGIC + lumber.to_wire()[len(b"LUM1") :] + struct.pack("<I", count)


def _unpack(wire: bytes) -> tuple[BareLumber, int]:
    off = len(MAGIC)
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    (count,) = struct.unpack_from("<I", wire, off)
    unit = LatticeUnit(symbols=symbols)
    return BareLumber(unit=unit, raw_len=raw_len), count


def _meta(data: bytes, lumber: BareLumber, count: int, *, mode: str) -> dict:
    return {
        "mode": mode,
        "raw_bytes": len(data),
        "symbol_count": count,
        "bare_lumber_bytes": lumber.stored_bytes,
        "stored_extra_bytes": 4,
        "total_wire_bytes": lumber.stored_bytes + 4,
        "walker_stored": 0,
        "ratio_x": round(len(data) / (lumber.stored_bytes + 4), 3),
    }
