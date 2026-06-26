"""
Dot-blob storage — structure compressed to bare lumber (not pattern matching).

We are not compressing repetition (zlib-style patterns). We strip raw data to:
  - **Bare lumber** — unique symbol catalog (e.g. 200 tokens for 200 symbols).
  - **Dot addresses** — (pair_origin, n) placement on rails; coords are formula-side.
  - **Scaffold** — n_tokens² pair origins exist procedurally; never stored.

Random or repetitive: same symbol set → same lumber. Formula rebuilds the full
structure (32-lattice coords + byte stream) from the dot blob on read.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginDot,
    PairOriginKey,
    SymbolAlphabet,
    _oriented_pair_catalog,
    _pack_pairs,
    _unpack_pairs,
    _bits_per_index,
    document_pair_walk,
    read_document_from_walk,
    regenerate_dot_from_formula,
)

MAGIC = b"LDOT"


@dataclass(frozen=True)
class DotBlob:
    """Compressed storage artifact — tokens + dot addresses, zero formula."""

    alphabet: SymbolAlphabet
    dots: tuple[tuple[int, int], ...]  # (origin_id, pair_n)
    raw_len: int

    @property
    def n_tokens(self) -> int:
        return self.alphabet.n

    @property
    def n_dots(self) -> int:
        return len(self.dots)


@dataclass(frozen=True)
class DotBlobLedger:
    raw_bytes: int
    n_tokens: int
    n_dots: int
    n_origins_max: int
    n_origins_used: int
    token_catalog_bytes: int
    dot_blob_bytes: int
    formula_stored_bytes: int
    coord_bytes_if_stored: int
    total_stored_bytes: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.total_stored_bytes if self.total_stored_bytes else 0.0

    @property
    def bare_lumber_bytes(self) -> int:
        """Irreducible structure: unique symbol catalog only (no patterns, no coords)."""
        return self.token_catalog_bytes

    @property
    def lumber_ratio(self) -> float:
        """How far raw data collapses to bare lumber (symbol count)."""
        return self.raw_bytes / self.bare_lumber_bytes if self.bare_lumber_bytes else 0.0

    def explain(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "n_tokens": self.n_tokens,
            "n_dots": self.n_dots,
            "n_origins_max": self.n_origins_max,
            "n_origins_used": self.n_origins_used,
            "bare_lumber_bytes": self.bare_lumber_bytes,
            "dot_blob_bytes": self.dot_blob_bytes,
            "token_catalog_bytes": self.token_catalog_bytes,
            "formula_stored_bytes": self.formula_stored_bytes,
            "coord_bytes_if_stored": self.coord_bytes_if_stored,
            "total_stored_bytes": self.total_stored_bytes,
            "compression_ratio_x": round(self.ratio, 3),
            "lumber_ratio_x": round(self.lumber_ratio, 3),
            "model": "bare lumber (tokens) + dot addresses; structure not patterns",
        }


def compress_to_dot_blob(data: bytes) -> tuple[DotBlob, DotBlobLedger, bytes]:
    """Compress raw data → dot blob wire bytes (no formula stored)."""
    alpha = SymbolAlphabet.from_bytes(data)
    walk = document_pair_walk(data, alpha)
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    dots = tuple((cat_index[d.origin], d.pair_n) for d in walk)
    blob = DotBlob(alphabet=alpha, dots=dots, raw_len=len(data))
    max_n = max((n for _, n in dots), default=1)
    wire = _encode_wire(blob, data=data, catalog_size=len(catalog), max_n=max_n)
    coords_if = len(walk) * 32 * 3 * 4
    origins_used = len({d.origin for d in walk})
    token_bytes = 2 + alpha.n
    dot_bytes = len(wire) - len(MAGIC) - 14 - token_bytes - (1 if len(data) <= 1 else 0)
    ledger = DotBlobLedger(
        raw_bytes=len(data),
        n_tokens=alpha.n,
        n_dots=len(dots),
        n_origins_max=len(catalog),
        n_origins_used=origins_used,
        token_catalog_bytes=token_bytes,
        dot_blob_bytes=max(0, dot_bytes),
        formula_stored_bytes=0,
        coord_bytes_if_stored=coords_if,
        total_stored_bytes=len(wire),
    )
    return blob, ledger, wire


def reconstruct_from_dot_blob(wire: bytes) -> bytes:
    """Read dot blob → formula regenerates symbols (no stored coords/formula)."""
    blob, pair_bits, n_bits = _decode_wire(wire)
    if blob.raw_len <= 1:
        off = len(wire) - blob.raw_len
        return wire[off:] if blob.raw_len else b""
    catalog = _oriented_pair_catalog(blob.alphabet)
    walk: list[PairOriginDot] = []
    for wi, (oid, pn) in enumerate(blob.dots):
        walk.append(regenerate_dot_from_formula(catalog[oid], pair_n=pn, walk_index=wi))
    return read_document_from_walk(tuple(walk))


def reconstruct_from_blob(blob: DotBlob) -> bytes:
    """Reconstruct from in-memory DotBlob."""
    catalog = _oriented_pair_catalog(blob.alphabet)
    if blob.raw_len <= 1:
        return bytes(blob.alphabet.symbols[:1]) if blob.raw_len == 1 else b""
    walk = tuple(
        regenerate_dot_from_formula(catalog[oid], pair_n=pn, walk_index=wi)
        for wi, (oid, pn) in enumerate(blob.dots)
    )
    return read_document_from_walk(walk)


def formula_regenerate_dot(
    blob: DotBlob,
    origin_id: int,
    pair_n: int,
) -> dict:
    """Glass-box: one dot from blob address → full lattice coords via formula."""
    catalog = _oriented_pair_catalog(blob.alphabet)
    dot = regenerate_dot_from_formula(catalog[origin_id], pair_n=pair_n, walk_index=pair_n - 1)
    return dot.explain()


def _encode_wire(blob: DotBlob, *, data: bytes, catalog_size: int, max_n: int) -> bytes:
    pair_bits = _bits_per_index(catalog_size)
    n_bits = max(1, math.ceil(math.log2(max_n + 1)))
    packed = _pack_pairs(list(blob.dots), pair_bits=pair_bits, n_bits=n_bits)
    header = struct.pack("<IIHBB", blob.raw_len, len(blob.dots), blob.alphabet.n, pair_bits, n_bits)
    tail = data if len(data) <= 1 else b""
    return MAGIC + header + bytes(blob.alphabet.symbols) + packed + tail


def bare_lumber_report(data: bytes) -> dict:
    """Show structure collapse: random or not → same lumber for same symbol set."""
    _, ledger, _ = compress_to_dot_blob(data)
    return ledger.explain()


def _decode_wire(wire: bytes) -> tuple[DotBlob, int, int]:
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    n_raw, n_dots, n_sym, pair_bits, n_bits = struct.unpack_from("<IIHBB", wire, off)
    off += 12
    symbols = wire[off : off + n_sym]
    off += n_sym
    alpha = SymbolAlphabet(symbols=tuple(symbols))
    if n_raw <= 1:
        return DotBlob(alphabet=alpha, dots=(), raw_len=n_raw), pair_bits, n_bits
    dots = tuple(_unpack_pairs(wire[off:], n_values=n_dots, pair_bits=pair_bits, n_bits=n_bits))
    return DotBlob(alphabet=alpha, dots=dots, raw_len=n_raw), pair_bits, n_bits
