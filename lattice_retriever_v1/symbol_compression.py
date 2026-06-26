"""
Pure symbol-stream compression — no RAG, no postings, no cages.

Model: map each unique symbol → one prime (formula side); store only:
  1. Alphabet table (unique symbols in corpus)
  2. Bit-packed order stream (symbol indices in read order)

Intersections / 3D placement recompute from the stream via lattice formula —
not stored per dot. Optional unique-pair index size reported separately
(unique bigrams only, not O(corpus) rows).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from aethos_sequences import SequenceKind, make_chain

MAGIC = b"LSC1"


@dataclass
class SymbolAlphabet:
    """Unique symbols → dense index → assigned prime (computed, not stored per token)."""

    symbols: tuple[int, ...]  # raw code units (0-255 bytes or unicode ord)
    index_by_symbol: dict[int, int] = field(init=False)
    prime_by_index: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.index_by_symbol = {s: i for i, s in enumerate(self.symbols)}
        primes = make_chain(SequenceKind.PRIMES, max(len(self.symbols), 1))
        self.prime_by_index = tuple(int(primes[i]) for i in range(len(self.symbols)))

    @classmethod
    def from_stream(cls, data: bytes) -> SymbolAlphabet:
        uniq = tuple(sorted(set(data)))
        return cls(symbols=uniq)

    @property
    def n(self) -> int:
        return len(self.symbols)

    def table_bytes(self) -> int:
        """Alphabet on wire: count + raw symbol bytes."""
        return 2 + len(self.symbols)

    def prime_table_bytes(self) -> int:
        """Primes derivable from index — counted separately for accounting."""
        return 4 * len(self.symbols)


def _bits_per_symbol(n: int) -> int:
    if n <= 1:
        return 1
    return max(1, math.ceil(math.log2(n)))


def pack_indices(indices: list[int], *, bits: int) -> bytes:
    """Pack symbol indices into a bit stream (MSB-first within each byte)."""
    if not indices:
        return b""
    out = bytearray()
    acc = 0
    nacc = 0
    mask = (1 << bits) - 1
    for idx in indices:
        acc = (acc << bits) | (idx & mask)
        nacc += bits
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
            acc &= (1 << nacc) - 1 if nacc else 0
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def unpack_indices(blob: bytes, *, n_symbols: int, n_values: int, bits: int) -> list[int]:
    if n_values == 0:
        return []
    out: list[int] = []
    acc = 0
    nacc = 0
    mask = (1 << bits) - 1
    pos = 0
    while len(out) < n_values and pos <= len(blob):
        if nacc < bits and pos < len(blob):
            acc = (acc << 8) | blob[pos]
            nacc += 8
            pos += 1
            continue
        if nacc < bits:
            break
        nacc -= bits
        out.append((acc >> nacc) & mask)
        acc &= (1 << nacc) - 1 if nacc else 0
    if len(out) != n_values:
        raise ValueError(f"unpack short: got {len(out)} want {n_values}")
    return out


@dataclass(frozen=True)
class CompressionLedger:
    raw_bytes: int
    n_symbols: int
    n_unique_pairs: int
    alphabet_bytes: int
    order_stream_bytes: int
    header_bytes: int
    prime_table_bytes: int
    unique_pair_index_bytes: int
    total_symbol_codec_bytes: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.total_symbol_codec_bytes if self.total_symbol_codec_bytes else 0.0

    @property
    def savings_pct(self) -> float:
        return 100.0 * (1.0 - self.total_symbol_codec_bytes / self.raw_bytes) if self.raw_bytes else 0.0

    def extrapolate(self, target_raw_bytes: int) -> CompressionLedger:
        if self.raw_bytes == 0:
            return self
        scale = target_raw_bytes / self.raw_bytes
        return CompressionLedger(
            raw_bytes=target_raw_bytes,
            n_symbols=self.n_symbols,
            n_unique_pairs=self.n_unique_pairs,
            alphabet_bytes=self.alphabet_bytes,
            order_stream_bytes=int(self.order_stream_bytes * scale),
            header_bytes=self.header_bytes,
            prime_table_bytes=self.prime_table_bytes,
            unique_pair_index_bytes=self.unique_pair_index_bytes,
            total_symbol_codec_bytes=int(
                self.alphabet_bytes
                + self.header_bytes
                + self.order_stream_bytes * scale
            ),
        )

    def explain(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "raw_gb": round(self.raw_bytes / 1e9, 4),
            "n_unique_symbols": self.n_symbols,
            "n_unique_bigrams": self.n_unique_pairs,
            "alphabet_bytes": self.alphabet_bytes,
            "order_stream_bytes": self.order_stream_bytes,
            "header_bytes": self.header_bytes,
            "prime_table_bytes_formula_side": self.prime_table_bytes,
            "unique_pair_index_bytes": self.unique_pair_index_bytes,
            "total_symbol_codec_bytes": self.total_symbol_codec_bytes,
            "total_symbol_codec_gb": round(self.total_symbol_codec_bytes / 1e9, 4),
            "compression_ratio_x": round(self.ratio, 3),
            "savings_pct": round(self.savings_pct, 2),
        }


def unique_bigrams(data: bytes) -> int:
    if len(data) < 2:
        return 0
    return len({(data[i], data[i + 1]) for i in range(len(data) - 1)})


def encode_bytes(data: bytes) -> tuple[bytes, CompressionLedger]:
    """Encode raw bytes → alphabet + packed order stream (lossless)."""
    alpha = SymbolAlphabet.from_stream(data)
    bits = _bits_per_symbol(alpha.n)
    indices = [alpha.index_by_symbol[b] for b in data]
    packed = pack_indices(indices, bits=bits)
    header = struct.pack("<IHB", len(data), alpha.n, bits)
    payload = MAGIC + header + bytes(alpha.symbols) + packed
    n_pairs = unique_bigrams(data)
    ledger = CompressionLedger(
        raw_bytes=len(data),
        n_symbols=alpha.n,
        n_unique_pairs=n_pairs,
        alphabet_bytes=alpha.table_bytes(),
        order_stream_bytes=len(packed),
        header_bytes=len(MAGIC) + len(header),
        prime_table_bytes=alpha.prime_table_bytes(),
        unique_pair_index_bytes=n_pairs * 8,
        total_symbol_codec_bytes=len(payload),
    )
    return payload, ledger


def decode_bytes(payload: bytes) -> bytes:
    """Decode symbol codec → original bytes (lossless roundtrip)."""
    if not payload.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    n_raw, n_sym, bits = struct.unpack_from("<IHB", payload, off)
    off += 7
    symbols = payload[off : off + n_sym]
    off += n_sym
    alpha = SymbolAlphabet(symbols=tuple(symbols))
    indices = unpack_indices(payload[off:], n_symbols=n_sym, n_values=n_raw, bits=bits)
    return bytes(alpha.symbols[i] for i in indices)


def analyze_stream(data: bytes) -> CompressionLedger:
    _, ledger = encode_bytes(data)
    return ledger
