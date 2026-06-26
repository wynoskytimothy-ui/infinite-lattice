"""
Wing-channel codec — 96-state formula readout encodes order.

Each 2-way dot reads (case × wing) = channel_id from formula.
When channel predicts symbol: store channel stream only (not walker).
Decompress: lazy_read_channel() → symbol from formula branch.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass

from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.stage03_rotation import wing_from_frequency_profile
from aethos_lattice import prime_pair_case
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit

MAGIC = b"WNG1"


@dataclass(frozen=True)
class WingChannelRead:
    """96-state readout per intersection — formula channel, not stored."""

    n: int
    case: int
    wing: int
    channel_id: int

    @property
    def bits_capacity(self) -> float:
        return 6.58


def wing_channel_at(
    alpha: SymbolAlphabet,
    left: int,
    right: int,
    *,
    n: int,
    sym_counts: Counter[int],
) -> WingChannelRead:
    lp, rp = alpha.prime_for(left), alpha.prime_for(right)
    case = prime_pair_case(lp, rp, n)
    profile = (sym_counts[left], sym_counts[right])
    wing = wing_from_frequency_profile(profile)
    return WingChannelRead(n=n, case=case, wing=wing, channel_id=(case - 1) * 32 + wing)


def _bits(n: int) -> int:
    return max(1, math.ceil(math.log2(max(n, 2))))


def _pack_bits(values: list[int], bits: int) -> bytes:
    if not values:
        return b""
    out = bytearray()
    acc = 0
    nacc = 0
    mask = (1 << bits) - 1
    for v in values:
        acc = (acc << bits) | (v & mask)
        nacc += bits
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
            acc &= (1 << nacc) - 1 if nacc else 0
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def _unpack_bits(blob: bytes, n: int, bits: int) -> list[int]:
    out: list[int] = []
    acc = 0
    nacc = 0
    pos = 0
    mask = (1 << bits) - 1
    while len(out) < n:
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
    return out


def _bit_bytes(n_vals: int, bits: int) -> int:
    return (n_vals * bits + 7) // 8


def channel_symbol_stream(data: bytes) -> tuple[list[int], list[int], SymbolAlphabet]:
    alpha = SymbolAlphabet.from_bytes(data)
    sym_index = {s: i for i, s in enumerate(alpha.symbols)}
    counts = Counter(data)
    channels: list[int] = []
    symbols: list[int] = []
    for i in range(len(data) - 1):
        ch = wing_channel_at(alpha, data[i], data[i + 1], n=i + 1, sym_counts=counts)
        channels.append(ch.channel_id)
        symbols.append(sym_index[data[i + 1]])
    return channels, symbols, alpha


def channel_predicts_symbol(alpha: SymbolAlphabet, channel_id: int) -> int:
    return channel_id % alpha.n


def encode_wing_channel(data: bytes) -> tuple[bytes, dict] | None:
    if len(data) < 2:
        return None
    channels, sym_idx, alpha = channel_symbol_stream(data)
    predicted = [channel_predicts_symbol(alpha, c) for c in channels]
    use_pure = all(predicted[i] == sym_idx[i] for i in range(len(sym_idx)))

    unit = LatticeUnit.from_data(data)
    lumber = BareLumber(unit=unit, raw_len=len(data))
    ch_bits = 7
    body = bytearray()
    body.extend(struct.pack("<B", data[0]))
    if use_pure:
        body.append(0)
        body.extend(_pack_bits(channels, ch_bits))
        mode = "wing_pure"
    else:
        body.append(1)
        body.extend(_pack_bits(channels, ch_bits))
        sym_bits = _bits(alpha.n)
        body.extend(_pack_bits(sym_idx, sym_bits))
        mode = "wing_plus_symbol"

    wire = MAGIC + lumber.to_wire()[len(b"LUM1") :] + bytes(body)
    if decode_wing_channel(wire) != data:
        return None
    return wire, {
        "mode": mode,
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "walker_stored": 0,
    }


def decode_wing_channel(wire: bytes) -> bytes:
    off = len(MAGIC)
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    alpha = SymbolAlphabet(symbols=symbols)
    seed = wire[off]
    off += 1
    hybrid = wire[off]
    off += 1
    n_ch = raw_len - 1
    ch_bits = 7
    ch_len = _bit_bytes(n_ch, ch_bits)
    channels = _unpack_bits(wire[off : off + ch_len], n_ch, ch_bits)
    off += ch_len
    if hybrid:
        sym_bits = _bits(alpha.n)
        sym_len = _bit_bytes(n_ch, sym_bits)
        sym_idx = _unpack_bits(wire[off : off + sym_len], n_ch, sym_bits)
    else:
        sym_idx = [channel_predicts_symbol(alpha, c) for c in channels]
    out = bytearray([seed])
    for idx in sym_idx:
        out.append(alpha.symbols[idx])
    return bytes(out)
