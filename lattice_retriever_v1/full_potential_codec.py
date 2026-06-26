"""
Full-potential lattice compress — all formula layers, portable wire, zero coords.

Layers tried (smallest lossless wins):
  1. formula_pure     — lumber + count; corridor formula reads order
  2. two_branch       — lumber + count; 2-prime branch formula
  3. wing_channel     — lumber + seed + 7-bit channel stream per step
  4. channel_residual — channels + sparse symbol locks only where ambiguous
  5. symbol_rail      — lumber + electron base-4 rail read (ceil log4 V per symbol)

Never stores coordinates. Never stores origin+pair walker. Cold decode from wire alone.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from dataclasses import dataclass

from lattice_retriever_v1.branch_order_codec import (
    lazy_reconstruct_branch_order,
    _two_branch_symbols,
)
from lattice_retriever_v1.formula_corridor_read import formula_can_lossless_read, formula_corridor_read
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit
from lattice_retriever_v1.wing_channel_codec import (
    _bit_bytes,
    _pack_bits,
    _unpack_bits,
    wing_channel_at,
)

MAGIC = b"FUL1"
MODE_PURE = 0
MODE_BRANCH2 = 1
MODE_WING = 2
MODE_CHANNEL_RESID = 3
MODE_SYMBOL_RAIL = 4
MODE_PROMOTED = 5


def _sym_bits(n: int) -> int:
    return max(1, math.ceil(math.log2(max(n, 2))))


def _lumber_body(lumber: BareLumber) -> bytes:
    return lumber.to_wire()[len(b"LUM1") :]


def _parse_lumber(wire: bytes, off: int) -> tuple[BareLumber, int]:
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    return BareLumber(unit=LatticeUnit(symbols=symbols), raw_len=raw_len), off


def _channel_candidates(
    alpha: SymbolAlphabet,
    left: int,
    ch_id: int,
    *,
    n: int,
    counts: Counter[int],
) -> list[int]:
    return [
        s
        for s in alpha.symbols
        if wing_channel_at(alpha, left, s, n=n, sym_counts=counts).channel_id == ch_id
    ]


def _encode_channel_residual(data: bytes, lumber: BareLumber) -> bytes:
    alpha = lumber.unit.alphabet
    sym_index = {s: i for i, s in enumerate(alpha.symbols)}
    counts = Counter(data)
    channels: list[int] = []
    resid_steps: list[int] = []
    resid_idx: list[int] = []
    for i in range(len(data) - 1):
        left, right = data[i], data[i + 1]
        ch = wing_channel_at(alpha, left, right, n=i + 1, sym_counts=counts).channel_id
        channels.append(ch)
        cands = _channel_candidates(alpha, left, ch, n=i + 1, counts=counts)
        if len(cands) != 1:
            resid_steps.append(i)
            resid_idx.append(sym_index[right])
    body = bytearray([data[0]])
    body.extend(struct.pack("<H", len(resid_steps)))
    body.extend(_pack_bits(channels, 7))
    if resid_steps:
        sb = _sym_bits(alpha.n)
        body.extend(_pack_bits(resid_idx, sb))
        step_bits = max(1, math.ceil(math.log2(max(len(data), 2))))
        body.extend(_pack_bits(resid_steps, step_bits))
    return bytes(body)


def _decode_channel_residual(
    alpha: SymbolAlphabet, count: int, body: bytes
) -> bytes:
    if count <= 0:
        return b""
    off = 0
    seed = body[off]
    off += 1
    (n_resid,) = struct.unpack_from("<H", body, off)
    off += 2
    n_ch = max(0, count - 1)
    ch_len = _bit_bytes(n_ch, 7)
    channels = _unpack_bits(body[off : off + ch_len], n_ch, 7)
    off += ch_len
    resid_map: dict[int, int] = {}
    if n_resid:
        sb = _sym_bits(alpha.n)
        idxs = _unpack_bits(body[off : off + _bit_bytes(n_resid, sb)], n_resid, sb)
        off += _bit_bytes(n_resid, sb)
        step_bits = max(1, math.ceil(math.log2(max(count, 2))))
        steps = _unpack_bits(body[off : off + _bit_bytes(n_resid, step_bits)], n_resid, step_bits)
        resid_map = dict(zip(steps, idxs))
    out = bytearray([seed])
    counts = Counter([seed])
    for i, ch_id in enumerate(channels):
        if i in resid_map:
            out.append(alpha.symbols[resid_map[i]])
            counts[out[-1]] += 1
            continue
        left = out[-1]
        cands = _channel_candidates(alpha, left, ch_id, n=i + 1, counts=counts)
        if len(cands) != 1:
            raise ValueError("channel residual missing")
        out.append(cands[0])
        counts[cands[0]] += 1
    return bytes(out)


def _encode_symbol_rail(data: bytes, lumber: BareLumber) -> bytes:
    alpha = lumber.unit.alphabet
    sym_index = {s: i for i, s in enumerate(alpha.symbols)}
    idxs = [sym_index[b] for b in data]
    return _pack_bits(idxs, _sym_bits(alpha.n))


def _decode_symbol_rail(alpha: SymbolAlphabet, count: int, body: bytes) -> bytes:
    sb = _sym_bits(alpha.n)
    idxs = _unpack_bits(body, count, sb)
    return bytes(alpha.symbols[i] for i in idxs)


def _encode_wing(data: bytes, lumber: BareLumber) -> bytes | None:
    alpha = lumber.unit.alphabet
    counts = Counter(data)
    channels: list[int] = []
    for i in range(len(data) - 1):
        ch = wing_channel_at(alpha, data[i], data[i + 1], n=i + 1, sym_counts=counts).channel_id
        channels.append(ch)
        cands = _channel_candidates(alpha, data[i], ch, n=i + 1, counts=counts)
        if len(cands) != 1:
            return None
    return bytes([data[0]]) + _pack_bits(channels, 7)


def _decode_wing(alpha: SymbolAlphabet, count: int, body: bytes) -> bytes:
    seed = body[0]
    channels = _unpack_bits(body[1:], max(0, count - 1), 7)
    out = bytearray([seed])
    counts = Counter([seed])
    for i, ch_id in enumerate(channels):
        cands = _channel_candidates(alpha, out[-1], ch_id, n=i + 1, counts=counts)
        if len(cands) != 1:
            raise ValueError("wing channel ambiguous")
        out.append(cands[0])
        counts[cands[0]] += 1
    return bytes(out)


@dataclass(frozen=True)
class FullPotentialResult:
    wire: bytes
    mode: str
    ratio: float


def encode_full_potential(data: bytes) -> tuple[bytes, dict]:
    from lattice_retriever_v1.lattice_compressor import PromotionTable

    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    count = len(data)
    base = _lumber_body(lumber) + struct.pack("<I", count)
    candidates: list[tuple[int, bytes, str]] = []

    # FTA promotion — collapse repeated patterns before formula encode
    prom = PromotionTable.mine(data, min_count=2, max_len=24)
    tok = prom.tokenize(data)
    if len(tok) < len(data) * 0.85:
        inner, _ = encode_full_potential(tok)
        pat_blob = bytearray()
        for pat in sorted(prom.patterns.keys(), key=len):
            pat_blob.append(len(pat))
            pat_blob.extend(pat)
        prom_body = struct.pack("<H", len(prom.patterns)) + bytes(pat_blob) + inner[5:]
        candidates.append((MODE_PROMOTED, prom_body, "promoted_formula"))

    if formula_can_lossless_read(alpha, data):
        candidates.append((MODE_PURE, b"", "formula_pure"))

    if len(alpha.symbols) == 2 and lazy_reconstruct_branch_order(alpha, count) == data:
        candidates.append((MODE_BRANCH2, b"", "two_branch_formula"))

    if len(alpha.symbols) == 1:
        candidates.append((MODE_PURE, b"", "one_symbol_branch"))

    wing_body = _encode_wing(data, lumber)
    if wing_body is not None:
        candidates.append((MODE_WING, wing_body, "wing_channel_pure"))

    ch_res = None
    try:
        ch_res = _encode_channel_residual(data, lumber)
        if len(ch_res) < len(data):
            candidates.append((MODE_CHANNEL_RESID, ch_res, "channel_residual"))
    except (ValueError, struct.error):
        pass

    rail = _encode_symbol_rail(data, lumber)
    candidates.append((MODE_SYMBOL_RAIL, rail, "symbol_rail"))

    if not candidates:
        raise ValueError("no lossless encoding path")

    best_mode, best_body, best_name = min(
        candidates,
        key=lambda t: len(MAGIC) + 1 + len(base) + len(t[1]),
    )
    wire = MAGIC + bytes([best_mode]) + base + best_body
    if decode_full_potential(wire) != data:
        raise ValueError("full potential roundtrip failed")
    return wire, {
        "mode": best_name,
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "walker_stored": 0,
        "coords_stored": 0,
        "n_symbols": unit.n_tokens,
    }


def decode_full_potential(wire: bytes) -> bytes:
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    mode = wire[4]
    lumber, off = _parse_lumber(wire, 5)
    (count,) = struct.unpack_from("<I", wire, off)
    off += 4
    alpha = lumber.unit.alphabet
    body = wire[off:]

    if mode == MODE_PURE or mode == MODE_BRANCH2:
        if len(alpha.symbols) == 1:
            return bytes([alpha.symbols[0]]) * count
        if mode == MODE_BRANCH2 and len(alpha.symbols) == 2:
            return lazy_reconstruct_branch_order(alpha, count)
        trial = formula_corridor_read(alpha, count)
        if trial is None:
            raise ValueError("formula pure decode failed")
        return trial
    if mode == MODE_WING:
        return _decode_wing(alpha, count, body)
    if mode == MODE_CHANNEL_RESID:
        return _decode_channel_residual(alpha, count, body)
    if mode == MODE_SYMBOL_RAIL:
        return _decode_symbol_rail(alpha, count, body)
    if mode == MODE_PROMOTED:
        from lattice_retriever_v1.lattice_compressor import PromotionTable

        (n_pat,) = struct.unpack_from("<H", body, 0)
        off = 2
        patterns: list[bytes] = []
        for _ in range(n_pat):
            ln = body[off]
            off += 1
            patterns.append(body[off : off + ln])
            off += ln
        prom = PromotionTable(patterns={p: 1 for p in patterns})
        inner = MAGIC + body[off:]
        tok = decode_full_potential(inner)
        return prom.detokenize(tok)
    raise ValueError(f"unknown mode {mode}")
