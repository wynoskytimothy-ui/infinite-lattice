"""
Trigger-formula codec — user's model (locked).

  2-way + transgressor n  → position on rail (vector holds WHEN triggered)
  3-way intersection      → symbol lock at meet
  Zero coordinates        → formula regenerates vectors on read

Wire: lumber + count + seed + gamma(pair_n stream) + sparse 3-way locks
      only where transgressor n alone is ambiguous (not a walker, not coords).
"""

from __future__ import annotations

import struct
from collections import defaultdict

from lattice_retriever_v1.formula_corridor_read import formula_can_lossless_read, formula_corridor_read
from lattice_retriever_v1.intersection_dot_codec import (
    SymbolAlphabet,
    document_pair_walk,
    dot_on_origin,
    read_document_from_walk,
)
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit
from lattice_retriever_v1.wing_channel_codec import _pack_bits, _unpack_bits, _bit_bytes

MAGIC = b"TFM1"
MODE_PURE = 0
MODE_TRANSGRESSOR_N = 1


def _pair_n_stream(data: bytes) -> tuple[int, ...]:
    walk = document_pair_walk(data)
    return tuple(d.pair_n for d in walk)


def _encode_pair_ns(values: tuple[int, ...]) -> tuple[bytes, int]:
    """Pack transgressor n stream — gamma when small, fixed width when rails grow."""
    if not values:
        return b"", 0
    max_pn = max(max(1, v) for v in values)
    fixed_bits = max(1, max_pn.bit_length())
    fixed = _pack_bits(list(values), fixed_bits)
    gamma_bits: list[int] = []
    for v in values:
        x = max(1, v)
        for _ in range(x - 1):
            gamma_bits.append(0)
        gamma_bits.append(1)
    gamma = _pack_bits(gamma_bits, 1)
    if len(gamma) <= len(fixed):
        return gamma, 0
    return fixed, fixed_bits


def _decode_pair_ns(blob: bytes, n_vals: int, *, fixed_bits: int) -> list[int]:
    if fixed_bits > 0:
        return _unpack_bits(blob, n_vals, fixed_bits)
    return _decode_gamma(blob, n_vals)


def _decode_gamma(blob: bytes, n_vals: int) -> list[int]:
    if n_vals <= 0:
        return []
    raw_bits: list[int] = []
    for b in blob:
        for bit in range(7, -1, -1):
            raw_bits.append((b >> bit) & 1)
    out: list[int] = []
    i = 0
    while len(out) < n_vals and i < len(raw_bits):
        run = 1
        while i < len(raw_bits) and raw_bits[i] == 0:
            run += 1
            i += 1
        if i < len(raw_bits):
            i += 1
        out.append(run)
    while len(out) < n_vals:
        out.append(1)
    return out[:n_vals]


def _pair_count_prefix(prefix: bytes, left: int, right: int) -> int:
    n = 0
    for i in range(len(prefix) - 1):
        if prefix[i] == left and prefix[i + 1] == right:
            n += 1
    return n


def _ambiguous_locks(data: bytes, alpha: SymbolAlphabet) -> list[tuple[int, int]]:
    """Steps where transgressor n alone doesn't unique-lock symbol — 3-way witness."""
    sym_index = {s: i for i, s in enumerate(alpha.symbols)}
    locks: list[tuple[int, int]] = []
    for wi in range(len(data) - 1):
        left, right = data[wi], data[wi + 1]
        pn = _pair_count_prefix(data[: wi + 1], left, right) + 1
        cands = [
            s
            for s in alpha.symbols
            if _pair_count_prefix(data[: wi + 1], left, s) + 1 == pn
        ]
        if len(cands) > 1:
            locks.append((wi, sym_index[right]))
    return locks


def _pack_locks(locks: list[tuple[int, int]], sym_bits: int, step_bits: int) -> bytes:
    if not locks:
        return b""
    steps, idxs = zip(*locks)
    return _pack_bits(list(idxs), sym_bits) + _pack_bits(list(steps), step_bits)


def _unpack_locks(
    blob: bytes, n_locks: int, sym_bits: int, step_bits: int
) -> dict[int, int]:
    if n_locks <= 0:
        return {}
    idx_len = _bit_bytes(n_locks, sym_bits)
    idxs = _unpack_bits(blob[:idx_len], n_locks, sym_bits)
    steps = _unpack_bits(blob[idx_len : idx_len + _bit_bytes(n_locks, step_bits)], n_locks, step_bits)
    return dict(zip(steps, idxs))


def _decode_from_transgressor_n(
    alpha: SymbolAlphabet,
    count: int,
    seed: int,
    pair_ns: list[int],
    locks: dict[int, int],
) -> bytes:
    if count <= 0:
        return b""
    if count == 1:
        return bytes([seed])
    out = bytearray([seed])
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    sym_bits = max(1, (alpha.n - 1).bit_length() + 1) if alpha.n > 1 else 1

    for wi, pn in enumerate(pair_ns):
        left = out[-1]
        if wi in locks:
            out.append(alpha.symbols[locks[wi]])
            pair_counts[(left, out[-1])] += 1
            continue
        found = None
        for s in alpha.symbols:
            if pair_counts[(left, s)] + 1 != pn:
                continue
            found = s
            break
        if found is None:
            raise ValueError("transgressor read failed")
        out.append(found)
        pair_counts[(left, found)] += 1

    result = bytes(out)
    dots = document_pair_walk(result, alpha)
    rebuilt = read_document_from_walk(
        tuple(dot_on_origin(d.origin, pair_n=d.pair_n, walk_index=d.walk_index) for d in dots)
    )
    if rebuilt != result:
        raise ValueError("3-way formula roundtrip failed")
    return result


def _build_transgressor_body(data: bytes, alpha: SymbolAlphabet) -> bytes:
    pair_ns = _pair_n_stream(data)
    locks = _ambiguous_locks(data, alpha)
    sym_bits = max(1, (alpha.n - 1).bit_length() + 1) if alpha.n > 1 else 1
    step_bits = max(1, (max(len(data), 2) - 1).bit_length())
    witness, fixed_bits = _encode_pair_ns(pair_ns)
    body = bytearray([data[0], fixed_bits & 0xFF])
    body.extend(struct.pack("<I", len(witness)))
    body.extend(witness)
    body.extend(struct.pack("<H", len(locks)))
    if locks:
        body.extend(_pack_locks(locks, sym_bits, step_bits))
    return bytes(body)


def encode_trigger_formula(data: bytes) -> tuple[bytes, dict]:
    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    count = len(data)
    base = lumber.to_wire()[len(b"LUM1") :] + struct.pack("<I", count)

    candidates: list[tuple[int, bytes, str]] = []

    if len(alpha.symbols) == 1 or formula_can_lossless_read(alpha, data):
        candidates.append((MODE_PURE, b"", "trigger_formula_pure"))

    if count > 1 and len(alpha.symbols) > 1:
        candidates.append(
            (MODE_TRANSGRESSOR_N, _build_transgressor_body(data, alpha), "transgressor_n_witness")
        )

    if not candidates:
        candidates.append((MODE_PURE, b"", "trigger_formula_pure"))

    mode, body, name = min(candidates, key=lambda t: len(MAGIC) + 1 + len(base) + len(t[1]))
    wire = MAGIC + bytes([mode]) + base + body
    if len(data) <= 4096 and decode_trigger_formula(wire) != data:
        raise ValueError("trigger formula roundtrip failed")
    return wire, {
        "mode": name,
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "coords_stored": 0,
        "origin_stored": 0,
        "walker_stored": 0,
    }


def decode_trigger_formula(wire: bytes) -> bytes:
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    mode = wire[4]
    off = 5
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    (count,) = struct.unpack_from("<I", wire, off)
    off += 4
    alpha = SymbolAlphabet(symbols=symbols)
    body = wire[off:]

    if mode == MODE_PURE:
        if count <= 1:
            return bytes([alpha.symbols[0]]) * count if count == 1 else b""
        trial = formula_corridor_read(alpha, count)
        if trial is None:
            raise ValueError("pure trigger read failed")
        return trial

    if mode == MODE_TRANSGRESSOR_N:
        seed = body[0]
        fixed_bits = body[1]
        n_steps = max(0, count - 1)
        (witness_len,) = struct.unpack_from("<I", body, 2)
        off_body = 6
        pair_ns = _decode_pair_ns(
            body[off_body : off_body + witness_len], n_steps, fixed_bits=fixed_bits
        )
        off_body += witness_len
        (n_locks,) = struct.unpack_from("<H", body, off_body)
        off_body += 2
        sym_bits = max(1, (alpha.n - 1).bit_length() + 1) if alpha.n > 1 else 1
        step_bits = max(1, (max(count, 2) - 1).bit_length())
        lock_blob = body[off_body:]
        locks = _unpack_locks(lock_blob, n_locks, sym_bits, step_bits)
        return _decode_from_transgressor_n(alpha, count, seed, pair_ns, locks)

    raise ValueError(f"unknown mode {mode}")
