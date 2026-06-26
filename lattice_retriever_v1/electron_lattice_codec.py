"""
Electron + entanglement lattice codec — 4-state coin reads, hidden pair patterns.

Each symbol carries a CoinState (WS/WH/BS/BH) = 2 bits from formula wing×case.
Co-occurring pairs entangle at imaginary intersection — opposites share ocean phase.

Compression paths:
  dit4_vocab   — vocab ≤4: one electron read (2 bits) per symbol
  entangle_ab  — oscillating entangled pair + count (lumber + pair witness + count)
  coin_order   — coin stream predicts every next symbol (formula readout only)
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass

from aethos_complex_plane import imaginary_start
from aethos_electron_tokenizer import (
    CoinState,
    ElectronVocabCodec,
    pack_states,
    state_to_bits,
    unpack_states,
)

from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit
from lattice_retriever_v1.wing_channel_codec import wing_channel_at, _pack_bits, _unpack_bits, _bit_bytes

MAGIC = b"ELC1"
MODE_DIT4 = 0
MODE_ENTANGLE_AB = 1
MODE_COIN_ORDER = 2


@dataclass(frozen=True)
class ElectronSymbol:
    """One catalog entry — byte value + 4-state electron read."""

    byte: int
    coin: CoinState

    @property
    def bits(self) -> tuple[int, int]:
        return state_to_bits(self.coin)


@dataclass(frozen=True)
class EntangleWitness:
    """Two symbols bound at intersection — formula-side entanglement."""

    left: int
    right: int
    intersection_imag: int
    opposite: bool


def wing_case_to_coin(case: int, wing: int) -> CoinState:
    """Map 3 branch cases × 32 wings → 4 electron coin states (membrane × spring)."""
    membrane = wing & 1
    spring = (case - 1) & 1
    return CoinState((membrane << 1) | spring)


def entangle_imag(left_prime: int, right_prime: int) -> int:
    """Composite intersection on imaginary axis — shared ocean coordinate."""
    a = imaginary_start(float(left_prime))
    b = imaginary_start(float(right_prime))
    return int(round(a.z.imag)) + int(round(b.z.imag))


def electrons_opposite(ca: CoinState, cb: CoinState) -> bool:
    """Opposite membrane = entangled pair in opposite phase (E1+E2 balanced)."""
    return (int(ca) >> 1) != (int(cb) >> 1)


def build_electron_alphabet(data: bytes) -> tuple[ElectronSymbol, ...]:
    """Assign each unique byte a coin state from lattice formula + frequency."""
    alpha = SymbolAlphabet.from_bytes(data)
    counts = Counter(data)
    ranked = sorted(alpha.symbols, key=lambda s: (-counts[s], s))
    out: list[ElectronSymbol] = []
    n = len(ranked)
    for rank, byte in enumerate(ranked):
        if rank == 0:
            coin = CoinState.WS
        elif rank == n - 1:
            coin = CoinState.BH
        elif alpha.prime_for(byte) % 2 == 0:
            coin = CoinState.WH
        else:
            coin = CoinState.BS
        out.append(ElectronSymbol(byte=byte, coin=coin))
    return tuple(out)


def _bits_to_bytes(bit_list: list[int]) -> bytes:
    return _pack_bits(bit_list, 1)


def _bytes_to_bits(blob: bytes, n_bits: int) -> list[int]:
    return _unpack_bits(blob, n_bits, 1)


def _pack_electron_lumber(
    lumber: BareLumber, catalog: tuple[ElectronSymbol, ...]
) -> bytes:
    """Lumber + packed 2-bit coin state per catalog symbol."""
    coin_bits = pack_states(s.coin for s in catalog)
    return lumber.to_wire()[len(b"LUM1") :] + _bits_to_bytes(coin_bits)


def _unpack_electron_lumber(
    wire: bytes, off: int
) -> tuple[BareLumber, tuple[ElectronSymbol, ...], int]:
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    coin_bits_len = _bit_bytes(n_sym * 2, 1)
    coin_bits = _bytes_to_bits(wire[off : off + coin_bits_len], n_sym * 2)
    off += coin_bits_len
    coins = unpack_states(coin_bits)
    catalog = tuple(
        ElectronSymbol(byte=symbols[i], coin=coins[i] if i < len(coins) else CoinState.WS)
        for i in range(n_sym)
    )
    unit = LatticeUnit(symbols=symbols)
    lumber = BareLumber(unit=unit, raw_len=raw_len)
    return lumber, catalog, off


def entangle_witness(a: int, b: int, alpha: SymbolAlphabet, catalog: tuple[ElectronSymbol, ...]) -> EntangleWitness:
    coin_map = {s.byte: s.coin for s in catalog}
    pa, pb = alpha.prime_for(a), alpha.prime_for(b)
    return EntangleWitness(
        left=a,
        right=b,
        intersection_imag=entangle_imag(pa, pb),
        opposite=electrons_opposite(coin_map[a], coin_map[b]),
    )


def _detect_entangle_oscillate(data: bytes) -> tuple[int, int] | None:
    if len(data) < 2:
        return None
    a, b = data[0], data[1]
    if a == b:
        return None
    for i, byte in enumerate(data):
        if byte != (a if i % 2 == 0 else b):
            return None
    return a, b


def _encode_dit4(data: bytes, lumber: BareLumber, catalog: tuple[ElectronSymbol, ...]) -> bytes | None:
    if len(catalog) > 4:
        return None
    sym_to_id = {s.byte: i for i, s in enumerate(catalog)}
    codec = ElectronVocabCodec([str(s.byte) for s in catalog])
    bits: list[int] = []
    for byte in data:
        bits.extend(pack_states(codec.encode_id(sym_to_id[byte])))
    body = struct.pack("<B", MODE_DIT4) + _bits_to_bytes(bits)
    return MAGIC + _pack_electron_lumber(lumber, catalog) + body


def _decode_dit4(wire: bytes) -> bytes:
    lumber, catalog, off = _unpack_electron_lumber(wire, len(MAGIC))
    mode = wire[off]
    off += 1
    if mode != MODE_DIT4:
        raise ValueError("not dit4 mode")
    codec = ElectronVocabCodec([str(s.byte) for s in catalog])
    width = codec.dits_per_token
    n_bits = lumber.raw_len * width * 2
    n_bytes = _bit_bytes(n_bits, 1)
    bits = _bytes_to_bits(wire[off : off + n_bytes], n_bits)
    states = unpack_states(bits)
    out = bytearray()
    for i in range(lumber.raw_len):
        dits = states[i * width : (i + 1) * width]
        tid = codec.decode_id(dits)
        out.append(catalog[min(tid, len(catalog) - 1)].byte)
    return bytes(out)


def _encode_entangle_ab(
    data: bytes, lumber: BareLumber, catalog: tuple[ElectronSymbol, ...], pair: tuple[int, int]
) -> bytes:
    alpha = lumber.unit.alphabet
    witness = entangle_witness(pair[0], pair[1], alpha, catalog)
    body = struct.pack(
        "<BBBI",
        MODE_ENTANGLE_AB,
        pair[0],
        pair[1],
        witness.intersection_imag & 0xFFFFFFFF,
    )
    body += struct.pack("<B", int(witness.opposite))
    body += struct.pack("<I", len(data))
    return MAGIC + _pack_electron_lumber(lumber, catalog) + body


def _decode_entangle_ab(wire: bytes) -> bytes:
    lumber, catalog, off = _unpack_electron_lumber(wire, len(MAGIC))
    mode = wire[off]
    if mode != MODE_ENTANGLE_AB:
        raise ValueError("not entangle_ab mode")
    a, b = wire[off + 1], wire[off + 2]
    count = struct.unpack_from("<I", wire, off + 8)[0]
    return bytes([a if i % 2 == 0 else b for i in range(count)])


def _encode_coin_order(
    data: bytes, lumber: BareLumber, catalog: tuple[ElectronSymbol, ...]
) -> bytes | None:
    if len(data) < 2:
        return None
    alpha = lumber.unit.alphabet
    cat_bytes = tuple(s.byte for s in catalog)
    sym_index = {b: i for i, b in enumerate(cat_bytes)}
    counts = Counter(data)
    coins: list[int] = []
    table: dict[tuple[int, int], int] = {}
    for i in range(len(data) - 1):
        ch = wing_channel_at(alpha, data[i], data[i + 1], n=i + 1, sym_counts=counts)
        coin = int(wing_case_to_coin(ch.case, ch.wing))
        coins.append(coin)
        key = (coin, sym_index[data[i]])
        nxt = sym_index[data[i + 1]]
        if key in table and table[key] != nxt:
            return None
        table[key] = nxt
    body = bytearray([MODE_COIN_ORDER, data[0]])
    body.extend(_pack_bits(coins, 2))
    return MAGIC + _pack_electron_lumber(lumber, catalog) + bytes(body)


def _decode_coin_order(wire: bytes) -> bytes:
    lumber, catalog, off = _unpack_electron_lumber(wire, len(MAGIC))
    mode = wire[off]
    if mode != MODE_COIN_ORDER:
        raise ValueError("not coin_order mode")
    seed = wire[off + 1]
    off += 2
    n_coins = max(0, lumber.raw_len - 1)
    coin_len = _bit_bytes(n_coins, 2)
    coins = _unpack_bits(wire[off : off + coin_len], n_coins, 2)
    cat_bytes = tuple(s.byte for s in catalog)
    alpha = lumber.unit.alphabet
    counts = Counter()
    out = bytearray([seed])
    prev = seed
    counts[prev] = 1
    for coin_val in coins:
        found = None
        for nxt in cat_bytes:
            ch = wing_channel_at(alpha, prev, nxt, n=len(out), sym_counts=counts)
            if int(wing_case_to_coin(ch.case, ch.wing)) == coin_val:
                found = nxt
                break
        if found is None:
            found = cat_bytes[coin_val % len(cat_bytes)]
        out.append(found)
        counts[found] += 1
        prev = found
    return bytes(out)


def encode_electron_entangle(data: bytes) -> tuple[bytes, dict] | None:
    if not data:
        return None
    unit = LatticeUnit.from_data(data)
    lumber = BareLumber(unit=unit, raw_len=len(data))
    catalog = build_electron_alphabet(data)

    candidates: list[tuple[bytes, str]] = []

    dit4 = _encode_dit4(data, lumber, catalog)
    if dit4 and _decode_dit4(dit4) == data:
        candidates.append((dit4, "electron_dit4"))

    pair = _detect_entangle_oscillate(data)
    if pair is not None:
        ent = _encode_entangle_ab(data, lumber, catalog, pair)
        if _decode_entangle_ab(ent) == data:
            candidates.append((ent, "electron_entangle_ab"))

    coin = _encode_coin_order(data, lumber, catalog)
    if coin is not None and _decode_coin_order(coin) == data:
        candidates.append((coin, "electron_coin_order"))

    if not candidates:
        return None
    wire, mode = min(candidates, key=lambda t: len(t[0]))
    if len(wire) >= len(data):
        return None
    return wire, {
        "mode": mode,
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "walker_stored": 0,
        "n_entangle_patterns": 1 if pair else 0,
        "electron_states": len(catalog),
    }


def decode_electron_entangle(wire: bytes) -> bytes:
    off = len(MAGIC)
    # peek mode after lumber — need to parse lumber first
    _, catalog, body_off = _unpack_electron_lumber(wire, off)
    mode = wire[body_off]
    if mode == MODE_DIT4:
        return _decode_dit4(wire)
    if mode == MODE_ENTANGLE_AB:
        return _decode_entangle_ab(wire)
    if mode == MODE_COIN_ORDER:
        return _decode_coin_order(wire)
    raise ValueError(f"unknown electron mode {mode}")


def electron_entangle_report(data: bytes) -> dict:
    """Expose hidden entanglement + coin structure for inspection."""
    catalog = build_electron_alphabet(data)
    alpha = SymbolAlphabet.from_bytes(data)
    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for i in range(len(data) - 1):
        a, b = data[i], data[i + 1]
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        w = entangle_witness(a, b, alpha, catalog)
        pairs.append(
            {
                "left": a,
                "right": b,
                "intersection_imag": w.intersection_imag,
                "opposite": w.opposite,
            }
        )
    enc = encode_electron_entangle(data)
    return {
        "catalog": [{"byte": s.byte, "coin": s.coin.name} for s in catalog],
        "entangled_pairs": pairs[:10],
        "encode": enc[1] if enc else None,
    }
