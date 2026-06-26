"""
Prime corridor codec — user's Nov 15 Prime Corridors sheet (locked).

  Main spine:     (i, 0, i)           — global walk index
  Prime corridor: (outer, k, outer)    — corridor P*; k transgresses until P
      outer = (pair_n - 1) + P + 1
      k     = min(pair_n, P)

  2-way: transgressor pair_n on left-prime corridor (when triggered)
  3-way: branch case at meet disambiguates symbol — not origin_id, not coords

Wire: lumber + count + seed + pair_n stream + sparse 2-bit branch cases at meets.
"""

from __future__ import annotations

import struct
from collections import defaultdict

from lattice_retriever_v1.aethos_n1_meet import meet_outer_fingerprint, pick_symbol_at_meet
from lattice_retriever_v1.formula_corridor_read import formula_can_lossless_read, formula_corridor_read
from lattice_retriever_v1.intersection_dot_codec import (
    SymbolAlphabet,
    document_pair_walk,
    dot_on_origin,
    read_document_from_walk,
)
from lattice_retriever_v1.trigger_formula_codec import _decode_pair_ns, _encode_pair_ns
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit
from lattice_retriever_v1.wing_channel_codec import _bit_bytes, _pack_bits, _unpack_bits

MAGIC = b"PCR1"
MODE_PURE = 0
MODE_CORRIDOR = 1


def spine_triplet(spine_i: int) -> tuple[int, int, int]:
    """Main spine row — (n, 0, n)."""
    return (spine_i, 0, spine_i)


def corridor_triplet(corridor_prime: int, pair_n: int) -> tuple[int, int, int]:
    """
    Prime corridor P* at transgressor pair_n (1-based).
    Middle transgresses 1..P then locks at P; outer rails track spine meet.
    """
    if pair_n < 1:
        raise ValueError("pair_n must be >= 1")
    middle = min(pair_n, corridor_prime)
    outer = (pair_n - 1) + corridor_prime + 1
    return (outer, middle, outer)


def corridor_meets_spine(spine_i: int, corridor_prime: int, pair_n: int) -> bool:
    """3-way meet: corridor outer rail crosses spine position."""
    outer, _, _ = corridor_triplet(corridor_prime, pair_n)
    s_outer, _, _ = spine_triplet(spine_i)
    return outer == s_outer + corridor_prime or outer == s_outer + 1


def _pair_n_stream(data: bytes) -> tuple[int, ...]:
    return tuple(d.pair_n for d in document_pair_walk(data))


def _ambiguous_meet_witness(data: bytes, alpha: SymbolAlphabet) -> list[tuple[int, int]]:
    """
    Steps where pair_n alone is ambiguous — witness VA1A outer rail (a+p), PDF §5.1.
    At n=1 case is always 1; outer rail still unique per oriented pair.
    """
    from lattice_retriever_v1.aethos_n1_meet import _candidates, _pair_counts

    witness: list[tuple[int, int]] = []
    prefix = bytes([data[0]])
    pc = _pair_counts(prefix)
    for wi in range(len(data) - 1):
        left, right = data[wi], data[wi + 1]
        pn = pc.get((left, right), 0) + 1
        cands = _candidates(alpha, prefix, pn, pc)
        if len(cands) > 1:
            lp, rp = alpha.prime_for(left), alpha.prime_for(right)
            witness.append((wi, meet_outer_fingerprint(lp, rp, pn)))
        prefix = prefix + bytes([right])
        pc[(left, right)] = pc.get((left, right), 0) + 1
    return witness


def _pack_meet_witness(witness: list[tuple[int, int]], step_bits: int) -> bytes:
    if not witness:
        return b""
    steps, outers = zip(*witness)
    body = bytearray()
    body.extend(struct.pack(f"<{len(outers)}H", *outers))
    body.extend(_pack_bits(list(steps), step_bits))
    return bytes(body)


def _unpack_meet_witness(
    blob: bytes, n_witness: int, step_bits: int
) -> dict[int, int]:
    if n_witness <= 0:
        return {}
    outers = struct.unpack_from(f"<{n_witness}H", blob, 0)
    off = 2 * n_witness
    steps = _unpack_bits(blob[off : off + _bit_bytes(n_witness, step_bits)], n_witness, step_bits)
    return dict(zip(steps, outers))


def _decode_corridor_walk(
    alpha: SymbolAlphabet,
    count: int,
    seed: int,
    pair_ns: list[int],
    meet_outer: dict[int, int],
) -> bytes:
    if count <= 0:
        return b""
    if count == 1:
        return bytes([seed])
    out = bytearray([seed])
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)

    for wi, pn in enumerate(pair_ns):
        left = out[-1]
        lp = alpha.prime_for(left)
        cands = [s for s in alpha.symbols if pair_counts[(left, s)] + 1 == pn]
        if wi in meet_outer:
            want = meet_outer[wi]
            cands = [
                s
                for s in cands
                if meet_outer_fingerprint(lp, alpha.prime_for(s), pn) == want
            ]
        elif len(cands) != 1:
            s = pick_symbol_at_meet(
                alpha, bytes(out), pair_ns, wi, count, pair_counts
            )
            out.append(s)
            pair_counts[(left, s)] += 1
            continue
        if len(cands) != 1:
            if len(cands) > 1:
                cands = [min(cands, key=lambda x: alpha.prime_for(x))]
            else:
                raise ValueError("prime corridor read failed")
        s = cands[0]
        out.append(s)
        pair_counts[(left, s)] += 1

    result = bytes(out)
    dots = document_pair_walk(result, alpha)
    rebuilt = read_document_from_walk(
        tuple(dot_on_origin(d.origin, pair_n=d.pair_n, walk_index=d.walk_index) for d in dots)
    )
    if rebuilt != result:
        raise ValueError("corridor roundtrip failed")
    return result


def _build_corridor_body(data: bytes, alpha: SymbolAlphabet) -> bytes:
    pair_ns = _pair_n_stream(data)
    meets = _ambiguous_meet_witness(data, alpha)
    step_bits = max(1, (max(len(data), 2) - 1).bit_length())
    witness, fixed_bits = _encode_pair_ns(pair_ns)
    body = bytearray([data[0], fixed_bits & 0xFF])
    body.extend(struct.pack("<I", len(witness)))
    body.extend(witness)
    body.extend(struct.pack("<H", len(meets)))
    if meets:
        body.extend(_pack_meet_witness(meets, step_bits))
    return bytes(body)


def encode_prime_corridor(data: bytes) -> tuple[bytes, dict]:
    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    count = len(data)
    base = lumber.to_wire()[len(b"LUM1") :] + struct.pack("<I", count)

    candidates: list[tuple[int, bytes, str]] = []
    if len(alpha.symbols) == 1 or formula_can_lossless_read(alpha, data):
        candidates.append((MODE_PURE, b"", "prime_corridor_pure"))

    if count > 1 and len(alpha.symbols) > 1:
        candidates.append(
            (MODE_CORRIDOR, _build_corridor_body(data, alpha), "prime_corridor_witness")
        )

    if not candidates:
        candidates.append((MODE_PURE, b"", "prime_corridor_pure"))

    mode, body, name = min(candidates, key=lambda t: len(MAGIC) + 1 + len(base) + len(t[1]))
    wire = MAGIC + bytes([mode]) + base + body
    if len(data) <= 8192 and decode_prime_corridor(wire) != data:
        raise ValueError("prime corridor roundtrip failed")
    return wire, {
        "mode": name,
        "raw_bytes": len(data),
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "coords_stored": 0,
        "origin_stored": 0,
        "walker_stored": 0,
    }


def decode_prime_corridor(wire: bytes) -> bytes:
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    mode = wire[4]
    off = 5
    _raw_len, n_sym = struct.unpack_from("<IH", wire, off)
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
            raise ValueError("pure corridor read failed")
        return trial

    if mode == MODE_CORRIDOR:
        seed = body[0]
        fixed_bits = body[1]
        n_steps = max(0, count - 1)
        (witness_len,) = struct.unpack_from("<I", body, 2)
        off_body = 6
        pair_ns = _decode_pair_ns(
            body[off_body : off_body + witness_len], n_steps, fixed_bits=fixed_bits
        )
        off_body += witness_len
        (n_meets,) = struct.unpack_from("<H", body, off_body)
        off_body += 2
        step_bits = max(1, (max(count, 2) - 1).bit_length())
        meet_outer = _unpack_meet_witness(body[off_body:], n_meets, step_bits)
        return _decode_corridor_walk(alpha, count, seed, pair_ns, meet_outer)

    raise ValueError(f"unknown mode {mode}")
