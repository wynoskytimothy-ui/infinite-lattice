"""
Deep branch codec — Stage 06 nested corridor functions.

Section 5 triple: anchors (a,p) + transgressor n; case 1/2/3 on rail.
Wire: lumber + symbol count + packed case witness (2 bits/case).
Decode: formula branch walk (lumber + count) — cases verify deep chain.
"""

from __future__ import annotations

import struct

from aethos_lattice import prime_pair_case

from lattice_retriever_v1.branch_order_codec import decode_branch_order, encode_branch_order
from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex
from lattice_retriever_v1.intersection_dot_codec import SymbolAlphabet
from lattice_retriever_v1.stage06_composites import RepeatedPrimeError, section5_triple_roles
from lattice_retriever_v1.wing_channel_codec import _pack_bits

MAGIC = b"DPB1"


def triple_case_stream(data: bytes, alpha: SymbolAlphabet) -> list[int]:
    cases: list[int] = []
    for i in range(len(data) - 2):
        a, b, c = data[i], data[i + 1], data[i + 2]
        primes = (alpha.prime_for(a), alpha.prime_for(b), alpha.prime_for(c))
        try:
            _, _, _, case = section5_triple_roles(primes)
        except RepeatedPrimeError:
            lp, rp = sorted((primes[0], primes[1]))
            case = prime_pair_case(lp, rp, primes[2] if len(set(primes)) > 2 else lp)
        cases.append(case)
    return cases


def encode_deep_branch(
    data: bytes,
    *,
    walk_index: FormulaWalkIndex | None = None,
) -> tuple[bytes, dict] | None:
    if len(data) < 3:
        return None
    alpha = SymbolAlphabet.from_bytes(data)
    idx = walk_index if walk_index is not None else FormulaWalkIndex()
    br_wire, br_meta = encode_branch_order(data, walk_index=idx)
    cases = triple_case_stream(data, alpha)
    inner = br_wire[4:]
    case_blob = _pack_bits(cases, 2)
    wire = MAGIC + inner + case_blob
    if decode_deep_branch(wire, walk_index=idx) != data:
        return None
    return wire, {
        "mode": "deep_branch_cases",
        "n_cases": len(cases),
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "walker_stored": 0,
        "branch_mode": br_meta.get("mode"),
    }


def decode_deep_branch(
    wire: bytes,
    *,
    walk_index: FormulaWalkIndex | None = None,
) -> bytes:
    off = len(MAGIC)
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    count_end = len(MAGIC) + 6 + n_sym + 4
    body = wire[len(MAGIC) : count_end]
    _, n_sym = struct.unpack_from("<IH", wire, len(MAGIC))
    magic = b"BRN1" if n_sym <= 2 else b"IDX1"
    return decode_branch_order(magic + body, walk_index=walk_index)
