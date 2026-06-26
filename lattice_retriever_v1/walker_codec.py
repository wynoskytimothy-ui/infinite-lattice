"""
Walker codec — 2-way is the walker; 3-way locks the symbol.

The walk is NOT stored separately from geometry:
  - **2-way intersection** (origin + transgressor n) = walker step on the rail.
  - **3-way intersection** = symbol witness (which symbol fired at this meet).
  - Vector + n tells you which intersection triggered — that IS the path.

Bare lumber (unit) + 2-way walker triggers = full lossless replay.
3-way witnesses recompute from the walk (formula-side symbol lock).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

from aethos_lattice import prime_pair_case

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginDot,
    SymbolAlphabet,
    _oriented_pair_catalog,
    _pack_pairs,
    _unpack_pairs,
    _bits_per_index,
    document_pair_walk,
    read_document_from_walk,
    regenerate_dot_from_formula,
)
from lattice_retriever_v1.stage02_intersections import intersect_primes
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit, MAGIC as LUMBER_MAGIC

WALK_MAGIC = b"WLK1"


@dataclass(frozen=True)
class TwoWayWalkerStep:
    """Walker dot — 2-way intersection on rail (origin, n)."""

    walk_index: int
    origin_id: int
    pair_n: int
    branch_case: int  # prime_pair_case regime 1/2/3
    dot: PairOriginDot

    def explain(self) -> dict:
        o = self.dot.origin
        return {
            "walk_index": self.walk_index,
            "walker": "2-way",
            "origin_id": self.origin_id,
            "pair_n": self.pair_n,
            "branch_case": self.branch_case,
            "pair": f"{chr(o.left_byte)!r}->{chr(o.right_byte)!r}",
            "L01": self.dot.address.lattice_coords[0],
        }


@dataclass(frozen=True)
class ThreeWaySymbolWitness:
    """Symbol lock — 3-way intersection names the symbol at this position."""

    position: int
    symbol: int
    chars: tuple[str, str, str]
    composite: int
    lattice_L01: tuple[int, int, int]

    def explain(self) -> dict:
        return {
            "position": self.position,
            "symbol_locked": self.symbol,
            "triple": "".join(self.chars),
            "composite": self.composite,
            "L01": self.lattice_L01,
            "role": "3-way symbol witness",
        }


@dataclass(frozen=True)
class WalkerSpan:
    """
    Single-rail walker: one origin, n branches from n_start to n_end.

    When one 2-way origin carries the whole file, storage is just (origin, n_end).
    """

    origin_id: int
    n_start: int
    n_end: int

    @property
    def stored_bytes(self) -> int:
        return 6  # origin_id u16 + n_end u32

    def explain(self) -> dict:
        return {
            "origin_id": self.origin_id,
            "n_start": self.n_start,
            "n_end": self.n_end,
            "stored_bytes": self.stored_bytes,
            "note": "one 2-way rail walks 0→EOF; n is the walker",
        }


def walk_2way(data: bytes, unit: LatticeUnit | None = None) -> tuple[TwoWayWalkerStep, ...]:
    """2-way walker — each step is one triggered intersection (origin, n)."""
    if len(data) < 2:
        return ()
    unit = unit or LatticeUnit.from_data(data)
    alpha = unit.alphabet
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    steps: list[TwoWayWalkerStep] = []
    for dot in document_pair_walk(data, alpha):
        o = dot.origin
        case = prime_pair_case(o.left_prime, o.right_prime, dot.pair_n)
        steps.append(
            TwoWayWalkerStep(
                walk_index=dot.walk_index,
                origin_id=cat_index[o],
                pair_n=dot.pair_n,
                branch_case=case,
                dot=dot,
            )
        )
    return tuple(steps)


def witness_3way(data: bytes, unit: LatticeUnit | None = None) -> tuple[ThreeWaySymbolWitness, ...]:
    """3-way symbol witnesses — one per sliding triple; locks middle symbol."""
    if len(data) < 3:
        return ()
    unit = unit or LatticeUnit.from_data(data)
    alpha = unit.alphabet
    out: list[ThreeWaySymbolWitness] = []
    for i in range(len(data) - 2):
        a, b, c = data[i], data[i + 1], data[i + 2]
        chars = (str(a), str(b), str(c))
        primes = (alpha.prime_for(a), alpha.prime_for(b), alpha.prime_for(c))
        addr = intersect_primes(chars, primes, start_index=i, n=i + 1)
        out.append(
            ThreeWaySymbolWitness(
                position=i + 1,
                symbol=b,
                chars=chars,
                composite=addr.composite,
                lattice_L01=addr.lattice_coords[0],
            )
        )
    return tuple(out)


def detect_walker_span(steps: tuple[TwoWayWalkerStep, ...]) -> WalkerSpan | None:
    """If one origin walks 0→EOF with n=1..k, return compact span."""
    if not steps:
        return None
    origins = {s.origin_id for s in steps}
    if len(origins) != 1:
        return None
    oid = next(iter(origins))
    ns = [s.pair_n for s in steps]
    if ns != list(range(1, len(ns) + 1)):
        return None
    return WalkerSpan(origin_id=oid, n_start=1, n_end=len(ns))


def encode_walker(data: bytes) -> tuple[bytes, dict]:
    """
    Encode: bare lumber + 2-way walker triggers (origin, n).
    3-way witnesses are formula-side — not stored separately.
    """
    unit = LatticeUnit.from_data(data)
    lumber = BareLumber(unit=unit, raw_len=len(data))
    steps = walk_2way(data, unit)
    witnesses = witness_3way(data, unit)
    span = detect_walker_span(steps)

    if span is not None:
        body = struct.pack("<HI", span.origin_id, span.n_end)
        mode = "single_rail_span"
    elif steps:
        catalog = _oriented_pair_catalog(unit.alphabet)
        pairs = [(s.origin_id, s.pair_n) for s in steps]
        max_n = max(n for _, n in pairs)
        bits_o = _bits_per_index(len(catalog))
        bits_n = max(1, (max_n).bit_length())
        packed = _pack_pairs(pairs, pair_bits=bits_o, n_bits=bits_n)
        body = struct.pack("<BB", bits_o, bits_n) + packed
        mode = "walker_triggers"
    else:
        body = b""
        mode = "lumber_only"

    wire = WALK_MAGIC + bytes([{"single_rail_span": 1, "walker_triggers": 2, "lumber_only": 0}[mode]]) + lumber.to_wire()[len(LUMBER_MAGIC) :] + body

    meta = {
        "mode": mode,
        "raw_bytes": len(data),
        "bare_lumber_bytes": lumber.stored_bytes,
        "walker_steps": len(steps),
        "witness_3way_count": len(witnesses),
        "witness_stored_bytes": 0,
        "walker_body_bytes": len(body),
        "total_wire_bytes": len(wire),
        "single_rail_span": span.explain() if span else None,
        "model": "2-way=walker; 3-way=symbol lock (formula); vector+n=path",
    }
    return wire, meta


def decode_walker(wire: bytes) -> bytes:
    """Decode from lumber + 2-way walker — 3-way symbols recompute on read."""
    if not wire.startswith(WALK_MAGIC):
        raise ValueError("bad magic")
    mode = wire[4]
    off = 5
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    unit = LatticeUnit(symbols=symbols)
    alpha = unit.alphabet

    if mode == 0:  # lumber_only
        return b"" if raw_len == 0 else bytes([symbols[0]]) if raw_len == 1 else b""

    catalog = _oriented_pair_catalog(alpha)

    if mode == 1:  # single_rail_span
        oid, n_end = struct.unpack_from("<HI", wire, off)
        n_start = 1
        if raw_len < 2:
            return bytes([symbols[0]]) if raw_len == 1 else b""
        origin = catalog[oid]
        dots = tuple(
            regenerate_dot_from_formula(origin, pair_n=n, walk_index=i)
            for i, n in enumerate(range(n_start, n_end + 1))
        )
        return read_document_from_walk(dots)

    # walker_triggers
    bits_o, bits_n = wire[off], wire[off + 1]
    off += 2
    n_steps = raw_len - 1 if raw_len > 1 else 0
    pairs = _unpack_pairs(wire[off:], n_values=n_steps, pair_bits=bits_o, n_bits=bits_n)
    dots = tuple(
        regenerate_dot_from_formula(catalog[oid], pair_n=pn, walk_index=wi)
        for wi, (oid, pn) in enumerate(pairs)
    )
    return read_document_from_walk(dots)


def verify_3way_from_walker(data: bytes) -> dict:
    """Show 3-way symbol locks match 2-way walker chain — no separate symbol store."""
    steps = walk_2way(data)
    witnesses = witness_3way(data)
    chain = read_document_from_walk(tuple(s.dot for s in steps)) if steps else data
    from_walker = [data[i + 1] for i in range(len(data) - 2)]
    locked = [w.symbol for w in witnesses]
    return {
        "roundtrip_2way": chain == data,
        "symbols_match_3way": locked == from_walker,
        "walker_steps": len(steps),
        "witnesses": len(witnesses),
        "first_walker": steps[0].explain() if steps else None,
        "first_witness": witnesses[0].explain() if witnesses else None,
    }
