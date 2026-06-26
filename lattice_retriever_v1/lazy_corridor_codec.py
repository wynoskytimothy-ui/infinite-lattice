"""
Lazy corridor walker — prime IS the function; order read on decompress.

  - Walker = corridor function keyed by walker_prime (e.g. 2).
  - 2-way branches find symbols; 3-way locks position; dots placed by formula.
  - Store: bare lumber + function params — NOT every step.
  - Decompress: lazy_eval opens corridor, branches until dots land → order appears.

Modes:
  constant_symbol  — one symbol, n transgresses 1→EOF (smallest)
  single_rail      — one 2-way origin, pair_n 1→EOF
  (fallback)       — compact walker triggers via walker_codec
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

from aethos_lattice import prime_pair_case

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginDot,
    PairOriginKey,
    SymbolAlphabet,
    _oriented_pair_catalog,
    dot_on_origin,
    read_document_from_walk,
    regenerate_dot_from_formula,
)
from lattice_retriever_v1.stage02_intersections import intersect_primes
from lattice_retriever_v1.unit_lattice_codec import BareLumber, LatticeUnit

MAGIC = b"LZY1"
MODE_CONSTANT = 0
MODE_SINGLE_RAIL = 1
DEFAULT_WALKER_PRIME = 2


@dataclass(frozen=True)
class CorridorWalkerFunction:
    """Minimal walker — prime opens corridor; n_end is how far rail transgresses."""

    mode: int
    walker_prime: int
    n_end: int
    seed_byte: int
    origin_id: int = 0

    def explain(self) -> dict:
        return {
            "mode": ("constant" if self.mode == MODE_CONSTANT else "single_rail"),
            "walker_prime": self.walker_prime,
            "n_end": self.n_end,
            "seed_byte": self.seed_byte,
            "origin_id": self.origin_id,
            "steps_stored": 0,
        }


@dataclass(frozen=True)
class LazyDotLanding:
    n: int
    left: int
    right: int
    branch_case: int
    L01: tuple[int, int, int]

    def explain(self) -> dict:
        return {
            "n": self.n,
            "pair": f"{chr(self.left)!r}->{chr(self.right)!r}",
            "branch_case": self.branch_case,
            "L01": self.L01,
            "coords_stored": 0,
        }


def _lazy_dot(alpha: SymbolAlphabet, left: int, right: int, pair_n: int, wi: int) -> LazyDotLanding:
    lp, rp = alpha.prime_for(left), alpha.prime_for(right)
    dot = dot_on_origin(PairOriginKey(left, right, lp, rp), pair_n=pair_n, walk_index=wi)
    return LazyDotLanding(
        n=pair_n,
        left=left,
        right=right,
        branch_case=prime_pair_case(lp, rp, pair_n),
        L01=dot.address.lattice_coords[0],
    )


def lazy_read_corridor(
    alpha: SymbolAlphabet,
    *,
    mode: int,
    walker_prime: int,
    seed_byte: int,
    n_end: int,
    origin_id: int,
) -> Iterator[LazyDotLanding]:
    """
    Lazy eval — open walker_prime corridor; branches place dots; no stored steps.
    """
    catalog = _oriented_pair_catalog(alpha)
    if mode == MODE_CONSTANT:
        for wi in range(n_end):
            yield _lazy_dot(alpha, seed_byte, seed_byte, wi + 1, wi)
        return
    if mode == MODE_SINGLE_RAIL:
        origin = catalog[origin_id]
        for wi in range(n_end):
            yield _lazy_dot(alpha, origin.left_byte, origin.right_byte, wi + 1, wi)
        return


def lazy_dots_to_bytes(
    alpha: SymbolAlphabet,
    mode: int,
    seed_byte: int,
    n_end: int,
    origin_id: int,
) -> bytes:
    """Reconstruct bytes from lazy corridor read."""
    if n_end <= 0:
        return bytes([seed_byte]) if seed_byte or n_end == 0 else b""
    landings = list(
        lazy_read_corridor(
            alpha,
            mode=mode,
            walker_prime=DEFAULT_WALKER_PRIME,
            seed_byte=seed_byte,
            n_end=n_end,
            origin_id=origin_id,
        )
    )
    dots: tuple[PairOriginDot, ...] = tuple(
        dot_on_origin(
            PairOriginKey(l.left, l.right, alpha.prime_for(l.left), alpha.prime_for(l.right)),
            pair_n=l.n,
            walk_index=i,
        )
        for i, l in enumerate(landings)
    )
    return bytes([seed_byte]) + read_document_from_walk(dots)[1:] if dots else bytes([seed_byte])


def _detect_mode(data: bytes, alpha: SymbolAlphabet) -> CorridorWalkerFunction:
    from lattice_retriever_v1.walker_codec import detect_walker_span, walk_2way

    seed = data[0]
    n_end = len(data) - 1
    uniq = set(data)
    if len(uniq) == 1:
        s = next(iter(uniq))
        return CorridorWalkerFunction(
            mode=MODE_CONSTANT,
            walker_prime=alpha.prime_for(s),
            n_end=n_end,
            seed_byte=s,
        )
    steps = walk_2way(data)
    span = detect_walker_span(steps)
    if span is not None:
        return CorridorWalkerFunction(
            mode=MODE_SINGLE_RAIL,
            walker_prime=DEFAULT_WALKER_PRIME,
            n_end=span.n_end,
            seed_byte=seed,
            origin_id=span.origin_id,
        )
    raise ValueError("needs walker fallback for multi-rail data")


def encode_lazy_corridor(data: bytes) -> tuple[bytes, dict]:
    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    lumber = BareLumber(unit=unit, raw_len=len(data))
    try:
        fn = _detect_mode(data, alpha)
    except ValueError:
        from lattice_retriever_v1.walker_codec import encode_walker

        wire, meta = encode_walker(data)
        meta["lazy_corridor"] = "fallback_triggers"
        return wire, meta

    body = struct.pack("<BIIBH", fn.mode, fn.walker_prime, fn.n_end, fn.seed_byte, fn.origin_id)
    wire = MAGIC + lumber.to_wire()[len(b"LUM1") :] + body
    back = decode_lazy_corridor(wire)
    if back != data:
        from lattice_retriever_v1.walker_codec import encode_walker

        wire, meta = encode_walker(data)
        meta["lazy_corridor"] = "fallback_triggers"
        return wire, meta

    meta = {
        "mode": fn.explain()["mode"],
        "raw_bytes": len(data),
        "bare_lumber_bytes": lumber.stored_bytes,
        "function_bytes": len(body),
        "total_wire_bytes": len(wire),
        "walker_steps_stored": 0,
        "walker": fn.explain(),
        "ratio_x": round(len(data) / len(wire), 3),
    }
    return wire, meta


def decode_lazy_corridor(wire: bytes) -> bytes:
    if wire.startswith(b"WLK1"):
        from lattice_retriever_v1.walker_codec import decode_walker

        return decode_walker(wire)
    if not wire.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    raw_len, n_sym = struct.unpack_from("<IH", wire, off)
    off += 6
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    mode, wp, n_end, seed, oid = struct.unpack_from("<BIIBH", wire, off)
    alpha = SymbolAlphabet(symbols=symbols)
    if raw_len <= 1:
        return bytes([seed]) if raw_len == 1 else b""
    if mode == MODE_CONSTANT:
        return bytes([seed]) * raw_len
    return lazy_dots_to_bytes(alpha, mode, seed, n_end, oid)


def symbol_position_dot(alpha: SymbolAlphabet, symbol: int) -> tuple[int, int, int]:
    """Each symbol's lattice position — formula only."""
    p = alpha.prime_for(symbol)
    addr = intersect_primes((str(symbol), "w"), (p, DEFAULT_WALKER_PRIME), n=1)
    return addr.lattice_coords[0]
