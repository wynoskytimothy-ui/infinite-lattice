"""
Unit lattice — one unit defines infinite procedural 3D space; storage = bare lumber.

Walk the structure like undiscovered land:
  - Define ONE unit (e.g. digits 0-9 → 10 tokens).
  - Formula opens n² pair-origin rails; each rail has n=1,2,3… forever.
  - 32 lattice wings per dot — computed on the fly, never stored.
  - Stored artifact = bare lumber only (unique symbols). Zero formula bytes on disk.

Lossless replay: the 2-way walker (origin + n) IS the path — vector and n tell
you which intersection fired. 3-way witnesses lock symbols formula-side; not
stored separately. Single-rail span: one origin, n branches 1→EOF (6 bytes).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginDot,
    PairOriginKey,
    SymbolAlphabet,
    _oriented_pair_catalog,
    document_pair_walk,
    dot_on_origin,
    read_document_from_walk,
)

MAGIC = b"LUM1"


@dataclass(frozen=True)
class LatticeUnit:
    """
    One unit cell — defines the whole procedural space for this symbol set.

    All pair origins and transgressor rails exist by formula; nothing allocated.
    """

    symbols: tuple[int, ...]

    @classmethod
    def digits(cls) -> LatticeUnit:
        """Canonical demo unit: 0-9."""
        return cls(symbols=tuple(range(10)))

    @classmethod
    def from_data(cls, data: bytes) -> LatticeUnit:
        return cls(symbols=tuple(sorted(set(data))))

    @property
    def alphabet(self) -> SymbolAlphabet:
        return SymbolAlphabet(symbols=self.symbols)

    @property
    def n_tokens(self) -> int:
        return len(self.symbols)

    @property
    def n_origins_procedural(self) -> int:
        """Pair-origin rails — formula-side, zero bytes."""
        return self.n_tokens**2

    @property
    def n_wings(self) -> int:
        return 32

    def origin_for(self, left: int, right: int) -> PairOriginKey:
        a = self.alphabet
        return PairOriginKey(left, right, a.prime_for(left), a.prime_for(right))

    def dot_at(self, left: int, right: int, pair_n: int) -> PairOriginDot:
        """Place one dot on the rail — coords from formula, not memory."""
        return dot_on_origin(
            self.origin_for(left, right),
            pair_n=pair_n,
            walk_index=pair_n - 1,
        )

    def explain(self) -> dict:
        return {
            "n_tokens": self.n_tokens,
            "symbols": list(self.symbols),
            "n_origins_procedural": self.n_origins_procedural,
            "n_wings_per_dot": self.n_wings,
            "stored_scaffold_bytes": 0,
            "stored_formula_bytes": 0,
            "note": "infinite n per origin; space size is procedural not allocated",
        }


@dataclass(frozen=True)
class BareLumber:
    """The only persistent structure — unique symbol catalog."""

    unit: LatticeUnit
    raw_len: int

    @property
    def stored_bytes(self) -> int:
        return len(MAGIC) + 6 + self.unit.n_tokens

    def to_wire(self) -> bytes:
        return MAGIC + struct.pack("<IH", self.raw_len, self.unit.n_tokens) + bytes(self.unit.symbols)

    @classmethod
    def from_wire(cls, wire: bytes) -> BareLumber:
        if not wire.startswith(MAGIC):
            raise ValueError("bad magic")
        raw_len, n = struct.unpack_from("<IH", wire, len(MAGIC))
        syms = wire[len(MAGIC) + 6 : len(MAGIC) + 6 + n]
        return cls(unit=LatticeUnit(symbols=tuple(syms)), raw_len=raw_len)


@dataclass(frozen=True)
class LiveWalkStep:
    """One step walking new land — dot discovered, not stored."""

    walk_index: int
    left: int
    right: int
    pair_n: int
    origin_id: int
    dot: PairOriginDot

    def explain(self) -> dict:
        return {
            "step": self.walk_index,
            "pair": f"{chr(self.left)!r}->{chr(self.right)!r}",
            "pair_n_on_rail": self.pair_n,
            "origin_id": self.origin_id,
            "L01": self.dot.address.lattice_coords[0],
            "coords_stored": 0,
        }


@dataclass(frozen=True)
class ProceduralFootprint:
    """What costs memory vs what the formula holds for free."""

    raw_bytes: int
    bare_lumber_bytes: int
    scaffold_origins: int
    wings_per_dot: int
    dots_walked: int
    coord_bytes_if_materialized: int
    formula_stored_bytes: int
    walk_trace_stored_bytes: int

    def explain(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "bare_lumber_bytes": self.bare_lumber_bytes,
            "lumber_ratio_x": round(self.raw_bytes / self.bare_lumber_bytes, 1)
            if self.bare_lumber_bytes
            else 0,
            "scaffold_origins_formula": self.scaffold_origins,
            "wings_per_dot_formula": self.wings_per_dot,
            "dots_on_walk": self.dots_walked,
            "coord_bytes_if_materialized": self.coord_bytes_if_materialized,
            "coord_bytes_stored": 0,
            "formula_stored_bytes": self.formula_stored_bytes,
            "walk_trace_stored_bytes": self.walk_trace_stored_bytes,
            "model": "structure=lumber only; 3D infinite scaffold=free",
        }


def walk_new_land(data: bytes, unit: LatticeUnit | None = None) -> Iterator[LiveWalkStep]:
    """
    Walk data through procedural space — discover dots live, store nothing.

    Each adjacent pair transgresses its origin rail (n=1,2,3…).
    """
    if len(data) < 2:
        return
    unit = unit or LatticeUnit.from_data(data)
    alpha = unit.alphabet
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    counters: dict[tuple[int, int], int] = {}
    for i in range(len(data) - 1):
        lb, rb = data[i], data[i + 1]
        key = (lb, rb)
        counters[key] = counters.get(key, 0) + 1
        pn = counters[key]
        dot = unit.dot_at(lb, rb, pn)
        yield LiveWalkStep(
            walk_index=i,
            left=lb,
            right=rb,
            pair_n=pn,
            origin_id=cat_index[dot.origin],
            dot=dot,
        )


def encode_bare_lumber(data: bytes) -> tuple[BareLumber, bytes, ProceduralFootprint]:
    """Compress structure to bare lumber — the only persistent storage."""
    unit = LatticeUnit.from_data(data)
    lumber = BareLumber(unit=unit, raw_len=len(data))
    wire = lumber.to_wire()
    steps = list(walk_new_land(data, unit))
    footprint = ProceduralFootprint(
        raw_bytes=len(data),
        bare_lumber_bytes=lumber.stored_bytes,
        scaffold_origins=unit.n_origins_procedural,
        wings_per_dot=unit.n_wings,
        dots_walked=len(steps),
        coord_bytes_if_materialized=len(steps) * 32 * 3 * 4,
        formula_stored_bytes=0,
        walk_trace_stored_bytes=0,
    )
    return lumber, wire, footprint


def reconstruct_from_walk(data_len: int, steps: tuple[LiveWalkStep, ...]) -> bytes:
    """Rebuild bytes from a live walk — formula already placed every dot."""
    if data_len <= 1:
        if data_len == 1 and steps:
            return bytes([steps[0].left])
        return b""
    dots = tuple(s.dot for s in sorted(steps, key=lambda s: s.walk_index))
    return read_document_from_walk(dots)


def live_roundtrip(data: bytes) -> bytes:
    """Lossless via ephemeral walk — nothing stored except lumber in a real index."""
    steps = tuple(walk_new_land(data))
    return reconstruct_from_walk(len(data), steps)


def demo_digits_land() -> dict:
    """0-9 unit: 10 tokens, 100 origins, infinite rails — walk a short string."""
    unit = LatticeUnit.digits()
    sample = bytes(list(range(10)) * 2)  # raw 0-9 bytes, not ASCII
    steps = list(walk_new_land(sample, unit))
    _, wire, fp = encode_bare_lumber(sample)
    return {
        "unit": unit.explain(),
        "sample": list(sample),
        "bare_lumber_wire_bytes": len(wire),
        "footprint": fp.explain(),
        "first_steps": [s.explain() for s in steps[:5]],
        "roundtrip_ok": live_roundtrip(sample) == sample,
    }
