"""
Pair-origin vector codec — order compressed via per-pair transgressor rails.

User model:
  - Every oriented 2-way intersection opens a **new origin** (pair bank).
  - Each time that pair is found together, n transgresses on *that* origin: 1, 2, 3…
  - "the" → (t,h)@n=1 on th-origin, (h,e)@n=1 on he-origin, (e, )@n=1 on e·-origin.
  - "the the" → th-origin vector [1,2], he-origin [1,2], …
  - Doc order = walk reading (pair_origin, n) dots; formula regenerates coords from n.
  - Storage: alphabet + literal small-n walk on pair origins — not stored coordinates.
"""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass, field

from aethos_sequences import SequenceKind, make_chain

from lattice_retriever_v1.stage02_intersections import (
    IntersectionAddress,
    intersect_primes,
)

MAGIC = b"LXV1"


@dataclass
class SymbolAlphabet:
    symbols: tuple[int, ...]
    index_by_symbol: dict[int, int] = field(init=False)
    prime_by_index: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.index_by_symbol = {s: i for i, s in enumerate(self.symbols)}
        primes = make_chain(SequenceKind.PRIMES, max(len(self.symbols), 1))
        self.prime_by_index = tuple(int(primes[i]) for i in range(len(self.symbols)))

    @classmethod
    def from_bytes(cls, data: bytes) -> SymbolAlphabet:
        return cls(symbols=tuple(sorted(set(data))))

    def prime_for(self, byte: int) -> int:
        return self.prime_by_index[self.index_by_symbol[byte]]

    @property
    def n(self) -> int:
        return len(self.symbols)


@dataclass(frozen=True)
class PairOriginKey:
    """Oriented 2-way origin — one vector per (left→right) prime pair."""

    left_byte: int
    right_byte: int
    left_prime: int
    right_prime: int

    @property
    def oriented_primes(self) -> tuple[int, int]:
        return (self.left_prime, self.right_prime)


@dataclass(frozen=True)
class PairOriginDot:
    """One dot on a pair-origin rail: literal n = k-th co-occurrence of this pair."""

    origin: PairOriginKey
    pair_n: int
    walk_index: int
    address: IntersectionAddress

    def explain(self) -> dict:
        return {
            "origin": f"{chr(self.origin.left_byte)!r}->{chr(self.origin.right_byte)!r}",
            "pair_n": self.pair_n,
            "walk_index": self.walk_index,
            "invoke_order": list(self.origin.oriented_primes),
            "composite": self.address.composite,
            "lattice_L01": self.address.lattice_coords[0],
        }


@dataclass
class PairOriginVector:
    """All dots placed on one pair origin — literal n sequence 1,2,3,…"""

    origin: PairOriginKey
    n_sequence: tuple[int, ...]
    walk_indices: tuple[int, ...]

    def explain(self) -> dict:
        return {
            "origin": f"{chr(self.origin.left_byte)!r}->{chr(self.origin.right_byte)!r}",
            "n_vector": list(self.n_sequence),
            "walk_at": list(self.walk_indices),
        }


def _oriented_pair_catalog(alpha: SymbolAlphabet) -> tuple[PairOriginKey, ...]:
    return tuple(
        PairOriginKey(
            left_byte=a,
            right_byte=b,
            left_prime=alpha.prime_for(a),
            right_prime=alpha.prime_for(b),
        )
        for a in alpha.symbols
        for b in alpha.symbols
    )


def dot_on_origin(
    origin: PairOriginKey,
    *,
    pair_n: int,
    walk_index: int,
) -> PairOriginDot:
    addr = intersect_primes(
        (str(origin.left_byte), str(origin.right_byte)),
        origin.oriented_primes,
        start_index=walk_index,
        n=pair_n,
    )
    return PairOriginDot(
        origin=origin,
        pair_n=pair_n,
        walk_index=walk_index,
        address=addr,
    )


def document_pair_walk(
    data: bytes,
    alpha: SymbolAlphabet | None = None,
) -> tuple[PairOriginDot, ...]:
    """
    Walk the document: each adjacent pair places a dot on its own origin.
    pair_n transgresses 1,2,3… every time that oriented pair appears again.
    """
    if len(data) < 2:
        return ()
    alpha = alpha or SymbolAlphabet.from_bytes(data)
    catalog = _oriented_pair_catalog(alpha)
    key_index = {(k.left_byte, k.right_byte): k for k in catalog}
    counters: dict[tuple[int, int], int] = defaultdict(int)
    out: list[PairOriginDot] = []
    for i in range(len(data) - 1):
        lb, rb = data[i], data[i + 1]
        origin = key_index[(lb, rb)]
        counters[(lb, rb)] += 1
        out.append(dot_on_origin(origin, pair_n=counters[(lb, rb)], walk_index=i))
    return tuple(out)


def pair_origin_vectors(
    walk: tuple[PairOriginDot, ...],
) -> tuple[PairOriginVector, ...]:
    """Group walk into per-origin n vectors — literal 1,2,3 on each pair rail."""
    by_key: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    origins: dict[tuple[int, int], PairOriginKey] = {}
    for d in walk:
        k = (d.origin.left_byte, d.origin.right_byte)
        origins[k] = d.origin
        by_key[k].append((d.pair_n, d.walk_index))
    vecs: list[PairOriginVector] = []
    for k in sorted(by_key.keys()):
        entries = sorted(by_key[k], key=lambda t: t[1])
        vecs.append(
            PairOriginVector(
                origin=origins[k],
                n_sequence=tuple(n for n, _ in entries),
                walk_indices=tuple(w for _, w in entries),
            )
        )
    return tuple(vecs)


def read_document_from_walk(walk: tuple[PairOriginDot, ...]) -> bytes:
    """Read dots in walk order — each origin dot contributes one step of the chain."""
    if not walk:
        return b""
    ordered = tuple(sorted(walk, key=lambda d: d.walk_index))
    buf = bytearray([ordered[0].origin.left_byte, ordered[0].origin.right_byte])
    for d in ordered[1:]:
        if d.origin.left_byte != buf[-1]:
            raise ValueError(
                f"walk break at {d.walk_index}: "
                f"{d.origin.left_byte} != {buf[-1]}"
            )
        buf.append(d.origin.right_byte)
    return bytes(buf)


def regenerate_dot_from_formula(
    origin: PairOriginKey,
    *,
    pair_n: int,
    walk_index: int,
) -> PairOriginDot:
    return dot_on_origin(origin, pair_n=pair_n, walk_index=walk_index)


def _bits_per_index(n: int) -> int:
    if n <= 1:
        return 1
    return max(1, math.ceil(math.log2(n)))


def _pack_pairs(items: list[tuple[int, int]], *, pair_bits: int, n_bits: int) -> bytes:
    if not items:
        return b""
    width = pair_bits + n_bits
    out = bytearray()
    acc = 0
    nacc = 0
    pm = (1 << pair_bits) - 1
    nm = (1 << n_bits) - 1
    for pid, pn in items:
        acc = (acc << width) | ((pid & pm) << n_bits) | (pn & nm)
        nacc += width
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
            acc &= (1 << nacc) - 1 if nacc else 0
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def _unpack_pairs(
    blob: bytes,
    *,
    n_values: int,
    pair_bits: int,
    n_bits: int,
) -> list[tuple[int, int]]:
    if n_values == 0:
        return []
    width = pair_bits + n_bits
    out: list[tuple[int, int]] = []
    acc = 0
    nacc = 0
    pos = 0
    nm = (1 << n_bits) - 1
    while len(out) < n_values:
        if nacc < width and pos < len(blob):
            acc = (acc << 8) | blob[pos]
            nacc += 8
            pos += 1
            continue
        if nacc < width:
            break
        nacc -= width
        word = (acc >> nacc) & ((1 << width) - 1)
        acc &= (1 << nacc) - 1 if nacc else 0
        out.append((word >> n_bits, word & nm))
    if len(out) != n_values:
        raise ValueError(f"unpack short: {len(out)} != {n_values}")
    return out


@dataclass(frozen=True)
class PairVectorLedger:
    raw_bytes: int
    n_symbols: int
    n_walk_steps: int
    n_pair_origins_used: int
    n_pair_origins_max: int
    max_pair_n: int
    alphabet_bytes: int
    walk_bytes: int
    header_bytes: int
    coord_bytes_if_stored: int
    total_bytes: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.total_bytes if self.total_bytes else 0.0

    def explain(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "n_symbols": self.n_symbols,
            "n_walk_steps": self.n_walk_steps,
            "n_pair_origins_used": self.n_pair_origins_used,
            "n_pair_origins_max": self.n_pair_origins_max,
            "max_pair_n": self.max_pair_n,
            "alphabet_bytes": self.alphabet_bytes,
            "walk_bytes": self.walk_bytes,
            "coord_bytes_if_stored": self.coord_bytes_if_stored,
            "total_bytes": self.total_bytes,
            "compression_ratio_x": round(self.ratio, 3),
            "model": "each oriented pair = origin; pair_n = 1,2,3 literal on that rail",
        }


def encode_pair_vectors(
    data: bytes,
) -> tuple[bytes, PairVectorLedger, tuple[PairOriginDot, ...], tuple[PairOriginVector, ...]]:
    alpha = SymbolAlphabet.from_bytes(data)
    walk = document_pair_walk(data, alpha)
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    steps = [(cat_index[d.origin], d.pair_n) for d in walk]
    max_n = max((pn for _, pn in steps), default=1)
    pair_bits = _bits_per_index(len(catalog))
    n_bits = max(1, math.ceil(math.log2(max_n + 1)))
    packed = _pack_pairs(steps, pair_bits=pair_bits, n_bits=n_bits)
    header = struct.pack("<IIHBB", len(data), len(walk), alpha.n, pair_bits, n_bits)
    tail = data if len(data) <= 1 else b""
    payload = MAGIC + header + bytes(alpha.symbols) + packed + tail
    vectors = pair_origin_vectors(walk)
    coords_if_stored = len(walk) * 32 * 3 * 4
    ledger = PairVectorLedger(
        raw_bytes=len(data),
        n_symbols=alpha.n,
        n_walk_steps=len(walk),
        n_pair_origins_used=len(vectors),
        n_pair_origins_max=len(catalog),
        max_pair_n=max_n,
        alphabet_bytes=2 + alpha.n,
        walk_bytes=len(packed),
        header_bytes=len(MAGIC) + len(header),
        coord_bytes_if_stored=coords_if_stored,
        total_bytes=len(payload),
    )
    return payload, ledger, walk, vectors


def decode_pair_vectors(payload: bytes) -> bytes:
    if not payload.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    n_raw, n_steps, n_sym, pair_bits, n_bits = struct.unpack_from("<IIHBB", payload, off)
    off += 12
    symbols = payload[off : off + n_sym]
    off += n_sym
    alpha = SymbolAlphabet(symbols=tuple(symbols))
    if n_raw <= 1:
        return payload[-n_raw:] if n_raw else b""
    catalog = _oriented_pair_catalog(alpha)
    steps = _unpack_pairs(payload[off:], n_values=n_steps, pair_bits=pair_bits, n_bits=n_bits)
    walk: list[PairOriginDot] = []
    for wi, (pid, pn) in enumerate(steps):
        walk.append(regenerate_dot_from_formula(catalog[pid], pair_n=pn, walk_index=wi))
    return read_document_from_walk(tuple(walk))


def analyze_pair_vectors(
    data: bytes,
) -> tuple[PairVectorLedger, tuple[PairOriginDot, ...], tuple[PairOriginVector, ...]]:
    _, ledger, walk, vectors = encode_pair_vectors(data)
    return ledger, walk, vectors


# ---------------------------------------------------------------------------
# Procedural index — dots live on formula rails; store alphabet only.
# ---------------------------------------------------------------------------


@dataclass
class ProceduralLatticeIndex:
    """
    Index bounded by unique symbols — not corpus size.

    Each oriented 2-way pair opens one origin; transgressor n = 1,2,3… already
    on that rail (procedural). Ingest *reads/places* dots via formula; nothing
    materialized. One new symbol branches all composites below via FTA.
    """

    alphabet: SymbolAlphabet

    @classmethod
    def from_bytes(cls, data: bytes) -> ProceduralLatticeIndex:
        return cls(alphabet=SymbolAlphabet.from_bytes(data))

    @property
    def n_symbols(self) -> int:
        return self.alphabet.n

    @property
    def n_pair_origins(self) -> int:
        return self.alphabet.n ** 2

    @property
    def stored_bytes(self) -> int:
        """Only the alphabet is persisted — same for 1 MB or 80 GB corpus."""
        return self.alphabet.n + 2

    def origin_for(self, left: int, right: int) -> PairOriginKey:
        return PairOriginKey(
            left_byte=left,
            right_byte=right,
            left_prime=self.alphabet.prime_for(left),
            right_prime=self.alphabet.prime_for(right),
        )

    def read_dot(self, origin: PairOriginKey, pair_n: int) -> PairOriginDot:
        """Dot at n on this pair's vector — already there; formula recomputes."""
        return dot_on_origin(origin, pair_n=pair_n, walk_index=pair_n - 1)

    def place_walk(self, data: bytes) -> tuple[PairOriginDot, ...]:
        """
        Walk corpus: order goes into each pair's n-vector (1,2,3…).
        Returns dots for glass-box readout; index size unchanged.
        """
        return document_pair_walk(data, self.alphabet)

    def branch_symbol(self, symbol: int) -> tuple[SymbolAlphabet, int]:
        """
        Hilbert append: one new symbol → prime branch → all new pair origins
        and composites open below via formula (0 stored rows).
        """
        if symbol in self.alphabet.index_by_symbol:
            return self.alphabet, 0
        old_n = self.alphabet.n
        new_symbols = tuple(sorted((*self.alphabet.symbols, symbol)))
        new_alpha = SymbolAlphabet(symbols=new_symbols)
        new_origins = new_alpha.n**2 - old_n**2
        self.alphabet = new_alpha
        return new_alpha, new_origins


@dataclass(frozen=True)
class ProceduralIndexLedger:
    """Same stored size regardless of how much data was walked."""

    n_symbols: int
    n_pair_origins: int
    stored_bytes: int
    n_dots_placed: int
    coord_bytes_if_stored: int
    model: str = "alphabet only; dots procedural on pair rails"

    def explain(self) -> dict:
        return {
            "n_symbols": self.n_symbols,
            "n_pair_origins_formula": self.n_pair_origins,
            "stored_bytes": self.stored_bytes,
            "n_dots_placed_this_walk": self.n_dots_placed,
            "coord_bytes_if_stored": self.coord_bytes_if_stored,
            "model": self.model,
        }


def procedural_index_ledger(data: bytes) -> tuple[ProceduralLatticeIndex, ProceduralIndexLedger]:
    idx = ProceduralLatticeIndex.from_bytes(data)
    walk = idx.place_walk(data)
    coords_if = len(walk) * 32 * 3 * 4
    return idx, ProceduralIndexLedger(
        n_symbols=idx.n_symbols,
        n_pair_origins=idx.n_pair_origins,
        stored_bytes=idx.stored_bytes,
        n_dots_placed=len(walk),
        coord_bytes_if_stored=coords_if,
    )


# Back-compat aliases for prior dot-chain naming
IntersectionDot = PairOriginDot
document_dots = document_pair_walk
encode_dots = encode_pair_vectors
decode_dots = decode_pair_vectors
analyze_dots = analyze_pair_vectors
read_document_from_dots = read_document_from_walk
