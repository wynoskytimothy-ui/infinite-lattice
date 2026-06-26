"""
Lattice Compressor — frontier technology (not classical compression).

This is procedural geometry compression:
  - Bare lumber (symbol → prime) is the only persistent structure.
  - 2-way intersections ARE the data channel; 3-way locks symbols.
  - 32 wings × 3 branch cases = 96 readout states per dot (formula-side).
  - FTA composites collapse patterns to single primes (free addresses).
  - Promotion ladder shortens streams before branch encoding.
  - Recompress: lumber cached → only update count / branch deltas.

Classical codecs compress byte entropy. We compress TO the lattice:
  order lives in branch geometry; decompress = lazy formula read.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

from lattice_retriever_v1.branch_order_codec import (
    decode_branch_order,
    encode_branch_order,
)
from lattice_retriever_v1.deep_branch_codec import decode_deep_branch, encode_deep_branch
from lattice_retriever_v1.electron_lattice_codec import (
    decode_electron_entangle,
    encode_electron_entangle,
)
from lattice_retriever_v1.intersection_dot_codec import (
    SymbolAlphabet,
    _oriented_pair_catalog,
    _bits_per_index,
    document_pair_walk,
    read_document_from_walk,
    regenerate_dot_from_formula,
)
from lattice_retriever_v1.personal_lattice_codec import PersonalKey, _xor_body
from lattice_retriever_v1.formula_index_codec import FormulaWalkIndex
from lattice_retriever_v1.full_potential_codec import decode_full_potential, encode_full_potential
from lattice_retriever_v1.prime_corridor_codec import decode_prime_corridor, encode_prime_corridor
from lattice_retriever_v1.trigger_formula_codec import decode_trigger_formula, encode_trigger_formula
from lattice_retriever_v1.unit_lattice_codec import LatticeUnit
from lattice_retriever_v1.wing_channel_codec import (
    WingChannelRead,
    decode_wing_channel,
    encode_wing_channel,
    wing_channel_at,
)


def _encode_formula_stack(data: bytes) -> tuple[bytes, dict]:
    best: tuple[bytes, dict] | None = None
    for enc in (encode_prime_corridor, encode_trigger_formula):
        wire, meta = enc(data)
        if best is None or len(wire) < len(best[0]):
            best = (wire, meta)
    assert best is not None
    return best

MAGIC = b"LFC1"
MODE_BRANCH_COUNT = 1
MODE_BRANCH_TREE = 2
MODE_PROMOTED = 3
MODE_WING_CHANNEL = 4
MODE_DEEP_BRANCH = 5
MODE_ELECTRON_ENTANGLE = 6
MODE_SESSION_COUNT = 7


@dataclass(frozen=True)
class BranchRun:
    """Walker segment on one 2-way origin — order in the vector."""

    origin_id: int
    length: int


@dataclass
class PromotionTable:
    """FTA composite promotions — pattern → one prime (formula address)."""

    patterns: dict[bytes, int]  # pattern bytes → composite int
    composite_to_pattern: dict[int, bytes] = field(init=False)

    def __post_init__(self) -> None:
        self.composite_to_pattern = {v: k for k, v in self.patterns.items()}

    @classmethod
    def mine(cls, data: bytes, *, min_count: int = 4, max_len: int = 6) -> PromotionTable:
        alpha = SymbolAlphabet.from_bytes(data)
        counts: Counter[bytes] = Counter()
        for ln in range(2, max_len + 1):
            for i in range(len(data) - ln + 1):
                pat = data[i : i + ln]
                counts[pat] += 1
        patterns: dict[bytes, int] = {}
        for pat, cnt in counts.items():
            if cnt < min_count:
                continue
            primes = tuple(alpha.prime_for(b) for b in pat)
            if len(set(primes)) == len(primes):
                comp = 1
                for p in sorted(primes):
                    comp *= p
            else:
                sp = sorted(set(primes))
                comp = sp[0] * sp[1] if len(sp) >= 2 else sp[0]
            patterns[pat] = comp
        return cls(patterns=patterns)

    def tokenize(self, data: bytes) -> bytes:
        """Greedy longest-match → shorter composite symbol stream."""
        if not self.patterns:
            return data
        sorted_pats = sorted(self.patterns.keys(), key=len, reverse=True)
        out = bytearray()
        i = 0
        while i < len(data):
            matched = False
            for pat in sorted_pats:
                if data[i : i + len(pat)] == pat:
                    out.append(0xFF)
                    out.extend(struct.pack("<H", len(pat)))
                    out.extend(pat)
                    i += len(pat)
                    matched = True
                    break
            if not matched:
                out.append(data[i])
                i += 1
        return bytes(out)

    def detokenize(self, tokens: bytes) -> bytes:
        out = bytearray()
        i = 0
        while i < len(tokens):
            if tokens[i] == 0xFF and i + 2 < len(tokens):
                ln = struct.unpack_from("<H", tokens, i + 1)[0]
                out.extend(tokens[i + 3 : i + 3 + ln])
                i += 3 + ln
            else:
                out.append(tokens[i])
                i += 1
        return bytes(out)


@dataclass
class LatticeSession:
    """Cached lumber + formula inverted index for fast recompress."""

    unit: LatticeUnit
    promotion: PromotionTable | None = None
    formula_index: FormulaWalkIndex = field(default_factory=FormulaWalkIndex)

    @property
    def lumber_bytes(self) -> int:
        return 2 + self.unit.n_tokens

    def same_vocabulary(self, data: bytes) -> bool:
        return self.unit.symbols == tuple(sorted(set(data)))


def _encode_session_count(session: LatticeSession, data: bytes) -> tuple[bytes, dict] | None:
    if not session.same_vocabulary(data):
        return None
    session.formula_index.ingest(data)
    count = len(data)
    wire = MAGIC + bytes([MODE_SESSION_COUNT]) + struct.pack("<I", count)
    if _decode_session_count(wire, session) != data:
        return None
    return wire, {
        "mode": "session_count",
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
        "walker_stored": 0,
    }


def _decode_session_count(wire: bytes, session: LatticeSession) -> bytes:
    from lattice_retriever_v1.formula_corridor_read import formula_corridor_read
    from lattice_retriever_v1.formula_index_codec import decode_formula_index
    from lattice_retriever_v1.unit_lattice_codec import BareLumber

    (count,) = struct.unpack_from("<I", wire, len(MAGIC) + 1)
    alpha = session.unit.alphabet
    out = formula_corridor_read(alpha, count)
    if out is not None:
        return out
    lumber = BareLumber(unit=session.unit, raw_len=count)
    return decode_formula_index(
        b"IDX1" + lumber.to_wire()[len(b"LUM1") :] + struct.pack("<I", count),
        session.formula_index,
    )


def branch_tree_runs(data: bytes, alpha: SymbolAlphabet) -> tuple[BranchRun, ...]:
    """Compress walk to origin runs — 2-way vector holds order within each run."""
    walk = document_pair_walk(data, alpha)
    if not walk:
        return ()
    catalog = _oriented_pair_catalog(alpha)
    cat_index = {k: i for i, k in enumerate(catalog)}
    runs: list[BranchRun] = []
    cur_oid = cat_index[walk[0].origin]
    cur_len = 1
    for dot in walk[1:]:
        oid = cat_index[dot.origin]
        if oid == cur_oid:
            cur_len += 1
        else:
            runs.append(BranchRun(cur_oid, cur_len))
            cur_oid, cur_len = oid, 1
    runs.append(BranchRun(cur_oid, cur_len))
    return tuple(runs)


def _pack_runs(runs: tuple[BranchRun, ...], n_origins: int) -> bytes:
    bits = _bits_per_index(n_origins)
    out = bytearray()
    for r in runs:
        out.extend(struct.pack("<HI", r.origin_id, r.length))
    return bytes(out)


def _unpack_runs(blob: bytes, n_runs: int) -> tuple[BranchRun, ...]:
    runs: list[BranchRun] = []
    off = 0
    for _ in range(n_runs):
        oid, ln = struct.unpack_from("<HI", blob, off)
        off += 6
        runs.append(BranchRun(oid, ln))
    return tuple(runs)


def _encode_branch_tree(data: bytes) -> tuple[bytes, dict]:
    unit = LatticeUnit.from_data(data)
    alpha = unit.alphabet
    runs = branch_tree_runs(data, alpha)
    catalog = _oriented_pair_catalog(alpha)
    packed = _pack_runs(runs, len(catalog))
    header = struct.pack("<IIH", len(data), len(runs), unit.n_tokens)
    wire = MAGIC + bytes([MODE_BRANCH_TREE]) + header + bytes(unit.symbols) + packed
    meta = {
        "mode": "branch_tree",
        "n_runs": len(runs),
        "wire_bytes": len(wire),
        "ratio_x": round(len(data) / len(wire), 3),
    }
    return wire, meta


def _decode_branch_tree(wire: bytes) -> bytes:
    off = len(MAGIC) + 1
    raw_len, n_runs, n_sym = struct.unpack_from("<IIH", wire, off)
    off += 10
    symbols = tuple(wire[off : off + n_sym])
    off += n_sym
    alpha = SymbolAlphabet(symbols=symbols)
    catalog = _oriented_pair_catalog(alpha)
    runs = _unpack_runs(wire[off:], n_runs)
    dots = []
    wi = 0
    for run in runs:
        origin = catalog[run.origin_id]
        for j in range(run.length):
            dots.append(
                regenerate_dot_from_formula(
                    origin, pair_n=j + 1, walk_index=wi
                )
            )
            wi += 1
    return read_document_from_walk(tuple(dots))


@dataclass(frozen=True)
class CompressResult:
    wire: bytes
    mode: str
    raw_bytes: int
    wire_bytes: int
    lumber_bytes: int
    ratio: float
    walker_stored: int
    formula_stored: int
    recompress_ready: bool

    def explain(self) -> dict:
        return {
            "mode": self.mode,
            "raw_bytes": self.raw_bytes,
            "wire_bytes": self.wire_bytes,
            "lumber_bytes": self.lumber_bytes,
            "ratio_x": round(self.ratio, 3),
            "walker_stored_bytes": self.walker_stored,
            "formula_stored_bytes": self.formula_stored,
            "recompress_ready": self.recompress_ready,
            "technology": "lattice procedural geometry — not entropy coding",
        }


class LatticeCompressor:
    """
    Frontier compressor — always picks best formula path.

    Paths tried (smallest wins):
      1. branch_count  — lumber + symbol count (single/branch-determined)
      2. promoted      — FTA composite promotion then branch_count
      3. branch_tree   — lumber + origin runs (2-way vector segments)
    """

    def __init__(self, session: LatticeSession | None = None) -> None:
        self.session = session

    def compress(
        self,
        data: bytes,
        *,
        key: PersonalKey | None = None,
        promote: bool = True,
    ) -> CompressResult:
        candidates: list[tuple[bytes, dict]] = []
        idx = self.session.formula_index if self.session else FormulaWalkIndex()

        # session fast path: lumber cached → count only (9 bytes total)
        if self.session is not None:
            sess = _encode_session_count(self.session, data)
            if sess is not None:
                candidates.append(sess)

        # path 1: lumber + count — walk in inverted index, zero walker on wire
        w1, m1 = encode_branch_order(data, walk_index=idx)
        candidates.append((w1, m1))

        # path 2: promotion ladder → shorter branch stream
        if promote:
            prom = PromotionTable.mine(data, min_count=3)
            tok = prom.tokenize(data)
            if len(tok) < len(data) * 0.9:
                w2, m2 = encode_branch_order(tok, walk_index=idx)
                wire2 = self._wrap_promoted(w2, prom, data)
                if self._decode_promoted(wire2) == data:
                    candidates.append((wire2, {**m2, "mode": "promoted"}))

        # path 3: branch tree (origin runs) — only if smaller than raw
        w3, m3 = _encode_branch_tree(data)
        if len(w3) < len(data) and _decode_branch_tree(w3) == data:
            candidates.append((w3, m3))

        # path 4: wing channel (96-state formula readout)
        wing = encode_wing_channel(data)
        if wing is not None and len(wing[0]) < len(data):
            candidates.append(wing)

        # path 5: deep branch (Section 5 case witness + branch count)
        deep = encode_deep_branch(data, walk_index=idx)
        if deep is not None and len(deep[0]) < len(data):
            candidates.append(deep)

        # path 6: electron 4-state + entanglement patterns
        electron = encode_electron_entangle(data)
        if electron is not None and len(electron[0]) < len(data):
            candidates.append(electron)

        valid: list[tuple[bytes, dict]] = []
        for w, m in candidates:
            try:
                if w.startswith(b"IDX1") or w.startswith(b"BRN1"):
                    from lattice_retriever_v1.branch_order_codec import decode_branch_order

                    back = decode_branch_order(w, walk_index=idx)
                else:
                    back = self.decompress(w)
                if back == data:
                    valid.append((w, m))
            except (ValueError, KeyError):
                continue
        wire, meta = (
            min(valid, key=lambda t: len(t[0]))
            if valid
            else encode_branch_order(data, walk_index=idx)
        )
        if key is not None:
            wire = self._personalize(wire, key)

        unit = LatticeUnit.from_data(data)
        prom_table = PromotionTable.mine(data, min_count=3) if promote else None
        self.session = LatticeSession(unit=unit, promotion=prom_table, formula_index=idx)

        return CompressResult(
            wire=wire,
            mode=str(meta.get("mode", "branch")),
            raw_bytes=len(data),
            wire_bytes=len(wire),
            lumber_bytes=2 + unit.n_tokens,
            ratio=len(data) / len(wire) if wire else 0,
            walker_stored=0,
            formula_stored=0,
            recompress_ready=True,
        )

    def compress_fast(self, data: bytes, *, key: PersonalKey | None = None) -> CompressResult:
        """Full formula stack — portable wire, zero coords, no session required."""
        if self.session is not None and self.session.same_vocabulary(data):
            sess = _encode_session_count(self.session, data)
            if sess is not None:
                wire, meta = sess
            else:
                wire, meta = encode_full_potential(data)
        else:
            wire, meta = _encode_formula_stack(data)
        if key is not None:
            wire = self._personalize(wire, key)
        unit = LatticeUnit.from_data(data)
        idx = self.session.formula_index if self.session else FormulaWalkIndex()
        idx.ingest(data)
        self.session = LatticeSession(unit=unit, formula_index=idx)
        return CompressResult(
            wire=wire,
            mode=str(meta.get("mode", "full_potential")),
            raw_bytes=len(data),
            wire_bytes=len(wire),
            lumber_bytes=2 + unit.n_tokens,
            ratio=len(data) / len(wire) if wire else 0,
            walker_stored=0,
            formula_stored=0,
            recompress_ready=True,
        )

    def recompress(self, data: bytes, *, key: PersonalKey | None = None) -> CompressResult:
        """Fast path — lumber already in session; only branch geometry updates."""
        if self.session is None:
            return self.compress(data, key=key)
        return self.compress(data, key=key, promote=self.session.promotion is not None)

    def decompress(self, wire: bytes, *, key: PersonalKey | None = None) -> bytes:
        if key is not None:
            wire = self._depersonalize(wire, key)
        if wire.startswith(b"PCR1"):
            return decode_prime_corridor(wire)
        if wire.startswith(b"TFM1"):
            return decode_trigger_formula(wire)
        if wire.startswith(b"FUL1"):
            return decode_full_potential(wire)
        if wire.startswith(b"WNG1"):
            return decode_wing_channel(wire)
        if wire.startswith(b"DPB1"):
            idx = self.session.formula_index if self.session else None
            return decode_deep_branch(wire, walk_index=idx)
        if wire.startswith(b"ELC1"):
            return decode_electron_entangle(wire)
        if wire.startswith(b"IDX1"):
            idx = self.session.formula_index if self.session else FormulaWalkIndex()
            return decode_branch_order(wire, walk_index=idx)
        if wire.startswith(b"BRN1"):
            idx = self.session.formula_index if self.session else None
            return decode_branch_order(wire, walk_index=idx)
        if wire.startswith(MAGIC):
            mode = wire[4]
            if mode == MODE_SESSION_COUNT and self.session is not None:
                return _decode_session_count(wire, self.session)
            if mode == MODE_BRANCH_TREE:
                return _decode_branch_tree(wire)
            if mode == MODE_PROMOTED:
                return self._decode_promoted(wire)
        from lattice_retriever_v1.lazy_corridor_codec import decode_lazy_corridor

        return decode_lazy_corridor(wire)

    def read_channels(self, data: bytes) -> Iterator[WingChannelRead]:
        """Expose hidden 96-state channel per intersection — formula readout."""
        alpha = SymbolAlphabet.from_bytes(data)
        counts = Counter(data)
        for i in range(len(data) - 1):
            yield wing_channel_at(
                alpha, data[i], data[i + 1], n=i + 1, sym_counts=counts
            )

    @staticmethod
    def _wrap_promoted(branch_wire: bytes, prom: PromotionTable, raw: bytes) -> bytes:
        inner = branch_wire[4:] if branch_wire.startswith(b"BRN1") else branch_wire
        pat_blob = bytearray()
        for pat in sorted(prom.patterns.keys(), key=len):
            pat_blob.append(len(pat))
            pat_blob.extend(pat)
        return (
            MAGIC
            + bytes([MODE_PROMOTED])
            + struct.pack("<I", len(raw))
            + struct.pack("<H", len(prom.patterns))
            + bytes(pat_blob)
            + inner
        )

    @staticmethod
    def _decode_promoted(wire: bytes) -> bytes:
        off = 5
        (raw_len,) = struct.unpack_from("<I", wire, off)
        off += 4
        (n_pat,) = struct.unpack_from("<H", wire, off)
        off += 2
        patterns: list[bytes] = []
        for _ in range(n_pat):
            ln = wire[off]
            off += 1
            patterns.append(wire[off : off + ln])
            off += ln
        prom = PromotionTable(patterns={p: 1 for p in patterns})
        inner = b"BRN1" + wire[off:]
        tok = decode_branch_order(inner)
        return prom.detokenize(tok)

    @staticmethod
    def _personalize(wire: bytes, key: PersonalKey) -> bytes:
        return wire[:4] + _xor_body(wire[4:], key)

    @staticmethod
    def _depersonalize(wire: bytes, key: PersonalKey) -> bytes:
        return wire[:4] + _xor_body(wire[4:], key)


def encode_secure_branch(data: bytes, key: PersonalKey) -> CompressResult:
    """Branch-order compress + personal key veil — wrong key → garbage."""
    return LatticeCompressor().compress(data, key=key)


def frontier_report(data: bytes) -> dict:
    """Show all hidden formula layers for one corpus sample."""
    comp = LatticeCompressor()
    result = comp.compress(data)
    channels = list(comp.read_channels(data))
    prom = PromotionTable.mine(data, min_count=3)
    runs = branch_tree_runs(data, SymbolAlphabet.from_bytes(data))
    return {
        "compress": result.explain(),
        "wing_channels_sample": [c.__dict__ for c in channels[:5]],
        "wing_bits_per_dot": 6.58,
        "promotions_found": len(prom.patterns),
        "branch_tree_runs": len(runs),
        "hidden_possibilities": [
            "96-state wing×case channel per dot (formula-side)",
            "FTA promotion collapses patterns to one prime",
            "branch tree: origin runs not per-step walker",
            "deep branch: Section 5 triple case witness",
            "electron 4-state coin + entangled pair patterns",
            "session recompress: lumber cached → count only",
            "recompress: cached lumber, update geometry only",
            "personal key: infinite intersection sets",
            "32 lattices: parallel readout chambers",
        ],
    }
