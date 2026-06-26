"""
Lattice storage codec — compress raw data for storage (not RAG).

Primary artifact: dot blob (see dot_blob_codec) — token catalog + dot addresses.
No formula on disk. Pattern promotion (LST1) optional layer for extra shrink.

Model:
  - Unique symbols → token catalog (raw token size — same random or not).
  - Dot blob = (pair_origin, n) addresses; formula in code reconstructs data + coords.
  - 32 lattice chambers on read — never stored.
"""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from aethos_sequences import SequenceKind, make_chain

from lattice_retriever_v1.intersection_dot_codec import (
    PairOriginKey,
    SymbolAlphabet,
    document_pair_walk,
    dot_on_origin,
)
from lattice_retriever_v1.stage02_intersections import intersect_primes
from lattice_retriever_v1.stage03_rotation import wing_from_frequency_profile

# Re-export dot-blob path (primary storage artifact per user model)
from lattice_retriever_v1.dot_blob_codec import (  # noqa: E402
    DotBlob,
    DotBlobLedger,
    compress_to_dot_blob,
    reconstruct_from_dot_blob,
    reconstruct_from_blob,
    formula_regenerate_dot,
)

MAGIC = b"LST1"
FORMULA_HEADER_BYTES = 8  # magic + version/constants slot
MAX_PATTERN_LEN = 8
MAX_PATTERNS = 4096


@dataclass(frozen=True)
class LatticePattern:
    """Promoted composite — one token replacing a byte sequence."""

    pattern_id: int
    bytes_seq: bytes
    quadrant: int  # 1..32
    composite: int
    lattice_L01: tuple[int, int, int]
    occurrence_count: int

    def explain(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "bytes": list(self.bytes_seq),
            "quadrant": self.quadrant,
            "composite": self.composite,
            "lattice_L01": self.lattice_L01,
            "occurrences": self.occurrence_count,
        }


@dataclass(frozen=True)
class StorageLedger:
    raw_bytes: int
    n_atomic_symbols: int
    n_patterns: int
    n_effective_tokens: int
    n_stream_tokens: int
    alphabet_bytes: int
    pattern_table_bytes: int
    stream_bytes: int
    formula_header_bytes: int
    total_stored_bytes: int
    coord_bytes_if_stored: int
    pair_dots_placed: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.total_stored_bytes if self.total_stored_bytes else 0.0

    @property
    def index_only_bytes(self) -> int:
        """Alphabet + patterns + formula — no occurrence stream."""
        return self.alphabet_bytes + self.pattern_table_bytes + self.formula_header_bytes

    def explain(self) -> dict:
        return {
            "raw_bytes": self.raw_bytes,
            "raw_mb": round(self.raw_bytes / 1e6, 3),
            "n_atomic_symbols": self.n_atomic_symbols,
            "n_patterns": self.n_patterns,
            "n_effective_tokens": self.n_effective_tokens,
            "n_stream_tokens": self.n_stream_tokens,
            "alphabet_bytes": self.alphabet_bytes,
            "pattern_table_bytes": self.pattern_table_bytes,
            "stream_bytes": self.stream_bytes,
            "index_only_bytes": self.index_only_bytes,
            "total_stored_bytes": self.total_stored_bytes,
            "total_stored_mb": round(self.total_stored_bytes / 1e6, 6),
            "compression_ratio_x": round(self.ratio, 3),
            "coord_bytes_if_stored": self.coord_bytes_if_stored,
            "pair_dots_formula_side": self.pair_dots_placed,
            "model": "storage: alphabet + lattice patterns + token stream; dots free",
        }


def _pattern_lattice(
    alpha: SymbolAlphabet,
    seq: bytes,
    *,
    symbol_counts: Counter[int],
) -> tuple[int, int, tuple[int, int, int]]:
    """32-chamber placement for a byte pattern via pair meet + frequency rotation."""
    if len(seq) == 2:
        lp, rp = alpha.prime_for(seq[0]), alpha.prime_for(seq[1])
        origin = PairOriginKey(seq[0], seq[1], lp, rp)
        dot = dot_on_origin(origin, pair_n=1, walk_index=0)
        q = wing_from_frequency_profile((symbol_counts[seq[0]], symbol_counts[seq[1]]))
        return dot.address.composite, q, dot.address.lattice_coords[0]
    # 3+ bytes: 3-way meet on first three + wing from profile
    chars = tuple(str(b) for b in seq[:3])
    primes = tuple(alpha.prime_for(b) for b in seq[:3])
    addr = intersect_primes(chars, primes, start_index=0, n=1)
    profile = tuple(symbol_counts[b] for b in seq[: min(3, len(seq))])
    q = wing_from_frequency_profile(profile)
    return addr.composite, q, addr.lattice_coords[0]


def _count_substrings(data: bytes, min_len: int, max_len: int) -> Counter[bytes]:
    """Sampled substring counts for large inputs."""
    c: Counter[bytes] = Counter()
    n = len(data)
    step = 1 if n <= 500_000 else max(1, n // 500_000)
    for ln in range(min_len, max_len + 1):
        for i in range(0, n - ln + 1, step):
            c[data[i : i + ln]] += 1
    if step > 1:
        for pat in list(c.keys()):
            c[pat] = data.count(pat)
    return c


def _cohesion(data: bytes, pat: bytes) -> float:
    """1.0 when pattern always appears as an unbroken block relative to its first byte."""
    if len(pat) < 2:
        return 1.0
    first = pat[0]
    positions = [i for i, b in enumerate(data) if b == first]
    if not positions:
        return 0.0
    hits = sum(1 for i in positions if i + len(pat) <= len(data) and data[i : i + len(pat)] == pat)
    return hits / len(positions)


def detect_patterns(
    data: bytes,
    alpha: SymbolAlphabet,
    *,
    min_count: int = 2,
    min_cohesion: float = 0.85,
    max_patterns: int = MAX_PATTERNS,
) -> tuple[LatticePattern, ...]:
    """
    Find byte sequences that always / often appear together → composite tokens.
    32-lattice quadrant separates patterns that share a composite shell.
    """
    if len(data) < 2:
        return ()
    sym_counts = Counter(data)
    counts = _count_substrings(data, 2, MAX_PATTERN_LEN)
    scored: list[tuple[float, bytes, int]] = []
    covered: set[bytes] = set()
    for pat, cnt in counts.items():
        if cnt < min_count:
            continue
        coh = _cohesion(data, pat)
        if coh < min_cohesion:
            continue
        save = (len(pat) - 1) * cnt
        scored.append((save * coh, pat, cnt))
    scored.sort(reverse=True)
    patterns: list[LatticePattern] = []
    used_spans: set[bytes] = set()
    for _, pat, cnt in scored:
        if len(patterns) >= max_patterns:
            break
        if pat in used_spans:
            continue
        if any(pat in u and pat != u for u in used_spans):
            continue
        comp, quad, l01 = _pattern_lattice(alpha, pat, symbol_counts=sym_counts)
        pid = len(patterns)
        patterns.append(
            LatticePattern(
                pattern_id=pid,
                bytes_seq=pat,
                quadrant=quad,
                composite=comp,
                lattice_L01=l01,
                occurrence_count=cnt,
            )
        )
        used_spans.add(pat)
    return tuple(patterns)


def _build_trie(patterns: tuple[LatticePattern, ...]) -> dict:
    root: dict = {}
    for p in sorted(patterns, key=lambda x: -len(x.bytes_seq)):
        node = root
        for b in p.bytes_seq:
            node = node.setdefault(b, {})
        node["__pid__"] = p.pattern_id
    return root


def tokenize_longest_match(
    data: bytes,
    patterns: tuple[LatticePattern, ...],
    alpha: SymbolAlphabet,
) -> list[int]:
    """Return token ids: 0..alpha.n-1 atomic, alpha.n+pid for patterns."""
    if not data:
        return []
    trie = _build_trie(patterns)
    pat_base = alpha.n
    out: list[int] = []
    i = 0
    n = len(data)
    while i < n:
        node = trie
        best_pid: int | None = None
        best_end = i
        j = i
        while j < n and (b := data[j]) in node:
            node = node[b]
            j += 1
            if "__pid__" in node:
                best_pid = node["__pid__"]
                best_end = j
        if best_pid is not None:
            out.append(pat_base + best_pid)
            i = best_end
        else:
            out.append(alpha.index_by_symbol[data[i]])
            i += 1
    return out


def detokenize(
    tokens: list[int],
    alpha: SymbolAlphabet,
    patterns: tuple[LatticePattern, ...],
) -> bytes:
    pat_base = alpha.n
    pat_by_id = {p.pattern_id: p for p in patterns}
    out = bytearray()
    for tid in tokens:
        if tid < pat_base:
            out.append(alpha.symbols[tid])
        else:
            out.extend(pat_by_id[tid - pat_base].bytes_seq)
    return bytes(out)


def _bits_per_token(n_tokens: int) -> int:
    return max(1, math.ceil(math.log2(max(n_tokens, 2))))


def _pack_u32_list(values: list[int], bits: int) -> bytes:
    if not values:
        return b""
    out = bytearray()
    acc = 0
    nacc = 0
    mask = (1 << bits) - 1
    for v in values:
        acc = (acc << bits) | (v & mask)
        nacc += bits
        while nacc >= 8:
            nacc -= 8
            out.append((acc >> nacc) & 0xFF)
            acc &= (1 << nacc) - 1 if nacc else 0
    if nacc:
        out.append((acc << (8 - nacc)) & 0xFF)
    return bytes(out)


def _unpack_u32_list(blob: bytes, n_values: int, bits: int) -> list[int]:
    if n_values == 0:
        return []
    out: list[int] = []
    acc = 0
    nacc = 0
    pos = 0
    mask = (1 << bits) - 1
    while len(out) < n_values:
        if nacc < bits and pos < len(blob):
            acc = (acc << 8) | blob[pos]
            nacc += 8
            pos += 1
            continue
        if nacc < bits:
            break
        nacc -= bits
        out.append((acc >> nacc) & mask)
        acc &= (1 << nacc) - 1 if nacc else 0
    return out


def encode_storage(
    data: bytes,
    *,
    min_pattern_count: int = 2,
    min_cohesion: float = 0.85,
    max_patterns: int = MAX_PATTERNS,
) -> tuple[bytes, StorageLedger, tuple[LatticePattern, ...]]:
    alpha = SymbolAlphabet.from_bytes(data)
    patterns = detect_patterns(
        data,
        alpha,
        min_count=min_pattern_count,
        min_cohesion=min_cohesion,
        max_patterns=max_patterns,
    )
    tokens = tokenize_longest_match(data, patterns, alpha)
    n_vocab = alpha.n + len(patterns)
    bits = _bits_per_token(n_vocab)
    packed = _pack_u32_list(tokens, bits)

    pat_table = bytearray()
    for p in patterns:
        pat_table.append(len(p.bytes_seq))
        pat_table.extend(p.bytes_seq)
        pat_table.append(p.quadrant)

    header = struct.pack(
        "<IIHHBB",
        len(data),
        len(tokens),
        alpha.n,
        len(patterns),
        bits,
        0,
    )
    payload = MAGIC + header + bytes(alpha.symbols) + bytes(pat_table) + packed

    walk = document_pair_walk(data, alpha)
    coords_if = len(walk) * 32 * 3 * 4
    pat_bytes = len(pat_table)
    alpha_bytes = 2 + alpha.n
    ledger = StorageLedger(
        raw_bytes=len(data),
        n_atomic_symbols=alpha.n,
        n_patterns=len(patterns),
        n_effective_tokens=n_vocab,
        n_stream_tokens=len(tokens),
        alphabet_bytes=alpha_bytes,
        pattern_table_bytes=pat_bytes,
        stream_bytes=len(packed),
        formula_header_bytes=FORMULA_HEADER_BYTES,
        total_stored_bytes=len(payload),
        coord_bytes_if_stored=coords_if,
        pair_dots_placed=len(walk),
    )
    return payload, ledger, patterns


def decode_storage(payload: bytes) -> bytes:
    if not payload.startswith(MAGIC):
        raise ValueError("bad magic")
    off = len(MAGIC)
    n_raw, n_tok, n_sym, n_pat, bits, _ = struct.unpack_from("<IIHHBB", payload, off)
    off += 14
    symbols = payload[off : off + n_sym]
    off += n_sym
    alpha = SymbolAlphabet(symbols=tuple(symbols))
    patterns: list[LatticePattern] = []
    sym_counts = Counter(symbols)
    for pid in range(n_pat):
        ln = payload[off]
        off += 1
        seq = payload[off : off + ln]
        off += ln
        quad = payload[off]
        off += 1
        comp, q2, l01 = _pattern_lattice(alpha, seq, symbol_counts=sym_counts)
        patterns.append(
            LatticePattern(
                pattern_id=pid,
                bytes_seq=seq,
                quadrant=quad or q2,
                composite=comp,
                lattice_L01=l01,
                occurrence_count=0,
            )
        )
    tokens = _unpack_u32_list(payload[off:], n_tok, bits)
    out = detokenize(tokens, alpha, tuple(patterns))
    if len(out) != n_raw:
        raise ValueError(f"length mismatch {len(out)} != {n_raw}")
    return out


def analyze_storage(data: bytes, **kwargs) -> StorageLedger:
    _, ledger, _ = encode_storage(data, **kwargs)
    return ledger


def extrapolate_ledger(ledger: StorageLedger, *, target_bytes: int) -> dict:
    """Scale stream cost to target corpus size; index stays constant."""
    if ledger.raw_bytes == 0:
        return ledger.explain()
    per_byte_stream = ledger.stream_bytes / ledger.raw_bytes
    per_byte_tokens = ledger.n_stream_tokens / ledger.raw_bytes
    est_stream = int(per_byte_stream * target_bytes)
    est_tokens = int(per_byte_tokens * target_bytes)
    est_total = ledger.index_only_bytes + est_stream
    return {
        **ledger.explain(),
        "extrapolate_target_bytes": target_bytes,
        "extrapolate_target_tb": round(target_bytes / 1e12, 3),
        "extrapolate_stream_bytes": est_stream,
        "extrapolate_total_bytes": est_total,
        "extrapolate_ratio_x": round(target_bytes / est_total, 3) if est_total else 0,
        "extrapolate_index_only_bytes": ledger.index_only_bytes,
        "extrapolate_tokens": est_tokens,
    }
