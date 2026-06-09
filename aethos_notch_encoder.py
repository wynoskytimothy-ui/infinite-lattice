"""
Notch encoder — 4×4 spring correlation matrix per hub word (P5).

Each VA branch is a complex spring state z = X + iY at transgressor n.
The 4×4 matrix M[i,j] = conj(z_i) · z_j captures cross-branch correlation.
We store only the top-K peaks (and optional nulls) as fixed 10-byte notches.

Opt-in storage: set environment variable STORAGE_BACKEND=notch
(default pipeline unchanged — hub signatures remain primary).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

from aethos_complex_spring import spring_states_at
from aethos_hub_signature import MIN_POOL_PRIME, pool_factor_jaccard, pool_factors_for_word
from aethos_lattice import BranchKind

NOTCH_BYTES = 10
NOTCH_PACK = ">BBHhhBB"  # ba, bb, amp_u16, re_i16, im_i16, kind, xor_chk
DEFAULT_TOP_K = 10
DEFAULT_DOC_NOTCH_BYTES = 120  # K=10 × 10 B + small header budget

BRANCH_ORDER: tuple[BranchKind, ...] = (
    BranchKind.VA1,
    BranchKind.VA2,
    BranchKind.VA3,
    BranchKind.VA4,
)

KIND_PEAK = 1
KIND_NULL = 2
KIND_DIAG = 4

_AMP_SCALE = 100.0  # corr magnitude → uint16
_Z_SCALE = 1.0  # depth stored in amp_u16 high bits when kind==DIAG — not used in v1


def storage_backend() -> str:
    """hub (default) | notch."""
    return os.environ.get("STORAGE_BACKEND", "hub").strip().lower()


def _xor_chk(
    ba: int,
    bb: int,
    amp_u16: int,
    re_i16: int,
    im_i16: int,
    kind: int,
) -> int:
    x = ba ^ bb ^ (amp_u16 & 0xFF) ^ ((amp_u16 >> 8) & 0xFF)
    x ^= re_i16 & 0xFF ^ (re_i16 >> 8) & 0xFF
    x ^= im_i16 & 0xFF ^ (im_i16 >> 8) & 0xFF
    x ^= kind
    return x & 0xFF


@dataclass(frozen=True)
class Notch:
    """One peak/null in the 4×4 branch correlation matrix."""

    branch_a: int  # 1..4
    branch_b: int  # 1..4
    amplitude: float
    re: float
    im: float
    kind: int = KIND_PEAK

    @property
    def branch_pair(self) -> tuple[int, int]:
        return (self.branch_a, self.branch_b)

    def is_cross_branch(self) -> bool:
        return self.branch_a != self.branch_b


def branch_index(branch: BranchKind) -> int:
    return int(branch)


def chain_for_word(registry, word: str) -> tuple[int, ...]:
    """Prime chain fed to spring_states_at for this token (distinct anchors only)."""
    tok = registry.resolve_token(word.lower())
    seen: set[int] = set()
    parents: list[int] = []
    for p in sorted(tok.parent_primes):
        if p >= 3 and p not in seen:
            seen.add(p)
            parents.append(p)
    parents_tuple = tuple(parents)
    if parents_tuple:
        return parents_tuple
    if tok.prime >= MIN_POOL_PRIME:
        return (tok.prime,)
    if tok.prime >= 3:
        return (tok.prime,)
    return (3,)


def correlation_matrix_4x4(
    chain: tuple[int, ...],
    n: int,
) -> dict[tuple[BranchKind, BranchKind], complex]:
    """4×4 complex correlation from spring states at transgressor n."""
    _, states = spring_states_at(chain, n)
    out: dict[tuple[BranchKind, BranchKind], complex] = {}
    for ba in BRANCH_ORDER:
        for bb in BRANCH_ORDER:
            za = states[ba].z
            zb = states[bb].z
            out[(ba, bb)] = za.conjugate() * zb
    return out


def _amp_to_u16(amplitude: float) -> int:
    return max(0, min(65535, int(round(amplitude * _AMP_SCALE))))


def _u16_to_amp(amp_u16: int) -> float:
    return amp_u16 / _AMP_SCALE


def pack_notch(
    branch_a: int,
    branch_b: int,
    correlation: complex,
    *,
    kind: int = KIND_PEAK,
) -> bytes:
    """Serialize one notch to exactly NOTCH_BYTES (10)."""
    if not (1 <= branch_a <= 4 and 1 <= branch_b <= 4):
        raise ValueError(f"branch indices must be 1..4, got {branch_a},{branch_b}")
    amp_u16 = _amp_to_u16(abs(correlation))
    re_i16 = max(-32767, min(32767, int(round(correlation.real * _AMP_SCALE))))
    im_i16 = max(-32767, min(32767, int(round(correlation.imag * _AMP_SCALE))))
    chk = _xor_chk(branch_a, branch_b, amp_u16, re_i16, im_i16, kind)
    return struct.pack(NOTCH_PACK, branch_a, branch_b, amp_u16, re_i16, im_i16, kind, chk)


def unpack_notch(blob: bytes) -> Notch:
    if len(blob) != NOTCH_BYTES:
        raise ValueError(f"notch must be {NOTCH_BYTES} bytes, got {len(blob)}")
    ba, bb, amp_u16, re_i16, im_i16, kind, chk = struct.unpack(NOTCH_PACK, blob)
    expect = _xor_chk(ba, bb, amp_u16, re_i16, im_i16, kind)
    if chk != expect:
        raise ValueError("notch checksum mismatch")
    amp = _u16_to_amp(amp_u16)
    return Notch(
        branch_a=int(ba),
        branch_b=int(bb),
        amplitude=amp,
        re=re_i16 / _AMP_SCALE,
        im=im_i16 / _AMP_SCALE,
        kind=int(kind),
    )


def pack_notches(notches: Sequence[Notch]) -> bytes:
    return b"".join(
        pack_notch(
            n.branch_a,
            n.branch_b,
            complex(n.re, n.im),
            kind=n.kind,
        )
        for n in notches
    )


def unpack_notches(blob: bytes) -> tuple[Notch, ...]:
    if len(blob) % NOTCH_BYTES:
        raise ValueError("notch blob length must be multiple of 10")
    return tuple(unpack_notch(blob[i : i + NOTCH_BYTES]) for i in range(0, len(blob), NOTCH_BYTES))


def extract_top_notches(
    matrix: dict[tuple[BranchKind, BranchKind], complex],
    *,
    top_k: int = DEFAULT_TOP_K,
    prefer_cross_branch: bool = True,
) -> tuple[Notch, ...]:
    """
    Top-K cells by |correlation|, preferring off-diagonal (cross-branch) peaks.
    """
    cells: list[tuple[float, BranchKind, BranchKind, complex, int]] = []
    for ba in BRANCH_ORDER:
        for bb in BRANCH_ORDER:
            c = matrix[(ba, bb)]
            mag = abs(c)
            if mag <= 0:
                continue
            kind = KIND_DIAG if ba == bb else KIND_PEAK
            # tiny boost so cross-branch wins ties
            score = mag + (0.001 if prefer_cross_branch and ba != bb else 0.0)
            cells.append((score, ba, bb, c, kind))
    cells.sort(key=lambda t: -t[0])
    out: list[Notch] = []
    for _, ba, bb, c, kind in cells[:top_k]:
        out.append(
            Notch(
                branch_a=branch_index(ba),
                branch_b=branch_index(bb),
                amplitude=abs(c),
                re=c.real,
                im=c.imag,
                kind=kind,
            )
        )
    return tuple(out)


def encode_word_notches(
    registry,
    word: str,
    *,
    n: int = 7,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Notch, ...]:
    """
    Build top-K notches for one hub word from its spring correlation matrix.
    """
    chain = chain_for_word(registry, word)
    matrix = correlation_matrix_4x4(chain, n)
    return extract_top_notches(matrix, top_k=top_k)


def notch_similarity(
    a: Sequence[Notch],
    b: Sequence[Notch],
) -> float:
    """
    Amplitude-weighted Jaccard on matching (branch_a, branch_b) pairs.
    """
    if not a or not b:
        return 0.0
    index_b = {n.branch_pair: n for n in b}
    inter = 0.0
    sum_a = 0.0
    sum_b = sum(n.amplitude for n in b)
    for na in a:
        sum_a += na.amplitude
        nb = index_b.get(na.branch_pair)
        if nb is not None:
            inter += min(na.amplitude, nb.amplitude)
    union = sum_a + sum_b - inter
    return inter / union if union > 0 else 0.0


def doc_notch_score(
    query_notches: Sequence[Notch],
    doc_notches: Sequence[Notch],
    query_pool_factors: frozenset[int],
    doc_pool_factors: frozenset[int],
    *,
    min_pool_jaccard: float = 0.0,
) -> float:
    """
    Signal gated like hub Signal 5b: require pool-prime factor overlap first.
    """
    if not query_pool_factors or not doc_pool_factors:
        return 0.0
    jacc = pool_factor_jaccard(query_pool_factors, doc_pool_factors)
    if jacc <= min_pool_jaccard:
        return 0.0
    return notch_similarity(query_notches, doc_notches) * jacc


@dataclass(frozen=True)
class NotchFingerprint:
    """Compact per-document notch storage (~K×10 bytes + header)."""

    doc_id: int
    notches: bytes
    top_hub: str = ""
    n_anchors: int = 7

    def encoded_size(self) -> int:
        # wire: 4 doc_id + 2 count + 1 n + 1 hub_len + notches + hub utf8
        hub_b = self.top_hub.encode("utf-8")
        n_notches = len(self.notches) // NOTCH_BYTES
        return 4 + 2 + 1 + 1 + len(self.notches) + len(hub_b)

    def decoded_notches(self) -> tuple[Notch, ...]:
        return unpack_notches(self.notches)

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "top_hub": self.top_hub,
            "n_notches": len(self.notches) // NOTCH_BYTES,
            "bytes": self.encoded_size(),
        }


def encode_doc_notch_fingerprint(
    doc_id: int,
    hub_words: Iterable[str],
    registry,
    *,
    n: int = 7,
    top_k: int = DEFAULT_TOP_K,
    max_hubs: int = 1,
    max_bytes: int = DEFAULT_DOC_NOTCH_BYTES,
) -> NotchFingerprint:
    """
    Encode document fingerprint from strongest hub word(s).

    Default: one hub × K=10 notches → 100 B payload (≤120 B budget).
    """
    words = [w for w in hub_words if len(w) >= 3]
    if not words:
        return NotchFingerprint(doc_id=doc_id, notches=b"", top_hub="", n_anchors=n)

    # use first hub only for stable 100-byte payload unless caller needs more
    top = words[0]
    notches = encode_word_notches(registry, top, n=n, top_k=top_k)
    blob = pack_notches(notches)
    max_payload = max(0, max_bytes - 20)  # reserve header + hub name
    max_notches = max_payload // NOTCH_BYTES
    if len(notches) > max_notches:
        notches = notches[:max_notches]
        blob = pack_notches(notches)

    return NotchFingerprint(
        doc_id=doc_id,
        notches=blob,
        top_hub=top[:48],
        n_anchors=n,
    )


def aggregate_query_notches(
    words: Sequence[str],
    registry,
    *,
    n: int = 7,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[Notch, ...]:
    """Merge top notches from query words (dedupe branch pairs, keep max amp)."""
    best: dict[tuple[int, int], Notch] = {}
    for w in words:
        if len(w) < 3:
            continue
        try:
            for notch in encode_word_notches(registry, w, n=n, top_k=top_k):
                key = notch.branch_pair
                if key not in best or notch.amplitude > best[key].amplitude:
                    best[key] = notch
        except Exception:
            continue
    ranked = sorted(best.values(), key=lambda x: -x.amplitude)
    return tuple(ranked[:top_k])


def fingerprint_document_notch(
    doc_id: int,
    hub_words: Iterable[str],
    registry,
    *,
    n: int = 7,
    top_k: int = DEFAULT_TOP_K,
) -> NotchFingerprint:
    """Entry used when STORAGE_BACKEND=notch."""
    return encode_doc_notch_fingerprint(
        doc_id,
        hub_words,
        registry,
        n=n,
        top_k=top_k,
    )
