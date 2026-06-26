"""
Electron-model tokenizer — binary substrate, quaternary token slots.

Physical layer: only 0 and 1 exist (±B branches, gate pins).
Read layer: pair two bits → one of four coin states (WH / WS / BH / BS).

    membrane bit  0 = White (W)   1 = Black (B)
    spring bit    0 = Soft  (S)   1 = Hard  (H)

    00 → WS    01 → WH    10 → BS    11 → BH

Each token in a stream occupies one electron read = 2 bits = 4 states.
Vocabulary IDs pack into base-4 dits (same 2-bit pairs) for arbitrary vocab size.

    python -m pytest tests/test_electron_tokenizer.py -q
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Iterator

from aethos_promotion import PromotionRegistry
from aethos_tokenize import TokenSpan, tokenize_spans


class CoinState(IntEnum):
    """Four deterministic coin states (Ch 8 membrane × spring)."""

    WS = 0  # 00  white membrane, soft spring  (superposition side)
    WH = 1  # 01  white membrane, hard spring  (pinned light side)
    BS = 2  # 10  black membrane, soft spring  (numeric / fluid branch)
    BH = 3  # 11  black membrane, hard spring  (dedicated prime, collapsed)


COIN_LABELS: dict[CoinState, str] = {
    CoinState.WS: "WS",
    CoinState.WH: "WH",
    CoinState.BS: "BS",
    CoinState.BH: "BH",
}


def bits_to_state(membrane: int, spring: int) -> CoinState:
    """Two 0/1 gate reads → one coin state."""
    return CoinState(((membrane & 1) << 1) | (spring & 1))


def state_to_bits(state: CoinState | int) -> tuple[int, int]:
    """Coin state → (membrane_bit, spring_bit)."""
    s = int(state) & 3
    return (s >> 1, s & 1)


def state_to_bit_pair(state: CoinState | int) -> tuple[int, int]:
    """Alias: one token slot as two binary outputs."""
    return state_to_bits(state)


def bit_pair_to_state(pair: tuple[int, int]) -> CoinState:
    return bits_to_state(pair[0], pair[1])


def iter_bits(pairs: Iterable[tuple[int, int]]) -> Iterator[int]:
    """Flatten (membrane, spring) pairs into a 0/1 stream."""
    for m, s in pairs:
        yield m & 1
        yield s & 1


def iter_pairs(bits: Iterable[int]) -> Iterator[tuple[int, int]]:
    """Regroup a 0/1 stream into electron reads (2 bits → 4 states)."""
    buf: list[int] = []
    for b in bits:
        buf.append(int(b) & 1)
        if len(buf) == 2:
            yield (buf[0], buf[1])
            buf.clear()
    if buf:
        yield (buf[0], 0)  # pad incomplete read with soft spring


def pack_states(states: Iterable[CoinState | int]) -> list[int]:
    """Sequence of coin states → flat 0/1 bit list."""
    return list(iter_bits(state_to_bits(s) for s in states))


def unpack_states(bits: Iterable[int]) -> list[CoinState]:
    """Flat 0/1 bit list → coin states (2 bits per token slot)."""
    return [bit_pair_to_state(p) for p in iter_pairs(bits)]


def tier_to_state(registry: PromotionRegistry, word: str, *, species: str = "WORD") -> CoinState:
    """
    Map promotion tier / species to a coin quadrant (semantic 4-state read).

    letters_only      → WS  (unsettled, soft superposition)
    intersection_only → WH  (shared white-side intersection, hard spring)
    dedicated_l3      → BH  (own prime pinned, hard collapse)
    NUM               → BS  (black-side numeric branch)
    """
    if species == "NUM":
        return CoinState.BS
    w = word.lower()
    from aethos_promotion import LatticeTier

    if (LatticeTier.L3_WORD, w) in registry.promoted:
        return CoinState.BH
    if w in registry.intersections:
        return CoinState.WH
    return CoinState.WS


@dataclass(frozen=True)
class ElectronToken:
    """One text token + its electron read (2 bits, 4 states)."""

    text: str
    species: str
    position: int
    state: CoinState
    membrane: int
    spring: int

    @property
    def label(self) -> str:
        return COIN_LABELS[self.state]

    @property
    def bits(self) -> tuple[int, int]:
        return (self.membrane, self.spring)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "species": self.species,
            "position": self.position,
            "state": self.label,
            "membrane": self.membrane,
            "spring": self.spring,
            "bits": list(self.bits),
        }


def encode_span(
    span: TokenSpan,
    registry: PromotionRegistry | None,
    *,
    position: int | None = None,
) -> ElectronToken:
    """One TokenSpan → ElectronToken."""
    pos = span.position if position is None else position
    state = tier_to_state(registry, span.text, species=span.species.value) if registry else CoinState(span.position % 4)
    m, s = state_to_bits(state)
    return ElectronToken(
        text=span.text,
        species=span.species.value,
        position=pos,
        state=state,
        membrane=m,
        spring=s,
    )


def tokenize_electron(
    text: str,
    registry: PromotionRegistry | None = None,
) -> list[ElectronToken]:
    """Tokenize text; attach a 4-state electron read to every token."""
    return [encode_span(span, registry) for span in tokenize_spans(text)]


def encode_bit_stream(tokens: Iterable[ElectronToken]) -> list[int]:
    """Document electron tokens → contiguous 0/1 stream (2 bits per token)."""
    return pack_states(t.state for t in tokens)


def decode_bit_stream(bits: Iterable[int]) -> list[CoinState]:
    """Contiguous 0/1 stream → coin states (vocabulary-agnostic layer)."""
    return unpack_states(bits)


class ElectronVocabCodec:
    """
    Fixed vocabulary: token ↔ base-4 dit sequence (still stored as 0/1 pairs).

    Each vocab index writes ceil(log4(V)) electron reads; short indices pad with WS (00).
    """

    def __init__(self, vocab: Iterable[str]):
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(sorted(set(vocab)))}
        self.id_to_token: list[str] = [""] * len(self.token_to_id)
        for t, i in self.token_to_id.items():
            self.id_to_token[i] = t
        self._width = max(1, _dit_width(len(self.id_to_token)))

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def dits_per_token(self) -> int:
        return self._width

    def encode_id(self, token_id: int) -> list[CoinState]:
        """One vocab index → fixed-width base-4 dit list (LSB first in stream order)."""
        if not self.id_to_token:
            return []
        tid = max(0, min(token_id, len(self.id_to_token) - 1))
        dits: list[CoinState] = []
        n = tid
        for _ in range(self._width):
            dits.append(CoinState(n & 3))
            n >>= 2
        return dits

    def decode_id(self, dits: Iterable[CoinState | int]) -> int:
        tid = 0
        shift = 0
        for d in dits:
            tid |= (int(d) & 3) << shift
            shift += 2
        return min(tid, max(0, len(self.id_to_token) - 1))

    def encode_token(self, token: str) -> list[int]:
        """Token string → flat 0/1 bit stream."""
        tid = self.token_to_id.get(token, 0)
        return pack_states(self.encode_id(tid))

    def decode_bits(self, bits: Iterable[int]) -> str:
        """One token's worth of bits → token string."""
        pairs = list(iter_pairs(bits))
        need = self._width
        if len(pairs) < need:
            pairs.extend([(0, 0)] * (need - len(pairs)))
        dits = [bit_pair_to_state(p) for p in pairs[:need]]
        return self.id_to_token[self.decode_id(dits)]

    def encode_document(self, tokens: Iterable[str]) -> list[int]:
        out: list[int] = []
        for t in tokens:
            out.extend(self.encode_token(t))
        return out

    def decode_document(self, bits: Iterable[int], *, n_tokens: int) -> list[str]:
        width_bits = self._width * 2
        flat = list(bits)
        out: list[str] = []
        for i in range(n_tokens):
            chunk = flat[i * width_bits : (i + 1) * width_bits]
            out.append(self.decode_bits(chunk))
        return out


def _dit_width(vocab_size: int) -> int:
    if vocab_size <= 1:
        return 1
    w = 0
    cap = 1
    while cap < vocab_size:
        cap <<= 2
        w += 1
    return w
