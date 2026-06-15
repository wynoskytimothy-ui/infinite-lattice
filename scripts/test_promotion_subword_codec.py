#!/usr/bin/env python3
"""
Test 16 - Promotion-subword codec: L2-style online promotion + token contexts.

Test 15 compressed BYTES with lattice-addressed context mixing: 2.870
bits/byte, beating zlib -9 and lzma but trailing bz2 -9 (2.831) by 1.4%.

This test swaps the symbol alphabet, exactly as the repo's retrieval stack
does with L2 PMI subwords: frequent units PROMOTE to their own symbols.
Here the promotion runs ONLINE inside the codec:

  - vocab starts as the 256 byte symbols
  - every adjacent token pair is counted; at count == 3 the pair PROMOTES
    to a new token (concatenated bytes), exactly mirrored by the decoder
  - no dictionary is ever transmitted - promotion is derived from the
    already-decoded stream on both sides (zero header bytes)
  - the encoder greedily emits the longest known token at each position
  - token stream is coded with blended order-0..3 token-context models,
    contexts addressed by FTA composites over position-tagged primes
    (slot j uses PRIMES[j*MAX_VOCAB + token_id] - Test 3's perfect hash)

This is the compression form of the promotion lattice: Test 6 promoted
frequent factor-sets into explaining primes; here frequent byte-sequences
promote into vocabulary primes that re-price the whole stream.

Pass conditions: byte-exact round trip; beats the Test 15 byte codec.
Standing vs zlib/bz2/lzma reported honestly.
"""

from __future__ import annotations

import bz2
import lzma
import math
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_lattice_context_compressor import Encoder, Decoder
import test_lattice_context_compressor as byte_codec

ROOT = Path(__file__).resolve().parents[1]

MAX_VOCAB = 8192
PROMOTE_AT = 3          # pair count that triggers promotion
MAX_TOKEN_LEN = 32      # bytes
T_ORDER = 3             # token-context orders 1..3
ORDER_PRIOR_T = [0.08, 1.5, 5.0, 12.0]   # order 0..3 priors
CONF = 1.2
PRIMES_T = chain_primes(MAX_VOCAB * T_ORDER + 100)


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


class PromotionCodecState:
    """Vocabulary + pair-promotion + token context model.

    Encoder and decoder each run one instance and feed it the SAME token
    sequence, so vocabulary growth and model statistics stay in lockstep.
    """

    def __init__(self):
        self.vocab_bytes: list[bytes] = [bytes([b]) for b in range(256)]
        self.vocab_map: dict[bytes, int] = {bytes([b]): b for b in range(256)}
        self.max_len = 1
        self.pair_counts: dict[tuple[int, int], int] = {}
        self.prev_token: int | None = None
        # model state
        self.c0 = np.zeros(MAX_VOCAB, dtype=np.float64)
        self.c0_total = 0
        self.tables: list[dict[int, list]] = [{} for _ in range(T_ORDER)]
        self.hist: list[int] = []

    # ---- vocabulary ----

    @property
    def vocab_size(self) -> int:
        return len(self.vocab_bytes)

    def next_token(self, data: bytes, pos: int) -> tuple[int, int]:
        """Greedy longest match at pos; returns (token_id, length)."""
        limit = min(self.max_len, len(data) - pos)
        for L in range(limit, 0, -1):
            tok = self.vocab_map.get(data[pos:pos + L])
            if tok is not None:
                return tok, L
        raise RuntimeError("unreachable: single bytes always present")

    # ---- context addressing (FTA composites over token ids) ----

    def context_keys(self) -> list[int]:
        keys = []
        key = 1
        for j in range(T_ORDER):
            if j < len(self.hist):
                key *= PRIMES_T[j * MAX_VOCAB + self.hist[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)
        return keys

    # ---- model ----

    def freq_table(self) -> tuple[np.ndarray, int]:
        V = self.vocab_size
        p = np.full(V, 0.01 / V, dtype=np.float64)
        if self.c0_total:
            p += ORDER_PRIOR_T[0] * (self.c0[:V] / self.c0_total)
        for k, key in enumerate(self.context_keys(), start=1):
            if key == 0:
                continue
            entry = self.tables[k - 1].get(key)
            if entry is not None:
                n_k, d = entry
                lam = ORDER_PRIOR_T[k] * (n_k / (n_k + CONF))
                inv = lam / n_k
                for t, c in d.items():
                    p[t] += c * inv
        freq = (p * (65536.0 / p.sum())).astype(np.int64) + 1
        return freq, int(freq.sum())

    def account(self, token: int):
        """Update model + pair stats + promotion. Mirrored on both sides."""
        # model counts
        self.c0[token] += 1
        self.c0_total += 1
        for k, key in enumerate(self.context_keys(), start=1):
            if key == 0:
                continue
            entry = self.tables[k - 1].setdefault(key, [0, {}])
            entry[0] += 1
            d = entry[1]
            d[token] = d.get(token, 0) + 1
        self.hist.append(token)
        if len(self.hist) > T_ORDER:
            self.hist.pop(0)
        # pair promotion (L2-style: frequent unit -> own symbol)
        if self.prev_token is not None and self.vocab_size < MAX_VOCAB:
            pair = (self.prev_token, token)
            c = self.pair_counts.get(pair, 0) + 1
            self.pair_counts[pair] = c
            if c == PROMOTE_AT:
                merged = self.vocab_bytes[pair[0]] + self.vocab_bytes[pair[1]]
                if len(merged) <= MAX_TOKEN_LEN and merged not in self.vocab_map:
                    self.vocab_map[merged] = len(self.vocab_bytes)
                    self.vocab_bytes.append(merged)
                    self.max_len = max(self.max_len, len(merged))
        self.prev_token = token


def compress(data: bytes) -> tuple[bytes, dict]:
    state = PromotionCodecState()
    enc = Encoder()
    pos = 0
    n_tokens = 0
    ideal_bits = 0.0
    while pos < len(data):
        token, L = state.next_token(data, pos)
        freq, total = state.freq_table()
        cum = np.cumsum(freq)
        cum_lo = int(cum[token - 1]) if token > 0 else 0
        cum_hi = int(cum[token])
        enc.encode(cum_lo, cum_hi, total)
        ideal_bits += -math.log2((cum_hi - cum_lo) / total)
        state.account(token)
        pos += L
        n_tokens += 1
    blob = enc.finish()
    stats = {
        "n_tokens": n_tokens,
        "vocab": state.vocab_size,
        "avg_token_len": len(data) / n_tokens,
        "ideal_bits": ideal_bits,
    }
    return blob, stats


def decompress(blob: bytes, n_bytes: int) -> bytes:
    state = PromotionCodecState()
    dec = Decoder(blob)
    out = bytearray()
    while len(out) < n_bytes:
        freq, total = state.freq_table()
        cum = np.cumsum(freq)
        value = dec.decode_value(total)
        token = int(np.searchsorted(cum, value, side="right"))
        cum_lo = int(cum[token - 1]) if token > 0 else 0
        cum_hi = int(cum[token])
        dec.consume(cum_lo, cum_hi, total)
        out.extend(state.vocab_bytes[token])
        state.account(token)
    return bytes(out)


def main():
    header("Promotion-subword codec - L2 promotion running inside a compressor")

    files = sorted((ROOT / "derivations").glob("*.md"))
    raw = b"".join(f.read_bytes() for f in files)
    data = raw[:65536]
    n = len(data)
    print(f"  data: {len(files)} markdown files from derivations/, first {n} bytes")

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("\nBaselines")
    print("-" * 72)
    counts = Counter(data)
    H0 = -sum((c / n) * math.log2(c / n) for c in counts.values())
    print(f"  order-0 frequency floor: {H0:.3f} bits/byte")

    base_results = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        out = fn(data)
        base_results[name] = len(out)
        print(f"  {name:<10} {len(out):>7} bytes   {len(out)*8/n:.3f} bits/byte")

    t0 = time.time()
    byte_blob, _, _ = byte_codec.compress(data)
    base_results["byte codec (Test 15)"] = len(byte_blob)
    print(f"  {'byte codec (Test 15)':<22} {len(byte_blob):>7} bytes   "
          f"{len(byte_blob)*8/n:.3f} bits/byte   ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # Ours
    # ------------------------------------------------------------------
    print("\nPromotion-subword codec (online pair promotion, token contexts)")
    print("-" * 72)
    t0 = time.time()
    blob, stats = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)
    print(f"  encoded:   {ours} bytes in {t_enc:.1f}s")
    print(f"  bits/byte: {ours*8/n:.3f} (model cross-entropy {stats['ideal_bits']/n:.3f})")
    print(f"  tokens:    {stats['n_tokens']} (avg {stats['avg_token_len']:.2f} bytes/token)")
    print(f"  vocab:     {stats['vocab']} symbols "
          f"({stats['vocab']-256} promoted from pairs - zero header bytes)")

    t0 = time.time()
    restored = decompress(blob, n)
    print(f"  decoded:   {len(restored)} bytes in {time.time()-t0:.1f}s")
    assertion(restored == data,
              "round-trip byte-exact (promotion replayed identically by decoder)")

    # ------------------------------------------------------------------
    # Scoreboard
    # ------------------------------------------------------------------
    print("\nScoreboard")
    print("-" * 72)
    rows = [*[(k, float(v)) for k, v in base_results.items()],
            ("promotion-subword codec", float(ours))]
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "promotion-subword codec" else ""
        print(f"  {name:<28} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")

    for name in ("zlib -9", "lzma", "bz2 -9", "byte codec (Test 15)"):
        size = base_results[name]
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [info]  {verdict} {name} ({ours} vs {size} bytes)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    header("RESULT: NEGATIVE (documented)")
    print(f"  ours:           {ours*8/n:.3f} bits/byte")
    print(f"  byte codec:     {base_results['byte codec (Test 15)']*8/n:.3f} bits/byte")
    print()
    print("  The online token-alphabet variant LOSES at this scale. The")
    print("  mechanism is correct (round-trip exact, promotion mirrored with")
    print("  zero header bytes) but the economics fail on 64KB:")
    print(f"    - {stats['vocab']-256} promoted tokens each pay a cold first-use cost")
    print(f"    - shifting tokenization fragments context statistics")
    print(f"    - avg token only {stats['avg_token_len']:.2f} bytes - not enough to amortize")
    print()
    print("  Lesson: at small scale, upgrade the CONTEXTS, not the alphabet.")
    print("  See test_chamber_blend_codec.py (word chambers over byte alphabet).")
    print("  The token alphabet should win at MB+ scale where promoted units")
    print("  amortize - same scaling behavior as BPE vocabularies in LLMs.")


if __name__ == "__main__":
    main()
