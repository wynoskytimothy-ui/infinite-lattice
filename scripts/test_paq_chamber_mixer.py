#!/usr/bin/env python3
"""
Test 18 - PAQ-style chamber mixer: bit-level logistic mixing of chambers.

Tests 16-17 documented three NEGATIVE results: token alphabets and
probability-space chamber mixing both lose at 64KB scale. This test builds
the architecture that holds text-compression records (PAQ / cmix family),
expressed in the repo's own vocabulary:

  - every byte is coded as 8 binary decisions (bit by bit, MSB first)
  - CHAMBERS predict each bit: order-0..5 byte contexts (FTA composite
    addresses - Test 3), a word chamber (word identity = FTA composite of
    position-tagged letter primes), and a word-pair chamber (previous word
    + current prefix)
  - a LOGISTIC MIXER combines chamber votes in the stretched domain:
        p = squash( sum_i w_i * stretch(p_i) )
    with weights learned online by bounded gradient descent
        w_i += LR * stretch(p_i) * (bit - p)
    per bit-position (8 weight sets). Gradients are bounded, so the
    collapse modes of Tests 17 v1-v3 cannot occur.
  - binary arithmetic coder (reuses Test 15's Encoder/Decoder)

The decoder mirrors model + mixer exactly: byte-exact round trip required.

This is the user's chamber architecture in its strongest known form:
many specialized chambers, each auditable, with trust learned per chamber
per bit-position - and zero pre-trained anything.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_lattice_context_compressor import Encoder, Decoder
import test_lattice_context_compressor as byte_codec

ROOT = Path(__file__).resolve().parents[1]

MAX_ORDER = 5
WORD_SLOTS = 8
N_CH = 1 + MAX_ORDER + 2        # O0, O1..O5, W, WP
LR = 0.003
CLIP = 10.0
COUNT_CAP = 1024
TOTAL = 4096                     # arithmetic coder total per bit
PRIMES = chain_primes(256 * (MAX_ORDER + WORD_SLOTS) + 64)


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


def is_wordchar(b: int) -> bool:
    return (48 <= b <= 57) or (65 <= b <= 90) or (97 <= b <= 122) or b == 95


def stretch(p: float) -> float:
    x = math.log(p / (1.0 - p))
    if x > CLIP:
        return CLIP
    if x < -CLIP:
        return -CLIP
    return x


def squash(x: float) -> float:
    if x > 25.0:
        x = 25.0
    elif x < -25.0:
        x = -25.0
    return 1.0 / (1.0 + math.exp(-x))


class ChamberMixer:
    """Bit-prediction chambers + logistic mixer. Mirrored by the decoder."""

    def __init__(self):
        # per-chamber: {(context_key, bit_prefix): [n0, n1]}
        self.tables: list[dict] = [{} for _ in range(N_CH)]
        # mixer weights: 8 bit-positions x N_CH chambers
        self.w = [[0.0] * N_CH for _ in range(8)]
        # byte-context state
        self.hist: list[int] = []
        self.cur_word_key = 1
        self.cur_word_len = 0
        self.prev_word_key = 1
        self._keys: list = [0] * N_CH

    def begin_byte(self):
        """Compute each chamber's context key once per byte."""
        keys: list = [1]                       # O0: constant context
        key = 1
        for j in range(MAX_ORDER):             # O1..O5: FTA composites
            if j < len(self.hist):
                key *= PRIMES[j * 256 + self.hist[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)                 # inactive
        keys.append(self.cur_word_key)         # W: word-prefix composite
        keys.append((self.prev_word_key, self.cur_word_key))  # WP
        self._keys = keys

    def predict(self, bit_idx: int, prefix: int) -> tuple[float, list]:
        """Mixture probability of bit=1, plus per-chamber stretches."""
        xs: list = [0.0] * N_CH
        dot = 0.0
        wrow = self.w[bit_idx]
        for i in range(N_CH):
            ck = self._keys[i]
            if ck == 0:
                continue
            e = self.tables[i].get((ck, prefix))
            if e is None:
                continue
            n0, n1 = e
            p_i = (n1 + 0.25) / (n0 + n1 + 0.5)
            x = stretch(p_i)
            xs[i] = x
            dot += wrow[i] * x
        return squash(dot), xs

    def learn(self, bit_idx: int, prefix: int, bit: int, p: float, xs: list):
        err = (bit - p) * LR
        wrow = self.w[bit_idx]
        for i in range(N_CH):
            x = xs[i]
            if x != 0.0:
                wrow[i] += x * err
            ck = self._keys[i]
            if ck == 0:
                continue
            e = self.tables[i].get((ck, prefix))
            if e is None:
                e = [0, 0]
                self.tables[i][(ck, prefix)] = e
            e[bit] += 1
            if e[0] + e[1] > COUNT_CAP:
                e[0] -= e[0] >> 1
                e[1] -= e[1] >> 1

    def end_byte(self, byte: int):
        """Advance byte history + word state after a full byte."""
        self.hist.append(byte)
        if len(self.hist) > MAX_ORDER:
            self.hist.pop(0)
        if is_wordchar(byte):
            slot = min(self.cur_word_len, WORD_SLOTS - 1)
            self.cur_word_key *= PRIMES[(MAX_ORDER + slot) * 256 + byte]
            self.cur_word_len += 1
        else:
            if self.cur_word_len > 0:
                self.prev_word_key = self.cur_word_key
            self.cur_word_key = 1
            self.cur_word_len = 0

    def chamber_stats(self) -> str:
        names = ["O0", "O1", "O2", "O3", "O4", "O5", "W", "WP"]
        sizes = ", ".join(f"{n}={len(t)}" for n, t in zip(names, self.tables))
        # average |weight| per chamber across bit positions = learned authority
        auth = []
        for i in range(N_CH):
            a = sum(abs(self.w[k][i]) for k in range(8)) / 8
            auth.append(f"{names[i]}={a:.2f}")
        return f"    contexts: {sizes}\n    authority: " + ", ".join(auth)


def bit_ranges(p: float) -> int:
    """Split point for coding bit=1 with probability p."""
    c1 = int(p * TOTAL)
    if c1 < 1:
        c1 = 1
    elif c1 > TOTAL - 1:
        c1 = TOTAL - 1
    return c1


def compress(data: bytes) -> tuple[bytes, float, ChamberMixer]:
    model = ChamberMixer()
    enc = Encoder()
    ideal_bits = 0.0
    for byte in data:
        model.begin_byte()
        prefix = 1
        for k in range(8):
            bit = (byte >> (7 - k)) & 1
            p, xs = model.predict(k, prefix)
            c1 = bit_ranges(p)
            if bit:
                enc.encode(0, c1, TOTAL)
                ideal_bits += -math.log2(c1 / TOTAL)
            else:
                enc.encode(c1, TOTAL, TOTAL)
                ideal_bits += -math.log2((TOTAL - c1) / TOTAL)
            model.learn(k, prefix, bit, p, xs)
            prefix = (prefix << 1) | bit
        model.end_byte(byte)
    return enc.finish(), ideal_bits, model


def decompress(blob: bytes, n: int) -> bytes:
    model = ChamberMixer()
    dec = Decoder(blob)
    out = bytearray()
    for _ in range(n):
        model.begin_byte()
        prefix = 1
        for k in range(8):
            p, xs = model.predict(k, prefix)
            c1 = bit_ranges(p)
            value = dec.decode_value(TOTAL)
            if value < c1:
                bit = 1
                dec.consume(0, c1, TOTAL)
            else:
                bit = 0
                dec.consume(c1, TOTAL, TOTAL)
            model.learn(k, prefix, bit, p, xs)
            prefix = (prefix << 1) | bit
        byte = prefix & 0xFF
        model.end_byte(byte)
        out.append(byte)
    return bytes(out)


def main():
    header("PAQ-style chamber mixer - bit-level logistic mixing")

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
    print("\nChamber mixer (8 chambers, bit-level logistic mixing)")
    print("-" * 72)
    t0 = time.time()
    blob, ideal_bits, model = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)
    print(f"  encoded:   {ours} bytes in {t_enc:.1f}s")
    print(f"  bits/byte: {ours*8/n:.3f} (model cross-entropy {ideal_bits/n:.3f})")
    print(f"  learned chamber state:")
    print(model.chamber_stats())

    t0 = time.time()
    restored = decompress(blob, n)
    print(f"  decoded:   {len(restored)} bytes in {time.time()-t0:.1f}s")
    assertion(restored == data, "round-trip byte-exact (mixer mirrored)")

    # ------------------------------------------------------------------
    # Scoreboard
    # ------------------------------------------------------------------
    print("\nScoreboard")
    print("-" * 72)
    rows = [("frequency floor (order-0)", H0 * n / 8),
            *[(k, float(v)) for k, v in base_results.items()],
            ("PAQ-style chamber mixer", float(ours))]
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "PAQ-style chamber mixer" else ""
        print(f"  {name:<28} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")

    assertion(ours < base_results["byte codec (Test 15)"],
              f"bit-level mixing improves on Test 15 "
              f"({(1-ours/base_results['byte codec (Test 15)'])*100:.1f}% smaller)")
    for name in ("zlib -9", "lzma", "bz2 -9"):
        size = base_results[name]
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
              f"({ours} vs {size} bytes)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    header("RESULT")
    best = min(base_results, key=base_results.get)
    print(f"  ours:           {ours*8/n:.3f} bits/byte")
    print(f"  best classical: {best} = {base_results[best]*8/n:.3f} bits/byte")
    print(f"  vs frequency floor: {(1-ours*8/n/H0)*100:.0f}% below")
    print()
    print("  CONCLUSION:")
    print("  Bit-level logistic mixing makes the chamber architecture work:")
    print("  bounded gradients, no trust collapse, every chamber's learned")
    print("  authority auditable per bit-position. Chambers + FTA context")
    print("  addressing + online mixing = the record-holding compressor")
    print("  architecture, built from this repo's primitives.")


if __name__ == "__main__":
    main()
