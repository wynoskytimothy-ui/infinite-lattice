#!/usr/bin/env python3
"""
Test 15 - A REAL lattice-context compressor that beats the frequency floor.

The user's intuition: a symbol should have many meanings depending on
rotation/position, letting us compress below "the limit". The correct form
of that intuition is CONTEXT MODELING: the same byte after different
histories has different predicted distributions. Conditional entropy is
always <= frequency entropy, so a context coder can land BELOW the order-0
"frequency map floor" - the number people usually call the Shannon limit
of a file.

Lattice's role: context ADDRESSING. The last k bytes map to position-tagged
primes (slot j gets primes[j*256 + byte]) and the context key is their
product - the FTA perfect hash from Test 3. Provably collision-free context
addresses, one integer per context.

This is a complete, working codec:
  - adaptive blended model over orders 0..4 (no pre-trained codebook)
  - integer arithmetic coder (CACM87-style)
  - decoder mirrors the model exactly; round-trip verified byte-exact

Tested on REAL data: this repo's own derivations markdown.

Honest scoreboard:
  (A) order-0 frequency floor (the naive "Shannon limit" of the file)
  (B) zlib -9 / bz2 -9 / lzma
  (C) our lattice-context coder
Pass condition: (C) decodes exactly AND lands below (A). Standing vs (B)
reported honestly, win or lose.
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

from core.primes import chain_primes

ROOT = Path(__file__).resolve().parents[1]

MASK = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREEQ = 0xC0000000

MAX_ORDER = 5
# order priors: higher orders dominate WHEN confident (normalized mixing)
ORDER_PRIOR = [0.05, 0.3, 1.0, 3.0, 9.0, 20.0]  # order 0..5
PRIMES = chain_primes(256 * MAX_ORDER + 64)


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# ----------------------------------------------------------------------
# The shared adaptive model (encoder and decoder run identical copies)
# ----------------------------------------------------------------------

class LatticeContextModel:
    """Blended order-0..5 byte model with FTA-composite context addresses.

    Mixing is confidence-normalized: each context's distribution is
    normalized to a probability first, then weighted by its order prior
    times a confidence factor n/(n+1.5). Sharp high-order contexts
    dominate when they have data; order-0 only fills the gaps.
    """

    def __init__(self):
        self.c0 = np.zeros(256, dtype=np.float64)
        self.c0_total = 0
        # tables[k-1]: composite-key -> [n_total, {byte: count}] for order k
        self.tables: list[dict[int, list]] = [{} for _ in range(MAX_ORDER)]

    @staticmethod
    def context_keys(hist: list[int]) -> list[int]:
        """FTA composite address for each order 1..MAX_ORDER.

        Slot j (j=0 is the most recent byte) draws from its own disjoint
        prime alphabet PRIMES[j*256 + b], so the product is collision-free
        across both byte values AND positions (Test 3 / Test 8 machinery).
        """
        keys = []
        key = 1
        for j in range(MAX_ORDER):
            if j < len(hist):
                key *= PRIMES[j * 256 + hist[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)  # not enough history yet
        return keys

    def freq_table(self, hist: list[int]) -> tuple[np.ndarray, int]:
        """Blended frequency table (all entries >= 1) and its total."""
        p = np.full(256, 0.01 / 256, dtype=np.float64)  # uniform escape mass
        if self.c0_total:
            p += ORDER_PRIOR[0] * (self.c0 / self.c0_total)
        for k, key in enumerate(self.context_keys(hist), start=1):
            if key == 0:
                continue
            entry = self.tables[k - 1].get(key)
            if entry is not None:
                n_k, d = entry
                lam = ORDER_PRIOR[k] * (n_k / (n_k + 1.5))
                inv = lam / n_k
                for b, c in d.items():
                    p[b] += c * inv
        freq = (p * (65536.0 / p.sum())).astype(np.int64) + 1
        return freq, int(freq.sum())

    def update(self, hist: list[int], byte: int):
        self.c0[byte] += 1
        self.c0_total += 1
        for k, key in enumerate(self.context_keys(hist), start=1):
            if key == 0:
                continue
            entry = self.tables[k - 1].setdefault(key, [0, {}])
            entry[0] += 1
            d = entry[1]
            d[byte] = d.get(byte, 0) + 1

    def distinct_contexts(self) -> int:
        return sum(len(t) for t in self.tables)


# ----------------------------------------------------------------------
# Integer arithmetic coder (CACM87-style)
# ----------------------------------------------------------------------

class Encoder:
    def __init__(self):
        self.low = 0
        self.high = MASK
        self.pending = 0
        self.bits: list[int] = []

    def _bit(self, b: int):
        self.bits.append(b)
        while self.pending:
            self.bits.append(1 - b)
            self.pending -= 1

    def encode(self, cum_lo: int, cum_hi: int, total: int):
        span = self.high - self.low + 1
        self.high = self.low + span * cum_hi // total - 1
        self.low = self.low + span * cum_lo // total
        while True:
            if self.high < HALF:
                self._bit(0)
            elif self.low >= HALF:
                self._bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREEQ:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK

    def finish(self) -> bytes:
        self.pending += 1
        self._bit(0 if self.low < QUARTER else 1)
        self.bits.extend([0] * 64)  # tail padding for the decoder
        out = bytearray()
        acc = 0
        n = 0
        for b in self.bits:
            acc = (acc << 1) | b
            n += 1
            if n == 8:
                out.append(acc)
                acc = 0
                n = 0
        if n:
            out.append(acc << (8 - n))
        return bytes(out)


class Decoder:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0
        self.low = 0
        self.high = MASK
        self.code = 0
        for _ in range(32):
            self.code = (self.code << 1) | self._next_bit()

    def _next_bit(self) -> int:
        byte_idx = self.pos >> 3
        if byte_idx >= len(self.data):
            self.pos += 1
            return 0
        bit = (self.data[byte_idx] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def decode_value(self, total: int) -> int:
        span = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // span

    def consume(self, cum_lo: int, cum_hi: int, total: int):
        span = self.high - self.low + 1
        self.high = self.low + span * cum_hi // total - 1
        self.low = self.low + span * cum_lo // total
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.code -= HALF
            elif self.low >= QUARTER and self.high < THREEQ:
                self.low -= QUARTER
                self.high -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.low = (self.low << 1) & MASK
            self.high = ((self.high << 1) | 1) & MASK
            self.code = ((self.code << 1) | self._next_bit()) & MASK


# ----------------------------------------------------------------------
# Compress / decompress
# ----------------------------------------------------------------------

def compress(data: bytes) -> tuple[bytes, float, int]:
    model = LatticeContextModel()
    enc = Encoder()
    hist: list[int] = []
    ideal_bits = 0.0
    for byte in data:
        freq, total = model.freq_table(hist)
        cum = np.cumsum(freq)
        cum_lo = int(cum[byte - 1]) if byte > 0 else 0
        cum_hi = int(cum[byte])
        enc.encode(cum_lo, cum_hi, total)
        ideal_bits += -math.log2((cum_hi - cum_lo) / total)
        model.update(hist, byte)
        hist.append(byte)
        if len(hist) > MAX_ORDER:
            hist.pop(0)
    return enc.finish(), ideal_bits, model.distinct_contexts()


def decompress(blob: bytes, n: int) -> bytes:
    model = LatticeContextModel()
    dec = Decoder(blob)
    hist: list[int] = []
    out = bytearray()
    for _ in range(n):
        freq, total = model.freq_table(hist)
        cum = np.cumsum(freq)
        value = dec.decode_value(total)
        byte = int(np.searchsorted(cum, value, side="right"))
        cum_lo = int(cum[byte - 1]) if byte > 0 else 0
        cum_hi = int(cum[byte])
        dec.consume(cum_lo, cum_hi, total)
        model.update(hist, byte)
        hist.append(byte)
        if len(hist) > MAX_ORDER:
            hist.pop(0)
        out.append(byte)
    return bytes(out)


def main():
    header("Lattice-context compressor - below the frequency floor, for real")

    # ------------------------------------------------------------------
    # Load REAL data: this repo's own derivations markdown
    # ------------------------------------------------------------------
    files = sorted((ROOT / "derivations").glob("*.md"))
    raw = b"".join(f.read_bytes() for f in files)
    data = raw[:65536]
    print(f"  data: {len(files)} markdown files from derivations/, "
          f"first {len(data)} bytes")

    # ------------------------------------------------------------------
    # Scoreboard part 1: the frequency floor and standard compressors
    # ------------------------------------------------------------------
    print("\nBaselines")
    print("-" * 72)
    counts = Counter(data)
    n = len(data)
    H0 = -sum((c / n) * math.log2(c / n) for c in counts.values())
    floor_bytes = H0 * n / 8
    print(f"  order-0 entropy (frequency-map floor): {H0:.3f} bits/byte"
          f" -> {floor_bytes:.0f} bytes")
    print(f"  this is the 'Shannon limit' a frequency map can ever reach")

    base_results = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        out = fn(data)
        base_results[name] = len(out)
        print(f"  {name:<10} {len(out):>7} bytes   "
              f"{len(out)*8/n:.3f} bits/byte   {n/len(out):.2f}x")

    # ------------------------------------------------------------------
    # Our codec
    # ------------------------------------------------------------------
    print("\nLattice-context codec (orders 0..5, FTA composite addressing)")
    print("-" * 72)
    t0 = time.time()
    blob, ideal_bits, n_contexts = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)
    print(f"  encoded:  {ours} bytes in {t_enc:.1f}s")
    print(f"  bits/byte: {ours*8/n:.3f} (model cross-entropy {ideal_bits/n:.3f})")
    print(f"  ratio:     {n/ours:.2f}x")
    print(f"  distinct contexts learned: {n_contexts}")
    print(f"  (your '32 meanings per symbol' -> here each byte effectively has")
    print(f"   thousands of context-meanings; that's what buys the extra bits)")

    t0 = time.time()
    restored = decompress(blob, n)
    t_dec = time.time() - t0
    print(f"  decoded:  {len(restored)} bytes in {t_dec:.1f}s")
    assertion(restored == data,
              "round-trip is byte-exact (this is a real, decodable codec)")

    # ------------------------------------------------------------------
    # The verdicts
    # ------------------------------------------------------------------
    print("\nScoreboard")
    print("-" * 72)
    rows = [("frequency floor (order-0)", floor_bytes),
            *[(k, float(v)) for k, v in base_results.items()],
            ("lattice-context codec", float(ours))]
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "lattice-context codec" else ""
        print(f"  {name:<28} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")

    assertion(ours < floor_bytes,
              f"BELOW the frequency-map floor by {(1 - ours/floor_bytes)*100:.0f}% "
              f"- the 'limit' moved because the model got better")
    for name, size in base_results.items():
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
              f"({ours} vs {size} bytes)")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    header("RESULT")
    best_base = min(base_results, key=base_results.get)
    print(f"  frequency floor:   {H0:.3f} bits/byte (naive 'Shannon limit')")
    print(f"  our codec:         {ours*8/n:.3f} bits/byte "
          f"({(1-ours*8/n/H0)*100:.0f}% below that floor)")
    print(f"  best classical:    {best_base} = "
          f"{base_results[best_base]*8/n:.3f} bits/byte")
    print()
    print("  CONCLUSION:")
    print("  'Compress further' is TRUE - relative to any fixed-model floor.")
    print("  Context conditioning moved the reachable limit from "
          f"{H0:.2f} to {ideal_bits/n:.2f} bits/byte")
    print("  on this data, and the codec actually achieves it, decodably.")
    print("  The lattice contributes the context machinery: FTA composites as")
    print("  provably collision-free context addresses (Test 3 doing real work).")
    print()
    print("  The wall that remains is the conditional entropy under the TRUE")
    print("  source model - unknowable for real data, so practical headroom")
    print("  almost always exists. It is zero only for encrypted / already-")
    print("  compressed / truly random bytes (Tests 13-14).")
    print()
    print("  Next frontier: replace raw bytes with promotion-mined units")
    print("  (L2 PMI subwords from the retrieval work) as the symbol alphabet,")
    print("  and blend the 32 chambers as parallel context models - both are")
    print("  assets this repo already has.")


if __name__ == "__main__":
    main()
