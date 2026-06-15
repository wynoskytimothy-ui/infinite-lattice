#!/usr/bin/env python3
"""
Test 19 - Chamber mixer v2: more chambers + match model + SSE calibration.

Test 18 (8 chambers, bit-level logistic mixing) = 2.612 bits/byte, beating
zlib/bz2/lzma. This version adds the remaining PAQ-family components, all
expressed in repo primitives:

  NEW chambers:
    S13 : sparse context (bytes at t-1 and t-3) - FTA composite, slots {0,2}
    S24 : sparse context (bytes at t-2 and t-4) - FTA composite, slots {1,3}
    LP  : line-prefix - composite of the first 4 bytes of the current line
          (markdown structure: '#', '- ', '| ' line types), dedicated slots
    MATCH: history match model - find the last occurrence of the current
          8-byte context (hash table), predict the byte that followed,
          with confidence growing in match length. This is LZ77 as a
          CHAMBER VOTE instead of a copy command.

  NEW stage:
    SSE/APM - the mixed probability is recalibrated through a learned
    33-bin table conditioned on the previous byte, interpolated, then
    blended 1:3 with the raw mixture. Fixes systematic over/under-
    confidence of the mixer.

  12 chambers total; weights learned per bit-position; decoder mirrors
  everything; zero pre-trained state.

Run 1: same 64KB as Tests 15-18 (comparable scoreboard).
Run 2: 256KB scale check from the full repo markdown corpus.
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

ROOT = Path(__file__).resolve().parents[1]

MAX_ORDER = 5
WORD_SLOTS = 8      # slots 5..12
LP_SLOTS = 4        # slots 13..16
N_CH = 12           # O0,O1..O5,W,WP,S13,S24,LP,MATCH
CH_NAMES = ["O0", "O1", "O2", "O3", "O4", "O5", "W", "WP",
            "S13", "S24", "LP", "MATCH"]
MATCH_CH = 11
LR = 0.003
CLIP = 10.0
COUNT_CAP = 1024
TOTAL = 4096
APM_RATE = 0.02
PRIMES = chain_primes(256 * (MAX_ORDER + WORD_SLOTS + LP_SLOTS) + 64)


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


class ChamberMixerV2:
    def __init__(self):
        self.tables: list[dict] = [{} for _ in range(N_CH)]   # MATCH unused
        self.w = [[0.0] * N_CH for _ in range(8)]
        self.hist: list[int] = []
        self.cur_word_key = 1
        self.cur_word_len = 0
        self.prev_word_key = 1
        self.line_key = 1
        self.line_len = 0
        # match model
        self.history = bytearray()
        self.h8: dict[int, int] = {}
        self.match_ptr = -1
        self.match_len = 0
        self.pred_byte = -1
        self.match_hits = 0
        self.match_active = 0
        # SSE/APM: 256 contexts x 33 bins
        self.apm = [[squash((i * 20.0 / 32.0) - 10.0) for i in range(33)]
                    for _ in range(256)]
        self._keys: list = [0] * N_CH

    # ---- per-byte setup ----

    def begin_byte(self):
        keys: list = [1]                                   # O0
        key = 1
        h = self.hist
        for j in range(MAX_ORDER):                         # O1..O5
            if j < len(h):
                key *= PRIMES[j * 256 + h[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)
        keys.append(self.cur_word_key)                     # W
        keys.append((self.prev_word_key, self.cur_word_key))  # WP
        if len(h) >= 3:                                    # S13: t-1, t-3
            keys.append(PRIMES[0 * 256 + h[-1]] * PRIMES[2 * 256 + h[-3]])
        else:
            keys.append(0)
        if len(h) >= 4:                                    # S24: t-2, t-4
            keys.append(PRIMES[1 * 256 + h[-2]] * PRIMES[3 * 256 + h[-4]])
        else:
            keys.append(0)
        keys.append(self.line_key)                         # LP
        keys.append(0)                                     # MATCH (special)
        self._keys = keys
        self.pred_byte = (self.history[self.match_ptr]
                          if 0 <= self.match_ptr < len(self.history) else -1)
        if self.pred_byte >= 0:
            self.match_active += 1

    # ---- per-bit prediction / learning ----

    def predict(self, bit_idx: int, prefix: int):
        xs = [0.0] * N_CH
        dot = 0.0
        wrow = self.w[bit_idx]
        for i in range(N_CH - 1):
            ck = self._keys[i]
            if ck == 0:
                continue
            e = self.tables[i].get((ck, prefix))
            if e is None:
                continue
            n0, n1 = e
            x = stretch((n1 + 0.25) / (n0 + n1 + 0.5))
            xs[i] = x
            dot += wrow[i] * x
        pb = self.pred_byte
        if pb >= 0:
            # only votes while the coded prefix still agrees with pb
            if prefix == (1 << bit_idx) | (pb >> (8 - bit_idx) if bit_idx else 0):
                bit_pred = (pb >> (7 - bit_idx)) & 1
                x = (1.0 if bit_pred else -1.0) * min(1.0 + 0.5 * self.match_len, CLIP)
                xs[MATCH_CH] = x
                dot += wrow[MATCH_CH] * x
        p_mix = squash(dot)
        # SSE/APM stage
        ctx = self.hist[-1] if self.hist else 0
        s = stretch(p_mix)
        pos = (s + 10.0) * 1.6                  # -> [0, 32]
        b0 = int(pos)
        if b0 > 31:
            b0 = 31
        frac = pos - b0
        row = self.apm[ctx]
        p2 = row[b0] * (1.0 - frac) + row[b0 + 1] * frac
        p = 0.25 * p_mix + 0.75 * p2
        if p < 0.0002:
            p = 0.0002
        elif p > 0.9998:
            p = 0.9998
        return p, (xs, ctx, b0, frac, p2)

    def learn(self, bit_idx: int, prefix: int, bit: int, p: float, aux):
        xs, ctx, b0, frac, p2 = aux
        err = (bit - p) * LR
        wrow = self.w[bit_idx]
        for i in range(N_CH):
            x = xs[i]
            if x != 0.0:
                wrow[i] += x * err
        # APM bins
        e2 = (bit - p2) * APM_RATE
        row = self.apm[ctx]
        row[b0] = min(max(row[b0] + e2 * (1.0 - frac), 1e-4), 1.0 - 1e-4)
        row[b0 + 1] = min(max(row[b0 + 1] + e2 * frac, 1e-4), 1.0 - 1e-4)
        # count tables
        for i in range(N_CH - 1):
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

    # ---- per-byte state advance ----

    def end_byte(self, byte: int):
        self.hist.append(byte)
        if len(self.hist) > MAX_ORDER:
            self.hist.pop(0)
        # word state
        if is_wordchar(byte):
            slot = min(self.cur_word_len, WORD_SLOTS - 1)
            self.cur_word_key *= PRIMES[(MAX_ORDER + slot) * 256 + byte]
            self.cur_word_len += 1
        else:
            if self.cur_word_len > 0:
                self.prev_word_key = self.cur_word_key
            self.cur_word_key = 1
            self.cur_word_len = 0
        # line state
        if byte == 10:
            self.line_key = 1
            self.line_len = 0
        elif self.line_len < LP_SLOTS:
            self.line_key *= PRIMES[(MAX_ORDER + WORD_SLOTS + self.line_len)
                                    * 256 + byte]
            self.line_len += 1
        # match state
        if self.pred_byte == byte and self.match_ptr >= 0:
            self.match_ptr += 1
            self.match_len += 1
            self.match_hits += 1
            relookup = False
        else:
            self.match_len = 0
            self.match_ptr = -1
            relookup = True
        self.history.append(byte)
        if len(self.history) >= 8:
            k8 = int.from_bytes(self.history[-8:], "little")
            if relookup:
                cand = self.h8.get(k8)
                if cand is not None:
                    self.match_ptr = cand
                    self.match_len = 8
            self.h8[k8] = len(self.history)

    def report(self) -> str:
        auth = ", ".join(
            f"{CH_NAMES[i]}={sum(abs(self.w[k][i]) for k in range(8))/8:.2f}"
            for i in range(N_CH))
        hit = (self.match_hits / self.match_active * 100
               if self.match_active else 0.0)
        return (f"    authority: {auth}\n"
                f"    match chamber: active {self.match_active} bytes, "
                f"correct {hit:.0f}%")


def bit_ranges(p: float) -> int:
    c1 = int(p * TOTAL)
    if c1 < 1:
        c1 = 1
    elif c1 > TOTAL - 1:
        c1 = TOTAL - 1
    return c1


def compress(data: bytes) -> tuple[bytes, float, ChamberMixerV2]:
    model = ChamberMixerV2()
    enc = Encoder()
    ideal_bits = 0.0
    for byte in data:
        model.begin_byte()
        prefix = 1
        for k in range(8):
            bit = (byte >> (7 - k)) & 1
            p, aux = model.predict(k, prefix)
            c1 = bit_ranges(p)
            if bit:
                enc.encode(0, c1, TOTAL)
                ideal_bits += -math.log2(c1 / TOTAL)
            else:
                enc.encode(c1, TOTAL, TOTAL)
                ideal_bits += -math.log2((TOTAL - c1) / TOTAL)
            model.learn(k, prefix, bit, p, aux)
            prefix = (prefix << 1) | bit
        model.end_byte(byte)
    return enc.finish(), ideal_bits, model


def decompress(blob: bytes, n: int) -> bytes:
    model = ChamberMixerV2()
    dec = Decoder(blob)
    out = bytearray()
    for _ in range(n):
        model.begin_byte()
        prefix = 1
        for k in range(8):
            p, aux = model.predict(k, prefix)
            c1 = bit_ranges(p)
            value = dec.decode_value(TOTAL)
            if value < c1:
                bit = 1
                dec.consume(0, c1, TOTAL)
            else:
                bit = 0
                dec.consume(c1, TOTAL, TOTAL)
            model.learn(k, prefix, bit, p, aux)
            prefix = (prefix << 1) | bit
        byte = prefix & 0xFF
        model.end_byte(byte)
        out.append(byte)
    return bytes(out)


def corpus() -> bytes:
    files = sorted(ROOT.glob("derivations/*.md")) + \
            sorted(ROOT.glob("book/**/*.md")) + \
            sorted(ROOT.glob("*.md"))
    return b"".join(f.read_bytes() for f in files)


def scoreboard(data: bytes, label: str, include_t18: bool = False):
    n = len(data)
    print(f"\n{label}: {n} bytes")
    print("-" * 72)
    base = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        out = fn(data)
        base[name] = len(out)

    t0 = time.time()
    blob, ideal_bits, model = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)

    t0 = time.time()
    restored = decompress(blob, n)
    t_dec = time.time() - t0
    assertion(restored == data, f"round-trip byte-exact ({label})")

    rows = [*[(k, float(v)) for k, v in base.items()],
            ("chamber mixer v2", float(ours))]
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "chamber mixer v2" else ""
        print(f"  {name:<22} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")
    print(f"  (encode {t_enc:.1f}s, decode {t_dec:.1f}s)")
    print(model.report())
    for name, size in base.items():
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
              f"by {abs(1-ours/size)*100:.1f}%")
    return ours, base


def main():
    header("Chamber mixer v2 - sparse + line + match chambers, SSE calibrated")

    full = corpus()
    print(f"  corpus: {len(full)/1024:.0f} KB of repo markdown")

    # Run 1: the comparable 64KB benchmark (same data as Tests 15-18)
    deriv = b"".join(f.read_bytes() for f in sorted((ROOT / "derivations").glob("*.md")))
    data64 = deriv[:65536]
    ours64, base64 = scoreboard(data64, "Run 1 - benchmark 64KB (same as Tests 15-18)")
    print(f"\n  Test 18 reference: 21394 bytes (2.612 bits/byte)")
    assertion(ours64 < 21394,
              f"v2 improves on Test 18 ({(1-ours64/21394)*100:.1f}% smaller)")

    # Run 2: 256KB scale
    data256 = full[:262144]
    ours256, base256 = scoreboard(data256, "Run 2 - scale 256KB (full corpus)")

    # ------------------------------------------------------------------
    header("RESULT")
    print(f"  64KB:  ours {ours64*8/65536:.3f} bits/byte vs best classical "
          f"{min(base64.values())*8/65536:.3f}")
    print(f"  256KB: ours {ours256*8/262144:.3f} bits/byte vs best classical "
          f"{min(base256.values())*8/262144:.3f}")
    print()
    print("  CONCLUSION:")
    print("  Twelve chambers, each a different geometry over the same stream")
    print("  (orders, words, sparse skips, line types, history matches),")
    print("  mixed by learned per-bit trust and recalibrated by SSE. All")
    print("  context addresses are FTA composites. The match chamber is")
    print("  LZ77 recast as a VOTE rather than a copy command - the lattice")
    print("  way: every mechanism is a chamber, and trust is earned.")


if __name__ == "__main__":
    main()
