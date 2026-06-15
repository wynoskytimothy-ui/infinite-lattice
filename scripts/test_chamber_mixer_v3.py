#!/usr/bin/env python3
"""
Test 21 - Chamber mixer v3: context-selected mixing + chained APM + 14 chambers.

Upgrades over v2 (Test 19, 2.534 / 2.196 bits/byte at 64KB / 256KB):

  1. CONTEXT-SELECTED MIXER (the cmix/PAQ8 trick): instead of one weight
     set per bit position, weights are selected by
        (match-active, last-byte-is-wordchar, at-line-start) x bit position
     = 8 selectors x 8 bit positions = 64 independent weight sets.
     The mixer learns DIFFERENT chamber trust in different regimes
     (e.g. trust MATCH when a match is running, trust W inside words).

  2. CHAINED APM: two calibration stages - APM1 conditioned on the last
     byte, APM2 conditioned on (selector, last-byte bucket) - each blended
     1:3 with its input.

  3. NEW chambers (14 total):
     O6     - order-6 context. Died in count-blending (Test 15 tuning);
              safe under bounded logistic mixing, which learns its weight.
     MATCH16 - second match model on 16-byte contexts: longer, rarer,
              higher-confidence matches; the corpus has heavy cross-file
              repetition for it to feast on.

Run modes:
    python test_chamber_mixer_v3.py          -> 64KB + 256KB benchmarks
    python test_chamber_mixer_v3.py full     -> full ~1.4MB corpus showdown
"""

from __future__ import annotations

import bz2
import lzma
import math
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_lattice_context_compressor import Encoder, Decoder

ROOT = Path(__file__).resolve().parents[1]

MAX_ORDER = 6                # O1..O6 use slots 0..5
WORD_SLOTS = 8               # slots 6..13
LP_SLOTS = 4                 # slots 14..17
N_CH = 14                    # O0..O6, W, WP, S13, S24, LP, M8, M16
CH_NAMES = ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "W", "WP",
            "S13", "S24", "LP", "M8", "M16"]
M8_CH, M16_CH = 12, 13
N_SEL = 8
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


def apm_table(n_ctx: int) -> list:
    return [[squash((i * 20.0 / 32.0) - 10.0) for i in range(33)]
            for _ in range(n_ctx)]


class Matcher:
    """One history-match chamber: hash of last `clen` bytes -> next byte."""

    def __init__(self, clen: int, base_conf: float, slope: float):
        self.clen = clen
        self.base_conf = base_conf
        self.slope = slope
        self.table: dict[int, int] = {}
        self.ptr = -1
        self.mlen = 0
        self.pred = -1
        self.active = 0
        self.hits = 0

    def begin(self, history: bytearray):
        self.pred = (history[self.ptr]
                     if 0 <= self.ptr < len(history) else -1)
        if self.pred >= 0:
            self.active += 1

    def vote(self, bit_idx: int, prefix: int) -> float:
        pb = self.pred
        if pb < 0:
            return 0.0
        if prefix != (1 << bit_idx) | ((pb >> (8 - bit_idx)) if bit_idx else 0):
            return 0.0
        bit_pred = (pb >> (7 - bit_idx)) & 1
        conf = self.base_conf + self.slope * self.mlen
        if conf > CLIP:
            conf = CLIP
        return conf if bit_pred else -conf

    def end(self, history: bytearray, byte: int):
        if self.pred == byte and self.ptr >= 0:
            self.ptr += 1
            self.mlen += 1
            self.hits += 1
            relookup = False
        else:
            self.mlen = 0
            self.ptr = -1
            relookup = True
        # history already includes `byte` when end() is called
        if len(history) >= self.clen:
            key = int.from_bytes(history[-self.clen:], "little")
            if relookup:
                cand = self.table.get(key)
                if cand is not None:
                    self.ptr = cand
                    self.mlen = self.clen
            self.table[key] = len(history)


class ChamberMixerV3:
    def __init__(self):
        self.tables: list[dict] = [{} for _ in range(12)]  # count chambers
        self.w = [[[0.0] * N_CH for _ in range(8)] for _ in range(N_SEL)]
        self.hist: list[int] = []
        self.cur_word_key = 1
        self.cur_word_len = 0
        self.prev_word_key = 1
        self.line_key = 1
        self.line_len = 0
        self.history = bytearray()
        self.m8 = Matcher(8, 1.0, 0.5)
        self.m16 = Matcher(16, 2.0, 0.6)
        self.apm1 = apm_table(256)
        self.apm2 = apm_table(N_SEL * 32)
        self._keys: list = [0] * 12
        self._sel = 0

    def begin_byte(self):
        h = self.hist
        keys: list = [1]                                   # O0
        key = 1
        for j in range(MAX_ORDER):                         # O1..O6
            if j < len(h):
                key *= PRIMES[j * 256 + h[-1 - j]]
                keys.append(key)
            else:
                keys.append(0)
        keys.append(self.cur_word_key)                     # W
        keys.append((self.prev_word_key, self.cur_word_key))  # WP
        if len(h) >= 3:
            keys.append(PRIMES[0 * 256 + h[-1]] * PRIMES[2 * 256 + h[-3]])
        else:
            keys.append(0)
        if len(h) >= 4:
            keys.append(PRIMES[1 * 256 + h[-2]] * PRIMES[3 * 256 + h[-4]])
        else:
            keys.append(0)
        keys.append(self.line_key)                         # LP
        self._keys = keys
        self.m8.begin(self.history)
        self.m16.begin(self.history)
        m_active = 1 if (self.m8.pred >= 0 or self.m16.pred >= 0) else 0
        w_state = 1 if (h and is_wordchar(h[-1])) else 0
        l_state = 1 if self.line_len == 0 else 0
        self._sel = m_active * 4 + w_state * 2 + l_state

    def predict(self, bit_idx: int, prefix: int):
        xs = [0.0] * N_CH
        dot = 0.0
        wrow = self.w[self._sel][bit_idx]
        for i in range(12):
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
        x = self.m8.vote(bit_idx, prefix)
        if x != 0.0:
            xs[M8_CH] = x
            dot += wrow[M8_CH] * x
        x = self.m16.vote(bit_idx, prefix)
        if x != 0.0:
            xs[M16_CH] = x
            dot += wrow[M16_CH] * x
        p_mix = squash(dot)
        # APM stage 1: conditioned on last byte
        ctx1 = self.hist[-1] if self.hist else 0
        s = stretch(p_mix)
        pos = (s + 10.0) * 1.6
        b1 = int(pos)
        if b1 > 31:
            b1 = 31
        f1 = pos - b1
        row1 = self.apm1[ctx1]
        a1 = row1[b1] * (1.0 - f1) + row1[b1 + 1] * f1
        q1 = 0.25 * p_mix + 0.75 * a1
        if q1 < 0.0002:
            q1 = 0.0002
        elif q1 > 0.9998:
            q1 = 0.9998
        # APM stage 2: conditioned on (selector, last-byte bucket)
        ctx2 = self._sel * 32 + (ctx1 >> 3)
        s2 = stretch(q1)
        pos2 = (s2 + 10.0) * 1.6
        b2 = int(pos2)
        if b2 > 31:
            b2 = 31
        f2 = pos2 - b2
        row2 = self.apm2[ctx2]
        a2 = row2[b2] * (1.0 - f2) + row2[b2 + 1] * f2
        p = 0.25 * q1 + 0.75 * a2
        if p < 0.0002:
            p = 0.0002
        elif p > 0.9998:
            p = 0.9998
        return p, (xs, ctx1, b1, f1, a1, ctx2, b2, f2, a2)

    def learn(self, bit_idx: int, prefix: int, bit: int, p: float, aux):
        xs, ctx1, b1, f1, a1, ctx2, b2, f2, a2 = aux
        err = (bit - p) * LR
        wrow = self.w[self._sel][bit_idx]
        for i in range(N_CH):
            x = xs[i]
            if x != 0.0:
                wrow[i] += x * err
        e1 = (bit - a1) * APM_RATE
        row = self.apm1[ctx1]
        row[b1] = min(max(row[b1] + e1 * (1.0 - f1), 1e-4), 1.0 - 1e-4)
        row[b1 + 1] = min(max(row[b1 + 1] + e1 * f1, 1e-4), 1.0 - 1e-4)
        e2 = (bit - a2) * APM_RATE
        row = self.apm2[ctx2]
        row[b2] = min(max(row[b2] + e2 * (1.0 - f2), 1e-4), 1.0 - 1e-4)
        row[b2 + 1] = min(max(row[b2 + 1] + e2 * f2, 1e-4), 1.0 - 1e-4)
        for i in range(12):
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
        if byte == 10:
            self.line_key = 1
            self.line_len = 0
        elif self.line_len < LP_SLOTS:
            self.line_key *= PRIMES[(MAX_ORDER + WORD_SLOTS + self.line_len)
                                    * 256 + byte]
            self.line_len += 1
        self.history.append(byte)
        self.m8.end(self.history, byte)
        self.m16.end(self.history, byte)

    def report(self) -> str:
        auth = []
        for i in range(N_CH):
            a = sum(abs(self.w[s][k][i]) for s in range(N_SEL)
                    for k in range(8)) / (N_SEL * 8)
            auth.append(f"{CH_NAMES[i]}={a:.2f}")
        h8 = (self.m8.hits / self.m8.active * 100) if self.m8.active else 0
        h16 = (self.m16.hits / self.m16.active * 100) if self.m16.active else 0
        return (f"    authority: {', '.join(auth)}\n"
                f"    M8: active {self.m8.active}, correct {h8:.0f}%   "
                f"M16: active {self.m16.active}, correct {h16:.0f}%")


def bit_ranges(p: float) -> int:
    c1 = int(p * TOTAL)
    if c1 < 1:
        c1 = 1
    elif c1 > TOTAL - 1:
        c1 = TOTAL - 1
    return c1


def compress(data: bytes):
    model = ChamberMixerV3()
    enc = Encoder()
    for byte in data:
        model.begin_byte()
        prefix = 1
        for k in range(8):
            bit = (byte >> (7 - k)) & 1
            p, aux = model.predict(k, prefix)
            c1 = bit_ranges(p)
            if bit:
                enc.encode(0, c1, TOTAL)
            else:
                enc.encode(c1, TOTAL, TOTAL)
            model.learn(k, prefix, bit, p, aux)
            prefix = (prefix << 1) | bit
        model.end_byte(byte)
    return enc.finish(), model


def decompress(blob: bytes, n: int) -> bytes:
    model = ChamberMixerV3()
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


def scoreboard(data: bytes, label: str, reference: dict | None = None):
    n = len(data)
    print(f"\n{label}: {n} bytes")
    print("-" * 72)
    base = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        base[name] = len(fn(data))

    t0 = time.time()
    blob, model = compress(data)
    t_enc = time.time() - t0
    ours = len(blob)
    t0 = time.time()
    restored = decompress(blob, n)
    t_dec = time.time() - t0
    assertion(restored == data, f"round-trip byte-exact ({label})")

    rows = [*[(k, float(v)) for k, v in base.items()],
            ("chamber mixer v3", float(ours))]
    if reference:
        rows.extend((k, float(v)) for k, v in reference.items())
    for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "chamber mixer v3" else ""
        print(f"  {name:<24} {size:>9.0f} bytes  "
              f"{size*8/n:>6.3f} bits/byte{marker}")
    print(f"  (encode {t_enc:.1f}s, decode {t_dec:.1f}s)")
    print(model.report())
    for name, size in base.items():
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
              f"by {abs(1-ours/size)*100:.1f}%")
    return ours


def main():
    full_mode = len(sys.argv) > 1 and sys.argv[1] == "full"
    header("Chamber mixer v3 - context-selected mixing, chained APM, 14 chambers")

    if not full_mode:
        deriv = b"".join(f.read_bytes()
                         for f in sorted((ROOT / "derivations").glob("*.md")))
        data64 = deriv[:65536]
        ours64 = scoreboard(data64, "Run 1 - benchmark 64KB",
                            reference={"mixer v2 (Test 19)": 20757})
        assertion(ours64 < 20757,
                  f"v3 improves on v2 at 64KB ({(1-ours64/20757)*100:.1f}% smaller)")

        data256 = corpus()[:262144]
        ours256 = scoreboard(data256, "Run 2 - scale 256KB",
                             reference={"mixer v2 (Test 19)": 71944})
        assertion(ours256 < 71944,
                  f"v3 improves on v2 at 256KB ({(1-ours256/71944)*100:.1f}% smaller)")

        header("BENCHMARK RESULT")
        print(f"  64KB:  {ours64*8/65536:.3f} bits/byte  (v2: 2.534, bz2: 2.831)")
        print(f"  256KB: {ours256*8/262144:.3f} bits/byte  (v2: 2.196, bz2: 2.451)")
        print()
        print("  Run with 'full' argument for the ~1.4MB corpus showdown.")
    else:
        data = corpus()
        ours = scoreboard(data, f"FULL CORPUS - {len(data)//1024}KB")
        n = len(data)
        header("FULL-CORPUS RESULT")
        print(f"  ours: {ours} bytes = {ours*8/n:.3f} bits/byte")
        below_one = ours * 8 / n < 1.0
        print(f"  below 1 bit/byte: {'YES' if below_one else 'no'}")
        print()
        print("  Long-range match chambers + context-selected mixing on a")
        print("  corpus with heavy cross-file repetition - this is where")
        print("  chamber architectures separate hardest from block coders.")


if __name__ == "__main__":
    main()
