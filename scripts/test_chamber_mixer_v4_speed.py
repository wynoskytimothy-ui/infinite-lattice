#!/usr/bin/env python3
"""
Test 22 - Chamber mixer v4: lazy branching, lazy evaluation, O(1)-style lookups.

Same model family as v3 (Test 21: 0.881 bits/byte at full corpus, but
~100s each way). v4 restructures the evaluation for speed, using the
lattice's own structure:

  1. LAZY BRANCH FETCH: a chamber's FTA composite key is CONSTANT within
     a byte. v3 looked up (key, bit_prefix) in a flat dict per BIT -
     8 big-key probes per chamber per byte. v4 fetches the chamber's
     context NODE once per byte (one big-key probe), then the 8 bit steps
     descend the fetched node with tiny int keys. 8x fewer big lookups.

  2. LAZY EVALUATION (early exit): chambers vote in trust order (M16, M8,
     O6..O1, words, sparse, line, O0). If the running vote already exceeds
     +/-EXIT_T (the arithmetic coder's probability is saturated), the
     remaining chambers are not evaluated at all. The rule is
     deterministic, so encoder and decoder skip identically - lossless.
     On match-heavy spans (79% of the full corpus) most chambers never run.

  3. TABLE-DRIVEN TRANSFORMS: stretch/squash via 4096/8192-entry quantized
     tables instead of math.log/math.exp per bit (deterministic on both
     sides by construction).

  4. PACKED COUNTS: (n0, n1) packed into one int per node slot - no list
     allocation per context cell.

Run modes:
    python test_chamber_mixer_v4_speed.py            -> 64KB bench, lazy on+off
    python test_chamber_mixer_v4_speed.py full       -> full corpus, lazy on
    python test_chamber_mixer_v4_speed.py full eager -> full corpus, lazy off
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

MAX_ORDER = 6
WORD_SLOTS = 8
LP_SLOTS = 4
N_CH = 14
CH_NAMES = ["O0", "O1", "O2", "O3", "O4", "O5", "O6", "W", "WP",
            "S13", "S24", "LP", "M8", "M16"]
M8_CH, M16_CH = 12, 13
EVAL_ORDER = [13, 12, 6, 5, 4, 3, 2, 1, 7, 8, 9, 10, 11, 0]
N_SEL = 8
LR = 0.003
CLIP = 10.0
COUNT_CAP = 1024
TOTAL = 4096
APM_RATE = 0.02
EXIT_T = 14.0
PRIMES = chain_primes(256 * (MAX_ORDER + WORD_SLOTS + LP_SLOTS) + 64)

# quantized transform tables (identical on encoder and decoder)
STRETCH_TAB = []
for i in range(4096):
    pc = (i + 0.5) / 4096.0
    x = math.log(pc / (1.0 - pc))
    STRETCH_TAB.append(max(-CLIP, min(CLIP, x)))
SQUASH_TAB = []
for i in range(8192):
    x = (i + 0.5) / 8192.0 * 50.0 - 25.0
    SQUASH_TAB.append(1.0 / (1.0 + math.exp(-x)))


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


def squash_q(dot: float) -> float:
    i = int((dot + 25.0) * 163.84)
    if i < 0:
        i = 0
    elif i > 8191:
        i = 8191
    return SQUASH_TAB[i]


def stretch_q(p: float) -> float:
    i = int(p * 4096.0)
    if i < 0:
        i = 0
    elif i > 4095:
        i = 4095
    return STRETCH_TAB[i]


def apm_table(n_ctx: int) -> list:
    return [[1.0 / (1.0 + math.exp(-((i * 20.0 / 32.0) - 10.0)))
             for i in range(33)] for _ in range(n_ctx)]


class Matcher:
    __slots__ = ("clen", "base_conf", "slope", "table", "ptr", "mlen",
                 "pred", "active", "hits")

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
        if len(history) >= self.clen:
            key = int.from_bytes(history[-self.clen:], "little")
            if relookup:
                cand = self.table.get(key)
                if cand is not None:
                    self.ptr = cand
                    self.mlen = self.clen
            self.table[key] = len(history)


class ChamberMixerV4:
    def __init__(self, lazy: bool):
        self.lazy = lazy
        # count chambers: chamber -> {composite_key: {prefix: packed_counts}}
        self.tables: list[dict] = [{} for _ in range(12)]
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
        self._nodes: list = [None] * 12      # fetched per byte (lazy branch)
        self._keys: list = [0] * 12
        self._sel = 0
        # per-bit scratch (avoid tuple allocation)
        self._act_i = [0] * N_CH
        self._act_x = [0.0] * N_CH
        self._act_n = 0
        self._apm_cells = [0, 0, 0.0, 0.0, 0, 0, 0.0, 0.0]
        self.evals = 0
        self.skips = 0

    # ---- per-byte: compute keys, fetch nodes ONCE (lazy branch fetch) ----

    def begin_byte(self):
        h = self.hist
        keys = self._keys
        keys[0] = 1
        key = 1
        nh = len(h)
        for j in range(MAX_ORDER):
            if j < nh:
                key *= PRIMES[j * 256 + h[-1 - j]]
                keys[j + 1] = key
            else:
                keys[j + 1] = 0
        keys[7] = self.cur_word_key
        keys[8] = (self.prev_word_key, self.cur_word_key)        # WP (exact)
        keys[9] = (PRIMES[h[-1]] * PRIMES[512 + h[-3]]) if nh >= 3 else 0
        keys[10] = (PRIMES[256 + h[-2]] * PRIMES[768 + h[-4]]) if nh >= 4 else 0
        keys[11] = self.line_key
        nodes = self._nodes
        tables = self.tables
        for i in range(12):
            k = keys[i]
            nodes[i] = tables[i].get(k) if k else None
        self.m8.begin(self.history)
        self.m16.begin(self.history)
        m_active = 1 if (self.m8.pred >= 0 or self.m16.pred >= 0) else 0
        w_state = 1 if (h and is_wordchar(h[-1])) else 0
        l_state = 1 if self.line_len == 0 else 0
        self._sel = m_active * 4 + w_state * 2 + l_state

    # ---- per-bit predict (lazy evaluation) ----

    def predict(self, bit_idx: int, prefix: int) -> float:
        wrow = self.w[self._sel][bit_idx]
        act_i = self._act_i
        act_x = self._act_x
        n_act = 0
        dot = 0.0
        lazy = self.lazy
        nodes = self._nodes
        for i in EVAL_ORDER:
            if i >= 12:                       # match chambers
                m = self.m16 if i == M16_CH else self.m8
                pb = m.pred
                if pb < 0:
                    continue
                if prefix != (1 << bit_idx) | ((pb >> (8 - bit_idx)) if bit_idx else 0):
                    continue
                conf = m.base_conf + m.slope * m.mlen
                if conf > CLIP:
                    conf = CLIP
                x = conf if ((pb >> (7 - bit_idx)) & 1) else -conf
            else:
                node = nodes[i]
                if node is None:
                    continue
                packed = node.get(prefix)
                if packed is None:
                    continue
                n0 = packed >> 16
                n1 = packed & 0xFFFF
                x = stretch_q((n1 + 0.25) / (n0 + n1 + 0.5))
            act_i[n_act] = i
            act_x[n_act] = x
            n_act += 1
            dot += wrow[i] * x
            if lazy and (dot > EXIT_T or dot < -EXIT_T):
                self.skips += 1
                break
        self.evals += 1
        self._act_n = n_act
        p_mix = squash_q(dot)
        # APM chain (table-driven stretch)
        ctx1 = self.hist[-1] if self.hist else 0
        pos = (stretch_q(p_mix) + 10.0) * 1.6
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
        ctx2 = self._sel * 32 + (ctx1 >> 3)
        pos2 = (stretch_q(q1) + 10.0) * 1.6
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
        c = self._apm_cells
        c[0], c[1], c[2], c[3] = ctx1, b1, f1, a1
        c[4], c[5], c[6], c[7] = ctx2, b2, f2, a2
        return p

    # ---- per-bit learn ----

    def learn(self, bit_idx: int, prefix: int, bit: int, p: float):
        err = (bit - p) * LR
        wrow = self.w[self._sel][bit_idx]
        act_i = self._act_i
        act_x = self._act_x
        n_act = self._act_n
        nodes = self._nodes
        tables = self.tables
        keys = self._keys
        # mixer weights: only chambers that actually voted
        for a in range(n_act):
            wrow[act_i[a]] += act_x[a] * err
        # counts: ALL valid-key chambers learn every bit (as v3), on the
        # already-fetched nodes - creating branches lazily on first touch
        inc_hi = 0x10000 if not bit else 1   # n0 in high 16 bits, n1 in low
        for i in range(12):
            if keys[i] == 0:
                continue
            node = nodes[i]
            if node is None:
                node = {}
                tables[i][keys[i]] = node
                nodes[i] = node
            packed = node.get(prefix, 0) + inc_hi
            if (packed >> 16) + (packed & 0xFFFF) > COUNT_CAP:
                n0 = packed >> 16
                n1 = packed & 0xFFFF
                n0 -= n0 >> 1
                n1 -= n1 >> 1
                packed = (n0 << 16) | n1
            node[prefix] = packed
        c = self._apm_cells
        ctx1, b1, f1, a1, ctx2, b2, f2, a2 = c
        e1 = (bit - a1) * APM_RATE
        row = self.apm1[ctx1]
        v = row[b1] + e1 * (1.0 - f1)
        row[b1] = 1e-4 if v < 1e-4 else (1.0 - 1e-4 if v > 1.0 - 1e-4 else v)
        v = row[b1 + 1] + e1 * f1
        row[b1 + 1] = 1e-4 if v < 1e-4 else (1.0 - 1e-4 if v > 1.0 - 1e-4 else v)
        e2 = (bit - a2) * APM_RATE
        row = self.apm2[ctx2]
        v = row[b2] + e2 * (1.0 - f2)
        row[b2] = 1e-4 if v < 1e-4 else (1.0 - 1e-4 if v > 1.0 - 1e-4 else v)
        v = row[b2 + 1] + e2 * f2
        row[b2 + 1] = 1e-4 if v < 1e-4 else (1.0 - 1e-4 if v > 1.0 - 1e-4 else v)

    # ---- per-byte state advance ----

    def end_byte(self, byte: int):
        self.hist.append(byte)
        if len(self.hist) > MAX_ORDER:
            self.hist.pop(0)
        if is_wordchar(byte):
            slot = self.cur_word_len
            if slot >= WORD_SLOTS:
                slot = WORD_SLOTS - 1
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

    def lazy_report(self) -> str:
        rate = self.skips / self.evals * 100 if self.evals else 0
        h8 = (self.m8.hits / self.m8.active * 100) if self.m8.active else 0
        h16 = (self.m16.hits / self.m16.active * 100) if self.m16.active else 0
        return (f"    early-exit rate: {rate:.0f}% of bit decisions   "
                f"M8 correct {h8:.0f}%, M16 correct {h16:.0f}%")


def bit_ranges(p: float) -> int:
    c1 = int(p * TOTAL)
    if c1 < 1:
        c1 = 1
    elif c1 > TOTAL - 1:
        c1 = TOTAL - 1
    return c1


def compress(data: bytes, lazy: bool):
    model = ChamberMixerV4(lazy)
    enc = Encoder()
    for byte in data:
        model.begin_byte()
        prefix = 1
        for k in range(8):
            bit = (byte >> (7 - k)) & 1
            p = model.predict(k, prefix)
            c1 = bit_ranges(p)
            if bit:
                enc.encode(0, c1, TOTAL)
            else:
                enc.encode(c1, TOTAL, TOTAL)
            model.learn(k, prefix, bit, p)
            prefix = (prefix << 1) | bit
        model.end_byte(byte)
    return enc.finish(), model


def decompress(blob: bytes, n: int, lazy: bool) -> bytes:
    model = ChamberMixerV4(lazy)
    dec = Decoder(blob)
    out = bytearray()
    for _ in range(n):
        model.begin_byte()
        prefix = 1
        for k in range(8):
            p = model.predict(k, prefix)
            c1 = bit_ranges(p)
            value = dec.decode_value(TOTAL)
            if value < c1:
                bit = 1
                dec.consume(0, c1, TOTAL)
            else:
                bit = 0
                dec.consume(c1, TOTAL, TOTAL)
            model.learn(k, prefix, bit, p)
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


def main():
    full_mode = len(sys.argv) > 1 and sys.argv[1] == "full"
    lazy_full = "eager" not in sys.argv
    header("Chamber mixer v4 - lazy branching + lazy evaluation + O(1) tables")

    if not full_mode:
        deriv = b"".join(f.read_bytes()
                         for f in sorted((ROOT / "derivations").glob("*.md")))
        data = deriv[:65536]
        n = len(data)
        print(f"  benchmark: 64KB (v3 reference: 20563 bytes, 4.8s encode)")

        for lazy in (False, True):
            tag = "lazy ON " if lazy else "lazy OFF"
            t0 = time.time()
            blob, model = compress(data, lazy)
            t_enc = time.time() - t0
            t0 = time.time()
            restored = decompress(blob, n, lazy)
            t_dec = time.time() - t0
            print(f"\n  v4 {tag}: {len(blob)} bytes "
                  f"({len(blob)*8/n:.3f} bits/byte), "
                  f"encode {t_enc:.1f}s, decode {t_dec:.1f}s")
            print(model.lazy_report())
            assertion(restored == data, f"round-trip byte-exact ({tag})")
            assertion(len(blob) < 20563 * 1.03,
                      f"ratio within 3% of v3 ({tag}: "
                      f"{(len(blob)/20563-1)*100:+.1f}%)")
        print(f"\n  run with 'full' for the corpus-scale speed test")
    else:
        data = corpus()
        n = len(data)
        print(f"  full corpus: {n} bytes, lazy={'ON' if lazy_full else 'OFF'}")
        print(f"  v3 reference (Test 21): 157143 bytes, 96.7s enc / 100.8s dec")

        base = {}
        for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                         ("bz2 -9", lambda d: bz2.compress(d, 9)),
                         ("lzma", lzma.compress)]:
            base[name] = len(fn(data))

        t0 = time.time()
        blob, model = compress(data, lazy_full)
        t_enc = time.time() - t0
        ours = len(blob)
        t0 = time.time()
        restored = decompress(blob, n, lazy_full)
        t_dec = time.time() - t0
        assertion(restored == data, "round-trip byte-exact (full corpus)")

        print()
        rows = [*[(k, float(v)) for k, v in base.items()],
                ("chamber mixer v4", float(ours))]
        for name, size in sorted(rows, key=lambda r: r[1], reverse=True):
            marker = "  <-- ours" if name == "chamber mixer v4" else ""
            print(f"  {name:<22} {size:>9.0f} bytes  "
                  f"{size*8/n:>6.3f} bits/byte{marker}")
        print(f"  (encode {t_enc:.1f}s, decode {t_dec:.1f}s)")
        print(model.lazy_report())
        speedup = 96.7 / t_enc
        for name, size in base.items():
            verdict = "BEATS" if ours < size else "loses to"
            print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
                  f"by {abs(1-ours/size)*100:.1f}%")

        header("RESULT")
        print(f"  size:    {ours} bytes = {ours*8/n:.3f} bits/byte "
              f"(sub-1-bit: {'YES' if ours*8/n < 1.0 else 'no'})")
        print(f"  speed:   {t_enc:.1f}s encode = {speedup:.1f}x faster than v3")
        print(f"  per-MB:  {t_enc/(n/1048576):.0f}s/MB encode")
        print()
        print("  Lazy branch fetch (node per byte), lazy evaluation (early")
        print("  exit on saturated votes), quantized transform tables, and")
        print("  packed counts - the lattice's within-byte key invariance is")
        print("  what makes the lazy fetch sound.")


if __name__ == "__main__":
    main()
