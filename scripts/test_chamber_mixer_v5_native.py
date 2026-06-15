#!/usr/bin/env python3
"""
Test 23 - Chamber mixer v5: native JIT (numba) - how fast can it go?

Same 14-chamber architecture as v4 (Test 22: 0.826 bits/byte, 57s encode
at 1.4MB), compiled to native code via numba's LLVM JIT:

  - count cells live in ONE flat hash-slot table (2^24 slots x 8 bytes):
    [check:32 | n0:16 | n1:16], indexed by mixing (chamber, context key,
    bit prefix). Collisions overwrite (PAQ-style); a 32-bit check tag
    keeps false hits at ~2^-32.
  - FTA composite keys fold into a 64-bit field via wrapping multiply-xor
    with the SAME position-tagged primes (the lattice addressing, carried
    into a finite field for native speed).
  - match models keep exact 8/16-byte rolling windows (no hashing of the
    window itself) with checked hash-slot position tables.
  - the arithmetic coder runs inside the JIT too - the whole per-bit loop
    never touches the Python interpreter.
  - lazy evaluation (early exit at |vote| > 14) retained from v4.

Accuracy gate: SHA-256(decompressed) == SHA-256(original), byte-exact,
plus determinism (two encodes -> identical blobs).
"""

from __future__ import annotations

import bz2
import hashlib
import lzma
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes

ROOT = Path(__file__).resolve().parents[1]

u = np.uint64
MAX_ORDER = 6
N_CH = 14
CTAB_BITS = 24
MTAB_BITS = 22
GOLD = u(0x9E3779B97F4A7C15)
C2 = u(0xC2B2AE3D27D4EB4F)
C3 = u(0xFF51AFD7ED558CCD)
MASK32 = u(0xFFFFFFFF)
MASK16 = u(0xFFFF)
SH32 = u(32)
SH16 = u(16)
CTAB_SHIFT = u(64 - CTAB_BITS)
MTAB_SHIFT = u(64 - MTAB_BITS)
EXIT_T = 14.0
LR = 0.003
APM_RATE = 0.02
COUNT_CAP = 1024
TOTAL = u(4096)
AC_HALF = u(0x80000000)
AC_QUARTER = u(0x40000000)
AC_3Q = u(0xC0000000)
AC_MASK = u(0xFFFFFFFF)
U0, U1, U8 = u(0), u(1), u(8)

_PR = chain_primes(256 * 18 + 64)
PR = np.array(_PR, dtype=np.uint64)

STRETCH = np.empty(4096, dtype=np.float64)
for _i in range(4096):
    _pc = (_i + 0.5) / 4096.0
    STRETCH[_i] = max(-10.0, min(10.0, math.log(_pc / (1.0 - _pc))))
SQUASH = np.empty(8192, dtype=np.float64)
for _i in range(8192):
    _x = (_i + 0.5) / 8192.0 * 50.0 - 25.0
    SQUASH[_i] = 1.0 / (1.0 + math.exp(-_x))

EVAL = np.array([13, 12, 6, 5, 4, 3, 2, 1, 7, 8, 9, 10, 11, 0],
                dtype=np.int64)


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


@njit(inline="always")
def _cell_slot(key, ch, prefix):
    h = (key ^ (u(ch) * C2) ^ (u(prefix) * C3)) * GOLD
    idx = h >> CTAB_SHIFT
    chk = (h * C2) >> SH32
    return idx, chk & MASK32


@njit(inline="always")
def _squash_q(dot, SQ):
    i = int((dot + 25.0) * 163.84)
    if i < 0:
        i = 0
    elif i > 8191:
        i = 8191
    return SQ[i]


@njit(inline="always")
def _stretch_q(p, ST):
    i = int(p * 4096.0)
    if i < 0:
        i = 0
    elif i > 4095:
        i = 4095
    return ST[i]


@njit(cache=True)
def _code_core(mode, data, blob_in, out_buf, n,
               CTAB, MT8, MT16, W, APM1, APM2, ST, SQ, PRm, EV, stats):
    """mode 0 = encode data->out_buf(bits); mode 1 = decode blob_in->out_buf.
    Returns compressed bit count (encode) or 0 (decode)."""
    # ---- arithmetic coder state ----
    low = U0
    high = AC_MASK
    pending = 0
    bitpos = 0
    code = U0
    rpos = 0
    nblob = blob_in.shape[0]
    if mode == 1:
        for _ in range(32):
            b = U0
            if (rpos >> 3) < nblob:
                b = u((blob_in[rpos >> 3] >> (7 - (rpos & 7))) & 1)
            rpos += 1
            code = ((code << U1) | b) & AC_MASK
    # ---- model state ----
    hl = U0                  # last 8 bytes packed (low byte = most recent)
    nh = 0
    wk = U1                  # word key
    wlen = 0
    pwk = U1                 # prev word key
    lk = U1                  # line key
    llen = 0
    k16a = U0
    k16b = U0
    ptr8 = np.int64(-1)
    mlen8 = 0
    pred8 = np.int64(-1)
    ptr16 = np.int64(-1)
    mlen16 = 0
    pred16 = np.int64(-1)
    keys = np.zeros(14, dtype=np.uint64)
    xs = np.empty(14, dtype=np.float64)
    acti = np.empty(14, dtype=np.int64)

    for t in range(n):
        # ---------------- begin byte: compute chamber keys ----------------
        keys[0] = U1
        kk = U1
        for j in range(MAX_ORDER):
            if j < nh:
                bj = np.int64((hl >> u(8 * j)) & u(0xFF))
                kk = (kk * GOLD) ^ PRm[j * 256 + bj]
                keys[j + 1] = kk
            else:
                keys[j + 1] = U0
        keys[7] = wk
        keys[8] = (pwk * GOLD) ^ wk
        if nh >= 3:
            b0 = np.int64(hl & u(0xFF))
            b2 = np.int64((hl >> u(16)) & u(0xFF))
            keys[9] = (PRm[b0] * GOLD) ^ PRm[512 + b2]
        else:
            keys[9] = U0
        if nh >= 4:
            b1 = np.int64((hl >> u(8)) & u(0xFF))
            b3 = np.int64((hl >> u(24)) & u(0xFF))
            keys[10] = (PRm[256 + b1] * GOLD) ^ PRm[768 + b3]
        else:
            keys[10] = U0
        keys[11] = lk
        # match predictions (history = data when encoding, out when decoding)
        if mode == 0:
            hist = data
        else:
            hist = out_buf
        pred8 = np.int64(-1)
        if 0 <= ptr8 < t:
            pred8 = np.int64(hist[ptr8])
            stats[2] += 1
        pred16 = np.int64(-1)
        if 0 <= ptr16 < t:
            pred16 = np.int64(hist[ptr16])
            stats[4] += 1
        # selector
        m_active = 1 if (pred8 >= 0 or pred16 >= 0) else 0
        w_state = 0
        if nh > 0:
            lb = np.int64(hl & u(0xFF))
            if (48 <= lb <= 57) or (65 <= lb <= 90) or (97 <= lb <= 122) or lb == 95:
                w_state = 1
        l_state = 1 if llen == 0 else 0
        sel = m_active * 4 + w_state * 2 + l_state
        ctx1 = 0
        if nh > 0:
            ctx1 = np.int64(hl & u(0xFF))
        ctx2 = sel * 32 + (ctx1 >> 3)

        byte = np.int64(0)
        if mode == 0:
            byte = np.int64(data[t])
        prefix = 1
        for k in range(8):
            # ---------------- predict ----------------
            wbase = (sel * 8 + k) * N_CH
            dot = 0.0
            nact = 0
            for e in range(14):
                i = EV[e]
                if i >= 12:
                    if i == 13:
                        pb = pred16
                        ml = mlen16
                        bc = 2.0
                        sl = 0.6
                    else:
                        pb = pred8
                        ml = mlen8
                        bc = 1.0
                        sl = 0.5
                    if pb < 0:
                        continue
                    # pb <= 255 so pb >> 8 == 0; no ternary needed at k == 0
                    pp = (1 << k) | (pb >> (8 - k))
                    if prefix != pp:
                        continue
                    conf = bc + sl * ml
                    if conf > 10.0:
                        conf = 10.0
                    if (pb >> (7 - k)) & 1:
                        x = conf
                    else:
                        x = -conf
                else:
                    ck = keys[i]
                    if ck == U0:
                        continue
                    idx, chk = _cell_slot(ck, i, prefix)
                    cell = CTAB[idx]
                    if (cell >> SH32) != chk:
                        continue
                    n0 = np.int64((cell >> SH16) & MASK16)
                    n1 = np.int64(cell & MASK16)
                    if n0 == 0 and n1 == 0:
                        continue
                    x = _stretch_q((n1 + 0.25) / (n0 + n1 + 0.5), ST)
                xs[nact] = x
                acti[nact] = i
                nact += 1
                dot += W[wbase + i] * x
                if dot > EXIT_T or dot < -EXIT_T:
                    stats[1] += 1
                    break
            stats[0] += 1
            p_mix = _squash_q(dot, SQ)
            # APM chain
            pos = (_stretch_q(p_mix, ST) + 10.0) * 1.6
            ib1 = int(pos)
            if ib1 > 31:
                ib1 = 31
            f1 = pos - ib1
            a1 = APM1[ctx1 * 33 + ib1] * (1.0 - f1) + APM1[ctx1 * 33 + ib1 + 1] * f1
            q1 = 0.25 * p_mix + 0.75 * a1
            if q1 < 0.0002:
                q1 = 0.0002
            elif q1 > 0.9998:
                q1 = 0.9998
            pos2 = (_stretch_q(q1, ST) + 10.0) * 1.6
            ib2 = int(pos2)
            if ib2 > 31:
                ib2 = 31
            f2 = pos2 - ib2
            a2 = APM2[ctx2 * 33 + ib2] * (1.0 - f2) + APM2[ctx2 * 33 + ib2 + 1] * f2
            p = 0.25 * q1 + 0.75 * a2
            if p < 0.0002:
                p = 0.0002
            elif p > 0.9998:
                p = 0.9998
            c1i = int(p * 4096.0)
            if c1i < 1:
                c1i = 1
            elif c1i > 4095:
                c1i = 4095
            c1 = u(c1i)
            # ---------------- code the bit ----------------
            if mode == 0:
                bit = (byte >> (7 - k)) & 1
                span = high - low + U1
                if bit:
                    high = low + (span * c1) // TOTAL - U1
                else:
                    low = low + (span * c1) // TOTAL
                while True:
                    if high < AC_HALF:
                        bitpos += 1  # bit 0: buffer pre-zeroed
                        while pending > 0:
                            out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
                            bitpos += 1
                            pending -= 1
                    elif low >= AC_HALF:
                        out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
                        bitpos += 1
                        while pending > 0:
                            bitpos += 1  # zero bits: buffer pre-zeroed
                            pending -= 1
                        low -= AC_HALF
                        high -= AC_HALF
                    elif low >= AC_QUARTER and high < AC_3Q:
                        pending += 1
                        low -= AC_QUARTER
                        high -= AC_QUARTER
                    else:
                        break
                    low = (low << U1) & AC_MASK
                    high = ((high << U1) | U1) & AC_MASK
            else:
                span = high - low + U1
                value = ((code - low + U1) * TOTAL - U1) // span
                if value < c1:
                    bit = 1
                    high = low + (span * c1) // TOTAL - U1
                else:
                    bit = 0
                    low = low + (span * c1) // TOTAL
                while True:
                    if high < AC_HALF:
                        pass
                    elif low >= AC_HALF:
                        low -= AC_HALF
                        high -= AC_HALF
                        code -= AC_HALF
                    elif low >= AC_QUARTER and high < AC_3Q:
                        low -= AC_QUARTER
                        high -= AC_QUARTER
                        code -= AC_QUARTER
                    else:
                        break
                    low = (low << U1) & AC_MASK
                    high = ((high << U1) | U1) & AC_MASK
                    nb = U0
                    if (rpos >> 3) < nblob:
                        nb = u((blob_in[rpos >> 3] >> (7 - (rpos & 7))) & 1)
                    rpos += 1
                    code = ((code << U1) | nb) & AC_MASK
            # ---------------- learn ----------------
            err = (bit - p) * LR
            for a in range(nact):
                W[wbase + acti[a]] += xs[a] * err
            e1 = (bit - a1) * APM_RATE
            v = APM1[ctx1 * 33 + ib1] + e1 * (1.0 - f1)
            if v < 1e-4:
                v = 1e-4
            elif v > 1.0 - 1e-4:
                v = 1.0 - 1e-4
            APM1[ctx1 * 33 + ib1] = v
            v = APM1[ctx1 * 33 + ib1 + 1] + e1 * f1
            if v < 1e-4:
                v = 1e-4
            elif v > 1.0 - 1e-4:
                v = 1.0 - 1e-4
            APM1[ctx1 * 33 + ib1 + 1] = v
            e2 = (bit - a2) * APM_RATE
            v = APM2[ctx2 * 33 + ib2] + e2 * (1.0 - f2)
            if v < 1e-4:
                v = 1e-4
            elif v > 1.0 - 1e-4:
                v = 1.0 - 1e-4
            APM2[ctx2 * 33 + ib2] = v
            v = APM2[ctx2 * 33 + ib2 + 1] + e2 * f2
            if v < 1e-4:
                v = 1e-4
            elif v > 1.0 - 1e-4:
                v = 1.0 - 1e-4
            APM2[ctx2 * 33 + ib2 + 1] = v
            # count updates: ALL valid-key chambers
            for i in range(12):
                ck = keys[i]
                if ck == U0:
                    continue
                idx, chk = _cell_slot(ck, i, prefix)
                cell = CTAB[idx]
                if (cell >> SH32) == chk:
                    n0 = np.int64((cell >> SH16) & MASK16)
                    n1 = np.int64(cell & MASK16)
                else:
                    n0 = np.int64(0)
                    n1 = np.int64(0)
                if bit:
                    n1 += 1
                else:
                    n0 += 1
                if n0 + n1 > COUNT_CAP:
                    n0 -= n0 >> 1
                    n1 -= n1 >> 1
                CTAB[idx] = (chk << SH32) | (u(n0) << SH16) | u(n1)
            prefix = (prefix << 1) | bit
        # ---------------- end byte ----------------
        byte_out = prefix & 0xFF
        if mode == 1:
            out_buf[t] = np.uint8(byte_out)
            byte = byte_out
        # word state
        if (48 <= byte <= 57) or (65 <= byte <= 90) or (97 <= byte <= 122) or byte == 95:
            slot = wlen if wlen < 8 else 7
            wk = (wk * GOLD) ^ PRm[(MAX_ORDER + slot) * 256 + byte]
            wlen += 1
        else:
            if wlen > 0:
                pwk = wk
            wk = U1
            wlen = 0
        # line state
        if byte == 10:
            lk = U1
            llen = 0
        elif llen < 4:
            lk = (lk * GOLD) ^ PRm[(MAX_ORDER + 8 + llen) * 256 + byte]
            llen += 1
        # rolling windows
        hl = (hl << U8) | u(byte)
        nh += 1
        k16a = (k16a << U8) | (k16b >> u(56))
        k16b = (k16b << U8) | u(byte)
        # match advance / relookup
        if mode == 0:
            hist = data
        else:
            hist = out_buf
        if pred8 == byte and ptr8 >= 0:
            ptr8 += 1
            mlen8 += 1
            stats[3] += 1
            re8 = False
        else:
            mlen8 = 0
            ptr8 = np.int64(-1)
            re8 = True
        if pred16 == byte and ptr16 >= 0:
            ptr16 += 1
            mlen16 += 1
            stats[5] += 1
            re16 = False
        else:
            mlen16 = 0
            ptr16 = np.int64(-1)
            re16 = True
        tp1 = t + 1
        if tp1 >= 8:
            h8 = hl * GOLD
            i8 = h8 >> MTAB_SHIFT
            c8 = ((h8 * C2) >> SH32) & MASK32
            ent = MT8[i8]
            if re8 and (ent >> SH32) == c8:
                cand = np.int64(ent & MASK32)
                if 0 <= cand < tp1:
                    ptr8 = cand
                    mlen8 = 8
            MT8[i8] = (c8 << SH32) | u(tp1)
        if tp1 >= 16:
            h16 = (k16a * GOLD) ^ (k16b * C3)
            h16 = h16 * GOLD
            i16 = h16 >> MTAB_SHIFT
            c16 = ((h16 * C2) >> SH32) & MASK32
            ent = MT16[i16]
            if re16 and (ent >> SH32) == c16:
                cand = np.int64(ent & MASK32)
                if 0 <= cand < tp1:
                    ptr16 = cand
                    mlen16 = 16
            MT16[i16] = (c16 << SH32) | u(tp1)

    if mode == 0:
        # finish arithmetic coder
        pending += 1
        if low < AC_QUARTER:
            bitpos += 1  # final 0
            while pending > 0:
                out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
                bitpos += 1
                pending -= 1
        else:
            out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
            bitpos += 1
            while pending > 0:
                bitpos += 1
                pending -= 1
        bitpos += 64  # zero tail padding
        return bitpos
    return 0


def _fresh_state():
    return (np.zeros(1 << CTAB_BITS, dtype=np.uint64),
            np.zeros(1 << MTAB_BITS, dtype=np.uint64),
            np.zeros(1 << MTAB_BITS, dtype=np.uint64),
            np.zeros(8 * 8 * N_CH, dtype=np.float64),
            np.array([1.0 / (1.0 + math.exp(-((i % 33) * 20.0 / 32.0 - 10.0)))
                      for i in range(256 * 33)], dtype=np.float64),
            np.array([1.0 / (1.0 + math.exp(-((i % 33) * 20.0 / 32.0 - 10.0)))
                      for i in range(256 * 33)], dtype=np.float64))


def compress(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    out = np.zeros(len(data) * 2 + 4096, dtype=np.uint8)
    CTAB, MT8, MT16, W, APM1, APM2 = _fresh_state()
    stats = np.zeros(8, dtype=np.int64)
    bits = _code_core(0, arr, np.zeros(1, dtype=np.uint8), out, len(data),
                      CTAB, MT8, MT16, W, APM1, APM2,
                      STRETCH, SQUASH, PR, EVAL, stats)
    nbytes = (bits + 7) >> 3
    return bytes(out[:nbytes]), stats


def decompress(blob: bytes, n: int):
    barr = np.frombuffer(blob, dtype=np.uint8)
    out = np.zeros(n, dtype=np.uint8)
    CTAB, MT8, MT16, W, APM1, APM2 = _fresh_state()
    stats = np.zeros(8, dtype=np.int64)
    _code_core(1, np.zeros(1, dtype=np.uint8), barr, out, n,
               CTAB, MT8, MT16, W, APM1, APM2,
               STRETCH, SQUASH, PR, EVAL, stats)
    return out.tobytes(), stats


def corpus() -> bytes:
    files = sorted(ROOT.glob("derivations/*.md")) + \
            sorted(ROOT.glob("book/**/*.md")) + \
            sorted(ROOT.glob("*.md"))
    return b"".join(f.read_bytes() for f in files)


def main():
    header("Chamber mixer v5 - native JIT: how fast, and how accurate")

    # JIT warm-up (compile)
    print("  compiling JIT kernels...")
    t0 = time.time()
    warm = b"warmup data for the JIT compiler " * 64
    blob, _ = compress(warm)
    restored, _ = decompress(blob, len(warm))
    assert restored == warm
    print(f"  compiled + warm-up round-trip OK in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # 64KB benchmark (same data as Tests 15-22)
    # ------------------------------------------------------------------
    deriv = b"".join(f.read_bytes()
                     for f in sorted((ROOT / "derivations").glob("*.md")))
    data = deriv[:65536]
    n = len(data)
    t0 = time.time()
    blob, st = compress(data)
    t_enc = time.time() - t0
    t0 = time.time()
    restored, _ = decompress(blob, n)
    t_dec = time.time() - t0
    print(f"\n  64KB bench: {len(blob)} bytes ({len(blob)*8/n:.3f} bits/byte)")
    print(f"  encode {t_enc*1000:.0f}ms, decode {t_dec*1000:.0f}ms "
          f"(v4: 3300ms/3600ms)")
    assertion(restored == data, "round-trip byte-exact at 64KB")
    print(f"  [info]  v4 reference 20561 bytes; v5 hash-slot delta: "
          f"{(len(blob)/20561-1)*100:+.1f}%")

    # ------------------------------------------------------------------
    # FULL CORPUS - speed + accuracy gauntlet
    # ------------------------------------------------------------------
    data = corpus()
    n = len(data)
    sha_orig = hashlib.sha256(data).hexdigest()
    print(f"\n  full corpus: {n} bytes, SHA-256 {sha_orig[:16]}...")

    base = {}
    for name, fn in [("zlib -9", lambda d: zlib.compress(d, 9)),
                     ("bz2 -9", lambda d: bz2.compress(d, 9)),
                     ("lzma", lzma.compress)]:
        t0 = time.time()
        out = fn(data)
        base[name] = (len(out), time.time() - t0)

    t0 = time.time()
    blob, st = compress(data)
    t_enc = time.time() - t0
    t0 = time.time()
    blob2, _ = compress(data)
    t_enc2 = time.time() - t0
    assertion(blob == blob2, "determinism: two encodes -> identical blobs")

    t0 = time.time()
    restored, _ = decompress(blob, n)
    t_dec = time.time() - t0
    sha_back = hashlib.sha256(restored).hexdigest()
    assertion(restored == data, "round-trip byte-exact at full corpus")
    assertion(sha_back == sha_orig,
              f"SHA-256 verified: {sha_back[:16]}... == original")

    ours = len(blob)
    print()
    rows = [*[(k, float(v[0]), v[1]) for k, v in base.items()],
            ("chamber mixer v5", float(ours), t_enc)]
    for name, size, tt in sorted(rows, key=lambda r: r[1], reverse=True):
        marker = "  <-- ours" if name == "chamber mixer v5" else ""
        print(f"  {name:<20} {size:>9.0f} bytes  {size*8/n:>6.3f} bits/byte"
              f"  {tt:>6.2f}s{marker}")
    print(f"\n  v5 encode: {t_enc:.2f}s ({n/1048576/t_enc:.1f} MB/s)   "
          f"decode: {t_dec:.2f}s ({n/1048576/t_dec:.1f} MB/s)")
    print(f"  vs v4 (57.0s/60.2s): {57.0/t_enc:.0f}x / {60.2/t_dec:.0f}x faster")
    print(f"  vs v3 (96.7s/100.8s): {96.7/t_enc:.0f}x / {100.8/t_dec:.0f}x faster")
    er = st[1] / st[0] * 100 if st[0] else 0
    print(f"  early-exit rate: {er:.0f}%   "
          f"M8 correct {st[3]/max(st[2],1)*100:.0f}%   "
          f"M16 correct {st[5]/max(st[4],1)*100:.0f}%")
    for name, (size, tt) in base.items():
        verdict = "BEATS" if ours < size else "loses to"
        print(f"  [{'PASS' if ours < size else 'info'}]  {verdict} {name} "
              f"by {abs(1-ours/size)*100:.1f}% on size")

    header("RESULT")
    print(f"  ratio:    {ours*8/n:.3f} bits/byte "
          f"(sub-1-bit: {'YES' if ours*8/n < 1.0 else 'no'})")
    print(f"  speed:    {n/1048576/t_enc:.1f} MB/s compress, "
          f"{n/1048576/t_dec:.1f} MB/s expand")
    print(f"  accuracy: byte-exact, SHA-256 verified, deterministic")
    print()
    print("  The full chamber pipeline - 14 chambers, learned mixing, APM,")
    print("  match models, arithmetic coding - now runs as native code.")
    print("  Same architecture, same lattice addressing (folded to a 64-bit")
    print("  field), hash-slot cell tables with 32-bit check tags.")


if __name__ == "__main__":
    main()
