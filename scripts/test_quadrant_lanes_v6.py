#!/usr/bin/env python3
"""
Test 24 - Quadrant lanes: the 32-chamber architecture as PARALLEL sub-codecs.

The user's thesis: the formula's 32 sub-quadrants should let this process
millions of tokens in milliseconds. The honest mechanism for that is lane
parallelism: a context-mixing codec is serial WITHIN a stream (every bit's
probability depends on all prior bytes), but K independent quadrant lanes
- each a full 14-chamber mixer with its own tables - run simultaneously.

Critically, DECODE parallelizes identically. Classic CM decoders are
strictly serial; quadrant lanes expand in parallel too.

The price: lanes share no context. Cross-lane matches are lost, models
start cold per lane, so ratio degrades as K grows. This test maps the
whole speed/ratio frontier honestly: K in {1, 2, 4, 8, 16, 32} on the
1.4MB corpus, with SHA-256 verification at every point.

Total table memory is held ~constant (~134MB) by shrinking per-lane
tables as K grows: ctab_bits = 24 - log2(K).
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_chamber_mixer_v5_native import (
    STRETCH, SQUASH, PR, EVAL, header, assertion, corpus,
)

u = np.uint64
MAX_ORDER = 6
N_CH = 14
GOLD = u(0x9E3779B97F4A7C15)
C2 = u(0xC2B2AE3D27D4EB4F)
C3 = u(0xFF51AFD7ED558CCD)
MASK32 = u(0xFFFFFFFF)
MASK16 = u(0xFFFF)
SH32 = u(32)
SH16 = u(16)
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


@njit(cache=True)
def _lane_core(mode, data, blob_in, out_buf, n,
               CTAB, MT8, MT16, W, APM1, APM2, ST, SQ, PRm, EV,
               stats, ctab_shift, mtab_shift):
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
    hl = U0
    nh = 0
    wk = U1
    wlen = 0
    pwk = U1
    lk = U1
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
                    h = (ck ^ (u(i) * C2) ^ (u(prefix) * C3)) * GOLD
                    idx = h >> ctab_shift
                    chk = ((h * C2) >> SH32) & MASK32
                    cell = CTAB[idx]
                    if (cell >> SH32) != chk:
                        continue
                    n0 = np.int64((cell >> SH16) & MASK16)
                    n1 = np.int64(cell & MASK16)
                    if n0 == 0 and n1 == 0:
                        continue
                    pi = (n1 + 0.25) / (n0 + n1 + 0.5)
                    si = int(pi * 4096.0)
                    if si > 4095:
                        si = 4095
                    x = ST[si]
                xs[nact] = x
                acti[nact] = i
                nact += 1
                dot += W[wbase + i] * x
                if dot > EXIT_T or dot < -EXIT_T:
                    stats[1] += 1
                    break
            stats[0] += 1
            qi = int((dot + 25.0) * 163.84)
            if qi < 0:
                qi = 0
            elif qi > 8191:
                qi = 8191
            p_mix = SQ[qi]
            si = int(p_mix * 4096.0)
            if si > 4095:
                si = 4095
            pos = (ST[si] + 10.0) * 1.6
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
            si = int(q1 * 4096.0)
            if si > 4095:
                si = 4095
            pos2 = (ST[si] + 10.0) * 1.6
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
            if mode == 0:
                bit = (byte >> (7 - k)) & 1
                span = high - low + U1
                if bit:
                    high = low + (span * c1) // TOTAL - U1
                else:
                    low = low + (span * c1) // TOTAL
                while True:
                    if high < AC_HALF:
                        bitpos += 1
                        while pending > 0:
                            out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
                            bitpos += 1
                            pending -= 1
                    elif low >= AC_HALF:
                        out_buf[bitpos >> 3] |= np.uint8(1 << (7 - (bitpos & 7)))
                        bitpos += 1
                        while pending > 0:
                            bitpos += 1
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
            for i in range(12):
                ck = keys[i]
                if ck == U0:
                    continue
                h = (ck ^ (u(i) * C2) ^ (u(prefix) * C3)) * GOLD
                idx = h >> ctab_shift
                chk = ((h * C2) >> SH32) & MASK32
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
        byte_out = prefix & 0xFF
        if mode == 1:
            out_buf[t] = np.uint8(byte_out)
            byte = np.int64(byte_out)
        if (48 <= byte <= 57) or (65 <= byte <= 90) or (97 <= byte <= 122) or byte == 95:
            slot = wlen if wlen < 8 else 7
            wk = (wk * GOLD) ^ PRm[(MAX_ORDER + slot) * 256 + byte]
            wlen += 1
        else:
            if wlen > 0:
                pwk = wk
            wk = U1
            wlen = 0
        if byte == 10:
            lk = U1
            llen = 0
        elif llen < 4:
            lk = (lk * GOLD) ^ PRm[(MAX_ORDER + 8 + llen) * 256 + byte]
            llen += 1
        hl = (hl << U8) | u(byte)
        nh += 1
        k16a = (k16a << U8) | (k16b >> u(56))
        k16b = (k16b << U8) | u(byte)
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
            i8 = h8 >> mtab_shift
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
            i16 = h16 >> mtab_shift
            c16 = ((h16 * C2) >> SH32) & MASK32
            ent = MT16[i16]
            if re16 and (ent >> SH32) == c16:
                cand = np.int64(ent & MASK32)
                if 0 <= cand < tp1:
                    ptr16 = cand
                    mlen16 = 16
            MT16[i16] = (c16 << SH32) | u(tp1)

    if mode == 0:
        pending += 1
        if low < AC_QUARTER:
            bitpos += 1
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
        bitpos += 64
        return bitpos
    return 0


@njit(parallel=True, cache=True)
def _encode_lanes(data, offs, lens, CTABp, MT8p, MT16p, Wp, A1p, A2p,
                  outp, outbits, ST, SQ, PRm, EV, statsp,
                  ctab_shift, mtab_shift):
    K = offs.shape[0]
    for L in prange(K):
        seg = data[offs[L]:offs[L] + lens[L]]
        bits = _lane_core(0, seg, outp[L][:1], outp[L], lens[L],
                          CTABp[L], MT8p[L], MT16p[L], Wp[L], A1p[L], A2p[L],
                          ST, SQ, PRm, EV, statsp[L], ctab_shift, mtab_shift)
        outbits[L] = bits


@njit(parallel=True, cache=True)
def _decode_lanes(blobbuf, boffs, blens, out, offs, lens,
                  CTABp, MT8p, MT16p, Wp, A1p, A2p,
                  ST, SQ, PRm, EV, statsp, ctab_shift, mtab_shift):
    K = offs.shape[0]
    for L in prange(K):
        blob = blobbuf[boffs[L]:boffs[L] + blens[L]]
        seg = out[offs[L]:offs[L] + lens[L]]
        _lane_core(1, blob[:1], blob, seg, lens[L],
                   CTABp[L], MT8p[L], MT16p[L], Wp[L], A1p[L], A2p[L],
                   ST, SQ, PRm, EV, statsp[L], ctab_shift, mtab_shift)


def _apm_init(K):
    base = np.array([1.0 / (1.0 + math.exp(-((i % 33) * 20.0 / 32.0 - 10.0)))
                     for i in range(256 * 33)], dtype=np.float64)
    return np.tile(base, (K, 1)).copy()


def _pools(K: int):
    cbits = max(19, 24 - max(0, (K - 1).bit_length()))
    mbits = max(16, 22 - max(0, (K - 1).bit_length()))
    return (np.zeros((K, 1 << cbits), dtype=np.uint64),
            np.zeros((K, 1 << mbits), dtype=np.uint64),
            np.zeros((K, 1 << mbits), dtype=np.uint64),
            np.zeros((K, 8 * 8 * N_CH), dtype=np.float64),
            _apm_init(K), _apm_init(K),
            u(64 - cbits), u(64 - mbits))


def compress_lanes(data: bytes, K: int):
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    lane = n // K
    lens = np.full(K, lane, dtype=np.int64)
    lens[K - 1] = n - lane * (K - 1)
    offs = np.zeros(K, dtype=np.int64)
    for i in range(1, K):
        offs[i] = offs[i - 1] + lens[i - 1]
    maxout = int(lens.max()) * 2 + 4096
    outp = np.zeros((K, maxout), dtype=np.uint8)
    outbits = np.zeros(K, dtype=np.int64)
    statsp = np.zeros((K, 8), dtype=np.int64)
    CTABp, MT8p, MT16p, Wp, A1p, A2p, cs, ms = _pools(K)
    _encode_lanes(arr, offs, lens, CTABp, MT8p, MT16p, Wp, A1p, A2p,
                  outp, outbits, STRETCH, SQUASH, PR, EVAL, statsp, cs, ms)
    parts = [struct.pack("<II", K, n)]
    for L in range(K):
        nbytes = (int(outbits[L]) + 7) >> 3
        parts.append(struct.pack("<II", int(lens[L]), nbytes))
        parts.append(outp[L][:nbytes].tobytes())
    return b"".join(parts)


def decompress_lanes(container: bytes):
    K, n = struct.unpack_from("<II", container, 0)
    pos = 8
    lens = np.zeros(K, dtype=np.int64)
    blens = np.zeros(K, dtype=np.int64)
    blobs = []
    for L in range(K):
        ln, bn = struct.unpack_from("<II", container, pos)
        pos += 8
        lens[L] = ln
        blens[L] = bn
        blobs.append(container[pos:pos + bn])
        pos += bn
    offs = np.zeros(K, dtype=np.int64)
    for i in range(1, K):
        offs[i] = offs[i - 1] + lens[i - 1]
    blobbuf = np.frombuffer(b"".join(blobs), dtype=np.uint8)
    boffs = np.zeros(K, dtype=np.int64)
    for i in range(1, K):
        boffs[i] = boffs[i - 1] + blens[i - 1]
    out = np.zeros(n, dtype=np.uint8)
    statsp = np.zeros((K, 8), dtype=np.int64)
    CTABp, MT8p, MT16p, Wp, A1p, A2p, cs, ms = _pools(K)
    _decode_lanes(blobbuf, boffs, blens, out, offs, lens,
                  CTABp, MT8p, MT16p, Wp, A1p, A2p,
                  STRETCH, SQUASH, PR, EVAL, statsp, cs, ms)
    return out.tobytes()


def main():
    header("Quadrant lanes - parallel sub-codecs across the 32 quadrants")

    data = corpus()
    n = len(data)
    sha_orig = hashlib.sha256(data).hexdigest()
    print(f"  corpus: {n} bytes ({n/1048576:.2f} MB), 8 CPU cores")
    print(f"  SHA-256: {sha_orig[:16]}...")

    print("\n  compiling parallel kernels...")
    t0 = time.time()
    warm = compress_lanes(data[:8192], 2)
    assert decompress_lanes(warm) == data[:8192]
    print(f"  compiled in {time.time()-t0:.1f}s")

    print(f"\n  {'K':>3} | {'size':>8} | {'b/B':>6} | {'enc ms':>7} | "
          f"{'dec ms':>7} | {'enc MB/s':>8} | {'dec MB/s':>8} | sha")
    print(f"  {'-'*3} | {'-'*8} | {'-'*6} | {'-'*7} | {'-'*7} | "
          f"{'-'*8} | {'-'*8} | ---")

    results = []
    for K in [1, 2, 4, 8, 16, 32]:
        t0 = time.time()
        blob = compress_lanes(data, K)
        t_enc = time.time() - t0
        t0 = time.time()
        restored = decompress_lanes(blob)
        t_dec = time.time() - t0
        ok = hashlib.sha256(restored).hexdigest() == sha_orig and restored == data
        results.append((K, len(blob), t_enc, t_dec, ok))
        print(f"  {K:>3} | {len(blob):>8} | {len(blob)*8/n:>6.3f} | "
              f"{t_enc*1000:>7.0f} | {t_dec*1000:>7.0f} | "
              f"{n/1048576/t_enc:>8.1f} | {n/1048576/t_dec:>8.1f} | "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            assertion(False, f"verification failed at K={K}")

    assertion(all(r[4] for r in results),
              "all lane configurations byte-exact + SHA-256 verified")

    # classical references on the same data
    import bz2 as _bz2
    import lzma as _lzma
    t0 = time.time()
    bzs = len(_bz2.compress(data, 9))
    bzt = time.time() - t0
    t0 = time.time()
    xzs = len(_lzma.compress(data))
    xzt = time.time() - t0

    header("FRONTIER")
    print(f"  reference: bz2 -9 {bzs} bytes ({bzs*8/n:.3f} b/B) in {bzt*1000:.0f}ms")
    print(f"             lzma   {xzs} bytes ({xzs*8/n:.3f} b/B) in {xzt*1000:.0f}ms")
    print()
    for K, size, te, td, _ in results:
        beats = []
        if size < bzs:
            beats.append("bz2")
        if size < xzs:
            beats.append("lzma")
        tag = "+".join(beats) if beats else "-"
        print(f"  K={K:>2}: {size*8/n:.3f} b/B, {te*1000:.0f}ms enc / "
              f"{td*1000:.0f}ms dec   beats: {tag}")
    print()
    best_speed = min(results, key=lambda r: r[2])
    K, size, te, td, _ = best_speed
    print(f"  fastest: K={K} -> {n/1048576:.2f} MB in {te*1000:.0f}ms encode, "
          f"{td*1000:.0f}ms decode")
    print(f"  'millions of tokens in ms': {n:,} bytes in {te*1000:.0f}ms = "
          f"{n/te/1e6:.1f}M bytes/sec")
    print()
    print("  The quadrant thesis verified in mechanism: K independent")
    print("  chamber-mixer lanes run (and DECODE) in parallel. The cost is")
    print("  context isolation - the frontier above shows exactly what each")
    print("  level of parallelism pays in ratio.")


if __name__ == "__main__":
    main()
