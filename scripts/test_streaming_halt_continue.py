#!/usr/bin/env python3
"""
Test 27 - Halt and continue, flawlessly: byte-granular suspend/resume.

The claim, made precise: the engine can HALT at any byte boundary,
serialize its complete state to disk, lose all in-memory state ("die"),
reload, and CONTINUE - producing output BIT-IDENTICAL to a run that was
never interrupted. Not "compatible output": the same compressed bits.

This is the strongest form of Test 26's checkpoint (which resumed at
rotation boundaries). Here the core is refactored into a true streaming
machine: every scalar it carries lives in a savable state vector -

  su (uint64): coder low/high, rolling byte window, word key, prev-word
               key, line key, 16-byte match window halves, decoder code
  si (int64):  pending bits, bit position, bytes seen, word/line lengths,
               match pointers + lengths, decoder bit position, started flag

plus the persistent arrays (cell table, mixer weights, APMs, match
tables, output buffer). Suspend = np.save everything. Resume = reload
and call again with the next byte range.

Tests:
  (A) encode in 23 slices with FULL disk round-trip + in-memory teardown
      between every slice -> blob bit-identical to single-shot
  (B) halt at 7 RANDOM byte offsets (including pathological tiny slices)
      -> still bit-identical
  (C) decode with halts at random offsets -> SHA-256 == original
  (D) the price of perfect memory: measured state size + save/load time
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_chamber_mixer_v5_native import (
    STRETCH, SQUASH, PR, EVAL, header, assertion, corpus,
)

u = np.uint64
N_CH = 14
MAX_ORDER = 6
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

CTAB_BITS = 20            # streaming profile: 8MB cells -> fast suspend
MTAB_BITS = 18
CS = u(64 - CTAB_BITS)
MS = u(64 - MTAB_BITS)

# su slots
SU_LOW, SU_HIGH, SU_HL, SU_WK, SU_PWK, SU_LK, SU_K16A, SU_K16B, SU_CODE = range(9)
# si slots
(SI_PEND, SI_BITPOS, SI_NH, SI_WLEN, SI_LLEN, SI_PTR8, SI_MLEN8,
 SI_PTR16, SI_MLEN16, SI_RPOS, SI_STARTED) = range(11)


@njit(cache=True)
def _stream_core(mode, data, blob_in, out_buf, start, count, finalize,
                 su, si, CTAB, MT8, MT16, W, APM1, APM2, ST, SQ, PRm, EV):
    low = su[SU_LOW]
    high = su[SU_HIGH]
    hl = su[SU_HL]
    wk = su[SU_WK]
    pwk = su[SU_PWK]
    lk = su[SU_LK]
    k16a = su[SU_K16A]
    k16b = su[SU_K16B]
    code = su[SU_CODE]
    pending = si[SI_PEND]
    bitpos = si[SI_BITPOS]
    nh = si[SI_NH]
    wlen = si[SI_WLEN]
    llen = si[SI_LLEN]
    ptr8 = si[SI_PTR8]
    mlen8 = si[SI_MLEN8]
    ptr16 = si[SI_PTR16]
    mlen16 = si[SI_MLEN16]
    rpos = si[SI_RPOS]
    nblob = blob_in.shape[0]
    if mode == 1 and si[SI_STARTED] == 0:
        for _ in range(32):
            b = U0
            if (rpos >> 3) < nblob:
                b = u((blob_in[rpos >> 3] >> (7 - (rpos & 7))) & 1)
            rpos += 1
            code = ((code << U1) | b) & AC_MASK
        si[SI_STARTED] = 1

    keys = np.zeros(14, dtype=np.uint64)
    xs = np.empty(14, dtype=np.float64)
    acti = np.empty(14, dtype=np.int64)

    for t in range(start, start + count):
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
        pred16 = np.int64(-1)
        if 0 <= ptr16 < t:
            pred16 = np.int64(hist[ptr16])
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
                    idx = h >> CS
                    chk = ((h * C2) >> SH32) & MASK32
                    cell = CTAB[idx]
                    if (cell >> SH32) != chk:
                        continue
                    n0 = np.int64((cell >> SH16) & MASK16)
                    n1 = np.int64(cell & MASK16)
                    if n0 == 0 and n1 == 0:
                        continue
                    pi = (n1 + 0.25) / (n0 + n1 + 0.5)
                    sidx = int(pi * 4096.0)
                    if sidx > 4095:
                        sidx = 4095
                    x = ST[sidx]
                xs[nact] = x
                acti[nact] = i
                nact += 1
                dot += W[wbase + i] * x
                if dot > EXIT_T or dot < -EXIT_T:
                    break
            qi = int((dot + 25.0) * 163.84)
            if qi < 0:
                qi = 0
            elif qi > 8191:
                qi = 8191
            p_mix = SQ[qi]
            sidx = int(p_mix * 4096.0)
            if sidx > 4095:
                sidx = 4095
            pos = (ST[sidx] + 10.0) * 1.6
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
            sidx = int(q1 * 4096.0)
            if sidx > 4095:
                sidx = 4095
            pos2 = (ST[sidx] + 10.0) * 1.6
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
                idx = h >> CS
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
        if pred8 == byte and ptr8 >= 0:
            ptr8 += 1
            mlen8 += 1
            re8 = False
        else:
            mlen8 = 0
            ptr8 = np.int64(-1)
            re8 = True
        if pred16 == byte and ptr16 >= 0:
            ptr16 += 1
            mlen16 += 1
            re16 = False
        else:
            mlen16 = 0
            ptr16 = np.int64(-1)
            re16 = True
        tp1 = t + 1
        if tp1 >= 8:
            h8 = hl * GOLD
            i8 = h8 >> MS
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
            i16 = h16 >> MS
            c16 = ((h16 * C2) >> SH32) & MASK32
            ent = MT16[i16]
            if re16 and (ent >> SH32) == c16:
                cand = np.int64(ent & MASK32)
                if 0 <= cand < tp1:
                    ptr16 = cand
                    mlen16 = 16
            MT16[i16] = (c16 << SH32) | u(tp1)

    if mode == 0 and finalize == 1:
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

    su[SU_LOW] = low
    su[SU_HIGH] = high
    su[SU_HL] = hl
    su[SU_WK] = wk
    su[SU_PWK] = pwk
    su[SU_LK] = lk
    su[SU_K16A] = k16a
    su[SU_K16B] = k16b
    su[SU_CODE] = code
    si[SI_PEND] = pending
    si[SI_BITPOS] = bitpos
    si[SI_NH] = nh
    si[SI_WLEN] = wlen
    si[SI_LLEN] = llen
    si[SI_PTR8] = ptr8
    si[SI_MLEN8] = mlen8
    si[SI_PTR16] = ptr16
    si[SI_MLEN16] = mlen16
    si[SI_RPOS] = rpos
    return bitpos


# ----------------------------------------------------------------------
# Python-side streaming state: create / save / destroy / load
# ----------------------------------------------------------------------

ARRAYS = ["CTAB", "MT8", "MT16", "W", "APM1", "APM2", "su", "si", "out"]


def new_state(n_out: int) -> dict:
    import math as _m
    apm = np.array([1.0 / (1.0 + _m.exp(-((i % 33) * 20.0 / 32.0 - 10.0)))
                    for i in range(256 * 33)], dtype=np.float64)
    su = np.zeros(16, dtype=np.uint64)
    su[SU_HIGH] = AC_MASK
    su[SU_WK] = U1
    su[SU_PWK] = U1
    su[SU_LK] = U1
    si = np.zeros(16, dtype=np.int64)
    si[SI_PTR8] = -1
    si[SI_PTR16] = -1
    return {
        "CTAB": np.zeros(1 << CTAB_BITS, dtype=np.uint64),
        "MT8": np.zeros(1 << MTAB_BITS, dtype=np.uint64),
        "MT16": np.zeros(1 << MTAB_BITS, dtype=np.uint64),
        "W": np.zeros(8 * 8 * N_CH, dtype=np.float64),
        "APM1": apm.copy(),
        "APM2": apm.copy(),
        "su": su,
        "si": si,
        "out": np.zeros(n_out, dtype=np.uint8),
    }


def save_state(state: dict, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    for k in ARRAYS:
        np.save(d / f"{k}.npy", state[k])


def load_state(d: Path) -> dict:
    return {k: np.load(d / f"{k}.npy") for k in ARRAYS}


def run_segment(state, mode, data_arr, blob_arr, start, count, finalize):
    return _stream_core(mode, data_arr, blob_arr, state["out"],
                        start, count, finalize,
                        state["su"], state["si"],
                        state["CTAB"], state["MT8"], state["MT16"],
                        state["W"], state["APM1"], state["APM2"],
                        STRETCH, SQUASH, PR, EVAL)


def main():
    header("Halt and continue, flawlessly - byte-granular suspend/resume")

    data = corpus()
    n = len(data)
    arr = np.frombuffer(data, dtype=np.uint8)
    dummy = np.zeros(1, dtype=np.uint8)
    sha_orig = hashlib.sha256(data).hexdigest()
    tmp = Path("logs") / "halt_continue_state"
    print(f"  corpus: {n} bytes, SHA-256 {sha_orig[:16]}...")

    # ------------------------------------------------------------------
    print("\n  compiling streaming kernel + single-shot reference...")
    t0 = time.time()
    ref = new_state(n * 2 + 4096)
    bits = run_segment(ref, 0, arr, dummy, 0, n, 1)
    ref_blob = ref["out"][: (bits + 7) >> 3].tobytes()
    print(f"  single-shot: {len(ref_blob)} bytes "
          f"({len(ref_blob)*8/n:.3f} b/B) in {time.time()-t0:.1f}s "
          f"(incl. compile)")

    # ------------------------------------------------------------------
    print("\nPart A - 23 slices, full disk round-trip + teardown between each")
    print("-" * 72)
    slice_size = 65536
    state = new_state(n * 2 + 4096)
    pos = 0
    n_halts = 0
    t_save = 0.0
    t0 = time.time()
    while pos < n:
        count = min(slice_size, n - pos)
        finalize = 1 if pos + count >= n else 0
        bits = run_segment(state, 0, arr, dummy, pos, count, finalize)
        pos += count
        if not finalize:
            ts = time.time()
            save_state(state, tmp)          # HALT: state -> disk
            del state                        # death: in-memory state gone
            state = load_state(tmp)          # CONTINUE: reload from disk
            t_save += time.time() - ts
            n_halts += 1
    blob_a = state["out"][: (bits + 7) >> 3].tobytes()
    t_total = time.time() - t0
    print(f"  halts survived:   {n_halts} (every 64KB)")
    print(f"  total time:       {t_total:.1f}s ({t_save:.1f}s of it save/load)")
    assertion(blob_a == ref_blob,
              f"23-slice blob is BIT-IDENTICAL to single-shot "
              f"({len(blob_a)} bytes, every bit equal)")

    # ------------------------------------------------------------------
    print("\nPart B - Halt at 7 RANDOM byte offsets (adversarial slicing)")
    print("-" * 72)
    rng = random.Random(0xFA7E)
    cuts = sorted(rng.sample(range(1, n - 1), 7))
    print(f"  cut points: {cuts}")
    state = new_state(n * 2 + 4096)
    prev = 0
    for ci, cut in enumerate(cuts + [n]):
        finalize = 1 if cut == n else 0
        bits = run_segment(state, 0, arr, dummy, prev, cut - prev, finalize)
        prev = cut
        if not finalize:
            save_state(state, tmp)
            del state
            state = load_state(tmp)
    blob_b = state["out"][: (bits + 7) >> 3].tobytes()
    assertion(blob_b == ref_blob,
              "random-cut blob BIT-IDENTICAL (halt anywhere, continue exactly)")

    # ------------------------------------------------------------------
    print("\nPart C - Decode with halts at random offsets")
    print("-" * 72)
    blob_arr = np.frombuffer(ref_blob, dtype=np.uint8)
    cuts_d = sorted(rng.sample(range(1, n - 1), 5))
    state = new_state(n)
    prev = 0
    for cut in cuts_d + [n]:
        run_segment(state, 1, dummy, blob_arr, prev, cut - prev, 0)
        prev = cut
        if cut != n:
            save_state(state, tmp)
            del state
            state = load_state(tmp)
    restored = state["out"][:n].tobytes()
    assertion(hashlib.sha256(restored).hexdigest() == sha_orig
              and restored == data,
              "decode halted 5x mid-stream, resumed from disk, SHA-256 exact")

    # ------------------------------------------------------------------
    print("\nPart D - The price of perfect memory")
    print("-" * 72)
    sz = sum((tmp / f"{k}.npy").stat().st_size for k in ARRAYS)
    print(f"  suspended state on disk: {sz/1048576:.1f} MB "
          f"(cells {8*(1<<CTAB_BITS)/1048576:.0f}MB + matches + mixer + APM + buffers)")
    print(f"  save+load per halt:      ~{t_save/max(n_halts,1)*1000:.0f}ms")
    shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------
    header("RESULT")
    print(f"  single-shot blob:  {len(ref_blob)} bytes")
    print(f"  23-slice blob:     bit-identical after {n_halts} disk deaths")
    print(f"  random-cut blob:   bit-identical after 7 adversarial halts")
    print(f"  decode w/ halts:   SHA-256 exact after 5 mid-stream deaths")
    print()
    print("  'Halt and continue flawlessly' - now proven at BYTE granularity:")
    print("  suspend anywhere, persist to disk, lose all memory, reload,")
    print("  continue - and the bits are the same bits. The distinction that")
    print("  stands: predicting whether OTHER programs halt is undecidable")
    print("  (Test 25); suspending and resuming THIS machine exactly is an")
    print("  engineering property the lattice architecture has by design -")
    print("  deterministic state, all of it nameable, all of it savable.")


if __name__ == "__main__":
    main()
