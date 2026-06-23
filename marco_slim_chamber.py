#!/usr/bin/env python3
"""Run the user's CHAMBER CODEC (scripts/test_chamber_mixer_v5_native: 0.83 bits/byte on text, beats
lzma/bz2/zlib) on the slim index's posting-GAP stream. THE compression north-star test: does context-mixing
crush the delta-gaps below FOR's 13.3 bits/posting (0.44 GB)? Honest caveat (the user's own correction):
pruning leaves the SPARSE rare-term tail (big gaps, high entropy) that may resist. Measure on a representative
sample of kept (idf>=4) terms: chamber bits/posting + bit-identical round-trip vs raw/varbyte/FOR/min-width.
"""
import sys, time, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from marco_full_eval import FullIndex
from scripts.test_chamber_mixer_v5_native import compress, decompress

KEEP_IDF = 4.0
SAMPLE_POST = 1_500_000


def vbyte(gaps):
    out = bytearray()
    for v in gaps.astype(np.int64).tolist():
        while True:
            b = v & 0x7f; v >>= 7
            out.append(b | 0x80 if v else b)
            if not v:
                break
    return bytes(out)


def row(name, nbytes, npost, extra=""):
    print(f"    {name:<24}{nbytes/1e6:>9.2f} MB  {nbytes*8/npost:>7.2f} bits/posting{extra}")


def main():
    idx = FullIndex()
    keep = np.where(idx.idfa >= KEEP_IDF)[0]
    order = list(keep); random.Random(0).shuffle(order)
    chunks = []; npost = 0; nterms = 0
    for t in order:
        s, e = int(idx.ptr[t]), int(idx.ptr[t + 1])
        if e - s < 2:
            continue
        d = np.diff(idx.di[s:e].astype(np.int64))
        chunks.append(d.astype(np.uint32)); npost += len(d); nterms += 1
        if npost >= SAMPLE_POST:
            break
    print(f"  sampled {nterms:,} kept terms (idf>={KEEP_IDF}), {npost:,} gap-postings\n", flush=True)

    raw_u32 = np.concatenate(chunks).tobytes()
    vb = vbyte(np.concatenate(chunks))
    for_bits = sum(len(d) * max(1, int(int(d.max()).bit_length())) for d in chunks)
    print("  BASELINES (what the slim index could ship):")
    row("raw uint32", len(raw_u32), npost)
    row("varbyte (delta)", len(vb), npost)
    row("FOR bit-pack (per-term)", for_bits // 8, npost, "   <- marco_slim_for ships this (0.44 GB)")

    print("\n  warming JIT...", flush=True)
    w = b"warmup " * 256; assert decompress(compress(w)[0], len(w))[0] == w

    print("\n  CHAMBER CODEC (your context-mixer) on the gap stream:", flush=True)
    best = None
    for nm, data in (("chamber(uint32-LE)", raw_u32), ("chamber(varbyte)", vb)):
        t0 = time.time(); blob, _ = compress(data); te = time.time() - t0
        t0 = time.time(); back, _ = decompress(blob, len(data)); td = time.time() - t0
        ok = back == data
        bpp = len(blob) * 8 / npost
        best = bpp if best is None else min(best, bpp)
        print(f"    {nm:<24}{len(blob)/1e6:>9.2f} MB  {bpp:>7.2f} bits/posting   "
              f"enc {te:.0f}s dec {td:.0f}s  round-trip {'OK' if ok else 'FAIL'}", flush=True)

    for_bpp = (for_bits // 8) * 8 / npost
    full_for = 178.9e6 * for_bpp / 8 / 1e9
    full_ch = 178.9e6 * best / 8 / 1e9
    print(f"\n  PROJECTED full di (178.9M kept postings): FOR {full_for:.3f} GB vs chamber-best {full_ch:.3f} GB")
    print(f"  -> chamber {'BEATS' if best < for_bpp else 'does NOT beat'} FOR on the pruned gap stream "
          f"({best:.2f} vs {for_bpp:.2f} bits/posting). The sparse-tail wall, MEASURED.")


if __name__ == "__main__":
    main()
