#!/usr/bin/env python3
"""Make the chamber win USABLE: block-frame it for random access. The global chamber (9.24 bits/posting,
beats FOR's 13.37) is a single sequential stream = no random access. Group kept terms into BLOCKS of B and
chamber-compress each block independently, so decoding ONE block recovers ~B terms. Measure: (1) block-framed
bits/posting vs global 9.24 / FOR 13.37 (blocking loses ratio -- each block resets the model), and (2) per-block
DECODE latency (incl. state init) = the hot-path cost. Verdict: hot tier (beats FOR at OK latency) or cold tier.
"""
import sys, time, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from marco_full_eval import FullIndex
from scripts.test_chamber_mixer_v5_native import compress, decompress

KEEP_IDF = 4.0


def vbyte(gaps):
    out = bytearray()
    for v in gaps.astype(np.int64).tolist():
        while True:
            b = v & 0x7f; v >>= 7
            out.append(b | 0x80 if v else b)
            if not v:
                break
    return bytes(out)


def main():
    idx = FullIndex()
    keep = np.where(idx.idfa >= KEEP_IDF)[0]
    order = list(keep); random.Random(0).shuffle(order)
    # gather per-term gap byte-streams (varbyte) for ~6000 kept terms
    term_vb = []; term_np = []
    for t in order:
        s, e = int(idx.ptr[t]), int(idx.ptr[t + 1])
        if e - s < 2:
            continue
        d = np.diff(idx.di[s:e].astype(np.int64)).astype(np.uint32)
        term_vb.append(vbyte(d)); term_np.append(len(d))
        if len(term_vb) >= 6000:
            break
    print(f"  {len(term_vb):,} kept terms, {sum(term_np):,} gap-postings\n", flush=True)

    w = b"warmup " * 256; assert decompress(compress(w)[0], len(w))[0] == w   # JIT

    print(f"  {'config':<22}{'bits/posting':>14}{'vs FOR':>9}{'decode/block':>15}")
    print(f"  {'FOR (random-access)':<22}{'13.37':>14}{'--':>9}{'~0.01 ms':>15}")
    print(f"  {'chamber global':<22}{'9.24':>14}{'-31%':>9}{'no random access':>15}")
    for B in (256, 1024, 4096):
        nblocks = (len(term_vb) + B - 1) // B
        tot_bytes = 0; tot_post = 0; blobs = []
        for bi in range(nblocks):
            grp = term_vb[bi * B:(bi + 1) * B]
            data = b"".join(grp)
            blob, _ = compress(data)
            blobs.append((blob, len(data)))
            tot_bytes += len(blob); tot_post += sum(term_np[bi * B:(bi + 1) * B])
        bpp = tot_bytes * 8 / tot_post
        # decode latency: median over the blocks (incl. fresh-state init)
        ts = []
        for blob, n in blobs[:min(8, len(blobs))]:
            t0 = time.perf_counter(); back, _ = decompress(blob, n); ts.append((time.perf_counter() - t0) * 1000)
        med = float(np.median(ts))
        print(f"  {f'block B={B}':<22}{bpp:>14.2f}{(bpp/13.37-1)*100:>+8.0f}%{med:>12.1f} ms", flush=True)
    print(f"\n  read: ratio degrades as B shrinks (model resets); decode/block is the per-query cost")
    print(f"  (a query touches ~5-10 terms = that many block decodes). Hot tier iff beats FOR at low ms.")


if __name__ == "__main__":
    main()
