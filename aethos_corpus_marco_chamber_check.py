#!/usr/bin/env python3
"""Validate the 9.24 bits/posting chamber projection on a REAL MARCO gap stream.

footprint() in aethos_algebraic_corpus multiplies postings x 9.24 (a projection
from commit 345d6b0, measured on the slim posting-gaps).  Here we BUILD a small
MARCO subset, serialize its posting-gap stream the same way (varbyte gaps over
dense ordinals), and run the actual chamber codec (compress/decompress, byte-exact)
to MEASURE bits/posting + confirm round-trip.  Subset is small because chamber
decode is ~0.4 MB/s (cold tier).
"""
import sys, time
import numpy as np
from aethos_algebraic_corpus import AlgebraicCorpus
from marco_full_eval import FullIndex
import scripts.test_chamber_mixer_v5_native as cham


def varbyte(vals):
    out = bytearray()
    for v in vals:
        v = int(v)
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def main():
    n_docs = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"building {n_docs} MARCO passages for the chamber gap-stream check...")
    idx = FullIndex()
    import random
    pids = random.Random(42).sample(range(idx.N), n_docs)
    ac = AlgebraicCorpus()
    for pid in pids:
        ac.add(str(pid), idx.text(pid))
    ac.build()

    # serialize posting-gaps over DENSE ordinals (same model footprint() uses)
    ord_of = {d: i for i, d in enumerate(ac.doc_len)}
    n_postings = 0
    gap_vals = []
    for p, pl in ac.postings.items():
        docs = sorted(ord_of[d] for d in pl)
        n_postings += len(docs)
        gap_vals.append(docs[0])
        for i in range(1, len(docs)):
            gap_vals.append(docs[i] - docs[i - 1])
    stream = varbyte(gap_vals)
    print(f"  {n_postings:,} postings -> varbyte gap stream {len(stream):,} bytes "
          f"({8*len(stream)/n_postings:.2f} varbyte bits/posting)")

    print("  compiling + running chamber codec (cold, ~0.4 MB/s)...")
    t0 = time.time()
    blob, _ = cham.compress(stream)
    t_c = time.time() - t0
    t0 = time.time()
    restored, _ = cham.decompress(blob, len(stream))
    t_d = time.time() - t0
    exact = restored == stream
    cham_bpp = 8 * len(blob) / n_postings
    print(f"  chamber: {len(blob):,} bytes  = {cham_bpp:.2f} bits/posting  "
          f"(round-trip byte-exact? {exact})")
    print(f"  compress {t_c:.1f}s  decompress {t_d:.1f}s  "
          f"({len(stream)/1e6/max(1e-9,t_d):.2f} MB/s decode)")
    print(f"  PROJECTION USED IN footprint(): 9.24 bits/posting")
    print(f"  MEASURED HERE: {cham_bpp:.2f} bits/posting "
          f"({'within' if abs(cham_bpp-9.24)<2 else 'OFF from'} the projection)")


if __name__ == "__main__":
    main()
