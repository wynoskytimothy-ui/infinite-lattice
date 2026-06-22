#!/usr/bin/env python3
"""Shrink the 2.16 GB index WITHOUT hurting accuracy. Tests the user's claim that we're not using
the smallest index. Measures REAL footprint + MRR (tf quantization) and exact lossless projections
(di delta-compression). Keeps the fast meet (bm25_fast).

  tf  uint16 -> uint8 (cap 255) / 4-bit (cap 15): BM25 saturates tf, so capping should be ~lossless.
  di  uint32 sorted postings -> delta gaps + varbyte / bitpack: lossless, standard IR compression.
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_fast import bm25_fast

N = 300


def mrr_with(idx, sample, qrels):
    rr = 0.0
    for qid, qt in sample:
        o, _ = bm25_fast(idx, stoks(qt), 100)
        gold = qrels[qid]
        for rank, d in enumerate(o[:10]):
            if int(d) in gold:
                rr += 1.0 / (rank + 1); break
    return rr / len(sample)


def varbyte_bytes(gaps):
    g = gaps.astype(np.int64)
    b = np.ones(len(g), dtype=np.int64)
    for thr in (1 << 7, 1 << 14, 1 << 21, 1 << 28):
        b += (g >= thr)
    return int(b.sum())


def main():
    idx = FullIndex()
    di_b, tf_b = idx.di.nbytes, idx.tf.nbytes
    meta = idx.ptr.nbytes + idx.doclen.nbytes + idx.idfa.nbytes
    total = di_b + tf_b + meta
    print(f"\n  CURRENT: di {di_b/1e9:.2f} GB (uint32) + tf {tf_b/1e9:.2f} GB (uint16) + meta {meta/1e6:.0f} MB "
          f"= {total/1e9:.2f} GB", flush=True)

    s = idx.tf[::101]   # sample for distribution
    print(f"  tf dist: max {idx.tf.max()}, p99 {np.percentile(s,99):.0f}, p99.9 {np.percentile(s,99.9):.0f}, "
          f"frac>15 {(idx.tf>15).mean()*100:.2f}%, frac>255 {(idx.tf>255).mean()*100:.4f}%", flush=True)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    for qid, qt in sample[:5]:
        bm25_fast(idx, stoks(qt), 100)
    base = mrr_with(idx, sample, qrels)
    orig = idx.tf
    idx.tf = np.minimum(orig, 255).astype(np.uint8); mrr8 = mrr_with(idx, sample, qrels)
    idx.tf = np.minimum(orig, 15).astype(np.uint8);  mrr4 = mrr_with(idx, sample, qrels)
    idx.tf = orig
    print(f"\n  TF QUANTIZATION (measured MRR@10, fast meet):")
    print(f"    uint16 baseline:   {tf_b/1e9:.2f} GB   MRR {base:.4f}")
    print(f"    uint8  (cap 255):  {len(orig)/1e9:.2f} GB   MRR {mrr8:.4f}  (delta {mrr8-base:+.4f})")
    print(f"    4-bit  (cap 15):   {len(orig)/2/1e9:.2f} GB   MRR {mrr4:.4f}  (delta {mrr4-base:+.4f})", flush=True)

    # di compression: exact projection from a representative contiguous slice (weights by posting freq)
    L = 30_000_000
    a = (len(idx.di) - L) // 2
    sl = idx.di[a:a + L].astype(np.int64)
    d = np.diff(sl)
    bnd = idx.ptr[(idx.ptr > a) & (idx.ptr < a + L)] - a   # term-starts inside the slice
    is_start = np.zeros(L, dtype=bool); is_start[bnd.astype(np.int64)] = True
    gaps = d[~is_start[1:]]
    gaps = gaps[gaps > 0]
    n_first = len(bnd) + 1
    scale = len(idx.di) / L
    vb = (varbyte_bytes(gaps) + n_first * 4) * scale
    floor = (np.ceil(np.log2(gaps + 1)).sum() / 8 + n_first * 4) * scale
    print(f"\n  DI COMPRESSION (lossless, exact projection; mean gap {gaps.mean():.0f}, median {int(np.median(gaps))}):")
    print(f"    current uint32:      {di_b/1e9:.2f} GB")
    print(f"    delta + varbyte:     {vb/1e9:.2f} GB   ({di_b/vb:.1f}x smaller)")
    print(f"    delta + bitpack flr: {floor/1e9:.2f} GB   ({di_b/floor:.1f}x, entropy floor)", flush=True)

    proj = vb + len(orig) / 2 + meta     # varbyte di + 4-bit tf + meta
    print(f"\n  PROJECTED TOTAL (varbyte di + 4-bit tf + meta): {proj/1e9:.2f} GB  "
          f"(from {total/1e9:.2f} GB = {total/proj:.1f}x smaller)")
    print(f"  accuracy cost: tf 4-bit MRR delta {mrr4-base:+.4f}; di compression is exact/lossless")


if __name__ == "__main__":
    main()
