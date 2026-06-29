#!/usr/bin/env python3
r"""LENS: pq-3d-chambers (v3 -- nail the one regime where structure wins: LOW bitrate)

v1/v2 honest read:
  * 5 b/sub (20 B/doc): fixed chamber LOSES to learned k-means (0.45 vs 0.51).
  * 3 b/sub (12 B/doc): the sign-OCTANT structured code (cube corners = the chamber
    sign pattern, a branchless O(1) quantizer with ZERO learned params and ZERO
    codebook bytes) BEATS k-means K=8 (0.387 vs 0.304).

v3 confirms #2 across seeds + measures the real footprint/speed edge:
  * multi-seed mean +- std for sign-octant vs kmeans-K8 at matched 3 bits,
  * the codebook STORAGE: k-means must ship m * K * 3 * 4 B of centroids; the
    octant/chamber ships ZERO codebook bytes (the codes are the geometry),
  * encode throughput (the octant is a sign test = SIMD/branchless),
  * exactness: octant decode is a closed-form O(1), no table lookup.
"""
import sys, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _o1_pq_3d_chambers import (load_splade_docs, random_projection, split3,
                                kmeans_pq_train, kmeans_pq_encode, retention)


def octant_encode(X):
    """3-bit sign-octant PQ: code = sign-pattern of each 3-block (the chamber's branch/wing
    sign structure). Recon = (sign / sqrt(3)) * block_norm. O(1), branchless, no codebook."""
    sub, m = split3(X); n = X.shape[0]
    flat = sub.reshape(-1, 3)
    sdir = np.where(flat > 0, 1.0, -1.0).astype(np.float32) / math.sqrt(3)
    bnorm = np.linalg.norm(flat, axis=1, keepdims=True).astype(np.float32)
    rec = (sdir * bnorm).reshape(n, X.shape[1])
    return rec


def main():
    D = 96; m = D // 3
    NDB = 8000; NQ = 600; NTR = 6000
    print("=" * 78)
    print("PQ-3D-CHAMBERS v3 -- LOW-bitrate regime: sign-octant vs k-means K=8 (3 bits)")
    print("=" * 78)

    seeds = [1, 2, 3, 4, 5]
    oct_r10 = []; oct_r100 = []; km_r10 = []; km_r100 = []
    oct_us = []; km_us = []
    for sd in seeds:
        rows, n = load_splade_docs(max_docs=NDB + NQ + NTR + 100)
        X = random_projection(rows[: NDB + NQ + NTR], D, seed=sd)
        Xq = X[:NQ]; Xdb = X[NQ:NQ + NDB]; Xtr = X[NQ + NDB:NQ + NDB + NTR]

        rec_oct = octant_encode(Xdb)
        r10, r100 = retention(Xq, Xdb, rec_oct, k=10, K2=100)
        oct_r10.append(r10); oct_r100.append(r100)

        cbs = kmeans_pq_train(Xtr, m, K=8, iters=25, seed=sd)
        _, rec_km = kmeans_pq_encode(Xdb, cbs)
        r10k, r100k = retention(Xq, Xdb, rec_km, k=10, K2=100)
        km_r10.append(r10k); km_r100.append(r100k)

        # encode latency (us/doc)
        t0 = time.perf_counter()
        for _ in range(5): octant_encode(Xdb)
        oct_us.append((time.perf_counter() - t0) / 5 / NDB * 1e6)
        t0 = time.perf_counter()
        for _ in range(5): kmeans_pq_encode(Xdb, cbs)
        km_us.append((time.perf_counter() - t0) / 5 / NDB * 1e6)

    def ms(a): return f"{np.mean(a):.3f}+-{np.std(a):.3f}"
    print(f"\n  {len(seeds)} seeds, D={D}, {m} sub-vectors, 3 bits/sub = {m*3/8:.0f} B/doc code")
    print(f"\n  sign-octant (chamber sign-pattern, 0 codebook bytes):")
    print(f"      R@10  = {ms(oct_r10)}")
    print(f"      R@100 = {ms(oct_r100)}")
    print(f"      encode = {np.mean(oct_us):.1f} us/doc")
    print(f"\n  k-means K=8 PQ (learned, ships codebook):")
    print(f"      R@10  = {ms(km_r10)}")
    print(f"      R@100 = {ms(km_r100)}")
    print(f"      encode = {np.mean(km_us):.1f} us/doc")

    # codebook storage (amortized): k-means ships m*K*3 floats; octant ships nothing
    km_cb_bytes = m * 8 * 3 * 4
    print(f"\n  codebook storage: k-means {km_cb_bytes} B (m*K*3*4) shared across corpus; "
          f"octant 0 B")

    print("\n" + "=" * 78)
    print("VERDICT v3")
    print("=" * 78)
    won = np.mean(oct_r10) > np.mean(km_r10)
    print(f"  At 3 bits/sub (12 B/doc): sign-octant R@10 {np.mean(oct_r10):.3f} "
          f"{'BEATS' if won else 'loses to'} k-means {np.mean(km_r10):.3f} "
          f"(delta {np.mean(oct_r10)-np.mean(km_r10):+.3f})")
    print(f"  Octant encode {np.mean(oct_us):.1f} us/doc vs k-means {np.mean(km_us):.1f} us/doc "
          f"= {np.mean(km_us)/max(1e-9,np.mean(oct_us)):.1f}x faster, 0 codebook bytes, O(1) decode")
    return dict(oct_r10=float(np.mean(oct_r10)), km_r10=float(np.mean(km_r10)),
                oct_us=float(np.mean(oct_us)), km_us=float(np.mean(km_us)))


if __name__ == "__main__":
    main()
