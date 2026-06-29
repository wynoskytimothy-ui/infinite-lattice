#!/usr/bin/env python3
r"""LENS: pq-3d-chambers (v2 -- give the fixed chamber codebook its BEST shot)

v1 result: fixed 32-chamber codebook LOSES to learned k-means PQ at 5b/sub
(R@10 0.415 vs 0.509) because (a) only 24 of 32 dirs are distinct, (b) the
chamber dirs don't match isotropic RP blocks, (c) k-means adapts to data.

v2 tests three honest "best-shot" variants for the FIXED codebook:
  A) OPQ-style: learn a single 3x3 rotation per sub-space (or one global D x D
     rotation) that aligns data to the fixed chamber dirs. Rotation is ~free at
     query time (apply once) and adds NO per-doc bytes.
  B) small-training-set robustness: when only N_tr is tiny, does the fixed
     codebook (zero params) beat under-trained k-means?
  C) chamber AS the lattice address: instead of cosine PQ, exploit that the
     chamber assignment IS a meet/sign-pattern -> O(1) bucket. Measure pure
     sign-octant code (3 bits/block, sign of X,Y,zeta) vs the 5-bit chamber and
     vs k-means at matched 3 bits.

Honest goal: find ANY regime where the chamber structure wins or ties on the
footprint/speed/O(1) axis (its home turf), not the accuracy axis.
"""
import sys, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _o1_pq_3d_chambers import (chamber_codebook, load_splade_docs, random_projection,
                                split3, chamber_encode, kmeans_pq_train, kmeans_pq_encode,
                                retention, l2norm)


def fit_rotation_to_chambers(Xtr, unit, m, iters=15):
    """OPQ-lite: learn ONE global D x D orthogonal R so that R @ x has 3-blocks that
    align to the fixed chamber dirs. Alternate: (assign blocks->chambers) then
    (Procrustes update R to best map data onto assigned chamber targets)."""
    n, D = Xtr.shape
    R = np.eye(D, dtype=np.float64)
    X = Xtr.astype(np.float64)
    for it in range(iters):
        Y = X @ R                       # rotated
        sub = Y.reshape(n, m, 3)
        flat = sub.reshape(-1, 3)
        bnorm = np.linalg.norm(flat, axis=1, keepdims=True)
        dots = flat @ unit.T
        codes = dots.argmax(1)
        target = (unit[codes] * bnorm).reshape(n, D)   # what we WANT R@x to equal
        # Procrustes: find R minimizing ||X R - target||  => R = U V^T from X^T target
        M = X.T @ target
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
    return R.astype(np.float32)


def main():
    unit, raw = chamber_codebook()
    D = 96; m = D // 3
    NDB = 8000; NQ = 600
    print("=" * 78)
    print("PQ-3D-CHAMBERS v2 -- best-shot for the FIXED chamber codebook")
    print("=" * 78)

    rows, n = load_splade_docs(max_docs=NDB + NQ + 8000 + 100)
    X = random_projection(rows[: NDB + NQ + 8000], D, seed=1)
    Xq = X[:NQ]; Xdb = X[NQ:NQ + NDB]; Xtr_full = X[NQ + NDB:]

    Bdoc = m * 5 / 8.0
    print(f"\n  D={D}  {m} sub x 5b = {Bdoc:.0f} B/doc")

    # ---- A) OPQ rotation aligning data to fixed chambers ----
    print("\n  [A] OPQ-lite rotation -> fixed chamber codebook (rotation is free/per-doc)")
    R = fit_rotation_to_chambers(Xtr_full[:4000], unit, m, iters=15)
    Xdb_r = Xdb @ R; Xq_r = Xq @ R
    _, rec_chR = chamber_encode(Xdb_r, unit)
    # query must live in the same rotated space; retention compares ranking, rotation
    # is orthogonal so cos(q,d) is preserved for the EXACT side -> use rotated full for both
    r10, r100 = retention(Xq_r, Xdb_r, rec_chR, k=10, K2=100)
    print(f"      chamber+OPQrot:  R@10={r10:.3f}  R@100={r100:.3f}")
    _, rec_ch0 = chamber_encode(Xdb, unit)
    r10_0, _ = retention(Xq, Xdb, rec_ch0, k=10)
    print(f"      chamber (no rot): R@10={r10_0:.3f}  (rotation gain = {r10-r10_0:+.3f})")

    # ---- B) small-training-set robustness ----
    print("\n  [B] robustness: chamber (0 params) vs k-means under tiny training sets")
    for ntr in (50, 200, 1000, 6000):
        cbs = kmeans_pq_train(Xtr_full[:ntr], m, K=32, iters=25, seed=0)
        _, rec_km = kmeans_pq_encode(Xdb, cbs)
        r10_km, _ = retention(Xq, Xdb, rec_km, k=10)
        print(f"      ntr={ntr:>5}:  kmeans R@10={r10_km:.3f}   (chamber fixed = {r10_0:.3f})")

    # ---- C) matched-bitrate at the LOW end: sign-octant (3 bits) ----
    print("\n  [C] low-bitrate: sign-octant 3b/block vs k-means K=8 (3b) -- chamber's O(1) sign code")
    sub, _ = split3(Xdb)
    flat = sub.reshape(-1, 3)
    # sign octant code = which of 8 sign combos (O(1), branchless), recon = sign*mean|block|
    signs = (flat > 0).astype(np.uint8)
    oct_code = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]
    # recon: unit-sign direction * block norm
    sdir = np.where(flat > 0, 1.0, -1.0) / math.sqrt(3)
    bnorm = np.linalg.norm(flat, axis=1, keepdims=True)
    rec_oct = (sdir * bnorm).reshape(Xdb.shape)
    r10_oct, _ = retention(Xq, Xdb, rec_oct, k=10)
    cbs8 = kmeans_pq_train(Xtr_full[:6000], m, K=8, iters=25, seed=0)
    _, rec_km8 = kmeans_pq_encode(Xdb, cbs8)
    r10_km8, _ = retention(Xq, Xdb, rec_km8, k=10)
    B3 = m * 3 / 8.0
    print(f"      sign-octant 3b ({B3:.0f} B/doc):  R@10={r10_oct:.3f}   "
          f"kmeans K=8 3b: R@10={r10_km8:.3f}")

    print("\n" + "=" * 78)
    print("VERDICT v2")
    print("=" * 78)
    best_ch = max(r10, r10_0)
    print(f"  best fixed-chamber R@10 = {best_ch:.3f} (OPQrot helps={r10-r10_0:+.3f})")
    print(f"  learned k-means (ntr=6000) R@10 = (see [B] ntr=6000 line)")
    print(f"  sign-octant vs kmeans-K8 at 3b: {r10_oct:.3f} vs {r10_km8:.3f}")
    print("  => chamber codebook is a FIXED structured quantizer; honest read in stdout.")
    return dict(r10_chamber=r10_0, r10_chamber_rot=r10, r10_oct=r10_oct, r10_km8=r10_km8)


if __name__ == "__main__":
    main()
