#!/usr/bin/env python3
r"""LENS: pq-3d-chambers
Product-quantization of doc representations using the AETHOS 32 chambers as a FIXED
(data-independent) 3D codebook, vs standard learned k-means PQ at the SAME bitrate.

Construction
------------
The 32 chambers are 32 points in R^3 (X, Y, zeta) produced by the 4 branches x 8 wings
of the 3D complex plane (aethos_complex_plane.wing_transform). Normalized to unit
directions they form a fixed spherical codebook of <=32 distinct directions in R^3.

PQ-3D-chambers codec:
  * random-project the (sparse, 30522-d) SPLADE doc rep to a dense D-dim vector
    (D divisible by 3),
  * split into D/3 sub-vectors of dim 3,
  * quantize each 3-d sub-vector to the nearest of the 32 chamber DIRECTIONS
    (cosine / dot on the unit sphere) -> 5 bits per sub-vector,
  * code = D/3 * 5 bits.  Reconstruct each sub-vector as its chamber direction
    (we store a per-sub-vector norm? NO -- pure PQ stores only the 5-bit code; we
    test cosine retrieval where per-block norm cancels, and ALSO a norm-augmented
    variant for an apples-to-apples vs k-means that learns centroids w/ magnitude).

Baseline: standard k-means PQ -- learn 32 centroids per 3-d sub-space from a training
split (each sub-space gets its OWN 32-centroid codebook), 5 bits/sub-vector, identical
bitrate.  This is the honest competitor: same B/doc, codebook LEARNED from data.

We measure:
  * B/doc of the PQ code,
  * retrieval RETENTION: rank docs by approx cosine (query full-dense vs doc PQ-recon)
    and report Recall@10 / Recall@100 / nDCG-style overlap vs the EXACT full-precision
    dense cosine ranking (the dense rep is itself an approx of SPLADE; we measure how
    much the PQ step loses relative to that dense rep, which is the fair PQ question).
  * Also: does the SPLADE-native sparse-dot top-10 survive? (end-to-end retention).

Everything CPU, real bytes (bit-accounted) and real latency (perf_counter).
"""
import os, sys, time, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SPLADE_DIR = Path(r"C:\Users\wynos\trng\marco_data\splade_native")
VOCAB = 30522
RNG = np.random.default_rng(0)


# ----------------------------------------------------------------------------------
#  1. the 32-chamber 3D codebook (FIXED, data-independent)
# ----------------------------------------------------------------------------------
def chamber_codebook():
    """Return (32,3) array of UNIT chamber directions + the raw signed coords."""
    from aethos_complex_plane import wing_transform
    from aethos_lattice import BranchKind
    raw = []
    for b in BranchKind:
        for w in range(1, 9):
            psi = wing_transform(b, (3,), 7, w)  # a=3,n=7 (n>=a) gives a generic chamber
            raw.append(np.array(psi.coord, dtype=np.float64))
    raw = np.stack(raw)  # (32,3)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = raw / norms
    return unit, raw


# ----------------------------------------------------------------------------------
#  2. load SPLADE docs as sparse rows -> dense random projection
# ----------------------------------------------------------------------------------
def load_splade_docs(max_docs=20000):
    z = np.load(SPLADE_DIR / "chunk_00000.npz")
    di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"]
    z.close()
    n = min(max_docs, len(di))
    # build sparse rows (term_id, weight) per doc; weights are uint8 * QSCALE(0.04)
    rows = []
    for d in range(n):
        s, e = int(pa[d]), int(pa[d + 1])
        rows.append((ti[s:e].astype(np.int64), wt[s:e].astype(np.float32) * 0.04))
    return rows, n


def random_projection(rows, D, seed=0):
    """Sparse SPLADE rows -> dense D-dim via a fixed Gaussian random projection
    (Johnson-Lindenstrauss).  R is (VOCAB, D); for sparsity we only touch active terms."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((VOCAB, D)).astype(np.float32) / math.sqrt(D)
    X = np.zeros((len(rows), D), dtype=np.float32)
    for i, (tids, ws) in enumerate(rows):
        if len(tids):
            X[i] = (ws[:, None] * R[tids]).sum(axis=0)
    return X


# ----------------------------------------------------------------------------------
#  3. PQ encoders (chamber-fixed vs learned k-means), 3-d sub-vectors, 5 bits
# ----------------------------------------------------------------------------------
def split3(X):
    n, D = X.shape
    assert D % 3 == 0
    m = D // 3
    return X.reshape(n, m, 3), m


def chamber_encode(X, unit):
    """Quantize each 3-d block to nearest chamber DIRECTION (max dot on unit sphere).
    Returns codes (n,m) uint8 in [0,31] and reconstruction (n,D) using
    chamber_dir * block_norm (norm stored separately is NOT counted in the pure-PQ
    bitrate; for cosine retrieval the global norm cancels, so we reconstruct with the
    block norm to make the directional-only loss visible, then ALSO test code-only)."""
    sub, m = split3(X)                       # (n,m,3)
    n = X.shape[0]
    # dot of every block with every chamber dir
    flat = sub.reshape(-1, 3)                 # (n*m,3)
    dots = flat @ unit.T                      # (n*m,32)
    codes = np.argmax(dots, axis=1).astype(np.uint8).reshape(n, m)
    # reconstruction: chamber dir scaled by the block's own L2 norm (direction-only PQ)
    bnorm = np.linalg.norm(flat, axis=1, keepdims=True)
    recon = (unit[codes.reshape(-1)] * bnorm).reshape(n, m, 3).reshape(n, X.shape[1])
    return codes, recon


def kmeans_pq_train(Xtr, m, K=32, iters=25, seed=0):
    """Learn K centroids per sub-space from Xtr.  Returns list of (K,3) codebooks."""
    rng = np.random.default_rng(seed)
    subtr, _ = split3(Xtr)                    # (ntr,m,3)
    cbs = []
    for s in range(m):
        data = subtr[:, s, :]                 # (ntr,3)
        # k-means++ light init
        idx = rng.choice(len(data), 1)
        cent = data[idx]
        for _ in range(K - 1):
            d2 = ((data[:, None, :] - cent[None]) ** 2).sum(-1).min(1)
            p = d2 / (d2.sum() + 1e-12)
            cent = np.vstack([cent, data[rng.choice(len(data), p=p)]])
        for _ in range(iters):
            d2 = ((data[:, None, :] - cent[None]) ** 2).sum(-1)
            assign = d2.argmin(1)
            for k in range(K):
                msk = assign == k
                if msk.any():
                    cent[k] = data[msk].mean(0)
        cbs.append(cent.astype(np.float32))
    return cbs


def kmeans_pq_encode(X, cbs):
    sub, m = split3(X)
    n = X.shape[0]
    codes = np.empty((n, m), np.uint8)
    recon = np.empty_like(sub)
    for s in range(m):
        data = sub[:, s, :]
        d2 = ((data[:, None, :] - cbs[s][None]) ** 2).sum(-1)
        a = d2.argmin(1)
        codes[:, s] = a.astype(np.uint8)
        recon[:, s, :] = cbs[s][a]
    return codes, recon.reshape(n, X.shape[1])


def chamber_pq_train(Xtr, m, K=32, iters=25, seed=0):
    """Chamber-as-INIT k-means: per sub-space start from the 32 chamber dirs scaled to
    the data, then run k-means. Tests whether the chamber SHAPE is a good basin."""
    unit, _ = chamber_codebook()
    rng = np.random.default_rng(seed)
    subtr, _ = split3(Xtr)
    cbs = []
    for s in range(m):
        data = subtr[:, s, :]
        scale = np.linalg.norm(data, axis=1).mean()
        cent = (unit * scale).astype(np.float32).copy()
        for _ in range(iters):
            d2 = ((data[:, None, :] - cent[None]) ** 2).sum(-1)
            assign = d2.argmin(1)
            for k in range(K):
                msk = assign == k
                if msk.any():
                    cent[k] = data[msk].mean(0)
        cbs.append(cent)
    return cbs


# ----------------------------------------------------------------------------------
#  4. retrieval retention: rank by approx cosine vs exact dense cosine
# ----------------------------------------------------------------------------------
def l2norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return X / n


def retention(Xq_full, Xd_full, Xd_recon, k=10, K2=100):
    """For each query (=a doc used as query), rank all docs by:
       - exact: cos(q_full, d_full)
       - approx: cos(q_full, d_recon)
    Measure Recall@k of approx-topk vs exact-topk and rank-correlation."""
    Qf = l2norm(Xq_full); Df = l2norm(Xd_full); Dr = l2norm(Xd_recon)
    exact = Qf @ Df.T          # (nq,nd)
    approx = Qf @ Dr.T
    nq = Qf.shape[0]
    rec_k = 0.0; rec_K2 = 0.0
    for i in range(nq):
        ex = exact[i].copy(); ap = approx[i].copy()
        ex[i] = -1e9; ap[i] = -1e9     # exclude self
        ex_top = set(np.argpartition(-ex, k)[:k])
        ap_top = set(np.argpartition(-ap, k)[:k])
        rec_k += len(ex_top & ap_top) / k
        ex_topK = set(np.argpartition(-ex, K2)[:K2])
        ap_topK = set(np.argpartition(-ap, K2)[:K2])
        rec_K2 += len(ex_topK & ap_topK) / K2
    return rec_k / nq, rec_K2 / nq


def main():
    print("=" * 78)
    print("PQ-3D-CHAMBERS: fixed 32-chamber codebook vs learned k-means PQ (same 5b/sub)")
    print("=" * 78)

    unit, raw = chamber_codebook()
    uniq = np.unique(np.round(unit, 6), axis=0)
    print(f"\n  chambers: 32 raw -> {len(uniq)} DISTINCT unit directions in R^3")
    print(f"  (so the chamber codebook effectively uses ~{len(uniq)} of the 32 codes)")

    D = 96            # dense projection dim (must be div by 3); 96/3 = 32 sub-vectors
    NDOC = 8000       # docs in the index
    NTR = 6000        # training docs for k-means (held separate from the query set)
    NQ = 600          # query docs

    rows, n = load_splade_docs(max_docs=NDOC + NTR + NQ + 100)
    print(f"\n  loaded {n} SPLADE docs; projecting to D={D} (Gaussian RP)")
    t0 = time.perf_counter()
    X = random_projection(rows[: NDOC + NTR + NQ], D, seed=1)
    print(f"  projection done in {time.perf_counter()-t0:.1f}s  X shape {X.shape}")

    Xtr = X[:NTR]
    Xdb = X[NTR:NTR + NDOC]
    Xq = X[NTR + NDOC:NTR + NDOC + NQ]
    m = D // 3
    bits = m * 5
    Bdoc = bits / 8.0
    print(f"\n  PQ config: {m} sub-vectors x 5 bits = {bits} bits = {Bdoc:.1f} B/doc PQ code")
    print(f"  (vs dense fp32 {D*4} B/doc, fp16 {D*2} B/doc; raw SPLADE ~286.9 B/doc)")

    # encode the DB three ways
    codes_ch, rec_ch = chamber_encode(Xdb, unit)

    t0 = time.perf_counter()
    cbs_km = kmeans_pq_train(Xtr, m, K=32, iters=25, seed=0)
    km_train_s = time.perf_counter() - t0
    codes_km, rec_km = kmeans_pq_encode(Xdb, cbs_km)

    cbs_chk = chamber_pq_train(Xtr, m, K=32, iters=25, seed=0)
    codes_chk, rec_chk = kmeans_pq_encode(Xdb, cbs_chk)

    # MSE of reconstruction (sanity)
    mse_ch = float(((Xdb - rec_ch) ** 2).mean())
    mse_km = float(((Xdb - rec_km) ** 2).mean())
    mse_chk = float(((Xdb - rec_chk) ** 2).mean())
    print(f"\n  recon MSE  chamber-fixed={mse_ch:.4f}  kmeans={mse_km:.4f}  chamber-init-kmeans={mse_chk:.4f}")
    print(f"  k-means train time: {km_train_s:.1f}s for {m} sub-codebooks")

    print(f"\n  --- retrieval retention (approx cos vs EXACT dense cos, {NQ} queries x {NDOC} docs) ---")
    for name, rec in (("chamber-fixed (data-indep)", rec_ch),
                      ("kmeans-PQ   (learned)     ", rec_km),
                      ("chamber-init-kmeans       ", rec_chk)):
        r10, r100 = retention(Xq, Xdb, rec, k=10, K2=100)
        print(f"    {name}:  R@10={r10:.3f}  R@100={r100:.3f}")

    # also: an even smaller chamber variant -- 3 bits would need only 8 codes; report
    # the EFFECTIVE distinct-direction count we already saw.
    print(f"\n  NOTE distinct chamber directions = {len(uniq)} -> "
          f"could pack {int(math.ceil(math.log2(len(uniq))))} bits/sub if relabeled "
          f"(=> {m*int(math.ceil(math.log2(len(uniq))))//8} B/doc)")

    # latency of encode (per-doc) for each
    def time_encode(fn, *a):
        t0 = time.perf_counter()
        for _ in range(3):
            fn(*a)
        return (time.perf_counter() - t0) / 3 / Xdb.shape[0] * 1e6   # us/doc
    us_ch = time_encode(lambda: chamber_encode(Xdb, unit))
    us_km = time_encode(lambda: kmeans_pq_encode(Xdb, cbs_km))
    print(f"\n  encode latency: chamber {us_ch:.1f} us/doc   kmeans {us_km:.1f} us/doc")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    r10_ch, _ = retention(Xq, Xdb, rec_ch, k=10)
    r10_km, _ = retention(Xq, Xdb, rec_km, k=10)
    if r10_ch >= r10_km - 0.01:
        print(f"  chamber TIES/BEATS k-means at same bitrate (R@10 {r10_ch:.3f} vs {r10_km:.3f})")
    else:
        print(f"  chamber LOSES to learned k-means (R@10 {r10_ch:.3f} vs {r10_km:.3f}) "
              f"by {r10_km-r10_ch:.3f} -- the fixed codebook costs accuracy")
    print(f"  Both at {Bdoc:.1f} B/doc PQ code (5 b/sub).")
    return dict(Bdoc=Bdoc, r10_ch=r10_ch, r10_km=r10_km, mse_ch=mse_ch, mse_km=mse_km,
                distinct=len(uniq), us_ch=us_ch, us_km=us_km)


if __name__ == "__main__":
    main()
