#!/usr/bin/env python3
"""VALIDATE on full 8.8M MARCO: doc-prune K=40 + 4-bit Lloyd-Max weights + varint di gaps.
Streams the full SPLADE encode chunks, keeps each doc's top-K terms, re-inverts, measures the REAL
on-disk B/doc (varint di + 4-bit weights + 64B codebook + 0.625 chamber tag) and serves dev-small
(MRR@10, recall@100, recall@1000). Turns the 50k projection into a measured full-corpus number."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

K = int(os.environ.get("K", "40"))
NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
WORK = m.WORK


def varint_bytes(gaps):
    g = gaps.astype(np.int64)
    b = np.ones(len(g), np.int64)
    b += (g >= (1 << 7)); b += (g >= (1 << 14)); b += (g >= (1 << 21)); b += (g >= (1 << 28))
    return int(b.sum())


def main():
    t0 = time.perf_counter()
    chunks = sorted(WORK.glob("chunk_*.npz"))
    # pass 1: stream, per-doc top-K, accumulate (term_id, global_doc, weight)
    Ti = []; Di = []; Wi = []
    ndoc = 0
    for ci, cp in enumerate(chunks):
        z = np.load(cp)
        di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"]
        ct = []; cd = []; cw = []
        for d in range(len(di)):
            s, e = int(pa[d]), int(pa[d + 1])
            tids = ti[s:e]; ws = wt[s:e]
            if len(tids) > K:
                idx = np.argpartition(-ws, K)[:K]; tids = tids[idx]; ws = ws[idx]
            ct.append(tids); cd.append(np.full(len(tids), int(di[d]), np.int64)); cw.append(ws)
            ndoc += 1
        if ct:
            Ti.append(np.concatenate(ct)); Di.append(np.concatenate(cd)); Wi.append(np.concatenate(cw))
        if (ci + 1) % 10 == 0:
            print(f"    chunk {ci+1}/{len(chunks)} docs={ndoc:,} ({time.perf_counter()-t0:.0f}s)", flush=True)
    T = np.concatenate(Ti); D = np.concatenate(Di); W = np.concatenate(Wi)
    del Ti, Di, Wi
    n_post = len(T)
    present = np.unique(D); n_present = len(present)
    locmap = np.full(int(present.max()) + 1, -1, np.int64); locmap[present] = np.arange(n_present)
    Dl = locmap[D].astype(np.int32)
    print(f"\n  pruned: {n_post:,} postings ({n_post/n_present:.1f}/doc), {n_present:,} docs ({time.perf_counter()-t0:.0f}s)", flush=True)

    # build inverted index: group by term
    o = np.argsort(T, kind="stable"); T = T[o]; Dl = Dl[o]; W = W[o]
    uniqT, first = np.unique(T, return_index=True); offs = np.append(first, n_post)
    # 4-bit Lloyd-Max codebook on a weight sample
    samp = W[np.random.RandomState(0).randint(0, n_post, min(2_000_000, n_post))].astype(np.float32)
    cb = np.quantile(samp, np.linspace(0, 1, 16))           # 16 levels init
    for _ in range(8):
        d = np.abs(samp[:, None] - cb[None, :]); a = d.argmin(1)
        for k in range(16):
            sel = a == k
            if sel.any(): cb[k] = samp[sel].mean()
    # footprint: varint di-gaps per term + 4-bit weights + codebook + chamber tag
    di_bytes = 0
    tcols = {}
    for j in range(len(uniqT)):
        s, e = offs[j], offs[j + 1]
        locs = np.sort(Dl[s:e].astype(np.int64))
        gaps = np.empty(len(locs), np.int64); gaps[0] = locs[0] + 1
        if len(locs) > 1: gaps[1:] = np.diff(locs)
        di_bytes += varint_bytes(gaps)
        tcols[int(uniqT[j])] = (locs.astype(np.int64), W[s:e].astype(np.float32))
    wt_bytes = n_post * 0.5                                  # 4-bit weights
    cb_bytes = 16 * 4
    chamber_bytes = n_present * 0.625                        # 5-bit glass-box tag
    total = di_bytes + wt_bytes + cb_bytes + chamber_bytes
    bdoc = total / n_present
    print(f"\n  FOOTPRINT (full 8.8M, K={K}): di(varint) {di_bytes/1e6:.0f}MB + wt(4b) {wt_bytes/1e6:.0f}MB "
          f"+ tag {chamber_bytes/1e6:.0f}MB = {total/1e9:.3f} GB", flush=True)
    print(f"  -> {bdoc:.1f} B/doc  (vs 286.9 baseline = {286.9/bdoc:.2f}x smaller)\n", flush=True)

    # serve dev-small over the pruned index
    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels: queries.append((a[0], a[1]))
    queries = queries[:NQ]
    acc = np.zeros(n_present, np.float32)
    # quantize stored weights through the codebook (the real lossy index)
    def deq(w):
        idx = np.abs(w[:, None] - cb[None, :]).argmin(1); return cb[idx]
    for tid in tcols:
        locs, w = tcols[tid]; tcols[tid] = (locs, deq(w).astype(np.float32))

    def search(ids, qw, k=1000):
        touched = []
        order = np.argsort(-qw)[:60]
        for i in order:
            t = int(ids[i]); tc = tcols.get(t)
            if tc is None: continue
            locs, w = tc; acc[locs] += float(qw[i]) * w; touched.append(locs)
        if not touched: return np.zeros(0, np.int64)
        cand = np.unique(np.concatenate(touched)); sc = acc[cand]; acc[cand] = 0.0
        sel = np.argpartition(-sc, min(k, len(cand)-1))[:k] if len(cand) > k else np.arange(len(cand))
        return present[cand[sel[np.argsort(-sc[sel])]]]

    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0+m.BATCH], m.QUERY_ML, topk=10_000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0+m.BATCH], reps):
            qenc[qid] = rep
    mrr = rec100 = rec1000 = 0.0; lat = []
    for qid, _ in queries:
        ids, qw = qenc[qid]
        t = time.perf_counter(); top = [int(x) for x in search(ids, np.asarray(qw, np.float32), 1000)]
        lat.append((time.perf_counter()-t)*1000)
        gold = qrels[qid]
        if any(d in gold for d in top[:100]): rec100 += 1
        if any(d in gold for d in top): rec1000 += 1
        for r, d in enumerate(top[:10]):
            if d in gold: mrr += 1.0/(r+1); break
    n = len(queries); lat = np.array(lat)
    print(f"  ===== FULL 8.8M VALIDATED (K={K}, n={n} dev-small) =====", flush=True)
    print(f"    footprint     : {bdoc:.1f} B/doc  ({286.9/bdoc:.2f}x smaller than 286.9)", flush=True)
    print(f"    MRR@10        : {mrr/n:.4f}   (baseline full ~0.391)", flush=True)
    print(f"    recall@100    : {rec100/n*100:.2f}%", flush=True)
    print(f"    recall@1000   : {rec1000/n*100:.2f}%", flush=True)
    print(f"    serve latency : median {np.median(lat):.1f} ms", flush=True)

if __name__ == "__main__":
    main()
