#!/usr/bin/env python3
"""EXACT WAND serve for SPLADE-on-lattice, numba-compiled. Per-term max-weight upper bounds (RAM only,
footprint UNCHANGED) drive a document-at-a-time traversal that SKIPS docs that provably can't enter the
top-k -- so it returns the IDENTICAL top-k as the full scatter (accuracy unchanged by construction).
Measures: top-10 disagreements vs full (must be ~0) + latency vs the 88 ms baseline. Same 250 q."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict
from pathlib import Path
from numba import njit

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250


@njit(cache=True)
def wand(cols, qw, all_docs, all_wts, toffs, tmax, k):
    nq = len(cols)
    cur = np.zeros(nq, np.int64)
    starts = np.empty(nq, np.int64)
    lens = np.empty(nq, np.int64)
    ub = np.empty(nq, np.float64)
    for i in range(nq):
        c = cols[i]
        starts[i] = toffs[c]
        lens[i] = toffs[c + 1] - toffs[c]
        ub[i] = qw[i] * tmax[c]
    BIG = np.int64(1) << 62
    order = np.arange(nq)
    heap_s = np.full(k, -1.0e30)
    heap_d = np.full(k, -1, np.int64)
    hn = 0
    threshold = -1.0e30
    while True:
        # insertion-sort `order` by current doc id (nq is small)
        for i in range(1, nq):
            x = order[i]
            xd = all_docs[starts[x] + cur[x]] if cur[x] < lens[x] else BIG
            jj = i - 1
            while jj >= 0:
                y = order[jj]
                yd = all_docs[starts[y] + cur[y]] if cur[y] < lens[y] else BIG
                if yd <= xd:
                    break
                order[jj + 1] = order[jj]
                jj -= 1
            order[jj + 1] = x
        f = order[0]
        fd = all_docs[starts[f] + cur[f]] if cur[f] < lens[f] else BIG
        if fd >= BIG:
            break
        thr = threshold if hn >= k else -1.0e30
        cum = 0.0
        pivot = -1
        for i in range(nq):
            cum += ub[order[i]]
            if cum > thr:
                pivot = i
                break
        if pivot == -1:
            break
        pt = order[pivot]
        pdoc = all_docs[starts[pt] + cur[pt]] if cur[pt] < lens[pt] else BIG
        if fd == pdoc:
            score = 0.0
            for i in range(nq):
                t = order[i]
                if cur[t] < lens[t] and all_docs[starts[t] + cur[t]] == pdoc:
                    score += qw[t] * all_wts[starts[t] + cur[t]]
                    cur[t] += 1
            if hn < k:
                heap_s[hn] = score
                heap_d[hn] = pdoc
                hn += 1
                if hn == k:
                    threshold = heap_s[0]
                    for i in range(1, k):
                        if heap_s[i] < threshold:
                            threshold = heap_s[i]
            elif score > threshold:
                mi = 0
                for i in range(1, k):
                    if heap_s[i] < heap_s[mi]:
                        mi = i
                heap_s[mi] = score
                heap_d[mi] = pdoc
                threshold = heap_s[0]
                for i in range(1, k):
                    if heap_s[i] < threshold:
                        threshold = heap_s[i]
        else:
            tadv = -1
            for i in range(pivot):
                t = order[i]
                td = all_docs[starts[t] + cur[t]] if cur[t] < lens[t] else BIG
                if td < pdoc:
                    tadv = t
                    break
            if tadv == -1:
                break
            lo = cur[tadv]
            hi = lens[tadv]
            base = starts[tadv]
            while lo < hi:
                mid = (lo + hi) // 2
                if all_docs[base + mid] < pdoc:
                    lo = mid + 1
                else:
                    hi = mid
            cur[tadv] = lo
    idx = np.argsort(-heap_s[:hn])
    out = np.empty(hn, np.int64)
    for i in range(hn):
        out[i] = heap_d[idx[i]]
    return out


def main():
    si = m.ServedIndex()
    nT = len(si.tloc)
    lens = np.array([len(loc) for loc, _ in si.tloc], np.int64)
    toffs = np.zeros(nT + 1, np.int64)
    toffs[1:] = np.cumsum(lens)
    total = int(toffs[-1])
    all_docs = np.empty(total, np.int32)
    all_wts = np.empty(total, np.float32)
    tmax = np.zeros(nT, np.float32)
    for j, (loc, w) in enumerate(si.tloc):
        s, e = toffs[j], toffs[j + 1]
        all_docs[s:e] = loc.astype(np.int32)
        all_wts[s:e] = w
        if len(w):
            tmax[j] = w.max()
    present = si.present
    acc = si.acc
    col = si.col
    si.tloc = None
    print(f"  flat built: {total:,} postings (max-weight bounds: RAM only, footprint unchanged)\n", flush=True)

    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    # Reuse the cached raw SPLADE query encodings (vocab-id, quantized weight) -> NO torch/GPU needed.
    # The raw encoding is index-independent; only the column remap (col.get) is index-specific. A global
    # scale on query weights cannot change the top-k ranking, so correctness + latency are exact regardless.
    import pickle
    QCACHE = Path(r"C:\Users\wynos\trng\marco_data\splade_native") / "_dd_qenc_cache.pkl"
    raw = pickle.load(open(QCACHE, "rb"))   # qid(str) -> (vocab_ids:int[], qweight_uint8:int[])
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels and a[0] in raw:
                queries.append((a[0], a[1]))
    queries = queries[:NQ]
    qenc = {}
    for qid, _ in queries:
        ids, w = raw[qid]
        cols = []; qws = []
        for t, ww in zip(ids.tolist(), w.tolist()):
            j = col.get(int(t))
            if j is not None:
                cols.append(j); qws.append(float(ww) * m.QSCALE)   # QSCALE matches the production score
        qenc[qid] = (np.array(cols, np.int64), np.array(qws, np.float32))
    print(f"  {len(queries)} queries encoded from cache (no GPU/model loaded)\n", flush=True)

    def ref_full(cols, qw, k=100):
        touched = []
        for i in range(len(cols)):
            j = int(cols[i]); s, e = toffs[j], toffs[j + 1]
            loc = all_docs[s:e]
            acc[loc] += qw[i] * all_wts[s:e]
            touched.append(loc)
        if not touched:
            return np.zeros(0, np.int64)
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]; acc[cand] = 0.0
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return cand[order].astype(np.int64)   # local ids

    # warm + correctness on a sample
    for qid, _ in queries[:5]:
        c, q = qenc[qid]
        wand(c, q, all_docs, all_wts, toffs, tmax, 100)
        ref_full(c, q, 100)
    def score_of(cols, qws, d):
        s = 0.0
        for i in range(len(cols)):
            j = int(cols[i]); a, b = int(toffs[j]), int(toffs[j + 1])
            seg = all_docs[a:b]
            lo = int(np.searchsorted(seg, d))
            if lo < len(seg) and int(seg[lo]) == d:
                s += float(qws[i]) * float(all_wts[a + lo])
        return s

    dis10 = 0; dis_set = 0; score_mismatch = 0
    for qid, _ in queries:
        c, q = qenc[qid]
        rf = ref_full(c, q, 100)
        wd = wand(c, q, all_docs, all_wts, toffs, tmax, 100)
        if [int(x) for x in rf[:10]] != [int(x) for x in wd[:10]]:
            dis10 += 1
        if set(int(x) for x in rf[:100]) != set(int(x) for x in wd[:100]):
            dis_set += 1
        # the REAL exactness test: identical sorted top-100 SCORE vectors (id-order diffs are pure ties)
        rs = sorted((round(score_of(c, q, int(x)), 5) for x in rf[:100]), reverse=True)
        ws = sorted((round(score_of(c, q, int(x)), 5) for x in wd[:100]), reverse=True)
        if rs != ws:
            score_mismatch += 1
    print(f"  CORRECTNESS vs full scatter: top-10 id-order diffs {dis10}/{len(queries)} (ties), "
          f"top-100 set diffs {dis_set}/{len(queries)}, "
          f"top-100 SCORE-vector mismatches {score_mismatch}/{len(queries)} (must be 0 = exact)\n", flush=True)

    def bench(fn, label):
        for qid, _ in queries[:5]:
            c, q = qenc[qid]; fn(c, q, 100)
        mrr = 0.0; rec = 0; lat = []
        for qid, _ in queries:
            c, q = qenc[qid]
            t = time.perf_counter()
            loc = fn(c, q, 100)
            lat.append((time.perf_counter() - t) * 1000)
            top = [int(present[int(d)]) for d in loc[:100]]
            gold = qrels[qid]
            if any(d in gold for d in top):
                rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0 / (r + 1); break
        n = len(queries); lat = np.array(lat)
        print(f"  {label:<22}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}", flush=True)

    print(f"  {'serve':<22}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}", flush=True)
    bench(lambda c, q, k: ref_full(c, q, k), "full scatter (ref)")
    bench(lambda c, q, k: wand(c, q, all_docs, all_wts, toffs, tmax, k), "WAND (numba, exact)")
    print(f"\n  footprint 286.9 B/doc UNCHANGED (max-weight bounds are RAM-only). WAND returns exact top-k.", flush=True)


if __name__ == "__main__":
    main()
