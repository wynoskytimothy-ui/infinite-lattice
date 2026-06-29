#!/usr/bin/env python3
"""ZERO-SHOT serve speedup at IDENTICAL accuracy: replace the score-phase (112 ms, the bottleneck) with a
numba two-pointer MERGE scorer. The lattice keeps postings AND the candidate set sorted, so scoring is a
linear merge (O(|C|+post)) not a searchsorted scatter (O(|C|·log)). Same candidates, same scores => same
MRR/recall, lower latency. Also numba the meet-phase merges. Full 8.8M, same 250 cached queries."""
import os, sys, time, pickle
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
from collections import defaultdict
from numba import njit
import marco_splade_native as m

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250


@njit(cache=True)
def merge_meet(x, y):
    # intersection of two sorted int arrays (the meet), two-pointer
    out = np.empty(min(len(x), len(y)), np.int64); k = 0; i = 0; j = 0
    while i < len(x) and j < len(y):
        if x[i] == y[j]: out[k] = x[i]; k += 1; i += 1; j += 1
        elif x[i] < y[j]: i += 1
        else: j += 1
    return out[:k]


@njit(cache=True)
def merge_score(C, locs, offs, wts, qws, nt):
    # for each term, two-pointer merge its sorted postings with sorted C, accumulate qw*w
    out = np.zeros(len(C), np.float32)
    for t in range(nt):
        s = offs[t]; e = offs[t + 1]; qw = qws[t]; i = 0; j = s
        while i < len(C) and j < e:
            lj = locs[j]
            if lj == C[i]: out[i] += qw * wts[j]; i += 1; j += 1
            elif lj < C[i]: j += 1
            else: i += 1
    return out


def main():
    si = m.ServedIndex()
    col, tloc, present = si.col, si.tloc, si.present
    print(f"  loaded {si.n_docs:,} docs\n", flush=True)
    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(int(p[2]))
    raw = pickle.load(open(Path(os.environ["WORK"].replace("_full", "")) / "_dd_qenc_cache.pkl", "rb"))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels and a[0] in raw: queries.append(a[0])
    queries = queries[:NQ]
    venc = {}
    for qid in queries:
        ids, w = raw[qid]
        venc[qid] = (np.array(ids.tolist(), np.int64), np.array([x * m.QSCALE for x in w.tolist()], np.float32))
    print(f"  {len(queries)} queries\n", flush=True)

    def prep(qid, topq=30, n_anchor=6):
        qids, qw = venc[qid]; qw = np.asarray(qw, np.float32); top = np.argsort(-qw)[:topq]
        terms = []
        for i in top:
            j = col.get(int(qids[i]))
            if j is None: continue
            loc, w = tloc[j]; terms.append((loc.astype(np.int64), w.astype(np.float32), np.float32(qw[i])))
        terms.sort(key=lambda t: len(t[0]))
        return terms

    def serve_numpy(qid, k=100, n_anchor=6):   # baseline search_corr (numpy)
        terms = prep(qid)
        if not terms: return np.zeros(0, np.uint32)
        anchors = terms[:n_anchor]; parts = []
        for a in range(len(anchors)):
            for b in range(a + 1, len(anchors)):
                x, y = anchors[a][0], anchors[b][0]
                if len(x) > len(y): x, y = y, x
                pos = np.searchsorted(y, x); pc = np.minimum(pos, len(y) - 1); ab = x[y[pc] == x]
                if len(ab): parts.append(ab)
        parts.append(anchors[0][0]); C = np.unique(np.concatenate(parts))
        sc = np.zeros(len(C), np.float32)
        for loc, w, qw in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc) - 1); hit = loc[pc] == C
            sc[hit] += qw * w[pc[hit]]
        sel = np.argpartition(-sc, k)[:k] if len(C) > k else np.arange(len(C))
        return present[C[sel[np.argsort(-sc[sel])]]]

    def serve_numba(qid, k=100, n_anchor=6):   # numba merge meet + merge score (identical math)
        terms = prep(qid)
        if not terms: return np.zeros(0, np.uint32)
        anchors = terms[:n_anchor]; parts = []
        for a in range(len(anchors)):
            for b in range(a + 1, len(anchors)):
                ab = merge_meet(anchors[a][0], anchors[b][0])
                if len(ab): parts.append(ab)
        parts.append(anchors[0][0]); C = np.unique(np.concatenate(parts))
        nt = len(terms)
        lens = np.array([len(t[0]) for t in terms], np.int64); offs = np.zeros(nt + 1, np.int64)
        offs[1:] = np.cumsum(lens)
        locs = np.concatenate([t[0] for t in terms]); wts = np.concatenate([t[1] for t in terms])
        qws = np.array([t[2] for t in terms], np.float32)
        sc = merge_score(C, locs, offs, wts, qws, nt)
        sel = np.argpartition(-sc, k)[:k] if len(C) > k else np.arange(len(C))
        return present[C[sel[np.argsort(-sc[sel])]]]

    def bench(fn, label):
        for qid in queries[:5]: fn(qid)
        mrr = rec = 0.0; lat = []
        for qid in queries:
            t = time.perf_counter(); ids = fn(qid); lat.append((time.perf_counter() - t) * 1000)
            top = [int(d) for d in ids[:100]]; gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold: mrr += 1.0 / (r + 1); break
        lat = np.array(lat); n = len(queries)
        print(f"  {label:<24}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.1f}{np.percentile(lat,90):>9.1f}{np.percentile(lat,99):>9.1f}", flush=True)

    print(f"  {'serve':<24}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'p99 ms':>9}", flush=True)
    bench(serve_numpy, "corr (numpy baseline)")
    bench(serve_numba, "corr (numba merge)")
    print(f"\n  same candidates + same scores => identical accuracy; the merge kernel cuts the score-phase.", flush=True)


if __name__ == "__main__":
    main()
