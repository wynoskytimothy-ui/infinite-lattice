#!/usr/bin/env python3
"""DEFINITIVE serve-speed shootout on the full 8.8M SPLADE index, SAME 250 cached queries, apples-to-apples.
Benches four serve paths and reports MRR@10 / recall@100 / median+p90 ms each:
  - full scatter (exact)      : touch every posting list, np.unique, argpartition  (the naive exact ref)
  - WAND (numba, exact)       : max-weight skip traversal -> identical top-k as full scatter
  - search_fast (approx)      : the lattice's OWN rarest-address candidate pooling
  - search_corr (approx)      : composite-meet anchor pooling
Answers: what is the fastest serve, is it exact, and does textbook WAND beat the lattice's pooled path?"""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import marco_splade_native as m
from _dd_wand import wand   # reuse the verified numba WAND (module sets WORK=full, defs only)

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250


def main():
    si = m.ServedIndex()
    nT = len(si.tloc)
    lens = np.array([len(loc) for loc, _ in si.tloc], np.int64)
    toffs = np.zeros(nT + 1, np.int64); toffs[1:] = np.cumsum(lens)
    total = int(toffs[-1])
    all_docs = np.empty(total, np.int32); all_wts = np.empty(total, np.float32)
    tmax = np.zeros(nT, np.float32)
    for j, (loc, w) in enumerate(si.tloc):
        s, e = toffs[j], toffs[j + 1]
        all_docs[s:e] = loc.astype(np.int32); all_wts[s:e] = w
        if len(w):
            tmax[j] = w.max()
    present = si.present; acc = si.acc; col = si.col
    print(f"  flat: {total:,} postings, {si.n_docs:,} docs\n", flush=True)

    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    raw = pickle.load(open(Path(r"C:\Users\wynos\trng\marco_data\splade_native") / "_dd_qenc_cache.pkl", "rb"))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels and a[0] in raw:
                queries.append(a[0])
    queries = queries[:NQ]
    # two encodings: vocab-id+weight (for si.* methods) and col+weight (for WAND)
    venc = {}; cenc = {}
    for qid in queries:
        ids, w = raw[qid]
        vids = []; vws = []; cols = []; cws = []
        for t, ww in zip(ids.tolist(), w.tolist()):
            fw = float(ww) * m.QSCALE
            vids.append(int(t)); vws.append(fw)
            j = col.get(int(t))
            if j is not None:
                cols.append(j); cws.append(fw)
        venc[qid] = (np.array(vids, np.int64), np.array(vws, np.float32))
        cenc[qid] = (np.array(cols, np.int64), np.array(cws, np.float32))
    print(f"  {len(queries)} queries (cached, no GPU)\n", flush=True)

    def ref_full(qid, k=100):
        c, q = cenc[qid]; touched = []
        for i in range(len(c)):
            j = int(c[i]); s, e = toffs[j], toffs[j + 1]
            loc = all_docs[s:e]; acc[loc] += q[i] * all_wts[s:e]; touched.append(loc)
        if not touched:
            return np.zeros(0, np.int64)
        cand = np.unique(np.concatenate(touched)); sc = acc[cand]; acc[cand] = 0.0
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return present[cand[order].astype(np.int64)]   # GLOBAL ids

    def run_wand(qid, k=100):
        c, q = cenc[qid]
        loc = wand(c, q, all_docs, all_wts, toffs, tmax, k)
        return present[loc.astype(np.int64)]

    def run_fast(qid, k=100):
        v, q = venc[qid]
        ids, _ = si.search_fast(v, q, k=k)
        return ids

    def run_corr(qid, k=100):
        v, q = venc[qid]
        ids, _ = si.search_corr(v, q, k=k)
        return ids

    methods = [("full scatter (exact)", ref_full), ("WAND (numba, exact)", run_wand),
               ("search_fast (approx)", run_fast), ("search_corr (approx)", run_corr)]

    print(f"  {'serve':<24}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'p99 ms':>9}", flush=True)
    for label, fn in methods:
        for qid in queries[:5]:   # warm
            fn(qid, 100)
        mrr = 0.0; rec = 0; lat = []
        for qid in queries:
            t = time.perf_counter(); ids = fn(qid, 100); lat.append((time.perf_counter() - t) * 1000)
            top = [int(d) for d in ids[:100]]; gold = qrels[qid]
            if any(d in gold for d in top):
                rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0 / (r + 1); break
        n = len(queries); lat = np.array(lat)
        print(f"  {label:<24}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.1f}"
              f"{np.percentile(lat,90):>9.1f}{np.percentile(lat,99):>9.1f}", flush=True)

    print(f"\n  footprint UNCHANGED (286.9 B/doc raw; codec ~160-195 B/doc). Speed only.", flush=True)


if __name__ == "__main__":
    main()
