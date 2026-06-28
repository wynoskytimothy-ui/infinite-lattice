#!/usr/bin/env python3
"""SELECTIVE champion: keep rare/discriminative terms FULL (they pin the gold), champion-cap only the
high-DF terms (long lists, low SPLADE weight = cheap to lose). Goal: hold MRR ~0.39 while cutting the
high-DF cost -> faster, footprint unchanged-or-smaller. Sweep (DFCAP gate, M cap). Same 250 q."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250
MMAX = 8192
CONFIGS = [(50000, 8192), (50000, 2048), (20000, 8192), (20000, 2048), (100000, 4096)]  # (DFCAP gate, M cap)

def main():
    t0 = time.perf_counter()
    si = m.ServedIndex()
    dfs = np.array([len(loc) for loc, _ in si.tloc], np.int64)
    # champion prefix (weight-desc, <=MMAX) only needed for high-DF terms
    champ = [None]*len(si.tloc)
    for j, (loc, w) in enumerate(si.tloc):
        if len(w) > MMAX:
            idx = np.argpartition(-w, MMAX)[:MMAX]; o = idx[np.argsort(-w[idx])]
            champ[j] = (loc[o].copy(), w[o].copy())
    print(f"  index+champ ready ({time.perf_counter()-t0:.0f}s); {len(si.term_ids):,} terms\n", flush=True)

    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO/"qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO/"queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels: queries.append((a[0], a[1]))
    queries = queries[:NQ]
    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0+m.BATCH], m.QUERY_ML, topk=10_000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0+m.BATCH], reps):
            qenc[qid] = rep

    acc = si.acc
    def search_sel(ids, qw, dfcap, M, topq=40, k=100):
        qw = np.asarray(qw, np.float32); top = np.argsort(-qw)[:topq]; touched = []
        for i in top:
            j = si.col.get(int(ids[i]))
            if j is None: continue
            if dfs[j] <= dfcap:                       # rare/discriminative -> FULL list
                cloc, cw = si.tloc[j]
            else:                                     # high-DF -> top-M champion only
                cloc, cw = champ[j]; cloc = cloc[:M]; cw = cw[:M]
            acc[cloc] += float(qw[i]) * cw; touched.append(cloc)
        if not touched: return np.zeros(0, np.uint32)
        cand = np.unique(np.concatenate(touched)); sc = acc[cand]; acc[cand] = 0.0
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        return si.present[cand[sel[np.argsort(-sc[sel])]]]

    def evalrun(dfcap, M):
        for qid, _ in queries[:5]:
            ids, qw = qenc[qid]; search_sel(ids, qw, dfcap, M)
        mrr = 0.0; rec = 0; lat = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter(); top = [int(d) for d in search_sel(ids, qw, dfcap, M)]
            lat.append((time.perf_counter()-t)*1000)
            gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold: mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        print(f"  dfcap={dfcap:<7} M={M:<6}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}", flush=True)

    print(f"  {'config':<22}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}", flush=True)
    for dfcap, M in CONFIGS:
        evalrun(dfcap, M)
    print(f"\n  baseline (full): MRR 0.391 / 88 ms / 287 B/doc.  Goal: hold ~0.39 at <88ms.", flush=True)

if __name__ == "__main__":
    main()
