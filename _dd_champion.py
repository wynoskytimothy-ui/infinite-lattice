#!/usr/bin/env python3
"""Champion lists (Timothy's UltraFast technique) on the native SPLADE-on-lattice MARCO index.
Keep only each term's top-M docs by SPLADE weight. This (a) SHRINKS the index (fewer postings ->
smaller footprint), (b) makes the serve faster (less work per term), at a small accuracy cost that
should be ~0 if M is large enough. Goal: a config that is FASTER and SMALLER while holding MRR ~0.39.
Sweep M, measure MRR@10 / recall@100 / median latency / postings -> B/doc. Same 250 q. Captured."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250
MS = [128, 256, 512, 1024, 2048]
MMAX = max(MS)
B_PER_POST = 2_536_480_416 / 1_059_501_065     # real index bytes per posting (FOR-packed)
NDOCS = 8_841_823

def main():
    t0 = time.perf_counter()
    si = m.ServedIndex()
    print(f"  index loaded ({time.perf_counter()-t0:.0f}s); {len(si.term_ids):,} terms, {si.n_post:,} postings", flush=True)

    # precompute, per term, its docs+weights in WEIGHT-DESC order, truncated to MMAX (champion prefix)
    t1 = time.perf_counter()
    champ = []; dfs = np.empty(len(si.tloc), np.int64)
    for j, (loc, w) in enumerate(si.tloc):
        dfs[j] = len(loc)
        if len(w) > MMAX:
            idx = np.argpartition(-w, MMAX)[:MMAX]
            o = idx[np.argsort(-w[idx])]
        else:
            o = np.argsort(-w)
        champ.append((loc[o].copy(), w[o].copy()))
    print(f"  champion prefixes built ({time.perf_counter()-t1:.0f}s)\n", flush=True)

    def postings_at(M):
        return int(np.minimum(dfs, M).sum())

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
    print(f"  {len(queries)} queries encoded\n", flush=True)

    acc = si.acc
    def search_champ(ids, qw, M, topq=40, k=100):
        qw = np.asarray(qw, np.float32)
        top = np.argsort(-qw)[:topq]
        touched = []
        for i in top:
            j = si.col.get(int(ids[i]))
            if j is None: continue
            cloc, cw = champ[j]
            if len(cloc) > M:
                cloc = cloc[:M]; cw = cw[:M]
            acc[cloc] += float(qw[i]) * cw
            touched.append(cloc)
        if not touched:
            return np.zeros(0, np.uint32)
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]; acc[cand] = 0.0
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return si.present[cand[order]]

    def evalrun(M):
        for qid, _ in queries[:5]:
            ids, qw = qenc[qid]; search_champ(ids, qw, M)
        mrr = 0.0; rec = 0; lat = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter(); top = [int(d) for d in search_champ(ids, qw, M)]
            lat.append((time.perf_counter()-t)*1000)
            gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold: mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat); post = postings_at(M)
        bpd = 287.0 * post / 1_059_501_065
        print(f"  M={M:<6}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}"
              f"{post/1e6:>11.1f}M{bpd:>9.1f}", flush=True)

    print(f"  {'champion M':<8}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'postings':>12}{'B/doc':>9}", flush=True)
    for M in MS:
        evalrun(M)
    print(f"\n  baseline (full, no champion): MRR 0.391 / 88 ms / 1059.5M postings / 287 B/doc (from MEASUREMENTS.md)", flush=True)
    print(f"  GOAL: an M that is FASTER (<88ms) and SMALLER (<287 B/doc) at MRR ~0.39.", flush=True)

if __name__ == "__main__":
    main()
