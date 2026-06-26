#!/usr/bin/env python3
"""Curated composite-meet, NO rebuild: a stored composite = docs where BOTH terms have HIGH weight
(the doc is genuinely about the compound). The index already holds per-posting weights, so take each
discriminative term's top-T docs by weight, then meet (intersect) those -> small, curated, precise pool.
'machine'(top-weight) MEET 'learning'(top-weight) = docs about ML, not every co-mention. Footprint
286.9 B/doc unchanged. Same 250 q vs shipped (0.3909/92ms) and full scatter (0.3977/3144ms)."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250

def inter(a, b):
    if len(a) > len(b): a, b = b, a
    if len(a) == 0: return a
    pos = np.searchsorted(b, a); pc = np.minimum(pos, len(b)-1)
    return a[b[pc] == a]

def main():
    si = m.ServedIndex()
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
    print(f"  {len(queries)} queries\n", flush=True)

    def resolve(ids, qw, topq=30):
        qw = np.asarray(qw, np.float32); terms = []
        for i in np.argsort(-qw)[:topq]:
            j = si.col.get(int(ids[i]))
            if j is None: continue
            loc, w = si.tloc[j]; terms.append((loc, w, float(qw[i])))
        terms.sort(key=lambda t: len(t[0]))
        return terms

    def refine(terms, C, k=100):
        score = np.zeros(len(C), np.float32)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc)-1)
            hit = loc[pc] == C; score[hit] += qweight*w[pc[hit]]
        sel = np.argpartition(-score, k)[:k] if len(C) > k else np.arange(len(C))
        return si.present[C[sel[np.argsort(-score[sel])]]]

    def search_corr3(ids, qw, top_t=2000, n_anchor=8, df_cap=200000, min_pool=400, fb_cap=80000, k=100):
        terms = resolve(ids, qw)
        if not terms: return np.zeros(0, np.uint32), 0
        anchors = [t for t in terms if len(t[0]) <= df_cap][:n_anchor]
        hi = []
        for loc, w, qweight in anchors:
            if len(loc) > top_t:
                idx = np.argpartition(-w, top_t)[:top_t]   # this term's top-weight docs (curated)
                hi.append(np.sort(loc[idx]))
            else:
                hi.append(loc)
        parts = []
        for a in range(len(hi)):
            for b in range(a+1, len(hi)):
                ab = inter(hi[a], hi[b])
                if len(ab): parts.append(ab)
        C = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        if len(C) < min_pool:
            cur = []; tot = 0
            for loc, w, qweight in terms:
                if cur and tot+len(loc) > fb_cap: break
                cur.append(loc); tot += len(loc)
            C = np.unique(np.concatenate(([C]+cur) if len(C) else cur))
        return refine(terms, C, k), len(C)

    def search_fast(ids, qw, pool_cap=80000, k=100):
        terms = resolve(ids, qw)
        if not terms: return np.zeros(0, np.uint32), 0
        cur = []; tot = 0
        for loc, w, qweight in terms:
            if cur and tot+len(loc) > pool_cap: break
            cur.append(loc); tot += len(loc)
        C = np.unique(np.concatenate(cur))
        return refine(terms, C, k), len(C)

    def evalrun(fn, label):
        for qid, _ in queries[:5]:
            ids, qw = qenc[qid]; fn(ids, qw)
        mrr = 0.0; rec = 0; lat = []; csz = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter(); top, nc = fn(ids, qw); lat.append((time.perf_counter()-t)*1000); csz.append(nc)
            top = [int(d) for d in top]; gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold: mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        print(f"  {label:<30}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}{int(np.median(csz)):>9}", flush=True)

    print(f"  {'method':<30}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'med|C|':>9}", flush=True)
    evalrun(lambda i,w: search_fast(i, w), "shipped rarest-union 80k")
    for tt in (1000, 2000, 4000):
        evalrun(lambda i,w,tt=tt: search_corr3(i, w, top_t=tt), f"curated-meet top_t={tt}")
    print(f"\n  footprint 286.9 B/doc unchanged (curated meet uses stored weights; no rebuild).", flush=True)

if __name__ == "__main__":
    main()
