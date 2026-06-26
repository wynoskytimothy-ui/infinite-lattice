#!/usr/bin/env python3
"""Use the LATTICE, not a generic heuristic: the pool = docs that share a COMPOSITE CORRELATION with
the query. A compound (upper prime) is the product of its word-primes; its doc-list is the MEET
(intersection) of the constituents' posting lists. Intersections are smaller + correlation-precise vs
the shortest-list UNION. Test on-the-fly (zero footprint change) vs the shipped search_fast, same 250 q.
If correlation-pooling wins, pre-store each doc's top composites within the 500 B/doc budget for O(1).
Captured to _serve_corr.log."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250

def inter(a, b):
    """meet of two sorted-unique local-id arrays (docs containing BOTH terms = the composite)."""
    if len(a) > len(b):
        a, b = b, a
    if len(a) == 0:
        return a
    pos = np.searchsorted(b, a)
    pc = np.minimum(pos, len(b) - 1)
    return a[b[pc] == a]

def main():
    si = m.ServedIndex()
    MARCO = m.MARCO
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    queries = queries[:NQ]
    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0+m.BATCH], m.QUERY_ML, topk=10_000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0+m.BATCH], reps):
            qenc[qid] = rep
    print(f"  {len(queries)} queries, same set for all methods\n", flush=True)

    def resolve(ids, qw, topq):
        qw = np.asarray(qw, np.float32)
        terms = []
        for i in np.argsort(-qw)[:topq]:
            j = si.col.get(int(ids[i]))
            if j is None:
                continue
            loc, w = si.tloc[j]
            terms.append((loc, w, float(qw[i])))
        terms.sort(key=lambda t: len(t[0]))
        return terms

    def refine_topk(terms, C, k):
        score = np.zeros(len(C), np.float32)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc) - 1)
            hit = loc[pc] == C; score[hit] += qweight * w[pc[hit]]
        if len(C) > k:
            sel = np.argpartition(-score, k)[:k]
        else:
            sel = np.arange(len(C))
        order = sel[np.argsort(-score[sel])]
        return si.present[C[order]]

    def search_corr(ids, qw, k=100, topq=30, n_anchor=6, add_single=True):
        terms = resolve(ids, qw, topq)
        if not terms:
            return np.zeros(0, np.uint32), 0
        anchors = terms[:n_anchor]
        parts = []
        for a in range(len(anchors)):
            for b in range(a + 1, len(anchors)):
                int_ab = inter(anchors[a][0], anchors[b][0])     # the meet = composite doc-list
                if len(int_ab):
                    parts.append(int_ab)
        if add_single:
            parts.append(anchors[0][0])                          # rarest single term = recall floor
        parts = [p for p in parts if len(p)]
        if not parts:
            return np.zeros(0, np.uint32), 0
        C = np.unique(np.concatenate(parts))
        return refine_topk(terms, C, k), len(C)

    def search_fast(ids, qw, k=100, topq=30, pool_cap=80000):   # the shipped control
        terms = resolve(ids, qw, topq)
        if not terms:
            return np.zeros(0, np.uint32), 0
        cur = []; tot = 0
        for loc, w, qweight in terms:
            if cur and tot + len(loc) > pool_cap:
                break
            cur.append(loc); tot += len(loc)
        C = np.unique(np.concatenate(cur))
        return refine_topk(terms, C, k), len(C)

    def evalrun(fn, label):
        for qid, _ in queries[:5]:
            ids, qw = qenc[qid]; fn(ids, qw)
        mrr = 0.0; rec = 0; lat = []; csz = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter()
            top, nc = fn(ids, qw)
            lat.append((time.perf_counter()-t)*1000); csz.append(nc)
            top = [int(d) for d in top]; gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        print(f"  {label:<28}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}{int(np.median(csz)):>10}", flush=True)

    print(f"  {'method':<28}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'med |C|':>10}", flush=True)
    evalrun(lambda i,w: search_fast(i, w), "shipped: rarest-union 80k")
    for na in (4, 6, 8):
        evalrun(lambda i,w,na=na: search_corr(i, w, n_anchor=na, add_single=True), f"composite-meet n_anchor={na}")
    evalrun(lambda i,w: search_corr(i, w, n_anchor=8, add_single=False), "composite-meet n=8 pairs-only")
    print(f"\n  footprint 286.9 B/doc unchanged (on-the-fly meet; pre-storing composites = O(1), <500 B/doc).", flush=True)

if __name__ == "__main__":
    main()
