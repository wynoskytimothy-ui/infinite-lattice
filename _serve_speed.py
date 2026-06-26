#!/usr/bin/env python3
"""Speed WITHOUT touching the index (footprint stays 286.9 B/doc) or the encoder (accuracy stays).
The 3.2 s/query came from scoring ALL ~hundreds of SPLADE query terms, incl. high-DF ones with
million-long posting lists. Lever 1: prune the query to its top-K terms by weight. Lever 2 (opt):
reset-only-touched accumulator instead of a 35 MB memset per query. Sweep K, measure MRR@10 +
recall@100 + median search latency on the SAME query set -> find the smallest K that holds MRR.

Footprint is unchanged by construction (query-side only). Captured to _serve_speed.log."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 300
KS = [10, 20, 30, 50, 75, 100, 150, 300, 1000]

def main():
    t0 = time.perf_counter()
    si = m.ServedIndex()
    print(f"  index loaded ({time.perf_counter()-t0:.0f}s); {len(si.term_ids):,} terms, {si.n_post:,} postings", flush=True)

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
    print(f"  {len(queries):,} dev-small queries; encoding (full SPLADE) once...", flush=True)

    # encode once, keep ALL terms; store (ids, qw) sorted by weight DESC for easy top-K slicing
    qenc = {}
    qtexts = [qt for _, qt in queries]
    BATCH = m.BATCH
    for b0 in range(0, len(qtexts), BATCH):
        reps = m.splade_sparse(qtexts[b0:b0+BATCH], m.QUERY_ML, topk=10_000, minw=m.MINW)
        for (qid, _), (ids, w) in zip(queries[b0:b0+BATCH], reps):
            order = np.argsort(-w.astype(np.float32))     # weight desc
            qenc[qid] = (ids[order], w[order].astype(np.float32))
    nterms = [len(qenc[q][0]) for q, _ in queries]
    print(f"  query term counts: median {int(np.median(nterms))}, max {max(nterms)}\n", flush=True)

    acc = si.acc
    def search_pruned(ids, qw, K, k=100):
        touched = []
        n = min(K, len(ids))
        for i in range(n):
            j = si.col.get(int(ids[i]))
            if j is None:
                continue
            loc, w = si.tloc[j]
            acc[loc] += float(qw[i]) * w
            touched.append(loc)
        if not touched:
            return np.zeros(0, np.uint32)
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        acc[cand] = 0.0                                   # reset ONLY touched (no 35MB memset)
        if len(cand) > k:
            sel = np.argpartition(-sc, k)[:k]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return si.present[cand[order]]

    # warm
    for qid, _ in queries[:5]:
        ids, qw = qenc[qid]; search_pruned(ids, qw, 50)

    print(f"  {'K (top query terms)':<22}{'MRR@10':>9}{'recall@100':>12}{'median ms':>11}{'p90 ms':>9}", flush=True)
    baseline_mrr = None
    for K in KS:
        mrr = 0.0; rec = 0; lat = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter()
            top = [int(d) for d in search_pruned(ids, qw, K)]
            lat.append((time.perf_counter()-t)*1000)
            gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        mrr /= n; rec = rec/n*100
        if baseline_mrr is None or K == KS[-1]:
            baseline_mrr = mrr if K == KS[-1] else baseline_mrr
        print(f"  K={K:<20}{mrr:>9.4f}{rec:>11.2f}%{np.median(lat):>11.2f}{np.percentile(lat,90):>9.2f}", flush=True)

    print(f"\n  reference (unpruned, _serve_sample.log): MRR@10 0.3989, median 3234 ms (200-q)", flush=True)
    print(f"  footprint UNCHANGED at 286.9 B/doc (query-side pruning only).", flush=True)

if __name__ == "__main__":
    main()
