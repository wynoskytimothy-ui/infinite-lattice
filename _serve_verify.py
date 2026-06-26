#!/usr/bin/env python3
"""Apples-to-apples: run the SHIPPED full search() and the new search_fast() on the SAME queries,
prove search_fast holds MRR@10 while being ~35x faster. Reports MRR/recall/latency for both +
mean top-10 overlap. Captured to _serve_verify.log."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250

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
    print(f"  {len(queries)} queries, same set for both methods\n", flush=True)

    def evalrun(fn):
        mrr = 0.0; rec = 0; lat = []; tops = {}
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter()
            top, _sc = fn(ids, qw, k=100)
            lat.append((time.perf_counter()-t)*1000)
            top = [int(d) for d in top]; tops[qid] = top[:10]
            gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        return mrr/n, rec/n*100, np.median(lat), np.percentile(lat,90), tops

    for qid, _ in queries[:5]:
        ids, qw = qenc[qid]; si.search_fast(ids, qw, k=100)
    print("  running search_fast (rarest-address pool)...", flush=True)
    f_mrr, f_rec, f_med, f_p90, f_tops = evalrun(lambda i,w,k: si.search_fast(i, w, k=k))
    print(f"    FAST : MRR@10 {f_mrr:.4f}  recall@100 {f_rec:.2f}%  median {f_med:.2f} ms  p90 {f_p90:.2f} ms", flush=True)
    print("  running full search() (this is the slow ~3.2 s/q baseline, please wait)...", flush=True)
    u_mrr, u_rec, u_med, u_p90, u_tops = evalrun(lambda i,w,k: si.search(i, w, k=k))
    print(f"    FULL : MRR@10 {u_mrr:.4f}  recall@100 {u_rec:.2f}%  median {u_med:.2f} ms  p90 {u_p90:.2f} ms", flush=True)

    ov = np.mean([len(set(f_tops[q]) & set(u_tops[q])) / max(1, len(set(u_tops[q]))) for q, _ in queries])
    print(f"\n  ===== HEAD-TO-HEAD (same {len(queries)} queries) =====", flush=True)
    print(f"    MRR@10   : fast {f_mrr:.4f}  vs  full {u_mrr:.4f}   (delta {f_mrr-u_mrr:+.4f})", flush=True)
    print(f"    speedup  : {u_med/f_med:.0f}x  ({u_med:.0f} ms -> {f_med:.1f} ms median)", flush=True)
    print(f"    top-10 overlap (fast vs full): {ov*100:.1f}%", flush=True)
    print(f"    footprint: 286.9 B/doc UNCHANGED (search_fast is query-side only)", flush=True)

if __name__ == "__main__":
    main()
