#!/usr/bin/env python3
"""FAST SPLADE serve, index UNTOUCHED (286.9 B/doc) and encoder UNTOUCHED (accuracy held).
The 3.2 s came from scatter-adding the FULL posting lists of a few high-DF query terms. Fix =
rarest-address candidate pooling: the SHORT (discriminative) query-term posting lists build a
small candidate set C; then for EVERY query term we searchsorted C against its (sorted) posting
list -- O(|C|*log|posting|), never O(|posting|). So a million-long common-term list costs ~|C|
lookups, not a million scatter-adds. Sweep (TOPQ, POOL_CAP) vs the 0.3989 / 3234 ms baseline.
Captured to _serve_fast.log."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 300
CONFIGS = [(20, 20000), (30, 40000), (30, 80000), (45, 80000), (45, 150000)]   # (TOPQ, POOL_CAP)

def main():
    t0 = time.perf_counter()
    si = m.ServedIndex()
    # per-term posting length (for choosing discriminative seeds) — from the decoded loc arrays
    tlen = np.array([len(loc) for loc, _ in si.tloc], dtype=np.int64)
    print(f"  index loaded ({time.perf_counter()-t0:.0f}s); {len(si.term_ids):,} terms", flush=True)

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
        for (qid, _), (ids, w) in zip(queries[b0:b0+m.BATCH], reps):
            order = np.argsort(-w.astype(np.float32))
            qenc[qid] = (ids[order], w[order].astype(np.float32))
    print(f"  {len(queries):,} queries encoded\n", flush=True)

    def search_fast(ids, qw, TOPQ, POOL_CAP, k=100):
        # resolve top-TOPQ query terms to (loc, w, qweight), drop OOV
        terms = []
        for i in range(min(TOPQ, len(ids))):
            j = si.col.get(int(ids[i]))
            if j is None:
                continue
            loc, w = si.tloc[j]
            terms.append((loc, w, float(qw[i])))
        if not terms:
            return np.zeros(0, np.uint32), 0
        # candidate pool C = union of the SHORTEST posting lists until POOL_CAP
        terms.sort(key=lambda t: len(t[0]))
        parts = []; tot = 0
        for loc, w, qweight in terms:
            if parts and tot + len(loc) > POOL_CAP:
                break
            parts.append(loc); tot += len(loc)
        C = np.unique(np.concatenate(parts))                  # sorted candidate local ids
        score = np.zeros(len(C), np.float32)
        # refine: every term contributes via searchsorted (small C vs long sorted loc)
        for loc, w, qweight in terms:
            pos = np.searchsorted(loc, C)
            pc = np.minimum(pos, len(loc) - 1)
            hit = loc[pc] == C
            score[hit] += qweight * w[pc[hit]]
        if len(C) > k:
            sel = np.argpartition(-score, k)[:k]
        else:
            sel = np.arange(len(C))
        order = sel[np.argsort(-score[sel])]
        return si.present[C[order]], len(C)

    # warm
    for qid, _ in queries[:5]:
        ids, qw = qenc[qid]; search_fast(ids, qw, 30, 40000)

    print(f"  {'config (TOPQ,POOL)':<22}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'med |C|':>10}", flush=True)
    for TOPQ, POOL_CAP in CONFIGS:
        mrr = 0.0; rec = 0; lat = []; csz = []
        for qid, _ in queries:
            ids, qw = qenc[qid]
            t = time.perf_counter()
            top, nc = search_fast(ids, qw, TOPQ, POOL_CAP)
            lat.append((time.perf_counter()-t)*1000); csz.append(nc)
            top = [int(d) for d in top]
            gold = qrels[qid]
            if any(d in gold for d in top): rec += 1
            for r, d in enumerate(top[:10]):
                if d in gold:
                    mrr += 1.0/(r+1); break
        n = len(queries); lat = np.array(lat)
        print(f"  TOPQ={TOPQ:<3} POOL={POOL_CAP:<9}{mrr/n:>9.4f}{rec/n*100:>11.2f}%"
              f"{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}{int(np.median(csz)):>10}", flush=True)

    print(f"\n  baseline (full scatter, _serve_sample.log): MRR 0.3989, median 3234 ms, footprint 286.9 B/doc", flush=True)
    print(f"  footprint UNCHANGED (candidate pooling is query-side; index not modified).", flush=True)

if __name__ == "__main__":
    main()
