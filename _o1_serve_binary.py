#!/usr/bin/env python3
"""THE BINARY READER (Timothy's geometry) for candidate generation — measured on full 8.8M.
His correction: don't SEARCH pairwise meets; the correlated docs are the dots where MULTIPLE query-term
spines light up at once. Read them by COUNTING intersection multiplicity, not pairwise searchsorted.
  baseline : pairwise searchsorted meets (current meet-phase) + numba merge score
  binary   : count how many anchor-spines hit each doc (the dots), candidates = count>=2 (2-way+) plus the
             rarest spine as recall floor; then numba merge score. SAME scoring => identical accuracy.
Measures candidate-gen time + total latency + MRR/recall. Full 8.8M, 250 cached queries."""
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
def merge_score(C, locs, offs, wts, qws, nt):
    out = np.zeros(len(C), np.float32)
    for t in range(nt):
        s = offs[t]; e = offs[t + 1]; qw = qws[t]; i = 0; j = s
        while i < len(C) and j < e:
            lj = locs[j]
            if lj == C[i]: out[i] += qw * wts[j]; i += 1; j += 1
            elif lj < C[i]: j += 1
            else: i += 1
    return out


@njit(cache=True)
def count_reader(anchor_flat, anchor_offs, na, cnt):
    """light up the dots: cnt[doc] += 1 for each anchor spine that hits it; return touched positions."""
    touched = np.empty(len(anchor_flat), np.int64); k = 0
    for a in range(na):
        for j in range(anchor_offs[a], anchor_offs[a + 1]):
            d = anchor_flat[j]
            if cnt[d] == 0: touched[k] = d; k += 1
            cnt[d] += 1
    return touched[:k]


def main():
    si = m.ServedIndex()
    col, tloc, present = si.col, si.tloc, si.present
    n_docs_local = len(present)
    cnt = np.zeros(n_docs_local, np.uint8)         # the dot-counter (reused), local-id space
    print(f"  loaded {si.n_docs:,} docs ({n_docs_local:,} present)\n", flush=True)
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

    def prep(qid, topq=30):
        qids, qw = venc[qid]; qw = np.asarray(qw, np.float32); top = np.argsort(-qw)[:topq]
        terms = []
        for i in top:
            j = col.get(int(qids[i]))
            if j is None: continue
            loc, w = tloc[j]; terms.append((loc.astype(np.int64), w.astype(np.float32), np.float32(qw[i])))
        terms.sort(key=lambda t: len(t[0]))
        return terms

    def score_and_rank(terms, C, k):
        nt = len(terms)
        lens = np.array([len(t[0]) for t in terms], np.int64); offs = np.zeros(nt + 1, np.int64); offs[1:] = np.cumsum(lens)
        locs = np.concatenate([t[0] for t in terms]); wts = np.concatenate([t[1] for t in terms])
        qws = np.array([t[2] for t in terms], np.float32)
        sc = merge_score(C, locs, offs, wts, qws, nt)
        sel = np.argpartition(-sc, k)[:k] if len(C) > k else np.arange(len(C))
        return present[C[sel[np.argsort(-sc[sel])]]]

    def serve_searchsorted(qid, k=100, n_anchor=6):
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
        return score_and_rank(terms, C, k)

    def serve_binary(qid, k=100, n_anchor=6):
        terms = prep(qid)
        if not terms: return np.zeros(0, np.uint32)
        anchors = terms[:n_anchor]
        af = np.concatenate([a[0] for a in anchors]); ao = np.zeros(len(anchors) + 1, np.int64)
        ao[1:] = np.cumsum([len(a[0]) for a in anchors])
        touched = count_reader(af, ao, len(anchors), cnt)
        c = cnt[touched]
        C = np.sort(np.concatenate([touched[c >= 2], anchors[0][0]]))   # 2-way+ intersections + rarest floor
        C = np.unique(C)
        cnt[touched] = 0                                                # reset for reuse
        return score_and_rank(terms, C, k)

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
        print(f"  {label:<26}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.1f}{np.percentile(lat,90):>9.1f}{np.percentile(lat,99):>9.1f}", flush=True)

    print(f"  {'serve':<26}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'p99 ms':>9}", flush=True)
    bench(serve_searchsorted, "searchsorted meets+merge")
    bench(serve_binary, "BINARY READER (count)+merge")
    print(f"\n  binary reader = count where anchor-spines intersect (the dots), no pairwise search.", flush=True)


if __name__ == "__main__":
    main()
