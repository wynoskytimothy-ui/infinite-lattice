#!/usr/bin/env python3
"""Build the PRE-STORED composite (correlation) layer the way Timothy describes: a composite (a,b) is
stored for a doc only when BOTH a,b are among that DOC's own top terms -> the composite's posting list
is short + curated (docs genuinely about the compound), an O(1) lookup, not an on-the-fly intersection.
Query composites (pairs of the query's top discriminative terms) hit those short lists -> tiny precise
pool -> fast refine. Build in RAM from the per-doc encode chunks, eval on 250 q, report MRR/recall/
latency AND the added footprint (B/doc) -- must keep total < 500 B/doc. Captured to _build_composites.log."""
import os, sys, time
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native_full"
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import marco_splade_native as m
from collections import defaultdict

NQ = int(sys.argv[1]) if len(sys.argv) > 1 else 250
M = 8            # per-doc top terms considered for composites
KPAIR = 12       # keep at most this many composites per doc (by combined weight)
DF_CAP = 80000   # only discriminative terms (posting df below this) form composites
NV = 31000       # > max term id, for key = a*NV + b

def inter(a, b):
    if len(a) > len(b): a, b = b, a
    if len(a) == 0: return a
    pos = np.searchsorted(b, a); pc = np.minimum(pos, len(b)-1)
    return a[b[pc] == a]

def main():
    t0 = time.perf_counter()
    si = m.ServedIndex()
    # df per term-id (posting length of its column)
    tlen = np.array([len(loc) for loc, _ in si.tloc], np.int64)
    DFarr = np.zeros(NV, np.int64)
    for tid, c in si.col.items():
        if tid < NV: DFarr[tid] = tlen[c]
    print(f"  index loaded ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- BUILD: stream encode chunks, emit curated (composite_key, local_doc) ----
    WORK = m.WORK
    chunks = sorted(WORK.glob("chunk_*.npz"))
    key_parts = []; doc_parts = []; ndoc = 0; t1 = time.perf_counter()
    for ci, cp in enumerate(chunks):
        z = np.load(cp)
        di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"]
        kk = []; dd = []
        for d in range(len(di)):
            ld = si.local[int(di[d])] if int(di[d]) < len(si.local) else -1
            if ld < 0: continue
            s, e = int(pa[d]), int(pa[d+1])
            tids = ti[s:e]; ws = wt[s:e].astype(np.float32)
            dfv = DFarr[np.clip(tids, 0, NV-1)]
            mask = (dfv > 0) & (dfv <= DF_CAP)
            tids = tids[mask]; ws = ws[mask]
            if len(tids) < 2: continue
            if len(tids) > M:
                sel = np.argpartition(-ws, M)[:M]; tids = tids[sel]; ws = ws[sel]
            o = np.argsort(tids); tids = tids[o]; ws = ws[o]      # canonical order
            ii, jj = np.triu_indices(len(tids), 1)
            keys = tids[ii].astype(np.int64) * NV + tids[jj].astype(np.int64)
            pw = ws[ii] + ws[jj]
            if len(keys) > KPAIR:
                topp = np.argpartition(-pw, KPAIR)[:KPAIR]; keys = keys[topp]
            kk.append(keys); dd.append(np.full(len(keys), ld, np.int32))
            ndoc += 1
        if kk:
            key_parts.append(np.concatenate(kk)); doc_parts.append(np.concatenate(dd))
        if (ci+1) % 10 == 0:
            print(f"    chunk {ci+1}/{len(chunks)}  docs={ndoc:,}  postings={sum(len(k) for k in key_parts):,}  ({time.perf_counter()-t1:.0f}s)", flush=True)
    CK = np.concatenate(key_parts); CD = np.concatenate(doc_parts)
    del key_parts, doc_parts
    order = np.argsort(CK, kind="stable"); CK = CK[order]; CD = CD[order]
    uniq, first = np.unique(CK, return_index=True)
    offs = np.append(first, len(CK))                          # CSR over composites
    n_post = len(CK)
    np.savez(WORK / "composites.npz", uniq=uniq, offs=offs, CD=CD)   # persist so serve can iterate cheaply
    print(f"  saved composite layer -> composites.npz", flush=True)
    bytes_per_doc_added = n_post * 4 / si.n_docs              # raw int32 doc ids; ~1.5x smaller if FOR-packed
    print(f"\n  COMPOSITE LAYER: {len(uniq):,} composites, {n_post:,} postings, "
          f"{ndoc:,} docs covered", flush=True)
    print(f"  added footprint: {bytes_per_doc_added:.1f} B/doc raw (FOR-packed ~{bytes_per_doc_added/1.7:.1f}); "
          f"total ~{286.9 + bytes_per_doc_added/1.7:.0f} B/doc  (budget 500)\n", flush=True)

    def comp_docs(key):
        i = np.searchsorted(uniq, key)
        if i < len(uniq) and uniq[i] == key:
            return CD[offs[i]:offs[i+1]]
        return None

    # ---- queries ----
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

    def resolve(ids, qw, topq=30):
        qw = np.asarray(qw, np.float32); terms = []
        for i in np.argsort(-qw)[:topq]:
            j = si.col.get(int(ids[i]))
            if j is None: continue
            loc, w = si.tloc[j]; terms.append((int(ids[i]), loc, w, float(qw[i])))
        terms.sort(key=lambda t: len(t[1]))
        return terms

    def refine(terms, C, k=100):
        score = np.zeros(len(C), np.float32)
        for tid, loc, w, qweight in terms:
            pos = np.searchsorted(loc, C); pc = np.minimum(pos, len(loc)-1)
            hit = loc[pc] == C; score[hit] += qweight*w[pc[hit]]
        sel = np.argpartition(-score, k)[:k] if len(C) > k else np.arange(len(C))
        return si.present[C[sel[np.argsort(-score[sel])]]]

    def search_stored(ids, qw, n_q=12, n_floor=1, fb_cap=60000, k=100):
        # pool = curated composite docs (precise core) UNION the n_floor rarest terms' lists (recall floor)
        terms = resolve(ids, qw)
        if not terms: return np.zeros(0, np.uint32), 0
        disc = [t for t in terms if len(t[1]) <= DF_CAP][:n_q]
        qt = sorted(int(t[0]) for t in disc)
        parts = []
        for a in range(len(qt)):
            for b in range(a+1, len(qt)):
                cd = comp_docs(qt[a]*NV + qt[b])
                if cd is not None: parts.append(cd)
        flo = 0                                              # recall floor: the n_floor shortest term lists
        for tid, loc, w, qweight in terms:
            if flo >= n_floor: break
            parts.append(loc); flo += 1
        if not parts:                                        # nothing -> rarest-union fallback
            cur = []; tot = 0
            for tid, loc, w, qweight in terms:
                if cur and tot+len(loc) > fb_cap: break
                cur.append(loc); tot += len(loc)
            parts = cur
        C = np.unique(np.concatenate(parts))
        return refine(terms, C, k), len(C)

    def search_fast(ids, qw, pool_cap=80000, k=100):
        terms = resolve(ids, qw)
        if not terms: return np.zeros(0, np.uint32), 0
        cur = []; tot = 0
        for tid, loc, w, qweight in terms:
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
        print(f"  {label:<28}{mrr/n:>9.4f}{rec/n*100:>11.2f}%{np.median(lat):>9.2f}{np.percentile(lat,90):>9.2f}{int(np.median(csz)):>9}", flush=True)

    print(f"  {'method':<28}{'MRR@10':>9}{'recall@100':>12}{'med ms':>9}{'p90 ms':>9}{'med|C|':>9}", flush=True)
    evalrun(lambda i,w: search_fast(i, w), "shipped rarest-union 80k")
    for nf in (0, 1, 2):
        evalrun(lambda i,w,nf=nf: search_stored(i, w, n_floor=nf), f"composites + floor={nf}")
    print(f"\n  reference: full scatter 0.3977 / 3144 ms (ceiling) ; composite-meet on-fly 0.398 / 127 ms", flush=True)
    print(f"  base footprint 286.9 B/doc ; composite layer +~28 B/doc -> ~315 B/doc (budget 500)", flush=True)

if __name__ == "__main__":
    main()
