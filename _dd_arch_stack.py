#!/usr/bin/env python3
"""
LEAD-ARCHITECT STACK: combine the three orthogonal levers into ONE config and
MEASURE end-to-end on the 50k SPLADE testbed.

Levers (orthogonal, verified individually by the four agents):
  (1) doc-prune  : per-DOC top-K SPLADE terms  -> cuts postings/doc (the multiplier)
  (2) wt 4-bit   : per-corpus Lloyd-Max weight codebook, monotone dequant (glass-box)
  (3) di codec   : FOR vs varint gap coder on the (now pruned) doc-gap stream
  (+) chamber tag: 5-bit content-computed glass-box region (0.625 B/doc), non-compressing

Why measure the stack rather than multiply ratios: pruning REMOVES the tiny-weight
tail postings, which (a) makes per-term lists SHORTER -> LARGER doc-gaps (di codec
behaves differently), and (b) is exactly the tail that quantization rounds to the
low codebook levels. The product of the three single-lever ratios is NOT the joint
ratio; only an end-to-end measurement is honest. Scoring uses dequant(weights) so
MRR reflects the actual served, lossy index.

B/doc convention is IDENTICAL to the verified doc-prune script:
  FOR di-gap bytes + weight-stream bytes, CONTIGUOUS docs (<50000), /50000.
Weight-stream bytes under W-bit packing = ceil(n_post_contig * W / 8).
"""
import os, sys, time, json
os.environ.setdefault("WORK", r"C:\Users\wynos\trng\marco_data\splade_native")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from collections import defaultdict

WORK = os.environ["WORK"]
import marco_splade_native as m
PROJ_MAX_PID = 50000

# ---------------------------------------------------------------- load docs
def load_docs():
    parts = []
    for nm in ("chunk_00000.npz", "chunk_gold.npz"):
        z = np.load(os.path.join(WORK, nm))
        parts.append((z["doc_ids"].astype(np.uint32), z["term_ids"].astype(np.uint16),
                      z["weights"].astype(np.uint8), z["ptr"].astype(np.int64)))
    return parts

# ---------------------------------------------------------------- di codecs
def for_bytes(sorted_docs):
    n = len(sorted_docs)
    if n <= 1: return 0
    d = np.diff(sorted_docs.astype(np.int64))
    w = max(1, int(int(d.max()).bit_length()))
    return int(np.ceil((n - 1) * w / 8.0))

def varint_bytes(sorted_docs):
    """LEB128 byte cost of the ascending gap stream (lossless, decodable)."""
    n = len(sorted_docs)
    if n <= 1: return 0
    d = np.diff(sorted_docs.astype(np.int64))
    # bytes per gap = ceil(bitlen/7), min 1
    bl = np.maximum(1, np.floor(np.log2(np.maximum(d, 1))).astype(np.int64) + 1)
    nb = np.ceil(bl / 7.0).astype(np.int64)
    nb = np.maximum(nb, 1)
    return int(nb.sum())

# ---------------------------------------------------------------- 4-bit Lloyd-Max weight codebook
def lloyd_max(vals, levels=16, iters=25):
    """1-D Lloyd-Max quantizer on the (uint8) weight population. Returns sorted
    reproduction points (monotone) and a function mapping uint8 -> level index."""
    vals = vals.astype(np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    # init centroids at quantiles for a good start
    qs = np.linspace(0, 1, levels)
    cent = np.quantile(vals, qs)
    cent = np.unique(cent)
    if len(cent) < levels:
        cent = np.linspace(lo, hi, levels)
    cent = np.sort(cent.astype(np.float64))
    for _ in range(iters):
        # boundaries = midpoints
        bnd = (cent[:-1] + cent[1:]) / 2.0
        idx = np.searchsorted(bnd, vals)
        new = cent.copy()
        for k in range(levels):
            sel = vals[idx == k]
            if len(sel): new[k] = sel.mean()
        if np.allclose(new, cent): cent = new; break
        cent = new
    cent = np.sort(cent)
    # build uint8 -> level lookup (monotone) over 0..255
    bnd = (cent[:-1] + cent[1:]) / 2.0
    u8 = np.arange(256, dtype=np.float64)
    lut_level = np.searchsorted(bnd, u8).astype(np.int32)           # uint8 -> level
    lut_dequant = cent[lut_level].astype(np.float32)                # uint8 -> dequant value
    return cent.astype(np.float32), lut_level, lut_dequant

# ---------------------------------------------------------------- build pruned index, optional dequant
def build_index(parts, K=None, dequant_lut=None, wbits=8):
    g_term, g_doc, g_wt = [], [], []
    for (di, ti, wt, pa) in parts:
        nd = len(di)
        for d in range(nd):
            s, e = int(pa[d]), int(pa[d + 1])
            if e <= s: continue
            ids = ti[s:e]; ws = wt[s:e]
            if K is not None and (e - s) > K:
                keep = np.argpartition(ws, e - s - K)[(e - s - K):]
                ids = ids[keep]; ws = ws[keep]
            doc = int(di[d])
            g_term.append(ids); g_doc.append(np.full(len(ids), doc, np.uint32)); g_wt.append(ws)
    gt = np.concatenate(g_term).astype(np.uint16)
    gd = np.concatenate(g_doc).astype(np.uint32)
    gw = np.concatenate(g_wt).astype(np.uint8)
    order = np.lexsort((gd, gt)); gt = gt[order]; gd = gd[order]; gw = gw[order]
    uniq, starts = np.unique(gt, return_index=True)
    ends = np.append(starts[1:], len(gt))
    col = {}; tloc = []
    di_for = di_var = wt_bytes_packed = 0
    n_post_contig = 0
    for ci, (t, s, e) in enumerate(zip(uniq.tolist(), starts.tolist(), ends.tolist())):
        docs = gd[s:e]; ws_u8 = gw[s:e]
        # served weights: dequantized (lossy) or raw uint8 float
        if dequant_lut is not None:
            wf = dequant_lut[ws_u8]                      # float32 dequant
        else:
            wf = ws_u8.astype(np.float32)
        col[int(t)] = ci
        tloc.append((docs, wf))
        cmask = docs < PROJ_MAX_PID; cdocs = docs[cmask]
        if len(cdocs):
            di_for += for_bytes(cdocs)
            di_var += varint_bytes(cdocs)
            n_post_contig += int(len(cdocs))
    wt_bytes_packed = int(np.ceil(n_post_contig * wbits / 8.0))
    return tloc, col, di_for, di_var, wt_bytes_packed, n_post_contig

class Idx:
    def __init__(self, tloc, col):
        self.col = col
        present = np.unique(np.concatenate([d for d, _ in tloc])) if tloc else np.zeros(0, np.uint32)
        self.present = present
        maxd = int(present.max()) + 1 if len(present) else 1
        local = np.full(maxd, -1, np.int64); local[present.astype(np.int64)] = np.arange(len(present))
        self.tloc = [(local[d.astype(np.int64)], w) for d, w in tloc]
        self.acc = np.zeros(len(present), np.float32)
    def search(self, qids, qw, k=100):
        acc = self.acc; acc[:] = 0.0; touched = []
        for tid, qweight in zip(qids, qw):
            j = self.col.get(int(tid))
            if j is None: continue
            loc, w = self.tloc[j]
            acc[loc] += float(qweight) * w; touched.append(loc)
        if not touched: return np.zeros(0, np.uint32)
        cand = np.unique(np.concatenate(touched)); sc = acc[cand]
        if len(cand) > k: sel = np.argpartition(-sc, k)[:k]
        else: sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return self.present[cand[order]]

# ---------------------------------------------------------------- eval harness
def load_queries(present_set):
    qrels = defaultdict(set)
    with open(os.path.join(m.MARCO, "qrels.dev.small.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0: qrels[p[0]].add(int(p[2]))
    queries = []
    with open(os.path.join(m.MARCO, "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels: queries.append((a[0], a[1]))
    return qrels, queries

def encode_queries(queries):
    qenc = {}; qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0 + m.BATCH], m.QUERY_ML, topk=10000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0 + m.BATCH], reps): qenc[qid] = rep
    return qenc

def evaluate(idx, qenc, queries, qrels):
    mrr = 0.0; rec = 0; lat = []
    for qid, _ in queries:
        ids, qw = qenc[qid]
        t = time.perf_counter(); top = idx.search(ids, qw, k=100); lat.append((time.perf_counter()-t)*1000)
        gold = qrels[qid]; top = [int(d) for d in top]
        if any(d in gold for d in top): rec += 1
        for r, d in enumerate(top[:10]):
            if d in gold: mrr += 1.0/(r+1); break
    n = max(1, len(queries))
    return mrr/n, rec/n*100, float(np.median(lat))

# ---------------------------------------------------------------- main
def main():
    print("="*86)
    print("LEAD-ARCHITECT STACK: doc-prune + 4-bit Lloyd-Max weights + best-di + chamber tag")
    print("="*86)
    t0 = time.time()
    parts = load_docs()
    # build the weight codebook from the FULL (unpruned) weight population
    allw = np.concatenate([wt[pa[0]:pa[-1]] if False else wt for (_, _, wt, pa) in parts])
    cent16, lut_lvl, lut_deq = lloyd_max(allw, levels=16, iters=30)
    print(f"loaded; weight codebook (16 levels) = {np.round(cent16,1).tolist()}")

    # baseline (all postings, raw uint8, FOR di)
    tl, col, di_for, di_var, wt_b8, npc = build_index(parts, K=None, dequant_lut=None, wbits=8)
    base_idx = Idx(tl, col)
    base_Bdoc = (di_for + wt_b8) / PROJ_MAX_PID
    base_ppd = npc / PROJ_MAX_PID
    present_set = set(int(d) for d in base_idx.present)
    qrels, queries = load_queries(present_set)
    qenc = encode_queries(queries)
    base_mrr, base_rec, base_lat = evaluate(base_idx, qenc, queries, qrels)
    print(f"\nBASELINE  ppd={base_ppd:.1f}  B/doc={base_Bdoc:.2f} (di_FOR {di_for/1e6:.2f}MB + wt8 {wt_b8/1e6:.2f}MB)"
          f"  MRR@10={base_mrr:.4f}  rec@100={base_rec:.2f}%  lat={base_lat:.2f}ms")

    # stacked sweep: K x {wt8 vs wt4-dequant} x {FOR vs varint di}
    print("\n[STACK] K | wbits | di-codec | ppd | B/doc | shrink | MRR@10 | ret% | rec@100 | lat")
    rows = []
    for K in [64, 48, 40, 32, 24]:
        # raw-weight (for isolating doc-prune effect on di) + dequant-weight build
        tlq, colq, di_for, di_var, wt_b4, npc = build_index(parts, K=K, dequant_lut=lut_deq, wbits=4)
        _,    _,    _,      _,      wt_b8, _   = build_index(parts, K=K, dequant_lut=None,    wbits=8)
        idxq = Idx(tlq, colq)                       # served with dequantized 4-bit weights
        mrr, rec, lat = evaluate(idxq, qenc, queries, qrels)
        ppd = npc / PROJ_MAX_PID
        chamber = 0.625
        for di_name, di_bytes in (("FOR", di_for), ("varint", di_var)):
            for wname, wbytes, served_mrr in (("w8", wt_b8, None), ("w4", wt_b4, mrr)):
                # NOTE: w8 rows are footprint-only (served MRR == baseline-codec dot of raw uint8);
                # we only have a measured served MRR for the w4 build (the lossy one). For w8 we
                # report the doc-prune MRR which we re-measure once with raw weights below.
                pass
        # measured served MRR is for w4-dequant; also measure w8 (raw) served MRR for this K
        tl8, col8, _, _, _, _ = build_index(parts, K=K, dequant_lut=None, wbits=8)
        idx8 = Idx(tl8, col8)
        mrr8, rec8, lat8 = evaluate(idx8, qenc, queries, qrels)
        for di_name, di_bytes in (("FOR", di_for), ("varint", di_var)):
            # config A: w8 raw weights
            Bd8 = (di_bytes + wt_b8)/PROJ_MAX_PID + chamber
            rows.append((K, 8, di_name, ppd, Bd8, base_Bdoc/Bd8, mrr8, 100*mrr8/base_mrr, rec8, lat8))
            # config B: w4 dequant weights
            Bd4 = (di_bytes + wt_b4)/PROJ_MAX_PID + chamber
            rows.append((K, 4, di_name, ppd, Bd4, base_Bdoc/Bd4, mrr, 100*mrr/base_mrr, rec, lat))
    for (K, wb, dn, ppd, Bd, sh, mrr, ret, rec, lat) in rows:
        print(f"  K={K:>3} w{wb} {dn:>6}  ppd={ppd:5.1f}  {Bd:7.2f}  {sh:5.2f}x  "
              f"{mrr:.4f}  {ret:5.1f}%  {rec:5.2f}%  {lat:.2f}ms")

    # pick smallest B/doc with retention >= 98% (within ~2%)
    ok = [r for r in rows if r[7] >= 98.0]
    best = min(ok, key=lambda r: r[4]) if ok else None
    print("\n[PICK] smallest B/doc with retention>=98%:")
    if best:
        K, wb, dn, ppd, Bd, sh, mrr, ret, rec, lat = best
        print(f"  K={K} w{wb} di={dn}  -> {Bd:.2f} B/doc  {sh:.2f}x  MRR@10={mrr:.4f}  ret={ret:.1f}%  rec={rec:.2f}%  lat={lat:.2f}ms")
        # project to full MARCO: scale shrink onto the canonical 286.9 B/doc
        full_proj = 286.9 / sh
        print(f"  PROJECTION to full MARCO: 286.9 / {sh:.2f} = {full_proj:.1f} B/doc at ~{ret:.1f}% of native MRR")
    print(f"\n[done] {time.time()-t0:.0f}s")
    print("\nRESULT_JSON " + json.dumps(dict(
        base_Bdoc=round(base_Bdoc,3), base_mrr=round(base_mrr,5), base_ppd=round(base_ppd,2),
        codebook=np.round(cent16,2).tolist(),
        rows=[dict(K=r[0],wbits=r[1],di=r[2],ppd=round(r[3],2),Bdoc=round(r[4],3),
                   shrink=round(r[5],3),mrr=round(r[6],5),ret=round(r[7],2),rec=round(r[8],2),lat=round(r[9],3)) for r in rows],
        best=(dict(K=best[0],wbits=best[1],di=best[2],Bdoc=round(best[4],3),shrink=round(best[5],3),
                   mrr=round(best[6],5),ret=round(best[7],2)) if best else None))))

if __name__ == "__main__":
    main()
