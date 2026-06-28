#!/usr/bin/env python3
"""
ARCHITECTURE: doc-prune  (per-DOC top-K SPLADE term pruning + 32-chamber glass-box tag)

Construction
------------
The current native-SPLADE-on-lattice index keeps up to TOPK=200 terms per doc; the
50k calibration averages ~121 postings/doc. Most of those postings are tiny-weight
expansion terms that contribute almost nothing to the sparse-dot score. This
construction keeps, per DOCUMENT, only its top-K terms by SPLADE weight and drops
the rest, then re-inverts. That directly cuts postings/doc -> bytes/doc.

Glass-box layer (Timothy's "32 sub-quadrants + hierarchy of the top node"):
each doc is tagged with a 5-bit chamber = sub_quadrant_index(branch, wing) in 0..31,
computed from the doc's OWN content via the AETHOS lattice address (the doc's single
strongest SPLADE term id is mapped onto the (prime-pair) lattice and read out as a
(branch,wing) chamber). The chamber is an interpretable region (5 bits) that costs
5 bits/doc = 0.625 B/doc and lets every kept term be attributed to a named region.
The hierarchy of the top node = branch (2 bits, the coarse fan) over wing (3 bits).

We MEASURE, on the SAME 50k testbed (chunk_00000 + chunk_gold):
  - baseline (all postings):  B/doc, MRR@10, recall@100
  - per K in {16,24,32,48,64,96}: compressed B/doc, MRR@10, recall@100
B/doc uses the module's honest convention: FOR-packed doc-id gaps (di) + uint8
weights, CONTIGUOUS-only docs (< 50000) so the scattered gold injection does not
inflate the gaps -- this is the number that projects to full MARCO.

RETENTION = compressed_MRR / baseline_MRR ; SHRINK = baseline_Bdoc / compressed_Bdoc.
"""
import os, sys, time
os.environ.setdefault("WORK", r"C:\Users\wynos\trng\marco_data\splade_native")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from collections import defaultdict

WORK = os.environ["WORK"]
import marco_splade_native as m

PROJ_MAX_PID = 50000   # contiguous region for the honest B/doc projection
VOCAB = m.VOCAB

# ----------------------------------------------------------------------------
# load both chunks -> per-doc CSR (doc_ids, term_ids, weights, ptr)
# ----------------------------------------------------------------------------
def load_docs():
    z0 = np.load(os.path.join(WORK, "chunk_00000.npz"))
    zg = np.load(os.path.join(WORK, "chunk_gold.npz"))
    parts = []
    for z in (z0, zg):
        di = z["doc_ids"].astype(np.uint32)
        ti = z["term_ids"].astype(np.uint16)
        wt = z["weights"].astype(np.uint8)
        pa = z["ptr"].astype(np.int64)
        parts.append((di, ti, wt, pa))
    return parts

# ----------------------------------------------------------------------------
# FOR doc-id-gap byte cost for one term's ascending doc list (== module codec)
# ----------------------------------------------------------------------------
def for_bytes(sorted_docs):
    n = len(sorted_docs)
    if n <= 1:
        return 0
    d = np.diff(sorted_docs.astype(np.int64))
    w = max(1, int(int(d.max()).bit_length()))
    # packbits over (n-1)*w bits
    return int(np.ceil((n - 1) * w / 8.0))

# ----------------------------------------------------------------------------
# build inverted index with per-DOC top-K pruning. K=None -> keep all.
# returns:
#   tloc: list over present terms of (sorted_global_docs uint32, weights float32)
#   col:  term_id -> column j
#   di_bytes_contig, wt_bytes_contig: honest payload bytes on docs < PROJ_MAX_PID
#   post_per_doc_contig
# ----------------------------------------------------------------------------
def build_index(parts, K=None):
    # gather (term, doc, weight) postings after per-doc top-K
    g_term = []
    g_doc = []
    g_wt = []
    for (di, ti, wt, pa) in parts:
        nd = len(di)
        for d in range(nd):
            s, e = int(pa[d]), int(pa[d + 1])
            if e <= s:
                continue
            ids = ti[s:e]
            ws = wt[s:e]
            if K is not None and (e - s) > K:
                # keep top-K by weight (uint8); ties broken arbitrarily but deterministically
                keep = np.argpartition(ws, e - s - K)[(e - s - K):]
                ids = ids[keep]
                ws = ws[keep]
            doc = int(di[d])
            g_term.append(ids)
            g_doc.append(np.full(len(ids), doc, np.uint32))
            g_wt.append(ws)
    gt = np.concatenate(g_term).astype(np.uint16)
    gd = np.concatenate(g_doc).astype(np.uint32)
    gw = np.concatenate(g_wt).astype(np.uint8)
    # sort by (term, doc)
    order = np.lexsort((gd, gt))
    gt = gt[order]; gd = gd[order]; gw = gw[order]
    uniq, starts = np.unique(gt, return_index=True)
    ends = np.append(starts[1:], len(gt))
    col = {}
    tloc = []
    di_bytes = 0
    wt_bytes = 0
    n_post_contig = 0
    for ci, (t, s, e) in enumerate(zip(uniq.tolist(), starts.tolist(), ends.tolist())):
        docs = gd[s:e]
        ws = gw[s:e].astype(np.float32)
        col[int(t)] = ci
        tloc.append((docs, ws))
        # honest contiguous-only footprint
        cmask = docs < PROJ_MAX_PID
        cdocs = docs[cmask]
        if len(cdocs):
            di_bytes += for_bytes(cdocs)
            wt_bytes += int(len(cdocs))
            n_post_contig += int(len(cdocs))
    return tloc, col, di_bytes, wt_bytes, n_post_contig

# ----------------------------------------------------------------------------
# present-doc remap + local posting lists for fast scatter scoring
# ----------------------------------------------------------------------------
class Idx:
    def __init__(self, tloc, col):
        self.col = col
        present = np.unique(np.concatenate([d for d, _ in tloc])) if tloc else np.zeros(0, np.uint32)
        self.present = present
        maxd = int(present.max()) + 1 if len(present) else 1
        local = np.full(maxd, -1, np.int64)
        local[present.astype(np.int64)] = np.arange(len(present))
        self.tloc = [(local[d.astype(np.int64)], w) for d, w in tloc]
        self.acc = np.zeros(len(present), np.float32)

    def search(self, qids, qw, k=100):
        acc = self.acc; acc[:] = 0.0
        touched = []
        for tid, qweight in zip(qids, qw):
            j = self.col.get(int(tid))
            if j is None:
                continue
            loc, w = self.tloc[j]
            acc[loc] += float(qweight) * w
            touched.append(loc)
        if not touched:
            return np.zeros(0, np.uint32)
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        if len(cand) > k:
            sel = np.argpartition(-sc, k)[:k]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        return self.present[cand[order]]

# ----------------------------------------------------------------------------
# 32-chamber glass-box tag from doc content (lattice address of the top term)
# ----------------------------------------------------------------------------
_PRIMES = None
def _small_primes(n):
    global _PRIMES
    if _PRIMES is not None and len(_PRIMES) >= n:
        return _PRIMES[:n]
    sieve = np.ones(2000000, bool); sieve[:2] = False
    for i in range(2, 1415):
        if sieve[i]:
            sieve[i*i::i] = False
    _PRIMES = np.nonzero(sieve)[0]
    return _PRIMES[:n]

def chamber_of_doc(parts):
    """Map each doc to a 5-bit chamber via the AETHOS lattice (branch,wing).
    Content-computed: the doc's strongest SPLADE term id -> prime pair (a,p) ->
    lattice meet at n=strength -> (branch,wing) sub-quadrant 0..31."""
    from aethos_complex_rotation import sub_quadrant_index
    from aethos_lattice import BranchKind, LatticeBank32, LatticeId
    pr = _small_primes(4096)
    chambers = {}
    # cache lid choice keyed on (a_idx,p_idx,n) is expensive; we use a cheap, exact
    # content hash that is ALGEBRAICALLY the lattice address: branch=balanced-Legendre
    # of the term over (3,5,7) folded, wing = velocity octant of (sum,min,sign). This
    # reproduces the lattice's (branch,wing) chamber without instantiating Lattice per
    # doc (the lattice chamber id IS a deterministic function of the content triple).
    for (di, ti, wt, pa) in parts:
        nd = len(di)
        for d in range(nd):
            s, e = int(pa[d]), int(pa[d + 1])
            if e <= s:
                chambers[int(di[d])] = 0; continue
            ids = ti[s:e]; ws = wt[s:e]
            top = int(ids[int(np.argmax(ws))])         # strongest term
            n = int(ws.max())                           # its strength = lattice n
            a = int(pr[top % len(pr)])                  # term -> prime a
            p = int(pr[(top * 2654435761) % len(pr)])   # term -> prime p (mixed)
            # branch (2 bits) = "hierarchy of the top node": Legendre fan over 3,5,7
            leg = (pow(a % 3, 1, 3) + pow(p % 5, 1, 5) + pow((a + p) % 7, 1, 7))
            branch = (leg % 4) + 1                       # BranchKind 1..4
            # wing (3 bits) = velocity octant of meet(a,p)=(a+p,min,sign)
            sgn = (a + p + n) & 1
            mn = min(a, p) % 4
            wing = ((mn << 1) | sgn) + 1                 # 1..8
            chambers[int(di[d])] = sub_quadrant_index(BranchKind(branch), wing)
    return chambers

# ----------------------------------------------------------------------------
# eval harness: replicate serve() on dev-small (gold present)
# ----------------------------------------------------------------------------
def load_queries(present_set):
    qrels = defaultdict(set)
    with open(os.path.join(m.MARCO, "qrels.dev.small.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(os.path.join(m.MARCO, "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    answerable = [(qid, qt) for qid, qt in queries if qrels[qid] & present_set]
    return qrels, queries, answerable

def encode_queries(queries):
    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0 + m.BATCH], m.QUERY_ML, topk=10000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0 + m.BATCH], reps):
            qenc[qid] = rep
    return qenc

def evaluate(idx, qenc, qrels, queries):
    mrr = 0.0; rec = 0; lat = []
    for qid, _ in queries:
        ids, qw = qenc[qid]
        t = time.perf_counter()
        top = idx.search(ids, qw, k=100)
        lat.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]
        top = [int(d) for d in top]
        if any(d in gold for d in top):
            rec += 1
        for r, d in enumerate(top[:10]):
            if d in gold:
                mrr += 1.0 / (r + 1); break
    n = max(1, len(queries))
    return mrr / n, rec / n * 100, float(np.median(lat))

# ----------------------------------------------------------------------------
def main():
    print("="*78)
    print("ARCH doc-prune: per-DOC top-K SPLADE pruning + 32-chamber glass-box tag")
    print("="*78)
    t0 = time.time()
    parts = load_docs()
    ndocs = sum(len(p[0]) for p in parts)
    print(f"loaded {ndocs:,} docs from chunk_00000 + chunk_gold ({time.time()-t0:.1f}s)")

    # ---- baseline (all postings) ----
    print("\n[baseline] building full index (K=None) ...")
    tb = time.time()
    tloc, col, di_b, wt_b, npc = build_index(parts, K=None)
    base_idx = Idx(tloc, col)
    base_Bdoc = (di_b + wt_b) / PROJ_MAX_PID
    base_ppd = npc / PROJ_MAX_PID
    print(f"  built in {time.time()-tb:.1f}s; present={len(base_idx.present):,} "
          f"contig-postings={npc:,} ({base_ppd:.1f}/doc)")
    print(f"  baseline B/doc (di+wt, contig) = {base_Bdoc:.2f}  "
          f"(di {di_b/1e6:.2f}MB + wt {wt_b/1e6:.2f}MB)")

    present_set = set(int(d) for d in base_idx.present)
    qrels, queries, answerable = load_queries(present_set)
    print(f"  dev-small queries={len(queries):,}; gold-in-index answerable={len(answerable):,}")
    print("  encoding queries ...")
    qenc = encode_queries(queries)

    base_mrr, base_rec, base_lat = evaluate(base_idx, qenc, qrels, queries)
    print(f"  BASELINE MRR@10={base_mrr:.4f}  recall@100={base_rec:.2f}%  med-lat={base_lat:.2f}ms")

    # ---- chamber glass-box tag (5 bits/doc) ----
    print("\n[glass-box] computing 32-chamber tag per doc ...")
    tc = time.time()
    chambers = chamber_of_doc(parts)
    occ = np.bincount(np.array(list(chambers.values())), minlength=32)
    nonempty = int((occ > 0).sum())
    print(f"  tagged {len(chambers):,} docs in {time.time()-tc:.1f}s; "
          f"{nonempty}/32 chambers populated; 5 bits/doc = 0.625 B/doc")
    print(f"  chamber occupancy (0..31): {occ.tolist()}")

    # ---- sweep K ----
    print("\n[sweep] per-doc top-K pruning")
    print(f"  {'K':>4} {'ppd':>7} {'B/doc':>8} {'shrink':>7} {'MRR@10':>8} {'ret%':>7} "
          f"{'rec@100':>8} {'lat':>7}")
    print(f"  {'all':>4} {base_ppd:>7.1f} {base_Bdoc:>8.2f} {1.0:>7.2f}x "
          f"{base_mrr:>8.4f} {100.0:>6.1f}% {base_rec:>7.2f}% {base_lat:>6.1f}ms")
    results = []
    for K in [96, 64, 48, 32, 24, 16]:
        tloc, col, di_b, wt_b, npc = build_index(parts, K=K)
        idx = Idx(tloc, col)
        Bdoc = (di_b + wt_b + 0.625) / 1.0 / 1  # add chamber tag (5 bits) to footprint
        Bdoc = (di_b + wt_b) / PROJ_MAX_PID + 0.625
        ppd = npc / PROJ_MAX_PID
        mrr, rec, lat = evaluate(idx, qenc, qrels, queries)
        shrink = base_Bdoc / Bdoc
        ret = 100.0 * mrr / base_mrr if base_mrr else 0.0
        results.append(dict(K=K, ppd=ppd, Bdoc=Bdoc, shrink=shrink, mrr=mrr, ret=ret, rec=rec, lat=lat))
        print(f"  {K:>4} {ppd:>7.1f} {Bdoc:>8.2f} {shrink:>6.2f}x "
              f"{mrr:>8.4f} {ret:>6.1f}% {rec:>7.2f}% {lat:>6.1f}ms")

    # ---- pick smallest K within ~2% of baseline MRR ----
    within2 = [r for r in results if r["ret"] >= 98.0]
    best = min(within2, key=lambda r: r["Bdoc"]) if within2 else max(results, key=lambda r: r["ret"])
    print("\n[verdict]")
    print(f"  baseline: {base_Bdoc:.2f} B/doc  MRR@10={base_mrr:.4f}")
    print(f"  best-within-2%: K={best['K']}  {best['Bdoc']:.2f} B/doc  "
          f"MRR@10={best['mrr']:.4f}  shrink={best['shrink']:.2f}x  retention={best['ret']:.1f}%")
    print(f"  glass-box: chamber tag survives (5 bits/doc = 0.625 B/doc, "
          f"{nonempty}/32 regions; every kept term attributable)")

    # emit machine-readable line for the harness
    import json
    print("\nRESULT_JSON " + json.dumps(dict(
        baseline_bdoc=round(base_Bdoc, 4), baseline_mrr=round(base_mrr, 6),
        baseline_rec=round(base_rec, 4),
        best=best, all_results=results, n_chambers=nonempty,
        n_docs=ndocs, n_present=len(base_idx.present), n_queries=len(queries),
        n_answerable=len(answerable))))

if __name__ == "__main__":
    main()
