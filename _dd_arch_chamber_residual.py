#!/usr/bin/env python3
"""
_dd_arch_chamber_residual.py
============================
NOVEL CONSTRUCTION: chamber-clustered residual coding of the native-SPLADE lattice index.

Idea (Timothy's "32 sub-quadrants + hierarchy of the top node"):
  The 50k SPLADE docs are clustered into K groups. Per cluster we store ONE shared
  BASE profile (the cluster's common high-weight terms + their mean quantized weight).
  Each doc then stores only its RESIDUAL relative to its cluster base:
     - a bitmask over the base terms it KEEPS (drops the rest),
     - the small weight DELTA on each kept base term (residual-quantized, few bits),
     - its EXTRA terms (term-id + weight) not in the base.
  At query time:
     score(doc) = base_cluster_score(cluster(doc))            [computed ONCE per cluster per query]
                + sum over kept-base-terms of qw * delta       [the residual correction]
                + sum over extra-terms     of qw * w
  The base-cluster-score is the dot of the query with the cluster's MEAN profile; every
  doc in the cluster inherits it, and the residual restores per-doc fidelity.

  Hierarchy ("top node"): K clusters are LABELLED by the lattice 32-chamber of their
  centroid's dominant term (aethos_complex_rotation.sub_quadrant_index). The top node =
  the 32-chamber partition; clusters refine it. Glass-box: every score decomposes into
  (chamber base contribution) + (per-doc residual terms), all interpretable.

WHAT WE MEASURE (same 50k testbed, vs the UNCOMPRESSED contiguous baseline):
  - compressed B/doc  (base amortized over its members + per-doc residual payload)
  - MRR@10 (all dev-small q whose gold is in this 50k index) and recall@100
  - vs baseline B/doc (di FOR gaps + uint8 weights, contiguous-only) and baseline MRR/recall
Reported as a SHRINK-vs-ACCURACY curve over K and residual bit-budgets. Honest: where
accuracy drops, we say so, and report the best point.

Run:
  python _dd_arch_chamber_residual.py
"""
import os, sys, time, math, glob
os.environ.setdefault("WORK", r"C:\Users\wynos\trng\marco_data\splade_native")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

WORK = os.environ["WORK"]
sys.path.insert(0, r"C:\Users\wynos\New folder (3)")
import marco_splade_native as m
from collections import defaultdict

RNG = np.random.default_rng(0)


def requant(w_u8, wbits):
    """Requantize uint8 SPLADE weights (1..255) to `wbits` linear levels, return reconstructed u8.
    wbits>=8 is a no-op. Levels span the SPLADE active range [1,86] (observed max) for fidelity."""
    if wbits >= 8:
        return np.asarray(w_u8, np.float32)
    w = np.asarray(w_u8, np.float32)
    lo, hi = 1.0, 86.0
    levels = (1 << wbits) - 1
    q = np.clip(np.round((np.clip(w, lo, hi) - lo) / (hi - lo) * levels), 0, levels)
    return (q / levels * (hi - lo) + lo).astype(np.float32)

# ----------------------------------------------------------------------------------------------
# 1. Load per-doc SPLADE reps (CSR) from both chunks, restrict to contiguous docs (pid < 50000)
#    so the footprint density matches the honest full-MARCO projection.
# ----------------------------------------------------------------------------------------------
def load_docs(proj_max=50000):
    """Load ALL present docs (contiguous slice + scattered gold) so eval ranks over the full
    served index. `contig` flags which docs are in the contiguous slice (pid < proj_max) -- the
    FOOTPRINT is amortized ONLY over those (honest full-MARCO density); the scattered gold docs
    inflate FOR gaps artificially and are excluded from B/doc, exactly like marco_splade_native."""
    chunks = sorted(glob.glob(os.path.join(WORK, "chunk_*.npz")))
    doc_ids = []; rows = []; contig = []
    seen = set()
    for cp in chunks:
        z = np.load(cp)
        di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"].astype(np.int64)
        for d in range(len(di)):
            pid = int(di[d])
            if pid in seen:
                continue
            seen.add(pid)
            s, e = int(pa[d]), int(pa[d + 1])
            doc_ids.append(pid); rows.append((ti[s:e].copy(), wt[s:e].copy()))
            contig.append(pid < proj_max)
        z.close()
    return doc_ids, rows, np.array(contig, bool)


# ----------------------------------------------------------------------------------------------
# 2. Baseline footprint (contiguous-only di FOR gaps + uint8 weights) -- the number to beat.
#    Also build the served-index baseline MRR via marco_splade_native.ServedIndex over the SAME
#    present-doc set (which actually includes the scattered gold; we restrict scoring to honest
#    present docs but ranking is over the full index -> we reuse the existing si for MRR baseline).
# ----------------------------------------------------------------------------------------------
def baseline_footprint(doc_ids, rows, contig):
    """Footprint over CONTIGUOUS docs only (honest density). invert contiguous postings, FOR-pack."""
    cset = set(int(p) for p, c in zip(doc_ids, contig) if c)
    by_term = defaultdict(list)   # term -> list of (pid, w)
    for pid, (ti, wt), c in zip(doc_ids, rows, contig):
        if not c:
            continue
        for t, w in zip(ti.tolist(), wt.tolist()):
            by_term[t].append((pid, w))
    di_b = 0; wt_b = 0; n_post = 0
    for t, lst in by_term.items():
        lst.sort()
        docs = np.array([p for p, _ in lst], np.uint32)
        _, _, _, packed = m._for_pack_gaps(docs)
        di_b += packed.nbytes; wt_b += len(lst); n_post += len(lst)
    nd = len(cset)
    return dict(di_b=di_b, wt_b=wt_b, n_post=n_post, n_docs=nd,
                B_per_doc=(di_b + wt_b) / nd, ppd=n_post / nd)


# ----------------------------------------------------------------------------------------------
# 3. Build dense doc matrix (sparse) for clustering. Use a reduced vocab of the top-V terms by
#    document frequency to keep k-means cheap; keep full reps for scoring fidelity.
# ----------------------------------------------------------------------------------------------
def build_sparse(doc_ids, rows, vocab=m.VOCAB):
    nd = len(doc_ids)
    # CSR over full vocab
    indptr = np.zeros(nd + 1, np.int64)
    allt = []; allw = []
    for i, (ti, wt) in enumerate(rows):
        allt.append(ti.astype(np.int32)); allw.append(wt.astype(np.float32))
        indptr[i + 1] = indptr[i] + len(ti)
    cols = np.concatenate(allt); vals = np.concatenate(allw)
    return indptr, cols, vals


def doc_topterms(rows, topn=1):
    """dominant term id per doc (highest weight) for chamber labelling."""
    out = np.zeros(len(rows), np.int64)
    for i, (ti, wt) in enumerate(rows):
        if len(ti) == 0:
            out[i] = 0; continue
        out[i] = int(ti[int(np.argmax(wt))])
    return out


# ----------------------------------------------------------------------------------------------
# 4. Chamber address of a term id (glass-box 5-bit region). Map term-id -> integer triple ->
#    lattice 32-chamber via aethos. Cheap deterministic content-hash into (sum,balance,octant).
# ----------------------------------------------------------------------------------------------
def term_chamber(term_id):
    """Map a bert term-id to one of 32 chambers using the lattice octant + branch parity.
    Glass-box: branch = (residue mod 4), wing = sign pattern of a 3-residue (Legendre-like)."""
    t = int(term_id) + 1
    # three small primes from the algebraic corpus: 3,5,7 -> octant via residues (Legendre-ish)
    r3, r5, r7 = t % 3, t % 5, t % 7
    # wing 1..8 from 3 bits; branch 1..4 from t mod 4
    bit0 = 1 if (r3 != 0) else 0
    bit1 = 1 if (r5 >= 3) else 0
    bit2 = 1 if (r7 >= 4) else 0
    wing = 1 + (bit0 | (bit1 << 1) | (bit2 << 2))
    branch = 1 + (t % 4)
    from aethos_complex_rotation import sub_quadrant_index
    from aethos_lattice import BranchKind
    return sub_quadrant_index(BranchKind(branch), wing)


# ----------------------------------------------------------------------------------------------
# 5. K-means (spherical, cosine) on L2-normalized SPLADE doc vectors, reduced vocab.
# ----------------------------------------------------------------------------------------------
def kmeans_assign(indptr, cols, vals, nd, K, vocab_keep, iters=12):
    # restrict to top vocab_keep terms by collection weight mass for the clustering space
    mass = np.zeros(m.VOCAB, np.float64)
    np.add.at(mass, cols, vals)
    keep = np.argsort(-mass)[:vocab_keep]
    remap = -np.ones(m.VOCAB, np.int64); remap[keep] = np.arange(len(keep))
    # build reduced normalized rows
    rrows = []
    for i in range(nd):
        s, e = indptr[i], indptr[i + 1]
        c = remap[cols[s:e]]; v = vals[s:e]
        msk = c >= 0
        c = c[msk]; v = v[msk]
        n = np.linalg.norm(v) + 1e-9
        rrows.append((c, v / n))
    # init centroids by random docs
    cent = np.zeros((K, len(keep)), np.float32)
    init = RNG.choice(nd, size=K, replace=False)
    for k, i in enumerate(init):
        c, v = rrows[i]; cent[k, c] = v
    assign = np.zeros(nd, np.int32)
    for it in range(iters):
        # assign: argmax dot
        for i in range(nd):
            c, v = rrows[i]
            if len(c) == 0:
                assign[i] = 0; continue
            sims = cent[:, c] @ v
            assign[i] = int(np.argmax(sims))
        # update
        cent[:] = 0
        cnt = np.zeros(K, np.int64)
        for i in range(nd):
            c, v = rrows[i]
            cent[assign[i], c] += v
            cnt[assign[i]] += 1
        for k in range(K):
            nrm = np.linalg.norm(cent[k]) + 1e-9
            cent[k] /= nrm
    return assign


# ----------------------------------------------------------------------------------------------
# 6. Build the chamber-residual encoding for a given assignment + bit budget; measure footprint
#    AND score the dev-small queries (decode-then-score, exact reconstruction path) to get MRR.
# ----------------------------------------------------------------------------------------------
def build_residual(doc_ids, rows, assign, K, contig, base_terms_cap, delta_bits,
                   base_min_frac=0.30, wbits=8):
    """
    For each cluster:
      - gather member docs, compute per-term presence frequency + mean weight
      - BASE = top `base_terms_cap` terms by (freq * mean_weight) that appear in >= base_min_frac of members
      - base weight stored as uint8 mean
    Per doc:
      - keep-bitmask over base terms (1 bit each) -> which base terms the doc actually has
      - delta on kept base terms: residual = clip(round((w - base_w)/dscale)) in `delta_bits`
      - extra terms (not in base): (term_id uint16-ish via gap, weight uint8)
    FOOTPRINT is amortized over CONTIGUOUS docs only (contig mask); reconstruction is for ALL docs.
    Returns: footprint dict + a SCORER that reconstructs each doc's (term->weight) for eval.
    """
    nd = len(doc_ids)
    n_contig = int(contig.sum())
    members = defaultdict(list)
    for i in range(nd):
        members[int(assign[i])].append(i)

    # build base profiles
    base_terms = {}      # cluster -> np.array term ids (sorted)
    base_w = {}          # cluster -> np.array uint8 base weight aligned to base_terms
    for k, mem in members.items():
        if not mem:
            base_terms[k] = np.zeros(0, np.int64); base_w[k] = np.zeros(0, np.uint8); continue
        freq = defaultdict(int); wsum = defaultdict(float)
        for i in mem:
            ti, wt = rows[i]
            for t, w in zip(ti.tolist(), wt.tolist()):
                freq[t] += 1; wsum[t] += w
        cand = []
        nm = len(mem)
        for t, f in freq.items():
            if f >= max(1, int(base_min_frac * nm)):
                mw = wsum[t] / f
                cand.append((f * mw, t, mw))
        cand.sort(reverse=True)
        cand = cand[:base_terms_cap]
        bt = np.array(sorted(t for _, t, _ in cand), np.int64)
        mwmap = {t: mw for _, t, mw in cand}
        bw = np.clip(np.round([mwmap[t] for t in bt.tolist()]), 1, 255).astype(np.uint8)
        base_terms[k] = bt; base_w[k] = bw

    # per-doc residual encoding + footprint accounting
    dscale = max(1.0, 255.0 / (2 ** delta_bits - 1) / 1.0)  # placeholder; set per-doc below
    # We use a fixed residual scale: deltas are (w - base_w), small. Quantize to signed delta_bits.
    dmax = (1 << (delta_bits - 1)) - 1   # symmetric signed range
    # choose a delta quant step so most residuals fit: study residual magnitudes
    resid_all = []
    for i in range(nd):
        k = int(assign[i]); bt = base_terms[k]
        ti, wt = rows[i]
        if len(bt):
            pos = np.searchsorted(bt, ti)
            pc = np.minimum(pos, len(bt) - 1)
            hit = bt[pc] == ti
            resid_all.append(wt[hit].astype(np.int32) - base_w[k][pc[hit]].astype(np.int32))
    resid_all = np.concatenate(resid_all) if resid_all else np.zeros(0, np.int32)
    # step so ~95th pct residual fits dmax
    p95 = np.percentile(np.abs(resid_all), 95) if len(resid_all) else 1.0
    dstep = max(1.0, p95 / max(1, dmax))

    # footprint
    keepmask_bits = 0    # bits for keep-bitmask
    delta_bytes_bits = 0  # bits for deltas
    extra_di_b = 0; extra_wt_b = 0; extra_n = 0
    # base footprint: stored once per cluster: term-ids (FOR gap) + uint8 weight
    base_di_b = 0; base_wt_b = 0; base_n = 0
    for k in members:
        bt = base_terms[k]; bw = base_w[k]
        base_n += len(bt)
        if len(bt):
            _, _, _, packed = m._for_pack_gaps(bt.astype(np.uint32))
            base_di_b += packed.nbytes; base_wt_b += len(bt)
    # decode structures for scoring (reconstruct each doc term->weight EXACTLY as the codec would)
    dec_terms = [None] * nd; dec_w = [None] * nd
    # extra terms inverted for footprint: collect per term gap pack
    extra_by_doc = []
    for i in range(nd):
        k = int(assign[i]); bt = base_terms[k]; bw = base_w[k]
        ti, wt = rows[i]
        if len(bt):
            pos = np.searchsorted(bt, ti)
            pc = np.minimum(pos, len(bt) - 1)
            in_base = bt[pc] == ti
        else:
            pc = np.zeros(len(ti), np.int64)
            in_base = np.zeros(len(ti), bool)
        is_contig = bool(contig[i])
        # SPARSE base reference: store kept base-term POSITIONS (indices into base), gap-coded.
        # Each kept base term costs ~ceil(log2 |base|) bits via its base-index (cheaper than a raw
        # term-id since |base| << vocab) -- NOT a dense 1-bit-per-base-term mask.
        kept_idx = pc[in_base]
        kept_w = wt[in_base]
        base_idx_bits = max(1, int(math.ceil(math.log2(max(2, len(bt)))))) if len(bt) else 1
        if is_contig:
            keepmask_bits += len(kept_idx) * base_idx_bits   # sparse base-index references
        # deltas for kept base terms
        base_kept_w = bw[kept_idx] if len(bt) else np.zeros(0, np.uint8)
        resid = kept_w.astype(np.int32) - base_kept_w.astype(np.int32)
        if delta_bits <= 0:
            qresid = np.zeros(len(resid), np.int32)          # no-delta: use base mean weight (lossy)
        else:
            qresid = np.clip(np.round(resid / dstep), -dmax, dmax).astype(np.int32)
            if is_contig:
                delta_bytes_bits += len(qresid) * delta_bits
        recon_kept_w = (base_kept_w.astype(np.int32) + (qresid * dstep)).round().clip(1, 255)
        # extra terms (not in base): term-id (gap-coded in inverted index) + weight quantized to wbits
        et = ti[~in_base]; ew = wt[~in_base]
        # requantize extra weights to wbits levels (linear over the observed 1..255 uint8 range)
        ew_rec = requant(ew, wbits)
        if is_contig:
            extra_n += len(et)
            extra_by_doc.append((doc_ids[i], et, ew))
        # reconstruct full doc rep for scoring (extra weights use the requantized value)
        rec_t = np.concatenate([bt[kept_idx] if len(bt) else np.zeros(0, np.int64),
                                et.astype(np.int64)])
        rec_w = np.concatenate([recon_kept_w.astype(np.float32), ew_rec.astype(np.float32)])
        order = np.argsort(rec_t)
        dec_terms[i] = rec_t[order].astype(np.int64)
        dec_w[i] = rec_w[order].astype(np.float32)

    # extra footprint: invert extras by term, FOR-pack doc gaps + uint8 weights
    ex_by_term = defaultdict(list)
    for pid, et, ew in extra_by_doc:
        for t, w in zip(et.tolist(), ew.tolist()):
            ex_by_term[t].append((pid, w))
    for t, lst in ex_by_term.items():
        lst.sort()
        docs = np.array([p for p, _ in lst], np.uint32)
        _, _, _, packed = m._for_pack_gaps(docs)
        extra_di_b += packed.nbytes; extra_wt_b += len(lst)

    # cluster-id per doc: ceil(log2 K) bits  (contiguous docs only)
    clusterid_bits = n_contig * max(1, int(math.ceil(math.log2(max(2, K)))))

    # base profile is shared structure stored ONCE for the whole index; amortize over contiguous pop
    # extra weights cost wbits each (not 8); base weights kept at 8 (tiny, shared once)
    total_bits = (keepmask_bits + delta_bytes_bits + clusterid_bits
                  + extra_di_b * 8 + extra_wt_b * wbits
                  + base_di_b * 8 + base_wt_b * 8)
    total_B = total_bits / 8
    nc = max(1, n_contig)
    B_per_doc = total_B / nc

    return dict(
        B_per_doc=B_per_doc,
        keepmask_B=keepmask_bits / 8 / nc,
        delta_B=delta_bytes_bits / 8 / nc,
        clusterid_B=clusterid_bits / 8 / nc,
        extra_di_B=extra_di_b / nc, extra_wt_B=extra_wt_b / nc, extra_ppd=extra_n / nc,
        base_di_B=base_di_b / nc, base_wt_B=base_wt_b / nc, base_ppd=base_n / max(1, len(members)),
        dstep=dstep, p95resid=float(p95),
        dec_terms=dec_terms, dec_w=dec_w,
    )


# ----------------------------------------------------------------------------------------------
# 7. Score dev-small queries over a reconstructed (term->weight) doc set -> MRR@10, recall@100.
#    We build an inverted index from the decoded reps and run the exact sparse-dot (search()).
# ----------------------------------------------------------------------------------------------
def eval_reps(doc_ids, dec_terms, dec_w, queries, qenc, qrels, k=100):
    nd = len(doc_ids)
    # invert decoded reps
    tloc = defaultdict(lambda: ([], []))
    present = np.array(doc_ids, np.int64)
    local = {int(d): i for i, d in enumerate(present)}
    inv = defaultdict(list)  # term -> list of (local, weight)
    for i in range(nd):
        for t, w in zip(dec_terms[i].tolist(), dec_w[i].tolist()):
            inv[int(t)].append((i, w))
    # to arrays
    inv_arr = {}
    for t, lst in inv.items():
        loc = np.array([x for x, _ in lst], np.int64)
        w = np.array([y for _, y in lst], np.float32)
        o = np.argsort(loc)
        inv_arr[t] = (loc[o], w[o])
    acc = np.zeros(nd, np.float32)
    mrr = 0.0; rec = 0; scored = 0
    present_set = set(int(d) for d in present)
    answerable = [(qid, qt) for qid, qt in queries if qrels[qid] & present_set]
    for qid, _ in answerable:
        ids, qw = qenc[qid]
        acc[:] = 0.0; touched = []
        for tid, qweight in zip(ids.tolist(), qw.tolist()):
            ta = inv_arr.get(int(tid))
            if ta is None:
                continue
            loc, w = ta
            acc[loc] += float(qweight) * w
            touched.append(loc)
        if not touched:
            continue
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        if len(cand) > k:
            sel = np.argpartition(-sc, k)[:k]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        top = [int(present[cand[j]]) for j in order]
        gold = qrels[qid]
        scored += 1
        if any(d in gold for d in top):
            rec += 1
        for r, d in enumerate(top[:10]):
            if d in gold:
                mrr += 1.0 / (r + 1); break
    n = max(1, scored)
    return mrr / n, rec / n * 100, scored


def load_queries():
    qrels = defaultdict(set)
    with open(os.path.join(str(m.MARCO), "qrels.dev.small.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(os.path.join(str(m.MARCO), "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    return queries, qrels


def main():
    t0 = time.time()
    print("== loading docs (all present; footprint amortized over contiguous <50000) ==", flush=True)
    doc_ids, rows, contig = load_docs(50000)
    nd = len(doc_ids); nc = int(contig.sum())
    print(f"  {nd} present docs ({nc} contiguous), ppd={sum(len(r[0]) for r in rows)/nd:.1f}", flush=True)

    print("== baseline footprint (contiguous FOR di + uint8 wt) ==", flush=True)
    base = baseline_footprint(doc_ids, rows, contig)
    print(f"  baseline B/doc = {base['B_per_doc']:.1f}  (di {base['di_b']/nc:.1f} + wt {base['wt_b']/nc:.1f}), "
          f"ppd={base['ppd']:.1f}", flush=True)

    print("== load queries + encode (SPLADE) ==", flush=True)
    queries, qrels = load_queries()
    present_set = set(doc_ids)
    answerable = [(qid, qt) for qid, qt in queries if qrels[qid] & present_set]
    print(f"  dev-small q={len(queries)}, gold-in-50k-index={len(answerable)}", flush=True)
    qenc = {}
    qtexts = [qt for _, qt in queries]
    for b0 in range(0, len(qtexts), m.BATCH):
        reps = m.splade_sparse(qtexts[b0:b0 + m.BATCH], m.QUERY_ML, topk=10000, minw=m.MINW)
        for (qid, _), rep in zip(queries[b0:b0 + m.BATCH], reps):
            qenc[qid] = rep

    # ---- baseline MRR: score on the ORIGINAL (uncompressed) reps ----
    print("== baseline MRR (uncompressed reps) ==", flush=True)
    dec_t0 = [r[0].astype(np.int64) for r in rows]
    dec_w0 = [r[1].astype(np.float32) for r in rows]
    bmrr, brec, bscored = eval_reps(doc_ids, dec_t0, dec_w0, queries, qenc, qrels)
    print(f"  baseline MRR@10={bmrr:.4f}  recall@100={brec:.2f}%  (scored {bscored})", flush=True)

    # ---- clustering ----
    indptr, cols, vals = build_sparse(doc_ids, rows)

    print("\n== CHAMBER-RESIDUAL SHRINK-vs-ACCURACY CURVE ==", flush=True)
    print(f"  {'config':<46}{'B/doc':>8}{'shrink':>8}{'MRR@10':>9}{'ret%':>7}{'rec@100':>9}", flush=True)
    print(f"  {'BASELINE (uncompressed)':<46}{base['B_per_doc']:>8.1f}{1.0:>8.2f}{bmrr:>9.4f}{100.0:>7.1f}{brec:>9.2f}", flush=True)

    results = []
    # (A) pure 32-chamber-by-dominant-term (Timothy's literal ask)
    dom = doc_topterms(rows)
    cham32 = np.array([term_chamber(t) for t in dom.tolist()], np.int32)
    configs = []
    configs.append(("32-chamber(dominant term)", cham32, 32, dict(base_terms_cap=64, delta_bits=4, base_min_frac=0.25)))

    # (B) k-means at K in {32,64,128,256} x bit budgets
    Ks = [int(x) for x in os.environ.get("KS", "64,128,256").split(",")]
    iters = int(os.environ.get("ITERS", "10"))
    for K in Ks:
        assign = kmeans_assign(indptr, cols, vals, nd, K, vocab_keep=4000, iters=iters)
        for cap, db, bmf in [(96, 4, 0.20), (128, 4, 0.15), (160, 5, 0.12), (200, 5, 0.10)]:
            configs.append((f"kmeans K={K} cap={cap} db={db} f={bmf}", assign, K,
                            dict(base_terms_cap=cap, delta_bits=db, base_min_frac=bmf)))

    for name, assign, K, kw in configs:
        rb = build_residual(doc_ids, rows, assign, K, contig, **kw)
        mrr, rec, scored = eval_reps(doc_ids, rb["dec_terms"], rb["dec_w"], queries, qenc, qrels)
        shrink = base["B_per_doc"] / rb["B_per_doc"]
        ret = 100.0 * mrr / max(1e-9, bmrr)
        results.append((name, rb["B_per_doc"], shrink, mrr, ret, rec, rb))
        print(f"  {name:<46}{rb['B_per_doc']:>8.1f}{shrink:>8.2f}{mrr:>9.4f}{ret:>7.1f}{rec:>9.2f}", flush=True)

    # pick best: highest shrink with retention >= 98%, else highest retention
    good = [r for r in results if r[4] >= 98.0]
    if good:
        best = max(good, key=lambda r: r[2])
    else:
        best = max(results, key=lambda r: r[4])
    print("\n== BEST POINT ==", flush=True)
    print(f"  {best[0]}: B/doc={best[1]:.1f}  shrink={best[2]:.2f}x  MRR={best[3]:.4f}  ret={best[4]:.1f}%  rec@100={best[5]:.2f}%", flush=True)
    rb = best[6]
    print(f"  payload breakdown (B/doc): base_di={rb['base_di_B']:.1f} base_wt={rb['base_wt_B']:.1f} "
          f"keepmask={rb['keepmask_B']:.1f} delta={rb['delta_B']:.1f} clusterid={rb['clusterid_B']:.1f} "
          f"extra_di={rb['extra_di_B']:.1f} extra_wt={rb['extra_wt_B']:.1f}", flush=True)
    print(f"\n  total wall {time.time()-t0:.0f}s", flush=True)

    # emit machine-readable summary
    print("\n__RESULT__", flush=True)
    import json
    print(json.dumps(dict(
        baseline_Bdoc=base["B_per_doc"], baseline_mrr=bmrr, baseline_rec=brec,
        best_name=best[0], best_Bdoc=best[1], best_shrink=best[2],
        best_mrr=best[3], best_ret=best[4], best_rec=best[5],
        n_docs=nd, n_answerable=len(answerable),
    )), flush=True)


if __name__ == "__main__":
    main()
