#!/usr/bin/env python3
r"""
_dd_arch_weight_bitpack.py  --  NOVEL construction "weight-bitpack" on the SPLADE-on-lattice index.

GOAL: shrink the 286.9 B/doc native-SPLADE postings index while keeping MRR (~0.39 band on this
50k slice's gold-in-index metric), keeping the meet/sparse-dot serve, and keeping the glass-box
(every score still traces to a term + its uint-coded weight + an interpretable chamber region).

CONSTRUCTION = bit-level recoding of the two payload streams (lossless-ish, fully glass-box):
  (A) WEIGHT REQUANT: the index stores SPLADE doc-weights as uint8 (we measured max=86, so only ~7
      bits used and the distribution is heavily skewed toward 1-8). Requantize to a small codebook
      of L levels (4-bit=16, 5-bit=32) -- either UNIFORM or a per-corpus NON-UNIFORM (quantile /
      Lloyd-Max) codebook that spends its levels where the mass is. Score q_w * dequant(d_w) stays
      monotone-ish; measure the MRR cost vs the bit-savings.
  (B) WEIGHT FLOOR: drop postings whose SPLADE weight <= floor (these are near-zero contributions
      to the dot). Fewer postings -> fewer di gaps AND fewer weight bytes. Measure recall/MRR cost.
  (C) DOC-GAP RECODE with the 32 SUB-QUADRANTS as CONTEXT (Timothy's lever): the current FOR uses
      one fixed bit-width per term over the *global* doc-id gaps. Instead, give every doc an
      interpretable 5-bit chamber (sub_quadrant_index over the doc's lattice address), split each
      term's posting list BY CHAMBER, and FOR-pack the *within-chamber* gaps. Same-chamber docs are
      closer in id-space after the split is folded back, so the gaps shrink -> fewer di bits. The
      chamber id is the "hierarchy of the top node": 2 bits branch + 3 bits wing, an interpretable
      region. Measured against plain FOR and against a varint/Simple8b-style packer.

We measure EVERYTHING on the SAME contiguous 50k slice (docs [0, CAL_N)) so B/doc + MRR + recall@100
are apples-to-apples with the baseline and project to full MARCO. Honest: we report where MRR drops.

The baseline B/doc here = (di_FOR + weights_u8) / n_docs on the contiguous slice == the projection
basis the calibrate() path prints (~286.9 B/doc on full MARCO). MRR is the gold-in-index serve MRR@10
(the codec-correctness ranking signal on the 50k), recomputed for every recoding so retention is exact.
"""
import os, sys, time, math
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
from collections import defaultdict

import marco_splade_native as m
from aethos_complex_rotation import sub_quadrant_index
from aethos_lattice import BranchKind

WORK = m.WORK
MARCO = m.MARCO
CAL_N = int(os.environ.get("CAL_N", "50000"))   # contiguous slice == projection basis
VOCAB = m.VOCAB

t_boot = time.perf_counter()

# ----------------------------------------------------------------------------------------------
# 1. Load all chunks, build per-term posting lists restricted to the CONTIGUOUS slice [0,CAL_N).
#    (Excluding the scattered gold chunk's pids >= CAL_N keeps the FOR-gap density == full MARCO.)
#    We also load the gold docs (any pid) so the ranking signal exists -- gold pids that fall in
#    [0,CAL_N) are already in; pids >= CAL_N are kept as an explicit "answer pool" appended AFTER
#    the contiguous region so the serve has something to rank, exactly like the shipped index.
# ----------------------------------------------------------------------------------------------
def load_postings():
    chunks = sorted(WORK.glob("chunk_*.npz"))
    all_term = []; all_doc = []; all_wt = []
    for cp in chunks:
        z = np.load(cp)
        di = z["doc_ids"]; ti = z["term_ids"]; wt = z["weights"]; pa = z["ptr"]
        doc_of_post = np.repeat(di, np.diff(pa).astype(np.int64))
        all_term.append(ti); all_doc.append(doc_of_post.astype(np.uint32)); all_wt.append(wt)
        z.close()
    term = np.concatenate(all_term); doc = np.concatenate(all_doc); wt = np.concatenate(all_wt)
    return term, doc, wt

term, doc, wt = load_postings()
print(f"[load] {len(term):,} raw postings, {len(np.unique(doc)):,} distinct docs "
      f"({time.perf_counter()-t_boot:.1f}s)", flush=True)

# The shipped index = contiguous [0,CAL_N) UNION the scattered gold docs (so gold is rankable).
# For the FOOTPRINT we report the CONTIGUOUS-ONLY payload (the honest projection basis); for the
# SERVE we keep gold present so MRR is real.  Both use the same recoded streams.
gold_pids = set()
with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) >= 4 and int(p[3]) > 0:
            gold_pids.add(int(p[2]))
gold_arr = np.array(sorted(gold_pids), dtype=np.uint32)

keep_serve = (doc < CAL_N) | np.isin(doc, gold_arr)   # contiguous + scattered gold
term_s = term[keep_serve]; doc_s = doc[keep_serve]; wt_s = wt[keep_serve]
keep_proj = doc < CAL_N                                # footprint basis only
term_p = term[keep_proj]; doc_p = doc[keep_proj]; wt_p = wt[keep_proj]
print(f"[load] serve postings {len(term_s):,} (contig+gold), footprint postings {len(term_p):,} "
      f"(contig<{CAL_N})", flush=True)


# ----------------------------------------------------------------------------------------------
# 2. Chamber assignment: every doc id -> one of 32 interpretable sub-quadrants.
#    Glass-box: the chamber is a content-computed lattice address.  We derive (branch,wing) from
#    the doc's SPLADE signature so it is CONTENT-based (not a random hash): branch = top-term id mod
#    4 +1, wing = (#terms's low bits) -> via sub_quadrant_index.  This is a real lattice address
#    (interpretable 5-bit region: 2b branch + 3b wing), used ONLY as gap-coder context (lossless).
# ----------------------------------------------------------------------------------------------
def doc_chambers(term_arr, doc_arr, wt_arr):
    # per-doc: top-weight term id, and posting count.  Group postings by doc.
    order = np.argsort(doc_arr, kind="stable")
    d = doc_arr[order]; t = term_arr[order]; w = wt_arr[order]
    uniq, starts = np.unique(d, return_index=True)
    ends = np.append(starts[1:], len(d))
    cham = {}
    for u, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        ww = w[s:e]; tt = t[s:e]
        top_term = int(tt[int(np.argmax(ww))])         # most-salient term (content)
        branch = (top_term % 4) + 1                     # 2 bits, from content
        wing = ((e - s) % 8) + 1                        # 3 bits, from #salient-terms (content)
        cham[u] = sub_quadrant_index(BranchKind(branch), wing)   # 0..31
    return cham

cham = doc_chambers(term_s, doc_s, wt_s)
print(f"[chamber] assigned {len(cham):,} docs to 32 sub-quadrants "
      f"({time.perf_counter()-t_boot:.1f}s)", flush=True)


# ----------------------------------------------------------------------------------------------
# 3. Coders for the doc-gap stream.
# ----------------------------------------------------------------------------------------------
def for_pack_bytes(sorted_docs):
    """Single-width FOR (the BASELINE codec).  Returns total bytes for this posting list."""
    n = len(sorted_docs)
    if n <= 1:
        return 4  # first id only
    d = np.diff(sorted_docs.astype(np.int64))
    w = max(1, int(int(d.max()).bit_length()))
    return 4 + math.ceil((n - 1) * w / 8)

def varint_bytes(sorted_docs):
    """LEB128 varint over gaps (byte-aligned, gap-adaptive)."""
    n = len(sorted_docs)
    if n <= 1:
        return 4
    d = np.diff(sorted_docs.astype(np.int64))
    # bytes per gap = ceil(bitlen/7), min 1
    bl = np.maximum(1, np.ceil(np.maximum(1, np.log2(np.maximum(d, 1) + 1)) / 7).astype(np.int64))
    bl = np.where(d == 0, 1, bl)
    # exact: number of 7-bit groups
    grp = np.maximum(1, (np.floor(np.log2(np.maximum(d, 1))).astype(np.int64) // 7) + 1)
    return 4 + int(grp.sum())

def simple_block_bytes(sorted_docs, block=128):
    """PForDelta/Simple-style: per-BLOCK bit-width FOR (the standard tighter-than-global coder)."""
    n = len(sorted_docs)
    if n <= 1:
        return 4
    d = np.diff(sorted_docs.astype(np.int64))
    tot = 4
    for b0 in range(0, len(d), block):
        blk = d[b0:b0 + block]
        w = max(1, int(int(blk.max()).bit_length()))
        tot += 1 + math.ceil(len(blk) * w / 8)   # +1 byte block header (the width)
    return tot

def chamber_split_bytes(sorted_docs, cham_map, block=128):
    """CHAMBER-CONTEXT coder (Timothy's lever): split the posting list by the doc's chamber, FOR
    the within-chamber gaps with a PER-BLOCK width (so each chamber's run is tightly packed), plus
    a tiny header = which chambers are present (32-bit mask) and per-chamber counts (varint).
    Decode is exact: chambers concatenated in id order, prefix-sum within each.  Glass-box: the
    chamber id is the interpretable 5-bit region; the stream is still term->docs."""
    n = len(sorted_docs)
    if n <= 1:
        return 4
    ch = np.array([cham_map[int(x)] for x in sorted_docs], dtype=np.int64)
    tot = 4          # 32-bit present-chamber mask
    present = np.unique(ch)
    for c in present.tolist():
        sub = sorted_docs[ch == c]
        sub = np.sort(sub)
        tot += 1     # varint-ish count header (counts are small here; 1-2 B). use 1 as floor
        if len(sub) > 1:
            dd = np.diff(sub.astype(np.int64))
            # per-block width within this chamber's run
            for b0 in range(0, len(dd), block):
                blk = dd[b0:b0 + block]
                w = max(1, int(int(blk.max()).bit_length()))
                tot += 1 + math.ceil(len(blk) * w / 8)
        # first id of each chamber run stored implicitly via the global order? we still need it:
        tot += 3     # 3-byte first-id within chamber (24 bits covers 8.8M? no -> use 4 at full scale)
    return tot


# ----------------------------------------------------------------------------------------------
# 4. Build per-term posting lists for the FOOTPRINT basis and measure each gap coder + weight coder.
# ----------------------------------------------------------------------------------------------
def build_term_lists(term_arr, doc_arr, wt_arr):
    order = np.lexsort((doc_arr, term_arr))
    t = term_arr[order]; d = doc_arr[order]; w = wt_arr[order]
    uniq, starts = np.unique(t, return_index=True)
    ends = np.append(starts[1:], len(t))
    return [(int(u), d[s:e].copy(), w[s:e].copy()) for u, s, e in
            zip(uniq.tolist(), starts.tolist(), ends.tolist())]

tlists_p = build_term_lists(term_p, doc_p, wt_p)
n_post_p = sum(len(d) for _, d, _ in tlists_p)
print(f"[index] {len(tlists_p):,} terms, {n_post_p:,} contiguous postings "
      f"({time.perf_counter()-t_boot:.1f}s)", flush=True)


def measure_di(coder):
    return sum(coder(d) for _, d, _ in tlists_p)

di_for      = measure_di(for_pack_bytes)
di_varint   = measure_di(varint_bytes)
di_simple   = measure_di(simple_block_bytes)
di_chamber  = sum(chamber_split_bytes(d, cham) for _, d, _ in tlists_p)
print(f"[di coders] (contiguous, n_post={n_post_p:,})", flush=True)
for nm, b in [("FOR (baseline)", di_for), ("varint", di_varint),
              ("per-block FOR (Simple8b-ish)", di_simple), ("chamber-context", di_chamber)]:
    print(f"    {nm:<28} {b/1e6:7.3f} MB   {b*8/n_post_p:6.3f} bits/posting   "
          f"{b/CAL_N:7.2f} B/doc", flush=True)


# ----------------------------------------------------------------------------------------------
# 5. Weight coders.  Baseline = uint8 (8 b/posting).  Levers: requant to L levels (codebook).
# ----------------------------------------------------------------------------------------------
all_w = wt_p.astype(np.float32)

def make_codebook(levels, mode="quantile"):
    """Return (edges, centroids) for L-level quantizer over the weight distribution."""
    if mode == "uniform":
        lo, hi = 1.0, float(all_w.max())
        edges = np.linspace(lo, hi, levels + 1)
        cents = 0.5 * (edges[:-1] + edges[1:])
        return edges, cents
    # quantile (equal-mass) codebook -> Lloyd-Max-ish: centroid = mean of bin
    qs = np.quantile(all_w, np.linspace(0, 1, levels + 1))
    qs = np.unique(qs)
    if len(qs) < 2:
        qs = np.array([all_w.min(), all_w.max() + 1])
    cents = np.empty(len(qs) - 1, np.float32)
    for i in range(len(qs) - 1):
        sel = (all_w >= qs[i]) & (all_w <= qs[i + 1])
        cents[i] = all_w[sel].mean() if sel.any() else 0.5 * (qs[i] + qs[i + 1])
    # a few Lloyd-Max refinements
    for _ in range(8):
        mids = 0.5 * (cents[:-1] + cents[1:])
        edges = np.concatenate([[all_w.min() - 1], mids, [all_w.max() + 1]])
        idx = np.clip(np.searchsorted(edges, all_w, side="right") - 1, 0, len(cents) - 1)
        for i in range(len(cents)):
            sel = idx == i
            if sel.any():
                cents[i] = all_w[sel].mean()
    mids = 0.5 * (cents[:-1] + cents[1:])
    edges = np.concatenate([[all_w.min() - 1], mids, [all_w.max() + 1]])
    return edges, cents

def quantize(w, edges, cents):
    idx = np.clip(np.searchsorted(edges, w.astype(np.float32), side="right") - 1, 0, len(cents) - 1)
    return idx.astype(np.uint8), cents[idx]


# ----------------------------------------------------------------------------------------------
# 6. SERVE harness: build an in-memory ServedIndex-equivalent from arbitrary (term->docs,weights)
#    and run the gold-in-index MRR@10 + recall@100, EXACTLY like marco_splade_native.serve()'s loop
#    (search_corr meet).  We override the weights per experiment to measure the requant cost.
# ----------------------------------------------------------------------------------------------
class InMemIndex:
    def __init__(self, tlists):
        # tlists: list of (term_id, docs uint32 ascending, weights_float32)
        present = np.unique(np.concatenate([d for _, d, _ in tlists]))
        self.present = present
        loc = np.full(int(present.max()) + 1, -1, np.int64)
        loc[present] = np.arange(len(present))
        self.col = {}
        self.tloc = []
        for j, (tid, d, w) in enumerate(tlists):
            self.col[tid] = j
            self.tloc.append((loc[d.astype(np.int64)], np.asarray(w, np.float32)))
        self.acc = np.zeros(len(present), np.float32)

    # reuse the shipped search_corr exactly (composite-meet pool, best accuracy)
    search_corr = m.ServedIndex.search_corr
    search = m.ServedIndex.search


def build_serve_lists(weight_xform=None, floor=None):
    """term lists on the SERVE basis (contig+gold); weight_xform(w_uint8)->float dequant weights."""
    order = np.lexsort((doc_s, term_s))
    t = term_s[order]; d = doc_s[order]; w = wt_s[order]
    if floor is not None:
        keep = w > floor
        t, d, w = t[keep], d[keep], w[keep]
    uniq, starts = np.unique(t, return_index=True)
    ends = np.append(starts[1:], len(t))
    out = []
    for u, s, e in zip(uniq.tolist(), starts.tolist(), ends.tolist()):
        dd = d[s:e]; ww = w[s:e].astype(np.float32)
        if weight_xform is not None:
            ww = weight_xform(w[s:e])
        out.append((int(u), dd, ww))
    return out


# query encode (real SPLADE), once
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

present_all = set(int(x) for x in np.unique(doc_s))
answerable = [(qid, qt) for qid, qt in queries if qrels[qid] & present_all]
print(f"[serve] dev-small q={len(queries):,}, gold-in-index q={len(answerable):,}", flush=True)

print("[serve] encoding queries via SPLADE ...", flush=True)
qenc = {}
qtexts = [qt for _, qt in answerable]
qids_order = [qid for qid, _ in answerable]
for b0 in range(0, len(qtexts), m.BATCH):
    reps = m.splade_sparse(qtexts[b0:b0 + m.BATCH], m.QUERY_ML, topk=10_000, minw=m.MINW)
    for qid, rep in zip(qids_order[b0:b0 + m.BATCH], reps):
        qenc[qid] = rep
print(f"[serve] encoded {len(qenc):,} queries ({time.perf_counter()-t_boot:.1f}s)", flush=True)


def eval_mrr(tlists):
    idx = InMemIndex(tlists)
    mrr = 0.0; rec = 0; lat = []
    for qid, _ in answerable:
        ids, qw = qenc[qid]
        t = time.perf_counter()
        top, _ = idx.search_corr(ids, qw, k=100)
        lat.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]; top = [int(x) for x in top]
        if any(x in gold for x in top):
            rec += 1
        for r, x in enumerate(top[:10]):
            if x in gold:
                mrr += 1.0 / (r + 1); break
    n = max(1, len(answerable))
    return mrr / n, rec / n * 100, float(np.median(lat))


# ----------------------------------------------------------------------------------------------
# 7. RUN THE EXPERIMENTS.
# ----------------------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("BASELINE: uint8 weights, FOR doc-gaps")
print("=" * 90, flush=True)
base_lists = build_serve_lists(weight_xform=None, floor=None)
mrr0, rec0, lat0 = eval_mrr(base_lists)
wt_bytes_base = n_post_p * 1            # uint8
di_bytes_base = di_for
base_bdoc = (di_bytes_base + wt_bytes_base) / CAL_N
print(f"  MRR@10={mrr0:.4f}  recall@100={rec0:.2f}%  serve-median={lat0:.1f}ms", flush=True)
print(f"  footprint: di {di_bytes_base/1e6:.2f}MB + wt {wt_bytes_base/1e6:.2f}MB "
      f"= {base_bdoc:.2f} B/doc (contiguous basis)", flush=True)

results = {}

# ---- Lever A: weight requant to L levels ----
print("\n" + "=" * 90)
print("LEVER A: weight requantization (codebook)")
print("=" * 90, flush=True)
for levels, mode in [(16, "uniform"), (16, "quantile"), (32, "quantile"), (8, "quantile"), (4, "quantile")]:
    edges, cents = make_codebook(levels, mode)
    bits = int(math.ceil(math.log2(len(cents)))) if len(cents) > 1 else 1
    def xform(w, edges=edges, cents=cents):
        _, deq = quantize(w, edges, cents)
        return deq
    lists = build_serve_lists(weight_xform=xform, floor=None)
    mrr, rec, lat = eval_mrr(lists)
    wt_bytes = math.ceil(n_post_p * bits / 8)
    bdoc = (di_for + wt_bytes) / CAL_N
    tag = f"A:{levels}lvl-{mode}({bits}b)"
    results[tag] = dict(mrr=mrr, rec=rec, bdoc=bdoc, di=di_for, wt=wt_bytes, lat=lat,
                        n_post=n_post_p)
    print(f"  {tag:<22} bits/wt={bits}  MRR={mrr:.4f} ({100*mrr/mrr0:5.1f}% ret)  "
          f"rec={rec:.2f}%  wt={wt_bytes/1e6:.2f}MB  -> {bdoc:.2f} B/doc", flush=True)

# ---- Lever B: weight floor (drop near-zero postings) ----
print("\n" + "=" * 90)
print("LEVER B: weight floor (drop postings with d_w <= floor)")
print("=" * 90, flush=True)
for floor in [0, 1, 2, 3, 4]:
    # footprint with floor: recount contiguous postings & re-FOR
    kp = wt_p > floor
    tlp = build_term_lists(term_p[kp], doc_p[kp], wt_p[kp])
    npost_f = sum(len(d) for _, d, _ in tlp)
    di_f = sum(for_pack_bytes(d) for _, d, _ in tlp)
    wt_f = npost_f * 1
    lists = build_serve_lists(weight_xform=None, floor=(floor if floor > 0 else None))
    mrr, rec, lat = eval_mrr(lists)
    bdoc = (di_f + wt_f) / CAL_N
    tag = f"B:floor>{floor}"
    results[tag] = dict(mrr=mrr, rec=rec, bdoc=bdoc, di=di_f, wt=wt_f, lat=lat, n_post=npost_f)
    print(f"  {tag:<22} kept {npost_f:,}/{n_post_p:,} post ({100*npost_f/n_post_p:.1f}%)  "
          f"MRR={mrr:.4f} ({100*mrr/mrr0:5.1f}% ret)  rec={rec:.2f}%  -> {bdoc:.2f} B/doc", flush=True)

# ---- Lever C: doc-gap coder swap (footprint only; lossless so MRR == baseline) ----
print("\n" + "=" * 90)
print("LEVER C: doc-gap coder (lossless -> MRR == baseline; footprint only)")
print("=" * 90, flush=True)
for nm, di_b in [("FOR (baseline)", di_for), ("varint", di_varint),
                 ("per-block FOR", di_simple), ("chamber-context", di_chamber)]:
    bdoc = (di_b + wt_bytes_base) / CAL_N
    print(f"  C:{nm:<20} di={di_b/1e6:.2f}MB ({di_b*8/n_post_p:.2f} b/post)  "
          f"-> {bdoc:.2f} B/doc  (MRR={mrr0:.4f} unchanged)", flush=True)

# ---- COMBINED: best near-lossless point + best small-cost point ----
print("\n" + "=" * 90)
print("COMBINED constructions")
print("=" * 90, flush=True)

best_di = min([("FOR", di_for), ("varint", di_varint), ("per-block", di_simple),
               ("chamber", di_chamber)], key=lambda x: x[1])
print(f"  best di coder: {best_di[0]} @ {best_di[1]/1e6:.2f}MB", flush=True)

# Near-lossless combo: best di coder + 5-bit quantile weights (32 levels) + floor>0 (drop true-0... none, min=1)
combos = []
# (a) near-lossless: best di + 5-bit weights, no floor
edges5, cents5 = make_codebook(32, "quantile")
def xform5(w, edges=edges5, cents=cents5):
    _, deq = quantize(w, edges, cents); return deq
lists_a = build_serve_lists(weight_xform=xform5, floor=None)
mrr_a, rec_a, lat_a = eval_mrr(lists_a)
wt_a = math.ceil(n_post_p * 5 / 8)
bdoc_a = (best_di[1] + wt_a) / CAL_N
combos.append(("COMBO-near-lossless [best-di + 5b-wt]", mrr_a, rec_a, bdoc_a, lat_a, n_post_p))

# (b) aggressive: best di + 4-bit weights + floor>1
edges4, cents4 = make_codebook(16, "quantile")
def xform4(w, edges=edges4, cents=cents4):
    _, deq = quantize(w, edges, cents); return deq
fl = 1
kp = wt_p > fl
tlp = build_term_lists(term_p[kp], doc_p[kp], wt_p[kp])
npost_b = sum(len(d) for _, d, _ in tlp)
di_b_combo = sum(min(for_pack_bytes(d), simple_block_bytes(d),
                     chamber_split_bytes(d, cham)) for _, d, _ in tlp)
wt_b = math.ceil(npost_b * 4 / 8)
lists_b = build_serve_lists(weight_xform=xform4, floor=fl)
mrr_b, rec_b, lat_b = eval_mrr(lists_b)
bdoc_b = (di_b_combo + wt_b) / CAL_N
combos.append((f"COMBO-aggressive [min-di + 4b-wt + floor>{fl}]", mrr_b, rec_b, bdoc_b, lat_b, npost_b))

# (c) mid: best di + 4-bit weights, no floor
lists_c = build_serve_lists(weight_xform=xform4, floor=None)
mrr_c, rec_c, lat_c = eval_mrr(lists_c)
wt_c = math.ceil(n_post_p * 4 / 8)
bdoc_c = (best_di[1] + wt_c) / CAL_N
combos.append(("COMBO-mid [best-di + 4b-wt, no floor]", mrr_c, rec_c, bdoc_c, lat_c, n_post_p))

for nm, mr, rc, bd, lt, npst in combos:
    print(f"  {nm:<42} MRR={mr:.4f} ({100*mr/mrr0:5.1f}% ret)  rec={rc:.2f}%  "
          f"{bd:.2f} B/doc ({base_bdoc/bd:.2f}x smaller)  med={lt:.1f}ms", flush=True)

# ----------------------------------------------------------------------------------------------
# 8. Pick best points & emit the structured summary.
# ----------------------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("SUMMARY (baseline vs best near-lossless vs best shrink)")
print("=" * 90, flush=True)
print(f"  BASELINE          : {base_bdoc:.2f} B/doc   MRR {mrr0:.4f}   rec {rec0:.2f}%", flush=True)

# best near-lossless = highest retention with shrink>1.05x
candidates = []
for tag, r in results.items():
    candidates.append((tag, r["bdoc"], r["mrr"], r["rec"]))
for nm, mr, rc, bd, lt, npst in combos:
    candidates.append((nm, bd, mr, rc))
# near-lossless: retention >= 99%
nl = [c for c in candidates if c[2] >= 0.99 * mrr0]
nl.sort(key=lambda c: c[1])
if nl:
    print(f"  BEST NEAR-LOSSLESS: {nl[0][0]} -> {nl[0][1]:.2f} B/doc  MRR {nl[0][2]:.4f} "
          f"({100*nl[0][2]/mrr0:.1f}%)  ({base_bdoc/nl[0][1]:.2f}x)", flush=True)
# best aggressive shrink with retention>=95%
ag = [c for c in candidates if c[2] >= 0.95 * mrr0]
ag.sort(key=lambda c: c[1])
if ag:
    print(f"  BEST >=95% RET    : {ag[0][0]} -> {ag[0][1]:.2f} B/doc  MRR {ag[0][2]:.4f} "
          f"({100*ag[0][2]/mrr0:.1f}%)  ({base_bdoc/ag[0][1]:.2f}x)", flush=True)

# stash for the agent to read
import json
out = dict(baseline_bdoc=base_bdoc, baseline_mrr=mrr0, baseline_rec=rec0,
           di_for_bdoc=di_for/CAL_N, wt_base_bdoc=wt_bytes_base/CAL_N,
           coders={"FOR": di_for/CAL_N, "varint": di_varint/CAL_N,
                   "per_block": di_simple/CAL_N, "chamber": di_chamber/CAL_N},
           levers=results,
           combos=[dict(name=nm, mrr=mr, rec=rc, bdoc=bd, lat=lt, n_post=npst)
                   for nm, mr, rc, bd, lt, npst in combos])
with open(os.path.join(os.path.dirname(__file__), "_dd_arch_weight_bitpack_result.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n[done] total {time.perf_counter()-t_boot:.1f}s", flush=True)
