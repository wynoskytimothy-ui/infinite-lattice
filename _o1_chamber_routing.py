#!/usr/bin/env python3
r"""LENS: chamber-routing -- 32-chamber routing for SUBLINEAR search.

CLAIM UNDER TEST
  Assign each doc a chamber/cluster signature (from its top terms' meet address).
  At query time, route to only the relevant chamber(s) and scan that subset.
  Does routing cut latency without dropping recall@100 by more than ~1-2%?

HONEST PRIOR (from repo memory)
  The lattice places by VALUE not semantics, so a value-based chamber may scatter
  the docs relevant to a query across many chambers -> routing misses. This script
  MEASURES whether recall survives.

METHOD
  * Baseline = full scatter (ServedIndex.search): exact sparse-dot over the union of
    all query-term postings. This is the gold candidate set + gold ranking.
  * Chamber signature per doc: the lattice "address" of a term is taken from the
    meet/octant structure. We use 32 = 4 branches x 8 wings chambers. A doc's
    chamber(s) come from its TOP-weighted terms' chamber ids (value-based, faithful
    to "place by value"). We test:
       R1  doc -> 1 chamber  (its single top term)            [max reduction]
       R3  doc -> chambers of its top-3 terms (multi-tag)      [more recall]
       R6  doc -> chambers of its top-6 terms
  * Query routes to the chambers of its OWN top-Q terms (Q=3,6). Candidate set =
    docs tagged with any routed chamber. We then run the EXACT same meet score on
    that restricted candidate set, so any recall loss is purely a ROUTING miss
    (relevant doc not in a routed chamber), not a scoring change.
  * Two chamber definitions are tried, both "by value":
       (a) octant-of-rarest: chamber from the lattice meet-address sign/octant of the
           term's value (col id) -> 32 buckets via the (sum,min) triple structure.
       (b) df-bucket x hash: a learned-free balanced hash of the term id into 32.
    (b) is the "balanced" control; (a) is the genuinely lattice-geometric one.

  Metrics on the 6,980 cached dev-small queries:
    - candidate-set reduction %  (median |C_route| / |C_full|)
    - median + p90 latency vs full scatter
    - recall@100 retention vs the FULL-scatter top-100 (routing-miss rate) AND
      vs qrels gold (end metric)
"""
import os, sys, time, pickle
os.environ.setdefault("WORK", r"C:\Users\wynos\trng\marco_data\splade_native")
import numpy as np
import marco_splade_native as m

WORK = os.environ["WORK"]
NCHAM = 32


# ---------------------------------------------------------------------------
#  Chamber assignment for a term column.
#  (a) lattice octant: the meet triple of a small value is (sum,min,..); we form a
#      32-way address from the value's residues that mimic the 4-branch x 8-wing
#      chamber structure (zeta sign x |zeta| mod, balanced over the value space).
# ---------------------------------------------------------------------------
def lattice_chamber(col_ids):
    """col_ids: int array of term column indices (a proxy for the term's lattice value,
    since the index assigns columns in vocab order = value order). Returns chamber in [0,32).
    Faithful-to-lattice: 8 wings = sign pattern of (v, v//k, v//k^2) mod, 4 branches = v mod 4."""
    v = col_ids.astype(np.int64)
    # 8 wings: a 3-coordinate octant of the value under three coprime strides (the
    # octant=Legendre-mod-3.5.7 idea -> here a balanced 3-bit code from coprime mods)
    w0 = (v % 3) >= 2
    w1 = (v % 5) >= 3
    w2 = (v % 7) >= 4
    wing = (w0.astype(np.int64) << 2) | (w1.astype(np.int64) << 1) | w2.astype(np.int64)  # 0..7
    branch = v % 4                                                                          # 0..3
    return (branch * 8 + wing).astype(np.int64)                                             # 0..31


def hash_chamber(col_ids):
    """Balanced control: a learned-free multiplicative hash of the term id into 32 buckets."""
    v = col_ids.astype(np.uint64)
    h = (v * np.uint64(2654435761)) >> np.uint64(27)
    return (h % np.uint64(NCHAM)).astype(np.int64)


def build_doc_chambers(si, term_chamber, top_terms_per_doc):
    """For each present doc, the set of chambers of its top-N highest-weight terms.
    Returns: list-of-arrays index = local doc id -> sorted unique chamber ids,
    plus a boolean tag matrix [n_docs x 32] for fast routing."""
    n = len(si.present)
    # accumulate (doc, chamber, weight) by scanning every term's posting list
    # we want, per doc, its top-`top_terms_per_doc` terms by weight.
    # Build per-doc top-weight term list via a max-heap-free approach: keep running
    # arrays of best weights. For 57k docs x small T this is cheap.
    best_w = np.full((n, top_terms_per_doc), -1.0, np.float32)
    best_c = np.full((n, top_terms_per_doc), -1, np.int64)
    for j in range(len(si.term_ids)):
        loc, w = si.tloc[j]
        ch = term_chamber[j]
        # for each doc touched by this term, try to insert (w, ch) into its top-T
        # vectorized: find docs where this term's weight beats their current min-of-top
        wmin = best_w[loc].min(axis=1)              # current weakest kept weight per touched doc
        better = w > wmin
        if not better.any():
            continue
        bl = loc[better]; bw = w[better]
        # replace the argmin slot
        slot = best_w[bl].argmin(axis=1)
        rows = np.arange(len(bl))
        best_w[bl, slot] = bw
        best_c[bl, slot] = ch
    # build tag matrix
    tag = np.zeros((n, NCHAM), np.bool_)
    for t in range(top_terms_per_doc):
        c = best_c[:, t]
        valid = c >= 0
        tag[np.arange(n)[valid], c[valid]] = True
    return tag


def route_candidates(si, vids, qw, term_chamber, doc_tag, topq):
    """Candidate local-doc set = docs whose tag intersects the chambers of the query's
    top-`topq` terms. Returns sorted unique local ids (the routed candidate pool)."""
    order = np.argsort(-qw)[:topq]
    qcols = []
    for i in order:
        j = si.col.get(int(vids[i]))
        if j is not None:
            qcols.append(j)
    if not qcols:
        return np.zeros(0, np.int64)
    qch = np.unique(term_chamber[np.array(qcols)])
    routed = doc_tag[:, qch].any(axis=1)            # docs in any routed chamber
    return np.nonzero(routed)[0]


def score_on_candidates(si, vids, qw, cand_local, k=100):
    """Exact meet score restricted to cand_local (sorted local ids). Same scoring as
    full scatter -> isolates routing misses."""
    if len(cand_local) == 0:
        return np.zeros(0, np.uint32)
    score = np.zeros(len(cand_local), np.float32)
    for i in range(len(vids)):
        j = si.col.get(int(vids[i]))
        if j is None:
            continue
        loc, w = si.tloc[j]
        pos = np.searchsorted(loc, cand_local)
        pc = np.minimum(pos, len(loc) - 1)
        hit = loc[pc] == cand_local
        score[hit] += float(qw[i]) * w[pc[hit]]
    kk = min(k, len(cand_local))
    sel = np.argpartition(-score, kk - 1)[:kk] if len(cand_local) > kk else np.arange(len(cand_local))
    order = sel[np.argsort(-score[sel])]
    return si.present[cand_local[order]]


def full_candidates(si, vids):
    """The full-scatter candidate set (union of all query-term posting lists) as local ids."""
    touched = []
    for v in vids:
        j = si.col.get(int(v))
        if j is not None:
            touched.append(si.tloc[j][0])
    if not touched:
        return np.zeros(0, np.int64)
    return np.unique(np.concatenate(touched))


def main():
    print("=" * 80)
    print("CHAMBER-ROUTING LENS -- 32-chamber sublinear routing on MARCO SPLADE 50k")
    print("=" * 80, flush=True)
    t0 = time.perf_counter()
    si = m.ServedIndex()
    print(f"  loaded {len(si.term_ids):,} terms, {len(si.present):,} present docs "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # cached queries (NO GPU)
    qcache = pickle.load(open(os.path.join(WORK, "_dd_qenc_cache.pkl"), "rb"))
    # qrels
    from collections import defaultdict
    qrels = defaultdict(set)
    with open(m.MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    present_set = set(int(d) for d in si.present)
    # eval on cached queries that exist (cap for tractable wall-time; many trials for latency)
    qids = [q for q in qcache.keys() if q in qrels]
    print(f"  cached queries with qrels: {len(qids):,}", flush=True)

    # precompute query arrays
    Q = {}
    for qid in qids:
        vids, qw8 = qcache[qid]
        Q[qid] = (vids.astype(np.int64), qw8.astype(np.float32) * m.QSCALE)

    # term chambers (per column j)
    cols = np.arange(len(si.term_ids))
    chambers = {
        "lattice-octant": lattice_chamber(cols),
        "balanced-hash": hash_chamber(cols),
    }
    for name, ch in chambers.items():
        cnt = np.bincount(ch, minlength=NCHAM)
        print(f"  chamber def [{name}]: 32 buckets, term spread min/med/max = "
              f"{cnt.min()}/{int(np.median(cnt))}/{cnt.max()}", flush=True)

    # ---- baseline: full scatter top-100 (gold candidate + ranking) ----
    print("\n  [baseline] full-scatter search() -- computing gold top-100 + latency", flush=True)
    gold_top = {}
    full_cand_sz = []
    lat_full = []
    for qid in qids:
        vids, qw = Q[qid]
        t = time.perf_counter()
        top, _ = si.search(vids, qw, k=100)
        lat_full.append((time.perf_counter() - t) * 1000)
        gold_top[qid] = set(int(d) for d in top)
        full_cand_sz.append(len(full_candidates(si, vids)))
    lat_full = np.array(lat_full); full_cand_sz = np.array(full_cand_sz)
    # baseline accuracy vs qrels
    def acc_vs_gold(top_sets):
        mrr = 0.0; rec = 0; n = 0
        for qid in qids:
            g = qrels[qid]
            if not (g & present_set):
                continue
            n += 1
            top = top_sets[qid]
            if any(d in g for d in top):
                rec += 1
        return rec / max(1, n) * 100, n
    # full-scatter recall@100 (ordered) needs the ordered list; recompute ordered for mrr-less recall
    rec_full, n_ans = None, None
    # recompute ordered recall@100 properly
    def recall100_vs_qrels(get_top_ordered):
        rec = 0; n = 0
        for qid in qids:
            g = qrels[qid]
            if not (g & present_set):
                continue
            n += 1
            top = get_top_ordered(qid)
            if any(int(d) in g for d in top[:100]):
                rec += 1
        return rec / max(1, n) * 100, n

    full_ordered = {}
    for qid in qids:
        vids, qw = Q[qid]
        top, _ = si.search(vids, qw, k=100)
        full_ordered[qid] = [int(d) for d in top]
    rec_full, n_ans = recall100_vs_qrels(lambda q: full_ordered[q])
    print(f"    full-scatter: median {np.median(lat_full):.2f} ms  p90 {np.percentile(lat_full,90):.2f} ms; "
          f"|C_full| median {int(np.median(full_cand_sz)):,}", flush=True)
    print(f"    full-scatter recall@100 vs qrels (gold-in-index, n={n_ans}): {rec_full:.2f}%", flush=True)

    # ---- routing configs ----
    results = []
    configs = [
        ("lattice-octant", 1, 3), ("lattice-octant", 3, 3), ("lattice-octant", 6, 6),
        ("balanced-hash",  1, 3), ("balanced-hash",  3, 3), ("balanced-hash",  6, 6),
    ]
    # cache doc-tag matrices per (chamber_def, top_terms_per_doc)
    tag_cache = {}
    for cname, doc_top, q_top in configs:
        key = (cname, doc_top)
        if key not in tag_cache:
            tb = time.perf_counter()
            tag_cache[key] = build_doc_chambers(si, chambers[cname], doc_top)
            print(f"\n  built doc-tags [{cname}, top{doc_top}/doc] in {time.perf_counter()-tb:.1f}s; "
                  f"avg chambers/doc = {tag_cache[key].sum(axis=1).mean():.2f}", flush=True)
        doc_tag = tag_cache[key]
        term_ch = chambers[cname]

        red = []; lat_r = []; routed_ordered = {}; cand_sz = []
        # warm
        for qid in qids[:5]:
            vids, qw = Q[qid]
            c = route_candidates(si, vids, qw, term_ch, doc_tag, q_top)
            score_on_candidates(si, vids, qw, c, k=100)
        for qi, qid in enumerate(qids):
            vids, qw = Q[qid]
            t = time.perf_counter()
            c = route_candidates(si, vids, qw, term_ch, doc_tag, q_top)
            top = score_on_candidates(si, vids, qw, c, k=100)
            lat_r.append((time.perf_counter() - t) * 1000)
            routed_ordered[qid] = [int(d) for d in top]
            cand_sz.append(len(c))
        cand_sz = np.array(cand_sz)
        # candidate reduction vs full
        ratio = cand_sz / np.maximum(1, full_cand_sz)
        # routing-miss: recall of full-scatter top100 retained
        retain = []
        for qid in qids:
            g = gold_top[qid]
            if not g:
                retain.append(1.0); continue
            r = set(routed_ordered[qid])
            retain.append(len(g & r) / len(g))
        retain = np.array(retain)
        rec_route, _ = recall100_vs_qrels(lambda q: routed_ordered[q])
        lat_r = np.array(lat_r)
        res = dict(cfg=f"{cname} doc-top{doc_top} q-top{q_top}",
                   cand_ratio_med=float(np.median(ratio)),
                   cand_med=int(np.median(cand_sz)),
                   lat_med=float(np.median(lat_r)), lat_p90=float(np.percentile(lat_r, 90)),
                   retain_full_top100=float(np.mean(retain)),
                   rec100_qrels=float(rec_route))
        results.append(res)
        print(f"    [{res['cfg']}]", flush=True)
        print(f"        |C| median {res['cand_med']:,} = {res['cand_ratio_med']*100:.1f}% of full "
              f"(reduction {100-res['cand_ratio_med']*100:.1f}%)", flush=True)
        print(f"        latency median {res['lat_med']:.2f} ms (full {np.median(lat_full):.2f} ms, "
              f"speedup {np.median(lat_full)/max(1e-9,res['lat_med']):.2f}x)  p90 {res['lat_p90']:.2f} ms", flush=True)
        print(f"        retain full-top100 = {res['retain_full_top100']*100:.2f}%  "
              f"(routing-miss {100-res['retain_full_top100']*100:.2f}%)", flush=True)
        print(f"        recall@100 vs qrels = {res['rec100_qrels']:.2f}%  "
              f"(full {rec_full:.2f}%, drop {rec_full-res['rec100_qrels']:+.2f} pp)", flush=True)

    # ---- verdict ----
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80, flush=True)
    print(f"  full-scatter baseline: median {np.median(lat_full):.2f} ms, recall@100 {rec_full:.2f}%", flush=True)
    # best config = highest recall retention with real reduction
    best = max(results, key=lambda r: (r["rec100_qrels"], -r["lat_med"]))
    for r in results:
        drop = rec_full - r["rec100_qrels"]
        ok = "PASS" if drop <= 1.5 else "FAIL"
        spd = np.median(lat_full) / max(1e-9, r["lat_med"])
        print(f"  {r['cfg']:<38} red {100-r['cand_ratio_med']*100:5.1f}%  "
              f"{spd:4.2f}x  rec drop {drop:+5.2f}pp  [{ok}]", flush=True)
    print(f"\n  Honest read: see whether ANY config gives >1x speedup with recall drop <=1.5pp.", flush=True)
    return results, rec_full, float(np.median(lat_full))


if __name__ == "__main__":
    main()
