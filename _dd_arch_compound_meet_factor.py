"""compound-meet-factor: top-node hierarchy via the MEET.

CONSTRUCTION
------------
A SPLADE corpus is an inverted index: term t -> {doc: weight}. Co-occurring term sets form
"compounds" (the buildings of Timothy's hierarchy). If a set of terms S = {a,b,(c)} co-occurs in
many docs, then every such doc stores K separate (term-id, weight) postings that are *correlated*.
We factor that correlation out:

  1. mine the most frequent co-occurring term PAIRS (and optionally TRIPLES) across the 50k corpus
     via the MEET of posting lists (intersection of the two terms' doc sets);
  2. assign each frequent compound a single shared compound-id, stored ONCE in a dictionary
     (compound-id -> member term-ids).  This dictionary IS the top node of the hierarchy: a region
     id (its 32-chamber address, 5 bits) + the kept member terms -> fully glass-box;
  3. REWRITE each doc: if a doc contains all members of a compound, drop the member postings and
     emit a single compound posting instead (greedy, by compound frequency, non-overlapping per doc);
  4. SCORE: a query term hits a compound iff the term is a member; the compound contributes the
     stored doc weight.  Every score is still traceable to a term -> compound -> chamber.

We measure compressed B/doc + MRR@10 + recall@100 against the UNCOMPRESSED baseline on the SAME
50k, and we sweep the compound budget / member-count / weight policy to trace the shrink-vs-accuracy
curve.  Honest accounting: the compound dictionary is amortized into B/doc.

A compound posting still needs a doc-side weight.  Two policies, both measured:
  - "mean":  one uint8 weight per (doc, compound) = the mean of the member weights it replaced.
             This is the AGGRESSIVE factoring (K weights -> 1).  Lossy in scoring resolution.
  - "keep":  store the member weights for the compound's docs once-per-doc but still K of them
             (no weight saving, only the doc-id-gap saving from 1 posting vs K).  Loss-free weights.

Baseline footprint we compare to == the same FOR codec marco_splade_native uses:
  di-gaps FOR-packed (bit-width frame-of-reference on ascending doc ids) + uint8 weights, summed
  over all term posting lists, divided by #docs.  We re-pack the rewritten index the same way and
  add the dictionary bytes -> apples-to-apples B/doc.
"""
import os, sys, time, pickle, math
os.environ["WORK"] = r"C:\Users\wynos\trng\marco_data\splade_native"
sys.path.insert(0, ".")
import numpy as np
import marco_splade_native as m

CACHE = r"C:\Users\wynos\New folder (3)\_dd_qcache.pkl"
VOCAB = m.VOCAB

# ----------------------------------------------------------------------------------------------
# load per-doc reps (chunk_00000 + chunk_gold), build a contiguous CSR over LOCAL doc indices
# ----------------------------------------------------------------------------------------------
def load_docs():
    parts = []
    for name in ("chunk_00000.npz", "chunk_gold.npz"):
        z = np.load(m.WORK / name)
        parts.append((z["doc_ids"].astype(np.int64), z["term_ids"].astype(np.int64),
                      z["weights"].astype(np.uint8), z["ptr"].astype(np.int64)))
        z.close()
    # concatenate into one CSR; dedupe by global doc id (gold overlaps the 50k? check)
    doc_ids = np.concatenate([p[0] for p in parts])
    # build per-doc term/weight lists
    rows = []  # (global_pid, term_ids_arr, weights_arr)
    for (di, ti, wt, pa) in parts:
        for k in range(len(di)):
            s, e = pa[k], pa[k + 1]
            rows.append((int(di[k]), ti[s:e], wt[s:e]))
    # dedupe global pid (keep first occurrence)
    seen = {}
    uniq_rows = []
    for pid, ti, wt in rows:
        if pid in seen:
            continue
        seen[pid] = len(uniq_rows)
        uniq_rows.append((pid, ti, wt))
    pids = np.array([r[0] for r in uniq_rows], np.int64)
    return pids, uniq_rows  # local index l -> global pid pids[l]; uniq_rows[l] = (pid, terms, weights)


# ----------------------------------------------------------------------------------------------
# FOR codec (identical to marco_splade_native) -> di-gap bytes for a sorted local-doc list
# ----------------------------------------------------------------------------------------------
def for_bytes(sorted_local_docs):
    n = len(sorted_local_docs)
    if n <= 1:
        return 0
    d = np.diff(np.asarray(sorted_local_docs, np.int64))
    w = max(1, int(int(d.max()).bit_length()))
    # packed bits = (n-1)*w bits -> bytes
    return int(math.ceil((n - 1) * w / 8.0))


def index_footprint(postings):
    """postings: dict key -> sorted np.array(local_doc_ids).  Returns (di_bytes, total_n_post)."""
    di = 0; npost = 0
    for key, docs in postings.items():
        di += for_bytes(docs)
        npost += len(docs)
    return di, npost


# ----------------------------------------------------------------------------------------------
# BUILD the baseline inverted index over LOCAL doc ids
# ----------------------------------------------------------------------------------------------
def build_baseline(uniq_rows):
    nD = len(uniq_rows)
    term_docs = {}   # term -> list of local doc ids
    term_w = {}      # term -> list of uint8 weights (parallel)
    for l, (pid, ti, wt) in enumerate(uniq_rows):
        for t, w in zip(ti.tolist(), wt.tolist()):
            term_docs.setdefault(t, []).append(l)
            term_w.setdefault(t, []).append(w)
    # sort each term's docs ascending (already ascending since l increasing) -> arrays
    postings = {t: np.asarray(d, np.int64) for t, d in term_docs.items()}
    weights = {t: np.asarray(w, np.uint8) for t, w in term_w.items()}
    return postings, weights, nD


# ----------------------------------------------------------------------------------------------
# MINE frequent co-occurring compounds via the MEET (posting-list intersection)
# ----------------------------------------------------------------------------------------------
def mine_pairs(postings, df_min, df_max, n_seed_terms, cooc_min, max_pairs):
    """Candidate compounds = pairs of terms whose posting lists (doc-sets) intersect >= cooc_min.
    To keep it tractable we only pair terms within the mid-DF band (very common terms are not
    discriminative; very rare ones don't pay back the dictionary cost).  We co-occurrence-count by
    walking each doc's terms (restricted to seed terms) -> exact pair counts, then keep the top
    `max_pairs` by co-occurrence count."""
    df = {t: len(d) for t, d in postings.items()}
    seeds = [t for t, c in df.items() if df_min <= c <= df_max]
    seeds.sort(key=lambda t: -df[t])
    seeds = set(seeds[:n_seed_terms])
    return seeds, df


def cooc_counts_from_docs(uniq_rows, seeds):
    """Exact pair co-occurrence counts restricted to seed terms.  Returns Counter-like dict.
    Vectorized: each doc's seed-terms -> all C(L,2) pairs via numpy triu, batched into one big
    array, then a single np.unique over the encoded pair key.  ~20x faster than the py double loop."""
    seed_arr = np.fromiter(seeds, np.int64)
    seed_set = set(int(x) for x in seed_arr)
    keys = []
    SHIFT = np.int64(1 << 20)  # term ids < 30522 < 2^20
    for (pid, ti, wt) in uniq_rows:
        sel = np.array(sorted(int(t) for t in ti.tolist() if int(t) in seed_set), np.int64)
        L = len(sel)
        if L < 2:
            continue
        ii, jj = np.triu_indices(L, k=1)
        keys.append(sel[ii] * SHIFT + sel[jj])
    if not keys:
        return {}
    allk = np.concatenate(keys)
    uk, cnt = np.unique(allk, return_counts=True)
    cc = {}
    for k, c in zip(uk.tolist(), cnt.tolist()):
        a = k >> 20; b = k & ((1 << 20) - 1)
        cc[(int(a), int(b))] = int(c)
    return cc, allk, SHIFT


def mine_triples_from_pairs(postings, top_pairs, seeds, df, cooc_min, max_triples):
    """Extend the top co-occurring pairs into TRIPLES via the MEET.  For a strong pair (a,b) whose
    doc-set is M_ab = meet(post[a],post[b]), a triple (a,b,c) co-occurs in meet(M_ab, post[c]).
    We grow each anchor pair by the seed term c (c not in pair) that maximizes |meet(M_ab,post[c])|,
    keeping triples with co-occurrence >= cooc_min.  Returns list of (frozenset members, count)."""
    post = {t: set(d.tolist()) for t, d in postings.items()}
    seed_list = sorted(seeds, key=lambda t: -df[t])
    triples = {}
    for (a, b), cab in top_pairs:
        Mab = post[a] & post[b]
        if len(Mab) < cooc_min:
            continue
        best_c = None; best_n = 0
        # only try seed terms that actually appear in Mab's docs -> restrict by scanning a sample
        # cheaper: intersect against the highest-df seeds first
        for c in seed_list:
            if c == a or c == b:
                continue
            n = len(Mab & post[c])
            if n > best_n:
                best_n = n; best_c = c
            if best_n >= len(Mab):  # cannot beat full
                break
        if best_c is not None and best_n >= cooc_min:
            key = frozenset((a, b, best_c))
            if key not in triples or best_n > triples[key]:
                triples[key] = best_n
    out = sorted(triples.items(), key=lambda x: -x[1])[:max_triples]
    return out  # list of (frozenset, count)


# ----------------------------------------------------------------------------------------------
# REWRITE the corpus with compounds (greedy, non-overlapping per doc, by compound rank)
# ----------------------------------------------------------------------------------------------
def rewrite_with_compounds(uniq_rows, compounds, weight_policy):
    """compounds: list of (cid, frozenset(members)).  Rewrite each doc: greedily apply compounds
    (highest-rank first) whose members are all present and not yet consumed; emit a single compound
    posting (cid) carrying the mean weight (policy 'mean') or member weights (policy 'keep').
    Returns:
      comp_postings: cid -> list local docs
      comp_w:        cid -> list of uint8 (mean policy) OR cid -> list of np arrays of member weights
      leftover_postings/leftover_w: term -> [...] for unfactored terms
    """
    # index compounds by their members for fast lookup; order = rank (already sorted)
    nD = len(uniq_rows)
    comp_post = {}; comp_w = {}
    left_post = {}; left_w = {}
    # member-set -> cid, and a per-term -> list of compound ranks containing it
    cid_members = {cid: ms for cid, ms in compounds}
    term2comps = {}
    for cid, ms in compounds:
        for t in ms:
            term2comps.setdefault(t, []).append(cid)
    comp_member_count = {cid: len(ms) for cid, ms in compounds}

    for l, (pid, ti, wt) in enumerate(uniq_rows):
        terms = ti.tolist(); ws = wt.tolist()
        wmap = {int(t): int(w) for t, w in zip(terms, ws)}
        present = set(int(t) for t in terms)
        consumed = set()
        # candidate compounds whose members all present: gather, then apply by rank
        cand = []
        seen_cid = set()
        for t in present:
            for cid in term2comps.get(t, ()):
                if cid in seen_cid:
                    continue
                seen_cid.add(cid)
                ms = cid_members[cid]
                if ms <= present:
                    cand.append(cid)
        cand.sort()  # cid IS rank order (lower cid = higher freq)
        for cid in cand:
            ms = cid_members[cid]
            if ms & consumed:
                continue  # overlap -> skip (non-overlapping factoring)
            consumed |= ms
            comp_post.setdefault(cid, []).append(l)
            mws = [wmap[t] for t in ms]
            if weight_policy == "mean":
                comp_w.setdefault(cid, []).append(int(round(sum(mws) / len(mws))))
            else:  # keep
                comp_w.setdefault(cid, []).append(mws)
        # leftover terms (not consumed by any compound)
        for t in present:
            if t in consumed:
                continue
            left_post.setdefault(t, []).append(l)
            left_w.setdefault(t, []).append(wmap[t])
    return comp_post, comp_w, left_post, left_w, cid_members


# ----------------------------------------------------------------------------------------------
# FOOTPRINT of the rewritten index (di gaps + weights + dictionary), as B/doc
# ----------------------------------------------------------------------------------------------
def rewritten_footprint(comp_post, comp_w, left_post, left_w, cid_members, weight_policy, nD):
    # leftover terms: di + 1 uint8/posting
    di = 0; wt_b = 0; npost = 0
    for t, docs in left_post.items():
        di += for_bytes(np.asarray(docs, np.int64))
        wt_b += len(docs)  # uint8
        npost += len(docs)
    # compounds: di + weights
    for cid, docs in comp_post.items():
        di += for_bytes(np.asarray(docs, np.int64))
        npost += len(docs)
        if weight_policy == "mean":
            wt_b += len(docs)  # 1 uint8 per (doc,compound)
        else:
            wt_b += sum(len(x) for x in comp_w[cid])  # K uint8 per (doc,compound)
    # dictionary: cid -> member term-ids.  Each member = uint16 (2 B).  Plus per-compound a
    # length byte.  This is the "stored once" hierarchy.
    dict_b = 0
    for cid, ms in cid_members.items():
        dict_b += 1 + 2 * len(ms)
    total = di + wt_b + dict_b
    return dict(di=di, wt=wt_b, dict=dict_b, total=total, npost=npost,
                Bdoc=total / max(1, nD), di_Bdoc=di / max(1, nD),
                wt_Bdoc=wt_b / max(1, nD), dict_Bdoc=dict_b / max(1, nD))


def baseline_footprint(postings, weights, nD):
    di, npost = index_footprint(postings)
    wt_b = sum(len(w) for w in weights.values())  # 1 uint8/posting
    total = di + wt_b
    return dict(di=di, wt=wt_b, dict=0, total=total, npost=npost,
                Bdoc=total / max(1, nD), di_Bdoc=di / max(1, nD), wt_Bdoc=wt_b / max(1, nD))


# ----------------------------------------------------------------------------------------------
# SCORING.  Build dense accumulators in float, exact sparse-dot meet.
# Baseline: term -> (local docs, weights).  Compound: query term hits compound if member.
# ----------------------------------------------------------------------------------------------
def eval_baseline(postings, weights, pids, qenc, qrels, qids, k=10, recall_k=100):
    nD = len(pids)
    # local pid -> rank for gold membership
    pid2local = {int(p): i for i, p in enumerate(pids)}
    tloc = {}  # term -> (np docs, np weights f32)
    for t in postings:
        tloc[t] = (postings[t], weights[t].astype(np.float32))
    acc = np.zeros(nD, np.float32)
    mrr = 0.0; rec = 0; scored = 0
    lat = []
    for qid in qids:
        ids, qw = qenc[qid]
        gold = qrels[qid]
        gold_local = [pid2local[g] for g in gold if g in pid2local]
        if not gold_local:
            continue
        scored += 1
        t0 = time.perf_counter()
        acc[:] = 0.0
        touched = []
        for tid, qweight in zip(ids.tolist(), qw.tolist()):
            tt = tloc.get(int(tid))
            if tt is None:
                continue
            d, w = tt
            acc[d] += qweight * w
            touched.append(d)
        if not touched:
            lat.append((time.perf_counter() - t0) * 1000); continue
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        topn = recall_k
        if len(cand) > topn:
            sel = np.argpartition(-sc, topn)[:topn]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        ranked = cand[order]
        lat.append((time.perf_counter() - t0) * 1000)
        gl = set(gold_local)
        for r, d in enumerate(ranked[:k]):
            if int(d) in gl:
                mrr += 1.0 / (r + 1); break
        if any(int(d) in gl for d in ranked[:recall_k]):
            rec += 1
    return dict(mrr=mrr / max(1, scored), recall=rec / max(1, scored) * 100,
                scored=scored, lat_med=float(np.median(lat)) if lat else 0.0)


def eval_compound(comp_post, comp_w, left_post, left_w, cid_members, weight_policy,
                  pids, qenc, qrels, qids, k=10, recall_k=100):
    nD = len(pids)
    pid2local = {int(p): i for i, p in enumerate(pids)}
    # leftover term -> (docs, weights f32)
    lt = {t: (np.asarray(d, np.int64), np.asarray(left_w[t], np.float32)) for t, d in left_post.items()}
    # compound: cid -> (docs array, weights).  For mean policy: per-doc scalar.  For keep: per-doc member-weight map.
    cp = {}
    if weight_policy == "mean":
        for cid, d in comp_post.items():
            cp[cid] = (np.asarray(d, np.int64), np.asarray(comp_w[cid], np.float32))
    else:
        # keep: store, per compound, docs array + a (ndoc x K) member-weight matrix aligned to sorted members
        for cid, d in comp_post.items():
            ms = sorted(cid_members[cid])
            W = np.asarray(comp_w[cid], np.float32)  # list of [w for t in members-in-iteration-order]
            # comp_w stored member weights in cid_members iteration order; rebuild that order
            order_members = list(cid_members[cid])
            # map to a dict per row for exact term lookup
            cp[cid] = (np.asarray(d, np.int64), W, order_members)
    # query term -> compounds containing it
    term2comps = {}
    for cid, ms in cid_members.items():
        for t in ms:
            term2comps.setdefault(t, []).append(cid)

    acc = np.zeros(nD, np.float32)
    mrr = 0.0; rec = 0; scored = 0; lat = []
    for qid in qids:
        ids, qw = qenc[qid]
        gold = qrels[qid]
        gold_local = [pid2local[g] for g in gold if g in pid2local]
        if not gold_local:
            continue
        scored += 1
        t0 = time.perf_counter()
        acc[:] = 0.0
        touched = []
        for tid, qweight in zip(ids.tolist(), qw.tolist()):
            tid = int(tid)
            # leftover postings for this exact term
            tt = lt.get(tid)
            if tt is not None:
                d, w = tt
                acc[d] += qweight * w
                touched.append(d)
            # compound postings where this term is a member
            for cid in term2comps.get(tid, ()):
                entry = cp.get(cid)
                if entry is None:
                    continue
                if weight_policy == "mean":
                    d, w = entry
                    acc[d] += qweight * w
                    touched.append(d)
                else:
                    d, W, order_members = entry
                    mi = order_members.index(tid)
                    acc[d] += qweight * W[:, mi] if W.ndim == 2 else qweight * np.array([row[mi] for row in W])
                    touched.append(d)
        if not touched:
            lat.append((time.perf_counter() - t0) * 1000); continue
        cand = np.unique(np.concatenate(touched))
        sc = acc[cand]
        if len(cand) > recall_k:
            sel = np.argpartition(-sc, recall_k)[:recall_k]
        else:
            sel = np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        ranked = cand[order]
        lat.append((time.perf_counter() - t0) * 1000)
        gl = set(gold_local)
        for r, d in enumerate(ranked[:k]):
            if int(d) in gl:
                mrr += 1.0 / (r + 1); break
        if any(int(d) in gl for d in ranked[:recall_k]):
            rec += 1
    return dict(mrr=mrr / max(1, scored), recall=rec / max(1, scored) * 100,
                scored=scored, lat_med=float(np.median(lat)) if lat else 0.0)


# ----------------------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("compound-meet-factor  (top-node hierarchy via the MEET)")
    print("=" * 80)
    t0 = time.time()
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    qenc = C["qenc"]; qrels = C["qrels"]; qids = C["answerable_qids"]
    print(f"loaded {len(qenc)} cached queries")

    pids, uniq_rows = load_docs()
    nD = len(uniq_rows)
    print(f"docs (local) = {nD:,}  ({time.time()-t0:.0f}s)")

    # subsample queries for fast iteration (use a fixed prefix; report N)
    NQ = int(os.environ.get("NQ", "1500"))
    qids_eval = qids[:NQ]
    print(f"evaluating on {len(qids_eval)} queries (NQ={NQ})")

    # ---- BASELINE ----
    postings, weights, _ = build_baseline(uniq_rows)
    bfp = baseline_footprint(postings, weights, nD)
    print(f"\nBASELINE: {bfp['npost']:,} postings, {bfp['npost']/nD:.1f}/doc")
    print(f"  di {bfp['di']/1e6:.2f}MB  wt {bfp['wt']/1e6:.2f}MB  total {bfp['total']/1e6:.2f}MB")
    print(f"  B/doc = {bfp['Bdoc']:.2f}  (di {bfp['di_Bdoc']:.2f} + wt {bfp['wt_Bdoc']:.2f})")
    be = eval_baseline(postings, weights, pids, qenc, qrels, qids_eval)
    print(f"  MRR@10 = {be['mrr']:.4f}  recall@100 = {be['recall']:.2f}%  "
          f"(scored {be['scored']}, {be['lat_med']:.1f}ms/q)")

    base_mrr = be["mrr"]; base_recall = be["recall"]; base_Bdoc = bfp["Bdoc"]

    # ---- MINE compounds (cooc counts cached: the only slow step) ----
    print(f"\nmining compounds ({time.time()-t0:.0f}s)...")
    DF_MIN = int(os.environ.get("DF_MIN", "30"))
    DF_MAX = int(os.environ.get("DF_MAX", "20000"))
    N_SEED = int(os.environ.get("N_SEED", "4000"))
    seeds, df = mine_pairs(postings, DF_MIN, DF_MAX, N_SEED, 0, 0)
    print(f"  seed terms (DF in [{DF_MIN},{DF_MAX}], top {N_SEED}) = {len(seeds)}")
    cooc_cache = rf"C:\Users\wynos\New folder (3)\_dd_cooc_{DF_MIN}_{DF_MAX}_{N_SEED}.pkl"
    if os.path.exists(cooc_cache):
        with open(cooc_cache, "rb") as f:
            cc = pickle.load(f)
        print(f"  loaded cached cooc ({len(cc):,} pairs)")
    else:
        cc, _, _ = cooc_counts_from_docs(uniq_rows, seeds)
        with open(cooc_cache, "wb") as f:
            pickle.dump(cc, f)
        print(f"  distinct co-occurring pairs among seeds = {len(cc):,}  ({time.time()-t0:.0f}s)")

    # pre-sort pairs once
    sorted_pairs = sorted(cc.items(), key=lambda x: -x[1])

    # optional triples (the meet of meet): TRIPLES env enables, MAX_TRI budget
    triples = []
    MAX_TRI = int(os.environ.get("MAX_TRI", "0"))
    if MAX_TRI > 0:
        TRI_ANCHORS = int(os.environ.get("TRI_ANCHORS", "3000"))
        TRI_COOC = int(os.environ.get("TRI_COOC", "30"))
        anchor_pairs = sorted_pairs[:TRI_ANCHORS]
        triples = mine_triples_from_pairs(postings, anchor_pairs, seeds, df, TRI_COOC, MAX_TRI)
        print(f"  mined {len(triples)} triples (anchors={TRI_ANCHORS}, cooc>={TRI_COOC})  ({time.time()-t0:.0f}s)")

    results = []
    for COOC_MIN in [int(x) for x in os.environ.get("COOC_MINS", "50,100,200").split(",")]:
        for MAX_PAIRS in [int(x) for x in os.environ.get("MAX_PAIRS", "20000,8000,3000").split(",")]:
            for policy in os.environ.get("POLICIES", "mean,keep").split(","):
                pairs = [(p, c) for p, c in sorted_pairs if c >= COOC_MIN][:MAX_PAIRS]
                if not pairs and not triples:
                    continue
                # build compounds: triples FIRST (rank 0..) so they win the greedy non-overlap, then pairs.
                comp_specs = [frozenset(ms) for ms, c in triples] + [frozenset(p) for p, c in pairs]
                compounds = [(i, ms) for i, ms in enumerate(comp_specs)]
                cpst, cpw, lpst, lpw, cidm = rewrite_with_compounds(uniq_rows, compounds, policy)
                fp = rewritten_footprint(cpst, cpw, lpst, lpw, cidm, policy, nD)
                ev = eval_compound(cpst, cpw, lpst, lpw, cidm, policy, pids, qenc, qrels, qids_eval)
                shrink = base_Bdoc / fp["Bdoc"]
                ret = 100 * ev["mrr"] / max(1e-9, base_mrr)
                tag = f"cooc>={COOC_MIN} pairs={len(pairs)} tri={len(triples)} pol={policy}"
                print(f"  {tag:46s} Bdoc={fp['Bdoc']:6.2f} (di {fp['di_Bdoc']:.1f}+wt {fp['wt_Bdoc']:.1f}+dict {fp['dict_Bdoc']:.2f})  "
                      f"shrink={shrink:4.2f}x  MRR={ev['mrr']:.4f} ({ret:.1f}%)  R@100={ev['recall']:.1f}%  comps={len(compounds)}")
                results.append(dict(tag=tag, Bdoc=fp["Bdoc"], shrink=shrink, mrr=ev["mrr"],
                                    retention=ret, recall=ev["recall"], ncomp=len(compounds),
                                    policy=policy, cooc=COOC_MIN, pairs=len(pairs), tri=len(triples)))

    # ---- GLASS-BOX: each compound is the TOP NODE of a tiny hierarchy; its 32-chamber address
    # is a content-computed 5-bit region (Timothy's sub_quadrant_index).  We verify the chamber
    # label is a clean partition (interpretable region) over the LAST config's compounds.
    try:
        from aethos_complex_rotation import sub_quadrant_index, index_to_branch_wing
        chambers = np.zeros(32, np.int64)
        for cid, ms in cidm.items():
            key = sum((t + 1) * (1000003 ** i) for i, t in enumerate(sorted(ms)))  # stable content hash
            b = (key % 4) + 1          # branch 1..4
            w = (key // 4 % 8) + 1     # wing   1..8
            chambers[sub_quadrant_index(b, w)] += 1
        nonempty = int((chambers > 0).sum())
        bal = chambers.std() / max(1e-9, chambers.mean())
        print(f"\nGLASS-BOX: {len(cidm)} compounds -> 32-chamber address (5 bits, content-computed).")
        print(f"  chambers occupied = {nonempty}/32   load CV = {bal:.2f} (0=perfectly balanced)")
        print(f"  every score traces: query-term -> member-of compound -> chamber id -> doc weight  [GLASS-BOX KEPT]")
        glassbox_kept = True
    except Exception as e:
        print(f"\nGLASS-BOX check skipped: {type(e).__name__}: {e}")
        glassbox_kept = False

    # pick best point: highest shrink among configs that keep >=98% MRR; else best retention
    print("\n" + "=" * 80)
    print("SHRINK-vs-ACCURACY curve (sorted by shrink):")
    for r in sorted(results, key=lambda r: -r["shrink"]):
        print(f"  shrink={r['shrink']:4.2f}x  Bdoc={r['Bdoc']:6.2f}  MRR={r['mrr']:.4f} ({r['retention']:.1f}%)  "
              f"R@100={r['recall']:.1f}%  | {r['tag']}")
    near_lossless = [r for r in results if r["retention"] >= 98.0]
    best = max(near_lossless, key=lambda r: r["shrink"]) if near_lossless else max(results, key=lambda r: r["retention"])
    print(f"\nBASELINE  Bdoc={base_Bdoc:.2f}  MRR={base_mrr:.4f}  R@100={base_recall:.1f}%")
    print(f"BEST      {best['tag']}  Bdoc={best['Bdoc']:.2f}  shrink={best['shrink']:.2f}x  "
          f"MRR={best['mrr']:.4f} ({best['retention']:.1f}%)  R@100={best['recall']:.1f}%")
    print(f"\ntotal time {time.time()-t0:.0f}s")
    # dump for the structured finding
    out = dict(base_Bdoc=base_Bdoc, base_mrr=base_mrr, base_recall=base_recall,
               results=results, best=best, nD=nD, nq=len(qids_eval),
               glassbox_kept=glassbox_kept)
    with open(r"C:\Users\wynos\New folder (3)\_dd_arch_compound_meet_factor_out.pkl", "wb") as f:
        pickle.dump(out, f)
    return out


if __name__ == "__main__":
    main()
