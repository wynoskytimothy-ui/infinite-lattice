#!/usr/bin/env python3
"""WAVE 1 — glass-box recall diagnostic + zero-shot lexical lever sweep, per corpus.
WHY does the gold doc miss the pool? DROWNED (shares query words but buried by common terms) vs GAP (shares no
query word -> needs semantic). Then sweep the structural levers Timothy named (rare-word anchor, 2-way/3-way
intersection, df-cap, idf^2) and report recall@100 each -> the per-corpus tuning map. No GPU, all zero-shot."""
import os, sys, json, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words


def score_variant(eng, query, mode="bm25", cap=1.0, beta=0.5):
    """Lexical score vector with a lever. matchcount tracks #query-terms each doc hits (for 2/3-way)."""
    N = eng.N; scores = np.zeros(N); mc = np.zeros(N)
    ip, sd, st, den, k1p1 = eng.indptr, eng.seg_doc, eng.seg_tf, eng._denom, eng._k1p1
    for w, qwt in Counter(words(query)).items():
        tid = eng._term_id(w)
        if tid is None: continue
        a, e = int(ip[tid]), int(ip[tid + 1]); dfp = e - a
        if dfp == 0: continue
        if mode == "dfcap" and dfp > cap * N: continue          # drop drowning common terms
        di = sd[a:e]; tf = st[a:e].astype(np.float64)
        idf = eng._idf(dfp)
        if mode == "idf2": idf = idf * idf                      # boost discriminative terms
        scores[di] += (qwt * idf * k1p1) * tf / (tf + den[di]); mc[di] += 1
    if mode == "2way": scores *= (1.0 + beta * np.clip(mc - 1, 0, None))   # boost multi-term matches
    if mode == "3way": scores *= (1.0 + beta * np.clip(mc - 2, 0, None))
    return scores


def recall_ndcg(eng, queries, test_q, ids, d2i, scorer):
    rc = nd = 0.0
    for qid in ids:
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        sc = scorer(queries[qid]); top = np.argsort(sc)[::-1][:100]
        rc += len(set(top) & gold) / len(gold)
        nd += ndcg10([eng.doc_ids[i] for i in top[:10]], test_q[qid])
    return rc / len(ids), nd / len(ids)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    N = eng.N

    # ---- failure diagnostic: drowned vs gap, + how far drowned (gold docs tokenized LAZILY -> any scale) ----
    hit = drowned = gap = 0; drown_ranks = []
    for qid in ids:
        qw = set(words(queries[qid]))
        golds = [d for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in corpus]
        sc = score_variant(eng, queries[qid]); top = set(np.argsort(sc)[::-1][:100])
        if any(d2i[g] in top for g in golds): hit += 1; continue
        shared = [g for g in golds if qw & set(words(corpus[g]))]   # gold shares a query word but missed
        if shared:                                              # -> DROWNED
            drowned += 1
            order = np.argsort(sc)[::-1]; pos = {int(d): r for r, d in enumerate(order)}
            drown_ranks.append(min(pos[d2i[g]] for g in shared))
        else:
            gap += 1                                            # shares no word -> semantic GAP
    nq = len(ids)
    med_drown = int(np.median(drown_ranks)) if drown_ranks else -1

    # ---- lever sweep: recall@100 + nDCG@10 each ----
    levers = {
        "bm25": lambda q: score_variant(eng, q, "bm25"),
        "idf2": lambda q: score_variant(eng, q, "idf2"),
        "dfcap30": lambda q: score_variant(eng, q, "dfcap", cap=0.30),
        "dfcap10": lambda q: score_variant(eng, q, "dfcap", cap=0.10),
        "2way": lambda q: score_variant(eng, q, "2way", beta=0.6),
        "3way": lambda q: score_variant(eng, q, "3way", beta=1.0),
        "anchor": None,
    }
    res = {}
    for k, fn in levers.items():
        if k == "anchor":
            if not getattr(eng, "_sorted", False): eng.sort_segments()
            rc = nd = 0.0
            for qid in ids:
                gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
                top = [d2i[d] for d in eng.search_anchored(queries[qid], 100)]
                rc += len(set(top) & gold) / len(gold); nd += ndcg10(eng.search_anchored(queries[qid], 10), test_q[qid])
            res[k] = (rc / nq, nd / nq)
        else:
            res[k] = recall_ndcg(eng, queries, test_q, ids, d2i, fn)

    base_r = res["bm25"][0]
    best = max(res, key=lambda k: res[k][0])
    out = {"corpus": name, "n_docs": N, "n_q": nq, "hit": hit, "drowned": drowned, "gap": gap,
           "drowned_frac": round(drowned / nq, 3), "gap_frac": round(gap / nq, 3), "median_drown_rank": med_drown,
           "levers": {k: {"recall": round(v[0], 4), "ndcg": round(v[1], 4)} for k, v in res.items()},
           "best_lever": best, "best_recall_gain": round(res[best][0] - base_r, 4)}
    print(f"\n  {name}: {N:,} docs, {nq} q | recall-fail breakdown: DROWNED {drowned}/{nq} ({out['drowned_frac']:.0%}, "
          f"median rank {med_drown}) | GAP {gap}/{nq} ({out['gap_frac']:.0%})")
    print(f"  {'lever':<10}{'recall@100':>12}{'nDCG@10':>10}{'  vs bm25':>10}")
    for k, v in res.items():
        print(f"  {k:<10}{v[0]:>12.4f}{v[1]:>10.4f}{v[0]-base_r:>+10.4f}")
    print(f"  BEST: {best} (recall {res[best][0]:.4f}, {out['best_recall_gain']:+.4f} vs bm25)")
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
