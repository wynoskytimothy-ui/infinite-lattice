#!/usr/bin/env python3
"""MEET POOLS (Timothy): the meet (intersection) of the rarest query terms = a small, PRECISE candidate pool,
built instantly at query time from the postings (free, anchored on the 'top prime' = rarest term). 2-way = rarest
pair's intersection; 3-way = rarest triple. FUSE these precise pools with the diffuse lexical union -> surface the
conjunctive-match gold the union DROWNS, without dropping anything. Tests recall@100 + nDCG@10 of the fusion."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from _o1_drift import build_drift, drift_bag, score_bag


def term_set(eng, w):
    tid = eng._term_id(w)
    if tid is None: return None
    a, e = int(eng.indptr[tid]), int(eng.indptr[tid + 1])
    return set(eng.seg_doc[a:e].tolist()), e - a


def meet_pool(eng, query, scores, nway=2, depth=300):
    """intersect the rarest `nway` query terms -> precise pool; rank by full BM25 score."""
    cand = []
    for w in set(words(query)):
        r = term_set(eng, w)
        if r: cand.append((r[1], r[0]))
    cand.sort(key=lambda x: x[0])                                   # rarest (top prime) first
    if len(cand) < nway: return []
    pool = set(cand[0][1])
    for i in range(1, nway): pool &= cand[i][1]                     # the MEET
    if not pool: return []
    p = np.fromiter(pool, np.int64)
    return list(p[np.argsort(scores[p])[::-1][:depth]])


def rrf(rankings, k0=60, topn=100):
    agg = {}
    for rk in rankings:
        for r, d in enumerate(rk): agg[d] = agg.get(d, 0.0) + 1.0 / (k0 + r)
    return [d for d, _ in sorted(agg.items(), key=lambda x: -x[1])[:topn]]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    drift, _, _ = build_drift(corpus, {w for q in ids for w in words(queries[q])})
    print("=" * 80); print(f"MEET POOLS — {name}: {len(corpus):,} docs, {len(ids)} q"); print("=" * 80)

    def evl(fn):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            ranked = fn(qid)
            rc += len(set(ranked[:100]) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in ranked[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    def lex_rank(qid):
        return list(np.argsort(eng.score(queries[qid]))[::-1][:300])

    def fuse(qid, ways=(), with_drift=False):
        sc = eng.score(queries[qid])
        rks = [list(np.argsort(sc)[::-1][:300])]
        for n in ways:
            mp = meet_pool(eng, queries[qid], sc, nway=n)
            if mp: rks.append(mp)
        if with_drift:
            rks.append(list(np.argsort(score_bag(eng, drift_bag(queries[qid], drift, alpha=0.4)))[::-1][:300]))
        return rrf(rks)

    rows = [("lexical", lambda qid: lex_rank(qid)[:100]),
            ("+ meet-2", lambda qid: fuse(qid, ways=(2,))),
            ("+ meet-2+3", lambda qid: fuse(qid, ways=(2, 3))),
            ("+ meet + drift", lambda qid: fuse(qid, ways=(2, 3), with_drift=True))]
    print(f"  {'config':<18}{'recall@100':>12}{'nDCG@10':>10}{'  vs lexical':>14}")
    base = None
    for nm, fn in rows:
        r, n = evl(fn)
        if base is None: base = r
        print(f"  {nm:<18}{r:>12.4f}{n:>10.4f}{r-base:>+14.4f}")
    print("  meet pool = intersect rarest query terms (free, query-time) -> precise pool fused into the union.")


if __name__ == "__main__":
    main()
