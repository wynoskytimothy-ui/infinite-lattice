#!/usr/bin/env python3
"""WAVE 4 — MULTI-HOP DRIFT (Timothy's 2-hop 'drift toward'). 1-hop drift expands query words to their direct
co-occurrence correlates. 2-hop follows the correlates' correlates -- reaching docs that share no DIRECT
correlate but a 2nd-order one. Mass accumulates where multiple paths converge (the higher-D drift-toward).
Tests whether 2-hop raises the recall CEILING (reaches gold 1-hop + the other sources miss)."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from _o1_drift import build_drift, drift_bag, score_bag


def drift_bag_2hop(query, drift, a1=0.4, a2=0.15, max1=24, max2=24):
    qw = list(dict.fromkeys(words(query)))
    bag = {w: 1.0 for w in qw}
    hop1 = Counter()
    for w in qw:
        for cw, pmi in drift.get(w, ()):
            if cw not in bag: hop1[cw] += pmi
    if not hop1: return bag
    mx1 = max(hop1.values())
    hop2 = Counter()                                          # 2nd hop: correlates of the correlates
    for cw, m in hop1.items():
        for cw2, pmi2 in drift.get(cw, ()):
            if cw2 not in bag and cw2 not in hop1: hop2[cw2] += (m / mx1) * pmi2   # weighted by hop1 mass
    for cw, m in hop1.most_common(max1): bag[cw] = bag.get(cw, 0.0) + a1 * (m / mx1)
    if hop2:
        mx2 = max(hop2.values())
        for cw, m in hop2.most_common(max2): bag[cw] = bag.get(cw, 0.0) + a2 * (m / mx2)
    return bag


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    seeds = {w for q in ids for w in words(queries[q])}
    drift1, df, N = build_drift(corpus, seeds)
    # expand seeds with hop-1 correlates so the table has drift[] for them too (needed for hop-2)
    hop1_words = {cw for w in seeds for cw, _ in drift1.get(w, ())}
    drift2, _, _ = build_drift(corpus, seeds | hop1_words)
    print("=" * 84); print(f"WAVE 4 MULTI-HOP DRIFT — {name}: {N:,} docs, {len(ids)} q"); print("=" * 84)

    def evl(scorer, depth=100):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = scorer(queries[qid]); top = np.argsort(sc)[::-1][:depth]
            rc += len(set(top) & gold) / len(gold)
            if depth == 100: nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        return rc / len(ids), (nd / len(ids) if depth == 100 else 0)

    rows = [("lexical", lambda q: eng.score(q)),
            ("drift 1-hop", lambda q: score_bag(eng, drift_bag(q, drift2, alpha=0.4))),
            ("drift 2-hop", lambda q: score_bag(eng, drift_bag_2hop(q, drift2)))]
    print(f"  {'config':<16}{'recall@100':>12}{'nDCG@10':>10}{'recall@300':>12}  (300=drift pool ceiling contribution)")
    for nm, fn in rows:
        r100, n100 = evl(fn, 100); r300, _ = evl(fn, 300)
        print(f"  {nm:<16}{r100:>12.4f}{n100:>10.4f}{r300:>12.4f}")
    print("\n  2-hop reaches 2nd-order correlates; if recall@300 rises over 1-hop, it raises the recoverable ceiling.")


if __name__ == "__main__":
    main()
