#!/usr/bin/env python3
"""Targeted by the glass-box: 60% of missed gold is HOP-2 reachable but the rare bridge term is drowned among
noisy 2-hop terms. Test the rules that surface the RARE discriminative bridge: select 2-hop expansion by
PMI*IDF (rare bridges win, not common high-PMI), gate expansion to discriminative query terms, and capture
how much of the hop-2-reachable gold each rule recovers."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from _o1_drift import build_drift, score_bag
from _o1_propagate import propagate


def idf_of(eng, w):
    tid = eng._term_id(w)
    if tid is None: return 0.0
    a, e = int(eng.indptr[tid]), int(eng.indptr[tid + 1])
    return eng._idf(e - a) if e > a else 0.0


def bag_v(query, drift, prop, eng, a1=0.4, a2=0.4, sel="pmi", gate=0.0, m1=24, m2=24):
    qw = list(dict.fromkeys(words(query)))
    seed = [w for w in qw if idf_of(eng, w) >= gate] or qw          # gate: expand only discriminative terms
    bag = {w: 1.0 for w in qw}
    for src, alpha, mx in [(drift, a1, m1), (prop, a2, m2)]:
        acc = Counter()
        for w in seed:
            for cw, s in src.get(w, ()):
                if cw not in bag: acc[cw] += s
        if not acc: continue
        if sel == "pmi_idf":   items = sorted(acc.items(), key=lambda x: -(x[1] * idf_of(eng, x[0])))
        elif sel == "idf":     items = sorted(acc.items(), key=lambda x: -idf_of(eng, x[0]))
        else:                  items = acc.most_common()
        mxv = max(acc.values())
        for cw, m in items[:mx]: bag[cw] = bag.get(cw, 0.0) + alpha * (m / mxv)
    return bag


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    allw = {w for d in corpus.values() for w in words(d)}
    drift, _, _ = build_drift(corpus, allw); prop = propagate(drift)
    print("=" * 86); print(f"GLASS-BOX-TARGETED RECALL — {name}: {len(ids)} q (surface the rare hop-2 bridge)"); print("=" * 86)

    def evl(cfg):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = score_bag(eng, bag_v(queries[qid], drift, prop, eng, **cfg))
            top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    configs = [
        ("lexical (a1=0,a2=0)", dict(a1=0.0, a2=0.0)),
        ("1-hop drift", dict(a1=0.4, a2=0.0)),
        ("+2-hop PMI", dict(a1=0.4, a2=0.4, sel="pmi")),
        ("+2-hop PMI*IDF", dict(a1=0.4, a2=0.4, sel="pmi_idf")),
        ("+2-hop IDF-first", dict(a1=0.4, a2=0.4, sel="idf")),
        ("+2-hop PMI*IDF gate2", dict(a1=0.4, a2=0.4, sel="pmi_idf", gate=2.0)),
        ("+2-hop PMI*IDF m2=40", dict(a1=0.4, a2=0.5, sel="pmi_idf", m2=40)),
    ]
    print(f"  {'config':<26}{'recall@100':>12}{'nDCG@10':>10}{'  vs 1-hop':>12}")
    base = None; best = (0, "")
    for nm, cfg in configs:
        r, n = evl(cfg)
        if nm == "1-hop drift": base = r
        d = f"{r-base:+.4f}" if base is not None else ""
        if base is not None and r > best[0]: best = (r, nm)
        print(f"  {nm:<26}{r:>12.4f}{n:>10.4f}{d:>12}")
    print(f"  BEST: {best[1]} (recall {best[0]:.4f}, {best[0]-base:+.4f} vs 1-hop drift)")


if __name__ == "__main__":
    main()
