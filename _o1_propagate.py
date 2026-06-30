#!/usr/bin/env python3
"""INGEST-TIME SEMANTIC PROPAGATION (Timothy): cross-reference rare terms ACROSS docs to build a transitive
correlation graph BEFORE any query. 1-hop drift = direct co-occurrence (cell<->breast). Propagate 2-hop GLOBALLY
(aggregate over ALL intermediate terms/docs: cell->breast->fatality) and CLEAN with rules (threshold, novelty,
mutual) so transitive noise is dropped at ingest. Tests whether cleaned propagation beats raw 1-hop drift / raw
2-hop (Wave 4). The 'right rules' are the cleaning gates."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from _o1_drift import build_drift, score_bag


def propagate(drift, K=24, mutual=False, min_strength=0.0):
    """2-hop GLOBAL aggregation: prop[i] = sum over intermediate k of pmi(i,k)*pmi(k,j), then clean."""
    prop = {}
    for w, h1 in drift.items():
        h1d = dict(h1); acc = Counter()
        for cw, p1 in h1:
            for cw2, p2 in drift.get(cw, ()):
                if cw2 != w and cw2 not in h1d: acc[cw2] += p1 * p2     # aggregate ALL paths (cross-doc)
        items = sorted(((j, s) for j, s in acc.items() if s >= min_strength), key=lambda x: -x[1])
        if items: prop[w] = items[:K]
    if mutual:                                                          # keep i->j only if j->i also propagated
        keep = {}
        for w, lst in prop.items():
            ml = [(j, s) for j, s in lst if any(k == w for k, _ in prop.get(j, ()))]
            if ml: keep[w] = ml
        prop = keep
    return prop


def exp_bag(query, drift, prop, a1=0.4, a2=0.0, max1=24, max2=16):
    qw = list(dict.fromkeys(words(query))); bag = {w: 1.0 for w in qw}
    h1 = Counter()
    for w in qw:
        for cw, pmi in drift.get(w, ()):
            if cw not in bag: h1[cw] += pmi
    if h1:
        mx = max(h1.values())
        for cw, m in h1.most_common(max1): bag[cw] = bag.get(cw, 0) + a1 * (m / mx)
    if a2 > 0:
        h2 = Counter()
        for w in qw:
            for cw, s in prop.get(w, ()):
                if cw not in bag: h2[cw] += s
        if h2:
            mx2 = max(h2.values())
            for cw, m in h2.most_common(max2): bag[cw] = bag.get(cw, 0) + a2 * (m / mx2)
    return bag


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    seeds = {w for q in ids for w in words(queries[q])}
    drift, _, _ = build_drift(corpus, seeds)
    hop1 = {cw for w in seeds for cw, _ in drift.get(w, ())}
    drift2, _, _ = build_drift(corpus, seeds | hop1)                   # drift table incl. hop-1 words (for 2-hop)
    t0 = time.perf_counter(); prop_raw = propagate(drift2); t_prop = time.perf_counter() - t0
    prop_mut = propagate(drift2, mutual=True)
    print("=" * 88); print(f"INGEST PROPAGATION — {name}: {len(corpus):,} docs, {len(ids)} q (prop built {t_prop:.1f}s)"); print("=" * 88)

    def evl(scorer):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = scorer(queries[qid]); top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    prop_thr = propagate(drift2, min_strength=np.median([s for w in prop_raw for _, s in prop_raw[w]]))
    rows = [("lexical", lambda q: eng.score(q)),
            ("1-hop drift", lambda q: score_bag(eng, exp_bag(q, drift2, prop_raw, a1=0.4, a2=0.0)))]
    for a2 in [0.1, 0.2, 0.3, 0.4]:                                    # RULE: blend weight sweep
        rows.append((f"+ prop raw a2={a2}", lambda q, a=a2: score_bag(eng, exp_bag(q, drift2, prop_raw, a1=0.4, a2=a))))
    rows.append(("+ prop mutual a2=.2", lambda q: score_bag(eng, exp_bag(q, drift2, prop_mut, a1=0.4, a2=0.2))))
    rows.append(("+ prop thresh a2=.2", lambda q: score_bag(eng, exp_bag(q, drift2, prop_thr, a1=0.4, a2=0.2))))
    print(f"  {'config':<24}{'recall@100':>12}{'nDCG@10':>10}{'  vs 1-hop':>12}")
    base = None; best = (0, "")
    for nm, fn in rows:
        r, n = evl(fn)
        if nm == "1-hop drift": base = r
        d = f"{r-base:+.4f}" if base is not None else ""
        if base is not None and r > best[0]: best = (r, nm)
        print(f"  {nm:<24}{r:>12.4f}{n:>10.4f}{d:>12}")
    print(f"  BEST rule: {best[1]} (recall {best[0]:.4f}, {best[0]-base:+.4f} vs 1-hop drift)")
    # glass-box: show a propagated chain
    ex = next((w for w in seeds if w in prop_raw), None)
    if ex:
        print(f"\n  glass-box: '{ex}' 1-hop -> {[c for c,_ in drift2[ex][:5]]}")
        print(f"             '{ex}' PROPAGATED 2-hop -> {[c for c,_ in prop_raw[ex][:5]]}")


if __name__ == "__main__":
    main()
