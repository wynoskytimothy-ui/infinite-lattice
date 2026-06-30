#!/usr/bin/env python3
"""WAVE 2 — DRIFT (Timothy's higher-D idea): a zero-shot, encoder-free attack on the semantic GAP.
Build a word-correlation graph from corpus CO-OCCURRENCE (PMI), then DIFFUSE query words across it. A word that
two query words both drift toward accumulates mass from both -> the 3-way convergence (A,B -> C) lifts C into the
expansion = the 'higher dimension' the correlated words create. Score the drifted query on the lexical lattice.
No model. Tests whether drift recovers GAP docs (gold sharing no literal query word) -- SPLADE's recall, model-free."""
import os, sys, time, math
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words


def build_drift(corpus, seeds, topk=24, df_hi=0.25, max_words=60):
    """Co-occurrence correlation graph -> per-word top-K PMI-correlated words. Built ONLY for `seeds` (the query
    vocab) since drift only expands query words -> the cooc table is bounded by |seeds| -> scales to any corpus."""
    N = len(corpus); df = Counter(); fwd = []
    for d, txt in corpus.items():
        ws = list(dict.fromkeys(words(txt)))[:max_words]
        fwd.append(ws)
        for w in ws: df[w] += 1
    seedset = set(seeds); cap = df_hi * N
    cooc = defaultdict(Counter)                             # keyed by seed words only
    for ws in fwd:
        g = [w for w in ws if 2 <= df[w] <= cap]            # drop hapax + super-common
        sin = [w for w in g if w in seedset]
        if not sin: continue
        for s in sin:
            for w in g:
                if w != s: cooc[s][w] += 1
    drift = {}
    for w, partners in cooc.items():
        dw = df[w]; scored = []
        for cw, c in partners.items():
            if c < 2: continue
            pmi = math.log((c * N) / (dw * df[cw]))         # PMI(w,cw)
            if pmi > 0: scored.append((cw, pmi))
        scored.sort(key=lambda x: -x[1])
        if scored: drift[w] = scored[:topk]
    return drift, df, N


def drift_bag(query, drift, alpha=0.4, max_exp=24):
    """base query words + diffused expansion; expansion mass ACCUMULATES across query words (3-way convergence)."""
    qw = list(dict.fromkeys(words(query)))
    bag = {w: 1.0 for w in qw}
    exp = Counter()
    for w in qw:
        for cw, pmi in drift.get(w, ()):                    # diffuse one hop
            if cw not in bag: exp[cw] += pmi                # accumulate -> words 2+ query terms point to win
    if exp:
        mx = max(exp.values())
        for cw, m in exp.most_common(max_exp):
            bag[cw] = bag.get(cw, 0.0) + alpha * (m / mx)
    return bag


def score_bag(eng, bag):
    s = np.zeros(eng.N); ip, sd, st, den, k1p1 = eng.indptr, eng.seg_doc, eng.seg_tf, eng._denom, eng._k1p1
    for w, qwt in bag.items():
        tid = eng._term_id(w)
        if tid is None: continue
        a, e = int(ip[tid]), int(ip[tid + 1]); dfp = e - a
        if dfp == 0: continue
        di = sd[a:e]; tf = st[a:e].astype(np.float64)
        s[di] += (qwt * eng._idf(dfp) * k1p1) * tf / (tf + den[di])
    return s


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    seeds = {w for q in ids for w in words(queries[q])}     # only expand query words -> bounded graph
    t0 = time.perf_counter(); drift, df, N = build_drift(corpus, seeds); tb = time.perf_counter() - t0
    print("=" * 84); print(f"WAVE 2 DRIFT — {name}: {N:,} docs, {len(ids)} q | graph built in {tb:.1f}s, {len(drift):,} words"); print("=" * 84)

    def evl(scorer):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = scorer(queries[qid]); top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        return rc / len(ids), nd / len(ids)

    import json
    base_r, base_n = evl(lambda q: eng.score(q))
    print(f"  {'config':<18}{'recall@100':>12}{'nDCG@10':>10}{'  vs lexical':>14}")
    print(f"  {'lexical':<18}{base_r:>12.4f}{base_n:>10.4f}{0.0:>+14.4f}")
    best = (base_r, 0.0, base_n); rows = {"lexical": round(base_r, 4)}
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        r, n = evl(lambda q, a=alpha: score_bag(eng, drift_bag(q, drift, alpha=a)))
        print(f"  {'drift a='+str(alpha):<18}{r:>12.4f}{n:>10.4f}{r-base_r:>+14.4f}")
        rows[f"drift_a{alpha}"] = round(r, 4)
        if r > best[0]: best = (r, alpha, n)
    # show an example drift expansion (glass-box)
    ex = next((q for q in ids), None)
    if ex:
        bag = drift_bag(queries[ex], drift, alpha=0.4)
        added = sorted(((w, round(v, 2)) for w, v in bag.items() if w not in set(words(queries[ex]))),
                       key=lambda x: -x[1])[:8]
        print(f"\n  example: '{queries[ex][:50]}...' -> drift adds {added}")
    print(f"\n  best drift recall {best[0]:.4f} (a={best[1]}) vs lexical {base_r:.4f} ({best[0]-base_r:+.4f}); zero-shot, no model.")
    print("JSON " + json.dumps({"corpus": name, "n_docs": N, "n_q": len(ids), "build_s": round(tb, 1),
                                "lexical_recall": round(base_r, 4), "best_drift_recall": round(best[0], 4),
                                "best_alpha": best[1], "recall_gain": round(best[0] - base_r, 4),
                                "best_drift_ndcg": round(best[2], 4), "lexical_ndcg": round(base_n, 4),
                                "all": rows}))


if __name__ == "__main__":
    main()
