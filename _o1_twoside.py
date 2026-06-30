#!/usr/bin/env python3
"""TWO-SIDED EXPANSION (Timothy) = model-free SPLADE. Use the co-occurrence semantic graph on BOTH sides:
DOC-side at ingest (enrich each doc's postings with its correlates: a 'breast cancer' doc gets 'female/fatality'
even though it never says them) AND QUERY-side. Both meet in the enriched space. Measures the 4 quadrants
(plain/expanded query x original/expanded index) to isolate where the recall comes from + the footprint cost."""
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

DOC_M = 20   # expansion terms appended per doc


def doc_expand(text, drift, doc_set, M=DOC_M):
    acc = Counter()
    for w in set(words(text)):
        for cw, pmi in drift.get(w, ()):
            if cw not in doc_set: acc[cw] += pmi             # accumulate correlates across the doc's terms
    return [cw for cw, _ in acc.most_common(M)]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]

    # one graph for BOTH sides (built from the whole corpus vocab, not just query seeds)
    allw = {w for d in corpus.values() for w in words(d)}
    t0 = time.perf_counter(); drift, _, _ = build_drift(corpus, allw); t_g = time.perf_counter() - t0

    eng = EdgeRAG().build(corpus)                            # original index
    t0 = time.perf_counter()
    expanded = {}
    for d, txt in corpus.items():
        ex = doc_expand(txt, drift, set(words(txt)))
        expanded[d] = txt + " " + " ".join(ex)
    eng_x = EdgeRAG().build(expanded)                        # doc-expanded index
    t_x = time.perf_counter() - t0
    grow = len(eng_x.seg_doc) / len(eng.seg_doc)
    print("=" * 88); print(f"TWO-SIDED EXPANSION — {name}: {len(corpus):,} docs (graph {t_g:.1f}s, doc-expand {t_x:.1f}s)"); print("=" * 88)

    def evl(engine, query_expand):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            sc = score_bag(engine, drift_bag(queries[qid], drift, alpha=0.4)) if query_expand else engine.score(queries[qid])
            top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    rows = [("A lexical (orig+plain)", eng, False),
            ("B query-side (orig+exp Q)", eng, True),
            ("C doc-side (expIdx+plain Q)", eng_x, False),
            ("D TWO-SIDED (expIdx+exp Q)", eng_x, True)]
    print(f"  {'config':<30}{'recall@100':>12}{'nDCG@10':>10}")
    base = None
    for nm, e, qe in rows:
        r, n = evl(e, qe)
        if base is None: base = r
        print(f"  {nm:<30}{r:>12.4f}{n:>10.4f}{('  '+format(r-base,'+.4f')) if base is not None else ''}")
    print(f"  index footprint grows {grow:.2f}x (doc-side expansion adds postings). graph + both sides = model-free SPLADE.")


if __name__ == "__main__":
    main()
