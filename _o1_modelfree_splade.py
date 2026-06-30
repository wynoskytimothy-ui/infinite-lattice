#!/usr/bin/env python3
"""MODEL-FREE SPLADE head-to-head vs GPU SPLADE. Optimized doc-side expansion (STRENGTH-WEIGHTED: strong
correlates get higher tf via repetition; tuned per-doc budget) built from the zero-shot co-occurrence graph,
vs distilled-SPLADE (the GPU model), on a corpus where SPLADE WINS. Kill-or-confirm: can the ingest semantic
graph replace the neural encoder?"""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words
from _o1_drift import build_drift, drift_bag, score_bag

CACHE = Path(os.environ.get("TEMP", "/tmp"))


def doc_expand_w(text, drift, doc_set, M, R=3):
    acc = Counter()
    for w in set(words(text)):
        for cw, pmi in drift.get(w, ()):
            if cw not in doc_set: acc[cw] += pmi
    if not acc: return ""
    mx = max(acc.values()); out = []
    for cw, s in acc.most_common(M):
        out.extend([cw] * max(1, round(s / mx * R)))                  # strength-weighted tf
    return " ".join(out)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]

    allw = {w for d in corpus.values() for w in words(d)}
    t0 = time.perf_counter(); drift, _, _ = build_drift(corpus, allw); t_g = time.perf_counter() - t0
    eng = EdgeRAG().build(corpus)
    t0 = time.perf_counter()
    expanded = {d: txt + " " + doc_expand_w(txt, drift, set(words(txt)), M) for d, txt in corpus.items()}
    eng_x = EdgeRAG().build(expanded); t_x = time.perf_counter() - t0
    grow = len(eng_x.seg_doc) / len(eng.seg_doc)

    spl = None
    vocab = sorted({w for q in ids for w in words(queries[q])})
    df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])
    print("=" * 90); print(f"MODEL-FREE vs GPU SPLADE — {name}: {len(corpus):,} docs, M={M} (graph {t_g:.0f}s, doc-exp {t_x:.0f}s, grow {grow:.2f}x)"); print("=" * 90)

    def evl(fn):
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            ranked = fn(qid)
            rc += len(set(ranked[:100]) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in ranked[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n

    rows = [("lexical", lambda qid: list(np.argsort(eng.score(queries[qid]))[::-1][:100])),
            ("model-free DOC-side", lambda qid: list(np.argsort(eng_x.score(queries[qid]))[::-1][:100])),
            ("model-free TWO-SIDED", lambda qid: list(np.argsort(score_bag(eng_x, drift_bag(queries[qid], drift, alpha=0.4)))[::-1][:100]))]
    if spl: rows.append(("GPU distilled-SPLADE", lambda qid: [d2i[d] for d in spl.search(queries[qid], 100)]))
    print(f"  {'retriever':<26}{'recall@100':>12}{'nDCG@10':>10}{'  vs lexical':>14}")
    base = None
    for nm, fn in rows:
        r, n = evl(fn)
        if base is None: base = r
        print(f"  {nm:<26}{r:>12.4f}{n:>10.4f}{r-base:>+14.4f}")
    print("  model-free = zero-shot co-occurrence doc-expansion (no GPU); distilled-SPLADE = the neural encoder.")


if __name__ == "__main__":
    main()
