#!/usr/bin/env python3
"""#2 — stronger reranker. The reranker orders the union pool -> drives recall@100 + nDCG (fiqa's bottleneck).
Compute the union pools ONCE per corpus, then rerank with progressively stronger cross-encoders and compare."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from aethos_splade_lattice import DistilledSpladeIndex
from scripts.bench_supervised_bridges import load, ndcg10
from _o1_drift import build_drift, drift_bag, score_bag
from _fast_tok import words

CACHE = Path(os.environ.get("TEMP", "/tmp")); DEPTH = 300
MODELS = ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2", "BAAI/bge-reranker-base"]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, train_q, corpus)
    seeds = {w for q in ids for w in words(queries[q])}; drift, _, _ = build_drift(corpus, seeds)
    vocab = sorted(seeds); df = CACHE / f"splade_{name}_doc128.npz"; vf = CACHE / f"distill_{name}_v{len(vocab)}.npz"
    spl = None
    if df.exists() and vf.exists():
        dz = np.load(df); vz = np.load(vf, allow_pickle=True)
        spl = DistilledSpladeIndex(); spl.doc_ids = doc_ids; spl.N = len(doc_ids)
        spl._set_docs(dz["t"], dz["w"], dz["o"]); spl._set_table(vz["words"].tolist(), vz["t"], vz["w"], vz["o"])

    # union pools, computed ONCE
    unions = {}
    for qid in ids:
        q = queries[qid]
        pools = [list(np.argsort(eng.score(q))[::-1][:DEPTH]), [d2i[d] for d in eng.search_bridged(q, DEPTH)],
                 list(np.argsort(score_bag(eng, drift_bag(q, drift, alpha=0.4)))[::-1][:DEPTH])]
        if spl: pools.append([d2i[d] for d in spl.search(q, DEPTH)])
        unions[qid] = list(set().union(*[set(p) for p in pools]))

    print("=" * 80); print(f"#2 STRONGER RERANKER — {name}: {len(ids)} q, union pools fixed (depth {DEPTH})"); print("=" * 80)
    print(f"  {'reranker':<38}{'recall@100':>12}{'nDCG@10':>10}")
    from sentence_transformers import CrossEncoder
    for m in MODELS:
        try:
            ce = CrossEncoder(m, max_length=256)
        except Exception as e:
            print(f"  {m:<38}  (load failed: {str(e)[:30]})"); continue
        rc = nd = 0.0
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            u = unions[qid]
            sc = ce.predict([(queries[qid], corpus[doc_ids[i]][:512]) for i in u], batch_size=256, show_progress_bar=False)
            order = [u[j] for j in np.argsort(sc)[::-1]]
            rc += len(set(order[:100]) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in order[:10]], test_q[qid])
        nq = len(ids); print(f"  {m.split('/')[-1]:<38}{rc/nq:>12.4f}{nd/nq:>10.4f}")
        del ce
    print("  same union pool, only the reranker changes -> isolates the reranker's effect on recall@100 + nDCG.")


if __name__ == "__main__":
    main()
