#!/usr/bin/env python3
"""OPTIONAL MODEL TIER — retrieve fast+small on EdgeRAG, then rerank top-K with a small cross-encoder.
The model runs on ~K pairs/query (not the corpus), so the lattice keeps ingest/footprint/candidate-gen cheap
and the CE only closes the accuracy gap on the hard semantic corpora. Measures plain EdgeRAG vs +CE rerank
and compares to the cited SOTA band."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10

SOTA = {  # cited published BEIR nDCG@10 band (GPU), for context only
    "scifact": "SPLADE~0.704 ColBERT~0.693 E5~0.704",
    "nfcorpus": "SPLADE~0.345 ColBERT~0.338 E5~0.366",
    "fiqa": "SPLADE~0.347 ColBERT~0.356 E5~0.386",
}


def _modes(eng, ce, corpus, queries, qids, qrels, rerank_k, K0=60):
    """Return mean nDCG for {none, ce, rrf} over qids (and CE latency)."""
    s = {"none": 0.0, "ce": 0.0, "rrf": 0.0}; lat = []; nq = 0
    for qid in qids:
        if qid not in queries: continue
        q = queries[qid]; cands = eng.search_bridged(q, rerank_k)
        if not cands: continue
        nq += 1
        s["none"] += ndcg10(cands[:10], qrels[qid])
        t0 = time.perf_counter()
        sc = ce.predict([(q, corpus[d][:512]) for d in cands], batch_size=128, show_progress_bar=False)
        lat.append((time.perf_counter() - t0) * 1000)
        ce_order = list(np.argsort(sc)[::-1]); ce_rank = {i: r for r, i in enumerate(ce_order)}
        s["ce"] += ndcg10([cands[i] for i in ce_order[:10]], qrels[qid])
        rrf = sorted(range(len(cands)), key=lambda i: -(1.0/(K0+i) + 1.0/(K0+ce_rank[i])))
        s["rrf"] += ndcg10([cands[i] for i in rrf[:10]], qrels[qid])
    return {k: v / max(1, nq) for k, v in s.items()}, (float(np.median(lat)) if lat else 0.0), nq


def run(name, ce, rerank_k=100, qcap=None):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    if qcap and len(test_ids) > qcap: test_ids = test_ids[:qcap]
    # split train qrels: bridge-train (learn bridges) vs val (pick the tier mode) -- both disjoint from test
    tr_ids = [q for q in train_q if q in queries]
    val_ids = tr_ids[::5][:120]; bt_ids = [q for q in tr_ids if q not in set(val_ids)]
    bt_qrels = {q: train_q[q] for q in bt_ids}
    eng = EdgeRAG().build(corpus); eng.learn_bridges(queries, bt_qrels, corpus)

    val, _, _ = _modes(eng, ce, corpus, queries, val_ids, train_q, rerank_k)   # SELECT on val
    pick = max(val, key=val.get)
    test, ms, nq = _modes(eng, ce, corpus, queries, test_ids, test_q, rerank_k)  # report on TEST
    return test, pick, ms, nq


def main():
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    print("=" * 96)
    print("OPTIONAL MODEL TIER — EdgeRAG candidate-gen + cross-encoder rerank (closes the semantic gap)")
    print("=" * 96)
    print(f"  {'corpus':<10}{'none':>8}{'ce':>8}{'rrf':>8}  {'val-pick':>9}{'SELECTED':>10}{'gain':>9}{'ms/q':>7}   cited SOTA (GPU)")
    plans = [("scifact", 100, None), ("nfcorpus", 100, None), ("fiqa", 100, 200)]
    for name, k, cap in plans:
        test, pick, ms, nq = run(name, ce, rerank_k=k, qcap=cap)
        sel = test[pick]; tag = f" (n={nq})" if cap else ""
        print(f"  {name:<10}{test['none']:>8.4f}{test['ce']:>8.4f}{test['rrf']:>8.4f}  {pick:>9}{sel:>10.4f}"
              f"{sel-test['none']:>+9.4f}{ms:>6.0f}ms   {SOTA[name]}{tag}")
    print("\n  SELECTED = the mode picked on held-out TRAIN val (disjoint from bridge-train and test), applied to test.")
    print("  Selection makes the tier SAFE: it picks 'none' where the CE would hurt, 'ce'/'rrf' where it helps.")
    print("\n  EdgeRAG = CPU candidate gen (fast/small). CE reranks K pairs/query (small model, NPU/GPU/CPU).")
    print("  The tier is OPTIONAL: pay the model only where the semantic gap matters; lattice stays the engine.")


if __name__ == "__main__":
    main()
