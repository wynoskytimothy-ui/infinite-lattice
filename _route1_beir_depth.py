#!/usr/bin/env python3
"""ROUTE 1 - lattice+CE rerank-depth ceiling on BEIR (scifact, nfcorpus, fiqa).

Same protocol as MARCO: the AppendOnlyLatticeIndex (BM25 word gear) supplies a
fast/tiny recall POOL of depth D; the SAME cross-encoder reranks the pool; report
nDCG@10 + recall@D at D in {100,200,500,1000}. Zero per-corpus tuning - identical
pipeline for all three. Two-sided: report the lexical-lattice nDCG@10 baseline
(no CE) alongside, so the CE's lift and the recall ceiling are both visible.
"""
import sys, math, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load

DEPTHS = [100, 200, 500, 1000]
MAXD = max(DEPTHS)
CORPORA = ["scifact", "nfcorpus", "fiqa"]


def ndcg10(ranked, rels):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    idcg = sum(r / math.log2(i + 2)
               for i, r in enumerate(sorted(rels.values(), reverse=True)[:10]))
    return dcg / idcg if idcg else 0.0


def main():
    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")

    print("\n  ROUTE 1  BEIR  lattice recall pool -> CE rerank, depth sweep (no per-corpus tuning)\n")
    summary = {}
    for name in CORPORA:
        corpus, queries, train_q, test_q = load(name)
        test_ids = [q for q in test_q if q in queries]
        idx = AppendOnlyLatticeIndex()
        for d, txt in corpus.items():
            idx.add(d, txt)
        idx.finalize() if hasattr(idx, "finalize") else None
        ncorp = len(corpus)
        cap = min(MAXD, ncorp)
        print(f"  {name}: {ncorp} docs | test {len(test_ids)} q | pool cap {cap}", flush=True)

        # warm
        idx.search(queries[test_ids[0]], cap)

        ret_lat = []
        ce_lat = {d: [] for d in DEPTHS}
        nd_lat = 0.0  # lattice-only nDCG@10 (no CE), pure BM25 order
        nd = {d: 0.0 for d in DEPTHS}
        rec = {d: set() for d in DEPTHS}  # placeholder
        recall = {d: 0.0 for d in DEPTHS}

        nq = len(test_ids)
        for qid in test_ids:
            q = queries[qid]
            rels = test_q[qid]
            goldset = {d for d, s in rels.items() if s > 0}
            t = time.perf_counter()
            pool = list(idx.search(q, cap))
            ret_lat.append((time.perf_counter() - t) * 1000)

            # lattice-only nDCG@10 (BM25 order, no CE)
            nd_lat += ndcg10(pool[:10], rels)

            texts = [corpus[d] for d in pool]
            t = time.perf_counter()
            sc_all = ce.predict([(q, tx) for tx in texts], batch_size=256, show_progress_bar=False)
            ce_full_ms = (time.perf_counter() - t) * 1000

            for d in DEPTHS:
                dd = min(d, cap)
                poold = pool[:dd]
                if goldset & set(poold):
                    recall[d] += len(goldset & set(poold)) / len(goldset) if goldset else 0.0
                scd = sc_all[:dd]
                rr = [poold[k] for k in np.argsort(-scd)]
                nd[d] += ndcg10(rr, rels)
                ce_lat[d].append(ce_full_ms * dd / len(pool) if pool else 0.0)

        ret_lat = np.array(ret_lat)
        print(f"    lattice-only nDCG@10 (BM25 order, no CE): {nd_lat/nq:.4f}")
        print(f"    {'depth':>6} {'recall@D':>10} {'nDCG@10':>9} {'CE ms/q(med)':>13} {'e2e ms/q(med)':>15}")
        best = (0, 0.0)
        for d in DEPTHS:
            ce_med = np.median(ce_lat[d])
            e2e = np.median(ret_lat) + ce_med
            ndv = nd[d] / nq
            print(f"    {d:>6} {recall[d]/nq*100:>9.2f}% {ndv:>9.4f} {ce_med:>13.1f} {e2e:>15.1f}")
            if ndv > best[1]:
                best = (d, ndv)
        summary[name] = {"lattice_only": nd_lat / nq, "best_depth": best[0],
                         "best_ndcg10": best[1], "ndcg_by_depth": {d: nd[d]/nq for d in DEPTHS}}
        print()

    print("  ===== BEIR ROUTE 1 SUMMARY =====")
    for name, s in summary.items():
        print(f"  {name:10s}: lattice-only {s['lattice_only']:.4f}  ->  +CE best nDCG@10 {s['best_ndcg10']:.4f} @ depth {s['best_depth']}")
    avg_best = np.mean([s["best_ndcg10"] for s in summary.values()])
    print(f"  3-corpus avg best nDCG@10: {avg_best:.4f}   (BEIR zero-shot SOTA avg ~0.50)")


if __name__ == "__main__":
    main()
