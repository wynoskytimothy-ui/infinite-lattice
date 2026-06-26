#!/usr/bin/env python3
"""TASK 3 -- use the LATTICE to push speed AND accuracy around the cross-encoder, footprint frozen.
The CE is the accuracy engine (the answer-ness wall says the lattice can't BE it) but its COST is linear
in #candidates reranked, and its CEILING is the lattice's pool recall. So sweep BOTH directions on
dev-small, one run:
  SPEED (cascade): MRR@10 vs CE-rerank-DEPTH d in {10,20,30,50,100,200}. The lattice's BM25 pre-rank gates
    the CE -- if MRR plateaus at a shallow d, we cut CE calls (=latency) ~Nx for ~free. Find the KNEE.
  ACCURACY (recall): recall@{50,100,200,500,1000} = the CE's CEILING at each depth (gold it never sees caps
    MRR). Shows how much a wider/deeper lattice pool could buy.
Both keep the 0.428 GB index + 88 MB CE untouched.
"""
import time
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_headline import load_for

RECALL_DEPTHS = [50, 100, 200, 500, 1000]
MRR_DEPTHS = [10, 20, 30, 50, 100, 200]      # rerank these prefixes; CE scores the top-200 once
CE_SCORE = 200
MRR_SAMPLE = 1500


def main():
    idx = FullIndex()
    slim = load_for(idx)

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.small.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    print(f"\n  dev-small {len(queries):,} q -- SPEED (shallow rerank) + ACCURACY (recall ceiling)\n", flush=True)

    # ---- recall CURVE over ALL 6980 (retrieval only, the CE's ceiling) ----
    rec = {d: 0 for d in RECALL_DEPTHS}; lat = []; pools = []
    for i, (qid, qt) in enumerate(queries):
        t = time.perf_counter()
        o = [int(x) for x in slim.retrieve(stoks(qt), 1000)]
        lat.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]
        for d in RECALL_DEPTHS:
            if any(x in gold for x in o[:d]):
                rec[d] += 1
        if i < MRR_SAMPLE:
            pools.append((qid, qt, o))
    n = len(queries); lat = np.array(lat)
    print("  ACCURACY LEVER -- recall@k (max MRR achievable if the CE ranked every in-pool gold #1):")
    for d in RECALL_DEPTHS:
        print(f"    recall@{d:<4} {rec[d]/n*100:5.2f}%")
    print(f"  retrieve top-1000 latency: median {np.median(lat):.2f} ms\n", flush=True)

    # ---- MRR vs CE-rerank-DEPTH on the sample (CE scores top-200 once) ----
    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    mrr = {d: 0.0 for d in MRR_DEPTHS}; ce_ms = []
    for qid, qt, o in pools:
        cs = o[:CE_SCORE]
        t = time.perf_counter()
        sc = np.asarray(ce.predict([(qt, idx.text(p)) for p in cs], batch_size=256, show_progress_bar=False))
        ce_ms.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]
        for d in MRR_DEPTHS:
            order = np.argsort(-sc[:d])
            rr = [cs[k] for k in order[:10]]
            for r, dd in enumerate(rr):
                if dd in gold:
                    mrr[d] += 1.0 / (r + 1); break
    m = len(pools); ce200 = np.median(ce_ms)
    print("  SPEED LEVER (cascade) -- MRR@10 vs CE-rerank-DEPTH (lattice pre-rank gates the CE):")
    best = max(mrr.values()) / m
    for d in MRR_DEPTHS:
        v = mrr[d] / m
        ce_cost = ce200 * d / CE_SCORE
        knee = "  <- KNEE" if v >= best - 0.002 else ""
        print(f"    rerank top-{d:<4} MRR@10 {v:.4f}  ({v-best:+.4f} vs best)  CE ~{ce_cost:5.1f}ms/q{knee}")
    print(f"\n  read: the shallowest depth still at ~best MRR = free speedup (fewer CE calls). recall@k rising")
    print(f"  above 64% = the accuracy headroom a wider pool (union meet+BM25=0.73) or deeper rerank would unlock.")


if __name__ == "__main__":
    main()
