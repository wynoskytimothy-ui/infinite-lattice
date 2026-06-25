#!/usr/bin/env python3
"""ROUTE 1 - lattice+CE rerank-depth ceiling on MARCO dev-small (canonical 6,980 q).

Retrieve a POOL of depth D from the persisted 0.428GB FOR slim index, rerank the
whole pool with cross-encoder/ms-marco-MiniLM-L-6-v2, measure MRR@10 + recall@D
+ latency at D in {100,200,500,1000}. The pool is retrieved ONCE at the deepest
depth and truncated for shallower depths so recall@D is the lattice ceiling at D
and rerank cost is the only thing that changes. Two-sided: report where MRR@10
plateaus (recall-capped) vs where it is still climbing.
"""
import time, sys
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_headline import load_for

DEPTHS = [100, 200, 500, 1000]
MAXD = max(DEPTHS)


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
    print(f"\n  MARCO dev-SMALL: {len(queries):,} queries (canonical leaderboard subset)\n", flush=True)

    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")

    # warm
    for qid, qt in queries[:5]:
        o = [int(d) for d in slim.retrieve(stoks(qt), MAXD)]
        ce.predict([(qt, idx.text(p)) for p in o[:128]], batch_size=128, show_progress_bar=False)

    n = len(queries)
    ret_lat = []                         # lattice retrieve latency (depth MAXD)
    ce_lat = {d: [] for d in DEPTHS}     # CE rerank latency per depth
    mrr = {d: 0.0 for d in DEPTHS}
    rec = {d: 0 for d in DEPTHS}         # recall@d on the POOL (lattice ceiling)

    t_start = time.perf_counter()
    for i, (qid, qt) in enumerate(queries):
        gold = qrels[qid]
        t = time.perf_counter()
        pool = [int(d) for d in slim.retrieve(stoks(qt), MAXD)]
        ret_lat.append((time.perf_counter() - t) * 1000)

        # score the full deepest pool ONCE, reuse truncations
        texts = [idx.text(p) for p in pool]
        t = time.perf_counter()
        sc_all = ce.predict([(qt, tx) for tx in texts], batch_size=256, show_progress_bar=False)
        ce_full_ms = (time.perf_counter() - t) * 1000

        for d in DEPTHS:
            poold = pool[:d]
            # recall@d (pool ceiling)
            if any(p in gold for p in poold):
                rec[d] += 1
            # rerank truncated pool -> MRR@10
            scd = sc_all[:d]
            rr = [poold[k] for k in np.argsort(-scd)]
            for r, dd in enumerate(rr[:10]):
                if dd in gold:
                    mrr[d] += 1.0 / (r + 1)
                    break
            # CE cost scales ~linearly; record proportional share for shallow depths
            ce_lat[d].append(ce_full_ms * d / len(pool) if pool else 0.0)

        if (i + 1) % 1000 == 0:
            el = time.perf_counter() - t_start
            line = "  ".join(f"D{d}:MRR{mrr[d]/(i+1):.4f}/R{rec[d]/(i+1)*100:.1f}" for d in DEPTHS)
            print(f"    {i+1:>5}/{n}  {line}  ({el:.0f}s)", flush=True)

    ret_lat = np.array(ret_lat)
    print(f"\n  ===== ROUTE 1  MARCO dev-small  (n={n}) =====")
    print(f"  lattice FOR index on disk : 0.428 GB ; retrieve(top-{MAXD}) median {np.median(ret_lat):.2f}ms p90 {np.percentile(ret_lat,90):.2f}ms")
    print(f"  {'depth':>6} {'recall@D':>10} {'MRR@10':>9} {'CE ms/q(med)':>13} {'end2end ms/q(med)':>18}")
    for d in DEPTHS:
        ce_med = np.median(ce_lat[d])
        e2e = np.median(ret_lat) + ce_med
        print(f"  {d:>6} {rec[d]/n*100:>9.2f}% {mrr[d]/n:>9.4f} {ce_med:>13.1f} {e2e:>18.1f}")
    print(f"\n  SOTA band MARCO dev-small MRR@10: dense ~0.34, ColBERT/SPLADE 0.38-0.40")


if __name__ == "__main__":
    main()
