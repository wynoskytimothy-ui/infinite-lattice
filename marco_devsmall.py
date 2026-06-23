#!/usr/bin/env python3
"""Leaderboard-comparable number, zero asterisks: run the EXACT canonical MS MARCO dev-SMALL subset
(6,980 queries, qrels.msmarco-passage.dev-subset) that BM25 0.187 / dense 0.34 / ColBERT 0.38-0.40 are
all reported on. Loads the PERSISTED 0.428 GB FOR index FROM DISK (so the shipped artifact is exactly
what's measured) + cross-encoder rerank. Reports MRR@10 + recall@100 on the same set the leaderboard uses.
"""
import time
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_headline import load_for


def main():
    idx = FullIndex()
    slim = load_for(idx)                               # query FROM slim_index_for.npz (0.428 GB on disk)

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
    print(f"\n  dev-SMALL: {len(queries):,} queries (canonical leaderboard subset)\n", flush=True)

    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")
    for qid, qt in queries[:5]:
        slim.retrieve(stoks(qt), 100)                  # warm

    t0 = time.perf_counter(); lat = []; rec = 0; mrr = 0.0
    for i, (qid, qt) in enumerate(queries):
        t = time.perf_counter()
        o = [int(d) for d in slim.retrieve(stoks(qt), 100)]
        lat.append((time.perf_counter() - t) * 1000)
        gold = qrels[qid]
        if any(d in gold for d in o):
            rec += 1
        sc = ce.predict([(qt, idx.text(p)) for p in o], batch_size=128, show_progress_bar=False)
        rr = [o[k] for k in np.argsort(-sc)]
        for r, d in enumerate(rr[:10]):
            if d in gold:
                mrr += 1.0 / (r + 1); break
        if (i + 1) % 1000 == 0:
            print(f"    {i+1:>5}/{len(queries)}  recall@100 {rec/(i+1)*100:5.1f}%  MRR@10 {mrr/(i+1):.4f}", flush=True)

    n = len(queries); lat = np.array(lat)
    print(f"\n  ===== DEV-SMALL CANONICAL (n={n}, leaderboard-comparable) =====")
    print(f"    index on disk      : 0.428 GB (slim_index_for.npz, queried directly)")
    print(f"    retrieval latency  : median {np.median(lat):.2f} ms, p90 {np.percentile(lat,90):.2f} ms")
    print(f"    recall@100         : {rec/n*100:.2f}%")
    print(f"    hybrid MRR@10      : {mrr/n:.4f}")
    print(f"    ladder: BM25 0.187 < dense ~0.34 < [us {mrr/n:.3f}] < ColBERT/SPLADE 0.38-0.40")


if __name__ == "__main__":
    main()
