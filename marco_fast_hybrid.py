#!/usr/bin/env python3
"""Verify the fast (meet) retrieval preserves HYBRID accuracy end-to-end, not just BM25 MRR.

bm25_fast matches full-scan BM25 MRR@10 exactly, but its top-100 pool is ~94% identical (the 6% are
low-rank tail docs). The cross-encoder reranks the WHOLE pool, so confirm the hybrid MRR@10 is the
same on the fast pool as on the full pool -- i.e. "faster without changing accuracy" holds for the
whole system. Also reports end-to-end latency (retrieve + fetch + CE) for both pipelines.
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_fast import bm25_fast

N = 200


def main():
    idx = FullIndex()
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
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
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    from sentence_transformers import CrossEncoder
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device=dev)

    def rerank(order, qt):
        texts = [idx.text(int(d)) for d in order[:100]]
        sc = ce.predict([(qt, t) for t in texts], batch_size=128, show_progress_bar=False)
        return order[:100][np.argsort(-sc)]

    for qid, qt in sample[:5]:             # warm
        rerank(idx.bm25_top(stoks(qt), 100)[0], qt)
        rerank(bm25_fast(idx, stoks(qt), 100)[0], qt)

    mrrF = mrrA = 0.0; eF = []; eA = []
    for qid, qt in sample:
        qts = stoks(qt); gold = qrels[qid]
        t0 = time.perf_counter(); of, _ = idx.bm25_top(qts, 100); rf = rerank(of, qt); eF.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); oa, _ = bm25_fast(idx, qts, 100); ra = rerank(oa, qt); eA.append((time.perf_counter() - t0) * 1000)
        for rank, d in enumerate(rf[:10]):
            if int(d) in gold: mrrF += 1.0 / (rank + 1); break
        for rank, d in enumerate(ra[:10]):
            if int(d) in gold: mrrA += 1.0 / (rank + 1); break
    eF = np.array(eF); eA = np.array(eA)
    print(f"\n  HYBRID (BM25 pool + cross-encoder) -- fast pool vs full pool, n={N}\n")
    print(f"  {'pipeline':<22}{'hybrid MRR@10':>15}{'median ms e2e':>16}{'p90':>9}")
    print(f"  {'full scan + CE':<22}{mrrF/N:>15.4f}{np.median(eF):>16.0f}{np.percentile(eF,90):>9.0f}")
    print(f"  {'fast meet + CE':<22}{mrrA/N:>15.4f}{np.median(eA):>16.0f}{np.percentile(eA,90):>9.0f}")
    print(f"\n  hybrid MRR delta: {(mrrA-mrrF)/N:+.4f}   end-to-end speedup: {np.median(eF)/np.median(eA):.1f}x median")


if __name__ == "__main__":
    main()
