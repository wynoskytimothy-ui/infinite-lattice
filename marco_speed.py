#!/usr/bin/env python3
"""Operational scorecard for the full-8.8M MARCO retrieval system: SPEED + ACCURACY + FOOTPRINT.

Measures, right now, on a dev sample:
  - symbolic index RAM footprint (the loaded CSR arrays)
  - BM25 top-100 retrieval latency over 8.8M docs (median/mean/p90 ms/query)
  - cross-encoder rerank latency (GPU, 100 passages/query)
  - accuracy: raw BM25 MRR@10 and the hybrid (BM25 recall + CE rerank) MRR@10 on the same queries
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO

N_TIME = 300   # queries for retrieval-latency timing
N_CE = 200     # queries for CE rerank latency + hybrid accuracy


def main():
    idx = FullIndex()
    fp = idx.di.nbytes + idx.tf.nbytes + idx.ptr.nbytes + idx.doclen.nbytes + idx.idfa.nbytes
    print(f"  symbolic index RAM: {fp/1e9:.2f} GB  (di {idx.di.nbytes/1e9:.2f} + tf {idx.tf.nbytes/1e9:.2f} "
          f"+ ptr/doclen/idf {(idx.ptr.nbytes+idx.doclen.nbytes+idx.idfa.nbytes)/1e6:.0f}MB)", flush=True)

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
    print(f"  dev queries with gold: {len(queries):,}; timing on {N_TIME}, CE on {N_CE}", flush=True)

    sample = queries[:N_TIME]
    for qid, qt in sample[:10]:           # warm
        idx.bm25_top(stoks(qt), 100)
    ts = []; rr = 0.0
    for qid, qt in sample:
        t0 = time.perf_counter()
        order, _ = idx.bm25_top(stoks(qt), 100)
        ts.append((time.perf_counter() - t0) * 1000)
        gold = qrels[qid]
        for rank, d in enumerate(order[:10]):
            if int(d) in gold:
                rr += 1.0 / (rank + 1); break
    ts = np.array(ts)
    print(f"\n  RETRIEVAL  BM25 top-100 over {idx.N/1e6:.1f}M docs:")
    print(f"    median {np.median(ts):.1f} ms/q | mean {ts.mean():.1f} | p90 {np.percentile(ts,90):.1f} | "
          f"max {ts.max():.0f}  (n={len(ts)})")
    print(f"    raw BM25 MRR@10 = {rr/len(sample):.4f}", flush=True)

    from sentence_transformers import CrossEncoder
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device=dev)
    sub = queries[:N_CE]
    cets = []; fts = []; rr_ce = 0.0
    for qid, qt in sub[:5]:               # warm GPU
        order, _ = idx.bm25_top(stoks(qt), 100)
        ce.predict([(qt, idx.text(int(d))) for d in order[:100]], batch_size=128, show_progress_bar=False)
    for qid, qt in sub:
        order, _ = idx.bm25_top(stoks(qt), 100)
        t1 = time.perf_counter()
        texts = [idx.text(int(d)) for d in order[:100]]
        fts.append((time.perf_counter() - t1) * 1000)
        t0 = time.perf_counter()
        scores = ce.predict([(qt, tx) for tx in texts], batch_size=128, show_progress_bar=False)
        cets.append((time.perf_counter() - t0) * 1000)
        rer = order[:100][np.argsort(-scores)]
        gold = qrels[qid]
        for rank, d in enumerate(rer[:10]):
            if int(d) in gold:
                rr_ce += 1.0 / (rank + 1); break
    cets = np.array(cets); fts = np.array(fts)
    print(f"\n  RERANK  cross-encoder ms-marco-MiniLM-L-6-v2 on {dev}, 100 passages/query:")
    print(f"    text fetch {np.median(fts):.1f} ms/q | CE predict median {np.median(cets):.0f} ms/q "
          f"| p90 {np.percentile(cets,90):.0f}  (n={len(sub)})")
    print(f"    HYBRID MRR@10 (BM25 recall + CE rerank) = {rr_ce/len(sub):.4f}", flush=True)

    end2end = np.median(ts) + np.median(fts) + np.median(cets)
    print(f"\n  END-TO-END median latency (retrieve + fetch + rerank): {end2end:.0f} ms/query")


if __name__ == "__main__":
    main()
