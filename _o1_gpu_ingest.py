#!/usr/bin/env python3
"""GPU INGEST PATH — the one place a GPU helps: the one-time SPLADE encode (docs + per-word distill table).
Serve stays CPU + encoder-free. Measures GPU encode throughput vs the CPU 4 docs/s, verifies the index serves
the same recall on CPU, and projects to MARCO scale."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_splade_lattice import SpladeEncoder, DistilledSpladeIndex
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    test_ids = [q for q in test_q if q in queries]; ndoc = len(corpus)
    print("=" * 88); print(f"GPU INGEST PATH — {name}: {ndoc:,} docs (GPU encode, CPU encoder-free serve)"); print("=" * 88)

    enc = SpladeEncoder()
    print(f"  encoder device: {enc.device}")
    enc.encode(["warm up the gpu kernels"], topk=128)     # warm

    # --- GPU encode: docs + per-word distill table ---
    t0 = time.perf_counter(); idx = DistilledSpladeIndex().build_docs(corpus, enc, topk=128); tb = time.perf_counter() - t0
    npost = idx.seg_doc.size
    vocab = sorted({w for q in test_ids for w in words(queries[q])})
    t0 = time.perf_counter(); idx.distill(vocab, enc, topk=64); td = time.perf_counter() - t0
    t0 = time.perf_counter(); qt, qw, qo = enc.encode([queries[q] for q in test_ids], topk=None); tq = time.perf_counter() - t0

    print(f"\n  GPU ENCODE (one-time, ingest):")
    print(f"    docs   : {ndoc:,} in {tb:.1f}s = {ndoc/tb:,.0f} docs/s  ({npost/ndoc:.0f} terms/doc)  [CPU was ~4 docs/s]")
    print(f"    distill: {len(vocab):,} words in {td:.1f}s = {len(vocab)/td:,.0f} words/s")
    print(f"    queries: {len(test_ids):,} in {tq:.1f}s")

    # --- CPU encoder-free serve: verify recall recovery on the GPU-built index ---
    eng = EdgeRAG().build(corpus)
    def rec(top, gold): return len(set(top) & gold) / len(gold) if gold else 0.0
    L = S = D = nq = 0; lat = []
    for qi, qid in enumerate(test_ids):
        gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
        if not gold: continue
        nq += 1
        L += rec([d2i[d] for d in eng.search(queries[qid], 100)], gold)
        sf = np.zeros(ndoc)
        for tid, w in zip(qt[qo[qi]:qo[qi+1]], qw[qo[qi]:qo[qi+1]]):
            a, e = int(idx.indptr[tid]), int(idx.indptr[tid+1])
            if e > a: sf[idx.seg_doc[a:e]] += float(w) * idx.seg_w[a:e].astype(np.float32)
        S += rec(list(np.argsort(sf)[::-1][:100]), gold)
        t0 = time.perf_counter(); top = idx.search(queries[qid], 100); lat.append((time.perf_counter()-t0)*1000)
        D += rec([d2i[d] for d in top], gold)
    L, S, D = L/nq, S/nq, D/nq
    print(f"\n  CPU SERVE (encoder-free, the GPU-built index):")
    print(f"    Recall@100: lexical {L:.4f} | distilled {D:.4f} | SPLADE-full {S:.4f} "
          f"-> recovers {(D-L)/(S-L)*100:.0f}% of gain, {np.median(lat):.2f} ms/q, NO query encoder")

    # --- projection to MARCO ---
    rate = ndoc / tb
    print(f"\n  PROJECTION (8.8M MARCO at {rate:,.0f} docs/s on this RTX 5080): one-time encode ~{8_841_823/rate/60:.0f} min")
    print(f"  GPU is needed ONLY for this one-time ingest encode; serve is CPU + encoder-free (numpy scatter-add).")


if __name__ == "__main__":
    main()
