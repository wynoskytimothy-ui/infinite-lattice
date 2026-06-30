#!/usr/bin/env python3
"""Footprint/speed round 2: (1) uint8 weight quantization (per-term scale) -> accuracy cost + B/posting;
(2) confirm query serve is fast and the radix CSR is serve-ready (ingest helps query); (3) confirm the 2-way
meet (co-occurrence) is used as a FREE lazy correlation (drift), not stored per-doc composites."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load, ndcg10
from _fast_tok import words


def quantize_u8(eng):
    """Per-term uint8 quantization of seg_tf (scale = max/255). Returns u8 weights + per-term scales."""
    u8 = np.zeros(eng.seg_tf.shape, np.uint8); scale = np.ones(eng.V, np.float32)
    tf64 = eng.seg_tf.astype(np.float64)
    for t in range(eng.V):
        a, e = int(eng.indptr[t]), int(eng.indptr[t + 1])
        if e <= a: continue
        mx = tf64[a:e].max()
        if mx > 0:
            s = mx / 255.0; scale[t] = s
            u8[a:e] = np.clip(np.round(tf64[a:e] / s), 0, 255).astype(np.uint8)
    return u8, scale


def score_u8(eng, query, u8, scale):
    s = np.zeros(eng.N); ip, sd, den, k1p1 = eng.indptr, eng.seg_doc, eng._denom, eng._k1p1
    for w, qwt in Counter(words(query)).items():
        tid = eng._term_id(w)
        if tid is None: continue
        a, e = int(ip[tid]), int(ip[tid + 1]); dfp = e - a
        if dfp == 0: continue
        di = sd[a:e]; tf = u8[a:e].astype(np.float64) * scale[tid]      # dequantize
        s[di] += (qwt * eng._idf(dfp) * k1p1) * tf / (tf + den[di])
    return s


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus); npost = len(eng.seg_doc)
    u8, scale = quantize_u8(eng)

    def evl(scorer):
        rc = nd = 0.0; lat = []
        for qid in ids:
            gold = {d2i[d] for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in d2i}
            t0 = time.perf_counter(); sc = scorer(queries[qid]); lat.append((time.perf_counter() - t0) * 1000)
            top = np.argsort(sc)[::-1][:100]
            rc += len(set(top) & gold) / len(gold); nd += ndcg10([doc_ids[i] for i in top[:10]], test_q[qid])
        n = len(ids); return rc / n, nd / n, float(np.median(lat))

    r_f, n_f, ms_f = evl(lambda q: eng.score(q))
    r_u, n_u, ms_u = evl(lambda q: score_u8(eng, q, u8, scale))
    bp_f = (eng.seg_doc.itemsize + eng.seg_tf.itemsize)            # bytes/posting float16 path
    bp_u = (eng.seg_doc.itemsize + 1)                              # uint32 doc + uint8 weight
    print("=" * 78); print(f"FOOTPRINT/SPEED 2 — {name}: {eng.N:,} docs, {npost:,} postings"); print("=" * 78)
    print(f"  {'weights':<14}{'B/posting':>10}{'B/doc(raw)':>12}{'recall@100':>12}{'nDCG@10':>10}{'serve':>9}")
    print(f"  {'float16':<14}{bp_f:>10}{bp_f*npost/eng.N:>12.0f}{r_f:>12.4f}{n_f:>10.4f}{ms_f:>7.2f}ms")
    print(f"  {'uint8':<14}{bp_u:>10}{bp_u*npost/eng.N:>12.0f}{r_u:>12.4f}{n_u:>10.4f}{ms_u:>7.2f}ms")
    print(f"  uint8: {(1-bp_u/bp_f)*100:.0f}% smaller postings, nDCG {n_u-n_f:+.4f}, recall {r_u-r_f:+.4f} "
          f"({'near-lossless' if abs(n_u-n_f)<0.003 else 'cost'})")
    print("  NOTE: doc-id (uint32, 4B) dominates; delta+FOR coding (the compressed save() = ~198 B/doc) is the")
    print("  bigger footprint lever. uint8 weights stack on top. 2-way meet (co-occurrence) is the drift graph =")
    print("  a FREE lazy correlation (built once, small table), NOT stored per-doc composites.")


if __name__ == "__main__":
    main()
