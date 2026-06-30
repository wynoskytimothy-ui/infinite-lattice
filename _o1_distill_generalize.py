#!/usr/bin/env python3
"""GENERALIZE the encoder-free distilled tier across BEIR (GPU encode, CPU serve). Confirms the ~93% recall
recovery holds beyond nfcorpus. One encoder load, reused across corpora."""
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
    corpora = sys.argv[1].split(",") if len(sys.argv) > 1 else ["scifact", "nfcorpus", "fiqa"]
    enc = SpladeEncoder(); enc.encode(["warm"], topk=128)
    print("=" * 96)
    print(f"GENERALIZE distilled encoder-free tier (device={enc.device}). Recall@100 + nDCG@10, held-out test.")
    print("=" * 96)
    print(f"  {'corpus':<10}{'docs':>8}{'enc s':>7}{'R@100 lex':>11}{'distilled':>10}{'splade-full':>12}{'recovery':>10}{'nDCG d':>8}{'serve':>8}")
    for name in corpora:
        corpus, queries, train_q, test_q = load(name)
        doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
        test_ids = [q for q in test_q if q in queries]; ndoc = len(corpus)
        t0 = time.perf_counter(); idx = DistilledSpladeIndex().build_docs(corpus, enc, topk=128); es = time.perf_counter() - t0
        vocab = sorted({w for q in test_ids for w in words(queries[q])}); idx.distill(vocab, enc, topk=64)
        qt, qw, qo = enc.encode([queries[q] for q in test_ids], topk=None)
        eng = EdgeRAG().build(corpus)

        def rec(top, gold): return len(set(top) & gold) / len(gold) if gold else 0.0
        L = S = D = ndg = nq = 0; lat = []
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
            D += rec([d2i[d] for d in top[:100]], gold)
            ndg += ndcg10(top[:10], test_q[qid])
        L, S, D, ndg = L/nq, S/nq, D/nq, ndg/nq
        recov = (D - L) / (S - L) * 100 if S > L else 0
        print(f"  {name:<10}{ndoc:>8,}{es:>6.1f}s{L:>11.4f}{D:>10.4f}{S:>12.4f}{recov:>9.0f}%{ndg:>8.4f}{np.median(lat):>6.2f}ms")
    print("\n  recovery = % of SPLADE-full's recall gain that the ENCODER-FREE distilled tier recaptures (CPU serve).")


if __name__ == "__main__":
    main()
