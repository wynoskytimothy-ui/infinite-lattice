#!/usr/bin/env python3
"""Is pruning the common-term bulk nearly FREE for the HYBRID? The CE reranks the pool, so what matters
is whether gold still lands in the top-100 (recall), not BM25's exact rank. Pruning COMMON terms should
keep gold (found via its RARE terms). Measure recall@100 + hybrid MRR@10 at prune thresholds vs baseline.
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO
from marco_prune import bm25_prune

N = 200


def main():
    idx = FullIndex(); val = np.zeros(idx.N, np.uint16)
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
    random.Random(0).shuffle(queries); sample = queries[:N]

    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")

    def rerank(pids, qt):
        sc = ce.predict([(qt, idx.text(p)) for p in pids], batch_size=128, show_progress_bar=False)
        return [pids[i] for i in np.argsort(-sc)]

    for qid, qt in sample[:5]:
        rerank([int(d) for d in bm25_prune(idx, stoks(qt), val, 1.0, 100)], qt)

    df = np.diff(idx.ptr.astype(np.int64)); npost = int(df.sum())
    print(f"\n  PRUNED HYBRID -- does the cross-encoder recover the pruning loss? 8.8M, n={N}\n")
    print(f"  {'keep idf>=':<12}{'index GB':>10}{'recall@100':>12}{'hybrid MRR@10':>15}")
    for T in (1.0, 3.0, 4.0):
        kept = int(df[idx.idfa >= T].sum())
        gb = (kept / npost) * (0.450 + 0.176) + 0.05
        rec = 0; mrr = 0.0
        for qid, qt in sample:
            o = bm25_prune(idx, stoks(qt), val, T, 100); base = [int(d) for d in o[:100]]
            gold = qrels[qid]
            if any(g in base for g in gold):
                rec += 1
            rr = rerank(base, qt)
            for r, d in enumerate(rr[:10]):
                if d in gold:
                    mrr += 1.0 / (r + 1); break
        print(f"  {T:<12.1f}{gb:>10.3f}{rec/N*100:>11.1f}%{mrr/N:>15.4f}")
    print(f"\n  if hybrid MRR holds while index shrinks, pruning the common bulk is nearly free for the full system.")


if __name__ == "__main__":
    main()
