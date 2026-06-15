#!/usr/bin/env python3
"""BM25 recall@k by rare-word depth -- the comparison for the meet's localization.

The pure meet got recall@10 0.13/0.25/0.38 for 1/2/3+ rare-word queries. Does BM25 (the
complete meet = idf-weighted over all terms with tf+length) localize to the top-10 better or
worse? Same stemmed full index, same 3000 dev queries (seed 42), same depth buckets.
"""
import random
from collections import defaultdict
from marco_full_eval import FullIndex, stoks, RARE, QGATE
import time


def main():
    idx = FullIndex()
    qrels = defaultdict(set)
    with open(idx.cf.name.replace("collection.tsv", "qrels.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(idx.cf.name.replace("collection.tsv", "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:3000]

    rec = defaultdict(float); rec10_d = defaultdict(float); cnt_d = defaultdict(int)
    mrr = 0.0; t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        order, _ = idx.bm25_top(qs, k=100)
        ranked = [str(int(d)) for d in order]
        mrr += next((1.0 / i for i, d in enumerate(ranked[:10], 1) if d in rel), 0.0)
        for k in (1, 5, 10, 100):
            rec[k] += len(rel & set(ranked[:k])) / len(rel)
        dep = min(len([w for w in qs if idx.idf_of(w) >= RARE]), 3)
        rec10_d[dep] += len(rel & set(ranked[:10])) / len(rel); cnt_d[dep] += 1
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/3000 | BM25 R@10 {rec[10]/(n+1):.4f} MRR {mrr/(n+1):.4f} | {time.perf_counter()-t0:.0f}s", flush=True)
    N = len(qids)
    print(f"\nBM25 (stemmed) recall@k on full 8.8M -- {N} dev queries\n")
    print(f"   {'R@1':>8}{'R@5':>8}{'R@10':>8}{'R@100':>8}{'MRR@10':>9}")
    print(f"   {rec[1]/N:>8.4f}{rec[5]/N:>8.4f}{rec[10]/N:>8.4f}{rec[100]/N:>8.4f}{mrr/N:>9.4f}")
    print(f"\n   BM25 recall@10 BY # rare words (vs meet 0.1290 / 0.2467 / 0.3786):")
    for d in (1, 2, 3):
        lbl = "3+" if d == 3 else str(d)
        print(f"     {lbl} rare words: {cnt_d[d]:>5} q   recall@10 {rec10_d[d]/max(1,cnt_d[d]):.4f}")


if __name__ == "__main__":
    main()
