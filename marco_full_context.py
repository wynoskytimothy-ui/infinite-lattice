#!/usr/bin/env python3
"""The CONTEXTUAL second lattice -- the user's thesis: a richer (context-encoding) address makes
the 2-way intersection a CONTEXTUAL similarity, the symbolic analog of ColBERT's MaxSim. Same
term in different contexts => different score (bank|river vs bank|money).

Operationalized as a symbolic MaxSim over the BM25 top-100 (the same pool the cross-encoder used):
for each query term qi, find where qi (or a corridor companion) occurs in the doc; score the BEST
local window by how much of the REST of the query (terms + companions) sits in that window. A rare
term in the right context (surrounded by the query's other concepts) scores high; the same term in
a different context (the polysemy / wrong-doc case) scores low.

  score(doc) = sum_i idf(qi) * (1 + max_pos  sum_{j!=i} idf(qj)*[concept_j in window(pos)])

Yardsticks (same BM25 pool): BM25 alone 0.189, first-order corridor (no position), cross-encoder 0.407.
Variants: proximity-only (query terms as context) vs +corridor (companions = semantic context).
"""
import sys, time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE

W = 8   # context window (+/- tokens)


def ctx_score(qts, idf, comp, dtok):
    """symbolic contextual MaxSim. comp[qt] = set of companion terms (empty = proximity-only)."""
    concept_pos = {}
    for qt in qts:
        ct = comp[qt] | {qt}
        concept_pos[qt] = [j for j, t in enumerate(dtok) if t in ct]
    score = 0.0
    n = len(dtok)
    for qi in qts:
        ps = concept_pos[qi]
        if not ps:
            continue
        best = 0.0
        for p in ps:
            win = set(dtok[max(0, p - W):min(n, p + W + 1)])
            ctx = 0.0
            for qj in qts:
                if qj == qi:
                    continue
                if win & (comp[qj] | {qj}):
                    ctx += idf[qj]
            if ctx > best:
                best = ctx
        score += idf[qi] * (1.0 + best)
    return score


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    idx = FullIndex()
    gold = train_corridors(idx)

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
    random.Random(42).shuffle(qids); qids = qids[:nq]

    mrr = defaultdict(float); n_eval = 0; t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        n_eval += 1
        idf = {w: idx.idf_of(w) for w in qs}
        comp_full = {w: set(dt for dt, _ in gold.get(w, [])) for w in qs}
        comp_none = {w: set() for w in qs}
        order, _ = idx.bm25_top(qs, k=100)
        pids = [int(d) for d in order]
        dtoks = [stoks(idx.text(p)) for p in pids]
        # BM25 baseline order
        mrr["bm25"] += next((1.0 / i for i, p in enumerate(pids[:10], 1) if str(p) in rel), 0.0)
        # first-order corridor (flat company, NO position) -- the "below" yardstick
        flat = []
        for p, dt in zip(pids, dtoks):
            ds = set(dt)
            sc = sum(idf[w] for w in qs if w in ds) + 0.3 * sum(wt for w in qs for d2, wt in gold.get(w, []) if d2 in ds)
            flat.append((sc, p))
        fo = [p for _, p in sorted(flat, key=lambda x: -x[0])]
        mrr["corridor"] += next((1.0 / i for i, p in enumerate(fo[:10], 1) if str(p) in rel), 0.0)
        # contextual: proximity-only and +corridor
        for tag, comp in (("ctx_prox", comp_none), ("ctx_corr", comp_full)):
            scored = [(ctx_score(qs, idf, comp, dt), p) for p, dt in zip(pids, dtoks)]
            ranked = [p for _, p in sorted(scored, key=lambda x: -x[0])]
            mrr[tag] += next((1.0 / i for i, p in enumerate(ranked[:10], 1) if str(p) in rel), 0.0)
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{nq} | bm25 {mrr['bm25']/n_eval:.3f} corridor {mrr['corridor']/n_eval:.3f} "
                  f"ctx_prox {mrr['ctx_prox']/n_eval:.3f} ctx_corr {mrr['ctx_corr']/n_eval:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nCONTEXTUAL SECOND LATTICE -- rerank BM25 top-100, full 8.8M, {N} dev q (window +/-{W})\n")
    print(f"   {'reranker':<26}{'MRR@10':>9}")
    print(f"   {'BM25 (first stage)':<26}{mrr['bm25']/N:>9.4f}")
    print(f"   {'first-order corridor':<26}{mrr['corridor']/N:>9.4f}")
    print(f"   {'contextual: proximity':<26}{mrr['ctx_prox']/N:>9.4f}")
    print(f"   {'contextual: + corridor':<26}{mrr['ctx_corr']/N:>9.4f}")
    print(f"   {'cross-encoder (ref)':<26}{0.4065:>9.4f}")
    print(f"\n   does context (same term, different surroundings) lift symbolic reranking toward the")
    print(f"   cross-encoder? climb past corridor/BM25 = your thesis; short of 0.41 = neural distill works.")


if __name__ == "__main__":
    main()
