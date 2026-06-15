#!/usr/bin/env python3
"""RECALL-FIRST: get the gold into the pool using the lattice meet + semantic corridors, THEN
narrow. The audit says the gold is reachable 97.7% by rare words -- the bottleneck is RANKING
it into the top-K, not membership. So score the WHOLE candidate pool by meet + corridors via
postings (no text fetch, so corridors can lift a gold buried at meet-rank 500), and also test
corridor EXPANSION (pool = rare-word docs UNION corridor-term docs) to catch no-rare-word golds.

Engines (all NO BM25), recall@{100,500,1000} + pool-membership ceiling, full 8.8M:
  A meet           : rare-word union, ranked by meet score (sum rare-idf x depth^1.5)
  B meet+corr rank : rare-word union, ranked by meet + corridor company (corridors LIFT)
  C meet+corr expand: (rare UNION corridor) docs, ranked by meet + corridor (corridors ADD docs)
vs BM25 recall@100 0.666.
"""
import sys, random, time
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE, B_COMP, G_XO


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    idx = FullIndex()
    gold = train_corridors(idx)
    sum_idf = np.zeros(idx.N, np.float32); cnt = np.zeros(idx.N, np.float32)
    corr = np.zeros(idx.N, np.float32)

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

    Ks = (100, 500, 1000)
    recA = defaultdict(float); recB = defaultdict(float); recC = defaultdict(float)
    memR = memC = 0.0           # pool membership: gold in rare-union / in rare+corr-union
    n_eval = 0
    t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = set(int(p) for p in qrels[q])
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        if not rare:
            continue
        n_eval += 1
        # corridor terms (semantic relatives) with weights
        cterms = defaultdict(float)
        for qt in qs:
            for dt, w in gold.get(qt, []):
                cterms[dt] += w
        rel_arr = np.fromiter(rel, dtype=np.int64)
        rare_post = []
        for w in rare:
            i = idx.tid.get(w)
            if i is None:
                continue
            dis = idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]
            sum_idf[dis] += idx.idfa[i]; cnt[dis] += 1.0; rare_post.append(dis)
        # gold reachable via corridor expansion? (cheap membership, no huge union built)
        gold_in_corr = False
        for dt in cterms:
            i = idx.tid.get(dt)
            if i is None:
                continue
            dis = idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]
            corr[dis] += cterms[dt]
            if not gold_in_corr:
                j = np.searchsorted(dis, rel_arr)
                j = np.clip(j, 0, len(dis) - 1)
                if np.any(dis[j] == rel_arr):
                    gold_in_corr = True
        rare_cat = np.concatenate(rare_post) if rare_post else np.empty(0, np.uint32)
        rare_cand = np.unique(rare_cat)
        meetA = sum_idf[rare_cand] * (cnt[rare_cand] ** G_XO)
        scoreB = meetA + B_COMP * corr[rare_cand]

        def topset(cand, score, K):
            if len(cand) <= K:
                return set(int(d) for d in cand)
            sel = np.argpartition(-score, K)[:K]
            return set(int(d) for d in cand[sel])
        for K in Ks:
            recA[K] += len(rel & topset(rare_cand, meetA, K)) / len(rel)
            recB[K] += len(rel & topset(rare_cand, scoreB, K)) / len(rel)
        in_rare = len(rel & set(int(d) for d in rare_cand)) / len(rel)
        memR += in_rare
        memC += 1.0 if (in_rare > 0 or gold_in_corr) else 0.0
        sum_idf[rare_cat] = 0.0; cnt[rare_cat] = 0.0
        for dt in cterms:
            i = idx.tid.get(dt)
            if i is not None:
                corr[idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]] = 0.0
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{nq} | A@100 {recA[100]/n_eval:.3f} B@100 {recB[100]/n_eval:.3f} "
                  f"B@1000 {recB[1000]/n_eval:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nRECALL-FIRST (no BM25) on full 8.8M -- {N} dev queries with a rare word\n")
    print(f"   pool-membership ceiling (gold reachable in the candidate set, ANY rank):")
    print(f"     rare-word union:        {memR/N:.4f}")
    print(f"     + corridor expansion:   {memC/N:.4f}   (expansion adds {(memC-memR)/N:+.4f})")
    print(f"\n   ranked recall (rare-word pool):")
    print(f"   {'engine':<22}{'R@100':>9}{'R@500':>9}{'R@1000':>9}")
    print(f"   {'A meet':<22}{recA[100]/N:>9.4f}{recA[500]/N:>9.4f}{recA[1000]/N:>9.4f}")
    print(f"   {'B meet+corridors':<22}{recB[100]/N:>9.4f}{recB[500]/N:>9.4f}{recB[1000]/N:>9.4f}")
    print(f"\n   BM25 ref: R@100 0.666. Gap from membership {memR/N:.3f} to ranked R@100 = pure ranking loss.")
    print(f"   Goal: a pool with ~0.9 recall, then rerank to narrow. Deeper K or better rank closes it.")


if __name__ == "__main__":
    main()
