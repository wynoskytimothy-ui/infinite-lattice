#!/usr/bin/env python3
"""RECALL-MAX: push the ranked recall of the rare-word pool toward its 0.979 membership ceiling
(the gold is reachable 0.979 but only ranked into top-1000 at 0.826 -- 0.15 of mis-ranked gold).

Levers tested (all NO BM25 over common terms; rare-word pool; full 8.8M, 3000 dev q):
  - meet      : sum rare-idf x depth^1.5            (current; ignores term-frequency)
  - bm25-rare : sum rare-idf x tf-saturation        (adds the tf the meet threw away)
  + corridor company, swept weight {0.3, 1, 3}.
Reports recall@100/500/1000 -> which ranking recovers the most reachable gold.
"""
import sys, random, time
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE, G_XO, K1, B


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)
    sum_idf = np.zeros(idx.N, np.float32); cnt = np.zeros(idx.N, np.float32)
    bm = np.zeros(idx.N, np.float32); corr = np.zeros(idx.N, np.float32)
    BSW = [0.3, 1.0, 3.0]
    engines = [("meet", "m"), ("bm25-rare", "b")]
    rec = {(e, bs, K): 0.0 for e, _ in engines for bs in BSW for K in (100, 500, 1000)}
    memR = 0.0; n_eval = 0

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

    t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = set(int(p) for p in qrels[q])
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        if not rare:
            continue
        n_eval += 1
        cterms = defaultdict(float)
        for qt in qs:
            for dt, w in gold.get(qt, []):
                cterms[dt] += w
        rare_post = []
        for w in rare:
            i = idx.tid.get(w)
            if i is None:
                continue
            s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
            dis = idx.di[s:e]; tfs = idx.tf[s:e].astype(np.float32); dl = idx.doclen[dis]
            wi = idx.idfa[i]
            sum_idf[dis] += wi; cnt[dis] += 1.0
            bm[dis] += wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / idx.avgdl))
            rare_post.append(dis)
        corr_post = []
        for dt, w in cterms.items():
            i = idx.tid.get(dt)
            if i is None:
                continue
            dis = idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]
            corr[dis] += w; corr_post.append(dis)
        rare_cat = np.concatenate(rare_post) if rare_post else np.empty(0, np.uint32)
        cand = np.unique(rare_cat)
        meet = sum_idf[cand] * (cnt[cand] ** G_XO)
        bmr = bm[cand]
        cc = corr[cand]
        memR += len(rel & set(int(d) for d in cand)) / len(rel)

        def topset(score, K):
            if len(cand) <= K:
                return set(int(d) for d in cand)
            return set(int(d) for d in cand[np.argpartition(-score, K)[:K]])
        for bs in BSW:
            sm = meet + bs * cc
            sb = bmr + bs * cc
            for K in (100, 500, 1000):
                rec[("meet", bs, K)] += len(rel & topset(sm, K)) / len(rel)
                rec[("bm25-rare", bs, K)] += len(rel & topset(sb, K)) / len(rel)
        sum_idf[rare_cat] = 0.0; cnt[rare_cat] = 0.0; bm[rare_cat] = 0.0
        if corr_post:
            corr[np.concatenate(corr_post)] = 0.0
        if (n + 1) % 500 == 0:
            best = max(rec[("bm25-rare", bs, 1000)] for bs in BSW) / n_eval
            print(f"    {n+1}/{nq} | bm25-rare best R@1000 {best:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nRECALL-MAX (rare-word pool, no common-term BM25) -- {N} dev q. membership ceiling {memR/N:.4f}\n")
    print(f"   {'engine':<12}{'B_COMP':>8}{'R@100':>9}{'R@500':>9}{'R@1000':>9}")
    for e, _ in engines:
        for bs in BSW:
            print(f"   {e:<12}{bs:>8.1f}{rec[(e,bs,100)]/N:>9.4f}{rec[(e,bs,500)]/N:>9.4f}{rec[(e,bs,1000)]/N:>9.4f}")
    print(f"\n   prior: meet+corr(0.3) R@1000 0.826. ceiling {memR/N:.3f}. BM25 ref R@100 0.666.")
    print(f"   tf-saturation (bm25-rare) recovers the term-frequency signal the bare meet ignores.")


if __name__ == "__main__":
    main()
