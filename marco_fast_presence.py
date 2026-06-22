#!/usr/bin/env python3
"""Integrate the user's O(1) membership into the retriever: replace the searchsorted meet with a
hash-free presence array (doc-id IS the address). Wins 1.8x single / 3.5x multi in the micro-bench.
Here: full-retrieval latency + identical MRR@10 vs the searchsorted fast meet, on the 8.8M index.
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B, SCORE_FLOOR
from marco_fast import bm25_fast

N = 300
DF_CAP = 100_000


def bm25_fast_presence(idx, qterms, val, k=100, df_cap=DF_CAP):
    terms = []
    for w in set(qterms):
        i = idx.tid.get(w)
        if i is None:
            continue
        wi = float(idx.idfa[i])
        if wi < SCORE_FLOOR:
            continue
        s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
        terms.append((wi, e - s, s, e))
    if not terms:
        return np.empty(0, np.uint32)
    disc = [(s, e) for (wi, df, s, e) in terms if df < df_cap]
    if not disc:
        wi, df, s, e = min(terms, key=lambda t: t[1]); disc = [(s, e)]
    cand = np.unique(np.concatenate([idx.di[s:e] for (s, e) in disc]))
    dlc = idx.doclen[cand]; sc = np.zeros(len(cand), np.float32)
    for (wi, df, s, e) in terms:
        dis = idx.di[s:e]; val[dis] = idx.tf[s:e]          # scatter tf -- doc-id is the address
        tfc = val[cand].astype(np.float32); hit = tfc > 0   # O(1)/check gather at candidates
        c = wi * tfc * (K1 + 1) / (tfc + K1 * (1 - B + B * dlc / idx.avgdl)); sc[hit] += c[hit]
        val[dis] = 0
    if len(cand) > k:
        sel = np.argpartition(-sc, k)[:k]
    else:
        sel = np.arange(len(cand))
    return cand[sel[np.argsort(-sc[sel])]]


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

    for qid, qt in sample[:5]:
        bm25_fast(idx, stoks(qt), 100); bm25_fast_presence(idx, stoks(qt), val, 100)

    tA, tB, mrrA, mrrB, mism = [], [], 0.0, 0.0, 0
    for qid, qt in sample:
        qts = stoks(qt); gold = qrels[qid]
        t0 = time.perf_counter(); oa, _ = bm25_fast(idx, qts, 100); tA.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); ob = bm25_fast_presence(idx, qts, val, 100); tB.append((time.perf_counter() - t0) * 1000)
        if set(int(x) for x in oa[:100]) != set(int(x) for x in ob[:100]):
            mism += 1
        for r, d in enumerate(oa[:10]):
            if int(d) in gold:
                mrrA += 1 / (r + 1); break
        for r, d in enumerate(ob[:10]):
            if int(d) in gold:
                mrrB += 1 / (r + 1); break
    tA = np.array(tA); tB = np.array(tB)
    print(f"\n  FULL RETRIEVAL -- searchsorted meet vs O(1) presence meet, 8.8M, n={N}\n")
    print(f"  {'method':<26}{'median ms':>12}{'mean':>9}{'p90':>9}{'MRR@10':>10}")
    print(f"  {'fast (searchsorted)':<26}{np.median(tA):>12.2f}{tA.mean():>9.2f}{np.percentile(tA,90):>9.2f}{mrrA/N:>10.4f}")
    print(f"  {'fast (O(1) presence)':<26}{np.median(tB):>12.2f}{tB.mean():>9.2f}{np.percentile(tB,90):>9.2f}{mrrB/N:>10.4f}")
    print(f"\n  top-100 pool mismatch: {mism}/{N}   median speedup over searchsorted: {np.median(tA)/np.median(tB):.1f}x")
    print(f"  footprint: presence reuses a {idx.N*2/1e6:.0f} MB uint16 scratch (transient, not stored) -- index unchanged")


if __name__ == "__main__":
    main()
