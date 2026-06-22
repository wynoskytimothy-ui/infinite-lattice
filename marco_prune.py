#!/usr/bin/env python3
"""COMPRESS THE BULK, done right: the 95% of postings live in COMMON (low-idf) terms -- the lattice's
rarest-first principle says those are low-discrimination dead weight. Static-prune them (drop terms below
an idf threshold from the index) and measure the compression/speed/accuracy curve. If the rare-term signal
carries the ranking, we keep a fraction of the postings at ~no MRR loss = smaller AND faster AND accurate.
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B

N = 400
DF_CAP = 100_000


def bm25_prune(idx, qterms, val, keep_idf, k=100):
    terms = []
    for w in set(qterms):
        i = idx.tid.get(w)
        if i is None:
            continue
        wi = float(idx.idfa[i])
        if wi < keep_idf:                      # term pruned from the index
            continue
        s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
        terms.append((wi, e - s, s, e))
    if not terms:
        return np.empty(0, np.uint32)
    disc = [(s, e) for (wi, df, s, e) in terms if df < DF_CAP]
    if not disc:
        wi, df, s, e = min(terms, key=lambda t: t[1]); disc = [(s, e)]
    cand = np.unique(np.concatenate([idx.di[s:e] for (s, e) in disc]))
    dlc = idx.doclen[cand]; sc = np.zeros(len(cand), np.float32)
    for (wi, df, s, e) in terms:
        dis = idx.di[s:e]; val[dis] = idx.tf[s:e]
        tfc = val[cand].astype(np.float32); hit = tfc > 0
        c = wi * tfc * (K1 + 1) / (tfc + K1 * (1 - B + B * dlc / idx.avgdl)); sc[hit] += c[hit]
        val[dis] = 0
    if len(cand) > k:
        sel = np.argpartition(-sc, k)[:k]
    else:
        sel = np.arange(len(cand))
    return cand[sel[np.argsort(-sc[sel])]]


def main():
    idx = FullIndex(); val = np.zeros(idx.N, np.uint16)
    df = np.diff(idx.ptr.astype(np.int64)); npost = int(df.sum())
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
        bm25_prune(idx, stoks(qt), val, 1.0, 100)

    print(f"\n  STATIC PRUNE THE COMMON-TERM BULK -- keep terms with idf>=T, 8.8M, n={N}\n")
    print(f"  {'keep idf>=':<12}{'postings kept':>16}{'%post':>8}{'index GB*':>11}{'median ms':>12}{'MRR@10':>10}")
    for T in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        kept = int(df[idx.idfa >= T].sum())
        gb = (kept / npost) * (0.450 + 0.176)        # quantized di+tf scaled by postings kept
        ts = []; mrr = 0.0
        for qid, qt in sample:
            t0 = time.perf_counter(); o = bm25_prune(idx, stoks(qt), val, T, 100); ts.append((time.perf_counter() - t0) * 1000)
            gold = qrels[qid]
            for r, d in enumerate(o[:10]):
                if int(d) in gold:
                    mrr += 1.0 / (r + 1); break
        print(f"  {T:<12.1f}{kept:>16,}{kept/npost*100:>7.1f}%{gb:>10.3f}{np.median(ts):>12.2f}{mrr/N:>10.4f}")
    print(f"\n  *index GB = quantized di+tf scaled by postings kept (+ ~0.05 vocab/ptr). T=1.0 is the current index.")
    print(f"  reading: if MRR holds while postings drop, the common-term bulk is dead weight -> smaller+faster+accurate.")


if __name__ == "__main__":
    main()
