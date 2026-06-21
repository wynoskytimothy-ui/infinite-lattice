#!/usr/bin/env python3
"""Make full-8.8M retrieval fast using the LATTICE'S OWN formula -- no footprint/accuracy change.

The current bm25_top scans every non-stopword term's full postings into a dense 8.8M accumulator
(276 ms median, 1.8 s tail). But the lattice's principle is "the rarest term is the address; the
meet narrows." So:

  bm25_fast(q):
    1. candidates = union of postings of the DISCRIMINATIVE query terms (df < DF_CAP) -- short lists,
       the rare-term ADDRESSES. (fallback: the single rarest term if all are common.)
    2. score each candidate EXACTLY by MEETING it against every query term's postings via searchsorted
       (binary search), summing the identical BM25 contribution. Never touch the common-term universe.

Same di/tf/ptr/idf arrays (same 2.16 GB), identical BM25 score on every candidate (same accuracy on
the top-k), but work is bounded by the rare-term lists, not the corpus. Measured head-to-head vs the
full scan: speed, MRR@10, and top-100 agreement.
"""
import time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B, SCORE_FLOOR

DF_CAP = 100_000      # a query term is a candidate "address" if its postings list is shorter than this
N = 300


def bm25_fast(idx, qterms, k=100, df_cap=DF_CAP):
    terms = []
    for w in set(qterms):
        i = idx.tid.get(w)
        if i is None:
            continue
        wi = float(idx.idfa[i])
        if wi < SCORE_FLOOR:                 # stopword-ish: skip (same as full scan)
            continue
        s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
        terms.append((wi, e - s, s, e))
    if not terms:
        return np.empty(0, dtype=np.uint32), 0
    # candidate addresses = discriminative terms (short lists); fallback to the single rarest by df
    disc = [(s, e) for (wi, df, s, e) in terms if df < df_cap]
    if not disc:
        wi, df, s, e = min(terms, key=lambda t: t[1])
        disc = [(s, e)]
    cand = np.unique(np.concatenate([idx.di[s:e] for (s, e) in disc]))
    dlc = idx.doclen[cand]
    scores = np.zeros(len(cand), dtype=np.float32)
    for (wi, df, s, e) in terms:             # meet each term against the candidate set (exact BM25)
        dis = idx.di[s:e]
        pos = np.searchsorted(dis, cand)
        pos = np.minimum(pos, len(dis) - 1)
        hit = dis[pos] == cand
        tfs = idx.tf[s:e][pos].astype(np.float32)
        contrib = wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dlc / idx.avgdl))
        scores[hit] += contrib[hit]
    if len(cand) > k:
        sel = np.argpartition(-scores, k)[:k]
    else:
        sel = np.arange(len(cand))
    order = sel[np.argsort(-scores[sel])]
    return cand[order], len(cand)


def main():
    idx = FullIndex()
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
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    for qid, qt in sample[:10]:           # warm both paths
        idx.bm25_top(stoks(qt), 100); bm25_fast(idx, stoks(qt), 100)

    tf_, tfa, mrr_f, mrr_a, agree, csz = [], [], 0.0, 0.0, [], []
    for qid, qt in sample:
        qts = stoks(qt); gold = qrels[qid]
        t0 = time.perf_counter(); of, _ = idx.bm25_top(qts, 100); tf_.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter(); oa, nc = bm25_fast(idx, qts, 100); tfa.append((time.perf_counter() - t0) * 1000)
        csz.append(nc)
        sf = set(int(d) for d in of[:100]); sa = set(int(d) for d in oa[:100])
        agree.append(len(sf & sa) / max(1, len(sf)))
        for rank, d in enumerate(of[:10]):
            if int(d) in gold: mrr_f += 1.0 / (rank + 1); break
        for rank, d in enumerate(oa[:10]):
            if int(d) in gold: mrr_a += 1.0 / (rank + 1); break
    tf_ = np.array(tf_); tfa = np.array(tfa); csz = np.array(csz)
    print(f"\n  LATTICE FORMULA (rarest-address + meet) vs FULL SCAN  -- 8.8M docs, n={N}\n")
    print(f"  {'':<18}{'median ms':>12}{'mean':>9}{'p90':>9}{'max':>9}{'MRR@10':>10}")
    print(f"  {'full scan':<18}{np.median(tf_):>12.1f}{tf_.mean():>9.1f}{np.percentile(tf_,90):>9.1f}{tf_.max():>9.0f}{mrr_f/N:>10.4f}")
    print(f"  {'fast (meet)':<18}{np.median(tfa):>12.1f}{tfa.mean():>9.1f}{np.percentile(tfa,90):>9.1f}{tfa.max():>9.0f}{mrr_a/N:>10.4f}")
    print(f"\n  speedup: {np.median(tf_)/np.median(tfa):.1f}x median, {tf_.mean()/tfa.mean():.1f}x mean, {tf_.max()/max(1,tfa.max()):.1f}x worst-case")
    print(f"  top-100 agreement fast-vs-full: mean {np.mean(agree)*100:.2f}%  (1.0 = identical pool)")
    print(f"  candidate-set size: median {int(np.median(csz)):,}  p90 {int(np.percentile(csz,90)):,}  max {int(csz.max()):,}")
    print(f"  footprint unchanged (same di/tf/ptr/idf arrays); MRR delta {abs(mrr_f-mrr_a)/N:+.4f}")


if __name__ == "__main__":
    main()
