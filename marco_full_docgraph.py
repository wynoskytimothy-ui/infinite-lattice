#!/usr/bin/env python3
"""DOC<->DOC cross-referencing (the user's idea): build a doc-similarity graph and let docs
reinforce each other's scores -- the cluster hypothesis (relevant docs cluster) as score
propagation. Distinct from the query->doc LSA test: here docs link to EACH OTHER by the specific
rare vocabulary they share BEYOND the query (so links mean 'about the same specific thing', not
just 'both on topic'). A candidate central in the relevant cluster gets lifted.

  s = bm25 ; repeat: s <- (1-lam)*bm25 + lam * (row-normalized doc-doc sim) @ s
Rerank BM25 top-100 by the propagated score. Full 8.8M, 500 dev q.
Yardsticks: BM25 0.189, binary-contextual 0.200, cross-encoder 0.407.
"""
import sys, time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, RARE, QGATE


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 500
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
    random.Random(42).shuffle(qids); qids = qids[:nq]

    CONFIGS = [(0.0, 1), (0.3, 1), (0.3, 3), (0.6, 3)]   # (lambda, steps); 0.0 = pure BM25
    mrr = defaultdict(float); n_eval = 0; t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        qset = set(qs)
        n_eval += 1
        order, anchor = idx.bm25_top(qs, k=100)
        pids = [int(d) for d in order]
        m = len(pids)
        bm = np.array([anchor[p] for p in pids], dtype=np.float32)
        # each doc's specific rare vocab BEYOND the query (idf-weighted)
        dvocab = []
        for p in pids:
            ws = {}
            for t in set(stoks(idx.text(p))):
                iw = idx.idf_of(t)
                if iw >= RARE and t not in qset:
                    ws[t] = iw
            dvocab.append(ws)
        # doc-doc similarity: shared specific-rare idf mass (cosine-ish)
        S = np.zeros((m, m), dtype=np.float32)
        for i in range(m):
            wi = dvocab[i]
            if not wi:
                continue
            ni = np.sqrt(sum(v * v for v in wi.values()))
            for j in range(i + 1, m):
                wj = dvocab[j]
                if not wj:
                    continue
                common = wi.keys() & wj.keys()
                if not common:
                    continue
                dot = sum(wi[t] * wj[t] for t in common)
                nj = np.sqrt(sum(v * v for v in wj.values()))
                s = dot / (ni * nj + 1e-9)
                S[i, j] = s; S[j, i] = s
        rowsum = S.sum(axis=1, keepdims=True); rowsum[rowsum == 0] = 1.0
        Sn = S / rowsum
        bm_order = pids
        mrr[("bm25", 0, 0)] += next((1.0 / i for i, p in enumerate(bm_order[:10], 1) if str(p) in rel), 0.0)
        for lam, steps in CONFIGS:
            if lam == 0.0:
                continue
            s = bm.copy()
            for _ in range(steps):
                s = (1 - lam) * bm + lam * (Sn @ s)
            ranked = [p for _, p in sorted(zip(s, pids), key=lambda x: -x[0])]
            mrr[("prop", lam, steps)] += next((1.0 / i for i, p in enumerate(ranked[:10], 1) if str(p) in rel), 0.0)
        if (n + 1) % 100 == 0:
            base = mrr[("bm25", 0, 0)] / n_eval
            best = max(mrr[("prop", l, s)] for l, s in CONFIGS if l > 0) / n_eval
            print(f"    {n+1}/{nq} | bm25 {base:.3f} | best-prop {best:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nDOC<->DOC GRAPH PROPAGATION rerank -- full 8.8M, {N} dev q\n")
    print(f"   {'config':<22}{'MRR@10':>9}")
    print(f"   {'BM25 (no propagation)':<22}{mrr[('bm25',0,0)]/N:>9.4f}")
    for lam, steps in CONFIGS:
        if lam == 0.0:
            continue
        print(f"   {'prop lam='+str(lam)+' steps='+str(steps):<22}{mrr[('prop',lam,steps)]/N:>9.4f}")
    print(f"   {'binary contextual (ref)':<22}{0.2000:>9.4f}")
    print(f"   {'cross-encoder (ref)':<22}{0.4065:>9.4f}")
    print(f"\n   does the cluster hypothesis (docs reinforce each other) lift the gold? > BM25 = the")
    print(f"   doc-doc correlation adds a real signal; ~= BM25 = the on-topic pool is too uniform to cluster.")


if __name__ == "__main__":
    main()
