#!/usr/bin/env python3
"""UNSUPERVISED transitive-triangle mining (Timothy's 'second ingest', no qrels, no SPLADE).
Build a doc-doc MEET graph: rare shared words create an edge (2-way meet), weighted by idf.
Then diffuse query relevance across the graph: 1-hop = direct meets, 2-hop = the TRIANGLE
(A meets B, B meets C => A,C linked even if they never directly meet) = hidden semantic links.
Rerank base lattice scores by (base + alpha*1hop + beta*2hop). Measure lift over base 0.7023,
no supervision. Footprint: the graph is mined then discardable; only the base index ships."""
import os, sys, time, math
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10, recall10

def mrr10(ranked, rels):
    for i, d in enumerate(ranked[:10]):
        if rels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0

def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    DFCAP = int(os.environ.get("DFCAP", "60"))
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    docs = sorted(idx.alive); idmap = {d: i for i, d in enumerate(docs)}
    Nn = len(docs)
    print(f"\n{name}: {N:,} docs | test {len(test_ids):,} q | DFCAP={DFCAP}", flush=True)

    # ---- mine the doc-doc meet graph from rare WORD primes (2-way meet = shared rare word) ----
    MINSHARE = int(os.environ.get("MINSHARE", "2"))   # edge only if docs share >= this many rare words (3-way meet)
    t0 = time.perf_counter()
    G = np.zeros((Nn, Nn), np.float32)
    Cnt = np.zeros((Nn, Nn), np.float32)
    rarew = 0
    for tok, p in idx.token_prime.items():
        if tok[0] != "w":
            continue
        dfp = idx.df[p]
        if dfp < 2 or dfp > DFCAP:
            continue
        dd = [idmap[d] for d in idx.postings[p] if d in idmap]
        if len(dd) < 2:
            continue
        idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
        ii = np.array(dd)
        G[np.ix_(ii, ii)] += idf
        Cnt[np.ix_(ii, ii)] += 1.0
        rarew += 1
    np.fill_diagonal(G, 0.0); np.fill_diagonal(Cnt, 0.0)
    G = G * (Cnt >= MINSHARE)                          # keep edges only where docs share >= MINSHARE rare words
    rs = G.sum(1, keepdims=True); rs[rs == 0] = 1.0
    Gn = (G / rs).astype(np.float32)
    print(f"  meet graph (MINSHARE={MINSHARE}): {rarew:,} rare words -> {int((G>0).sum()):,} edges, "
          f"build {time.perf_counter()-t0:.1f}s", flush=True)

    def basevec(q):
        sc = idx._score(q)
        b = np.zeros(Nn, np.float32)
        if sc:
            mx = max(sc.values()) or 1.0
            for d, v in sc.items():
                if d in idmap:
                    b[idmap[d]] = v / mx
        return b
    bvecs = {qid: basevec(queries[qid]) for qid in test_ids}

    def rank(b, alpha, beta):
        if alpha == 0 and beta == 0:
            final = b
        else:
            s1 = Gn @ b
            final = b + alpha * s1
            if beta:
                final = final + beta * (Gn @ s1)            # 2-hop = the triangle
        order = np.argsort(-final)[:20]
        return [docs[i] for i in order]

    def ev(alpha, beta):
        nd = rc = mr = 0.0
        for qid in test_ids:
            r = rank(bvecs[qid], alpha, beta)
            nd += ndcg10(r, test_q[qid]); rc += recall10(r, test_q[qid]); mr += mrr10(r, test_q[qid])
        n = len(test_ids); return nd / n, rc / n, mr / n

    print(f"  {'config':<22}{'nDCG@10':>9}{'R@10':>8}{'MRR@10':>9}", flush=True)
    base = None
    for alpha, beta in [(0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0.2, 0.1), (0.3, 0.2), (0.5, 0.3)]:
        nd, rc, mr = ev(alpha, beta)
        if base is None: base = nd
        tag = "  base" if (alpha == 0 and beta == 0) else (f"  ({nd-base:+.4f})")
        lbl = "base (alpha=0)" if (alpha == 0 and beta == 0) else f"1hop a={alpha} 2hop b={beta}"
        print(f"  {lbl:<22}{nd:>9.4f}{rc:>8.4f}{mr:>9.4f}{tag}", flush=True)
    print(f"\n  reference: base 0.7023 | +SUPERVISED bridges 0.7375 | +glass_box 0.7645", flush=True)

if __name__ == "__main__":
    main()
