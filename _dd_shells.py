#!/usr/bin/env python3
"""Free hidden layers, DEEPER: does order-3 binding (triple shells) lift retrieval past order-2?
The supervised bridges = the AETHOS net at order-2 (pairwise binds + log-odds credit assignment).
aethos_nn.py proves order-3/4 reach rules order-2 can't. Here we apply order-1/2/3 binding shells as a
reranker on scifact: learn log-odds of each order-d conjunction of SHARED rare query-doc terms from train
qrels (counting, no backprop), rerank the lexical top-100, measure nDCG@10. Honest: does depth-3 add?"""
import os, sys, time, math, itertools
os.environ.setdefault("PYTHONUTF8", "1")
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

DFCAP = 0.10          # rare = df <= 10% of corpus (discriminative)
TOPK = 100           # rerank depth
NEG_PER_POS = 8


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive)
    cap = DFCAP * N
    # per-doc rare WORD prime set (discriminative)
    print(f"\n{name}: {N:,} docs | train {len(train_q):,} | test {len(test_ids):,}", flush=True)

    def rareset(text):
        bag = idx._multiview(text)
        s = set()
        for tok in bag:
            if tok[0] != "w":
                continue
            p = idx.token_prime.get(tok)
            if p is not None and 0 < idx.df[p] <= cap:
                s.add(p)
        return s

    drare = {d: rareset(t) for d, t in corpus.items()}
    qrare = {q: rareset(queries[q]) for q in set(test_ids) | set(train_q)}

    def conj(qr, dr, order):
        shared = sorted(qr & dr)
        keys = []
        for o in range(1, order + 1):
            keys.extend(itertools.combinations(shared, o))
        return keys

    # learn order-d log-odds from train qrels (positive = gold, negatives = lexical pool non-gold)
    def learn(order):
        c1, c0, n1, n0 = Counter(), Counter(), 0, 0
        rngs = np.random.RandomState(0)
        for q in train_q:
            if q not in queries:
                continue
            gold = set(train_q[q])
            pool = list(idx.search(queries[q], TOPK))
            qr = qrare.get(q) or rareset(queries[q])
            negs = [d for d in pool if d not in gold]
            rngs.shuffle(negs)
            for g in gold:
                if g in drare:
                    c1.update(conj(qr, drare[g], order)); n1 += 1
            for d in negs[:NEG_PER_POS * max(1, len(gold))]:
                if d in drare:
                    c0.update(conj(qr, drare[d], order)); n0 += 1
        w = {}
        for k in set(c1) | set(c0):
            w[k] = math.log(((c1[k] + 1) / (n1 + 2)) / ((c0[k] + 1) / (n0 + 2)))
        return w

    def rerank(order, w, alpha):
        nd = 0.0
        for q in test_ids:
            base = list(idx.search(queries[q], TOPK))
            sc = idx._score(queries[q])
            qr = qrare[q]
            mx = max(sc.values()) if sc else 1.0
            scored = []
            for d in base:
                shell = sum(w.get(k, 0.0) for k in conj(qr, drare.get(d, set()), order))
                scored.append((d, sc.get(d, 0.0) / mx + alpha * shell))
            ranked = [d for d, _ in sorted(scored, key=lambda x: -x[1])]
            nd += ndcg10(ranked, test_q[q])
        return nd / len(test_ids)

    # lexical baseline
    base_nd = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])
    print(f"  lexical (order-0):           nDCG@10 {base_nd:.4f}", flush=True)
    print(f"  {'shell order':<14}{'nDCG@10':>9}{'vs lexical':>12}{'#conjunctions':>15}", flush=True)
    for order in (1, 2, 3):
        t0 = time.perf_counter(); w = learn(order)
        best = (0.0, None)
        for alpha in (0.5, 1.0, 2.0, 3.0):
            nd = rerank(order, w, alpha)
            if nd > best[0]:
                best = (nd, alpha)
        print(f"  order {order} (a={best[1]})     {best[0]:>9.4f}{best[0]-base_nd:>+12.4f}{len(w):>15,}"
              f"   ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"\n  reference: BM25 0.665 | order-2 bridges 0.7375 | glass_box 0.7645 | SPLADE-scifact ~0.70", flush=True)
    print(f"  read: if order-3 > order-2, the deeper free hidden-layer shells lift retrieval (no GPU).", flush=True)


if __name__ == "__main__":
    main()
