#!/usr/bin/env python3
"""Wire the corridor the RIGHT way: as BRIDGES through the proven bridge_search
mechanism (query-term -> correlated doc-term, rerank+pool-expand the lexical
candidates) -- the same machine the supervised bridges use to get +6.5pp -- but
with weights from UNSUPERVISED corpus co-occurrence instead of qrels. No qrels.

This is the test of 'you didn't wire it right': same proven mechanism, corridor
weights. If it lifts retrieval, the earlier expansion-RRF wiring was the problem."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10


class CorridorBridges:
    """bridge[qt] = the rare doc-terms that co-occur with qt in the CORPUS, weighted
    P(dt|qt)*idf(dt) -- the unsupervised twin of RelevanceBridges (which counts
    co-occurrence in RELEVANT pairs). Same shape, so bridge_search consumes it."""

    def __init__(self, idx, N, min_co=3, top_per_term=12, idf_gate=2.5):
        self.idx, self.N = idx, N
        self.min_co, self.top_per_term, self.idf_gate = min_co, top_per_term, idf_gate
        self.bridge = {}
        self._idf = {}

    def idf(self, w):
        v = self._idf.get(w)
        if v is None:
            p = self.idx.token_prime.get(("w", w))
            v = self.idx._idf(p, self.N) if p else 0.0
            self._idf[w] = v
        return v

    def learn(self, corpus):
        cooc, tdf = defaultdict(Counter), Counter()
        for t in corpus.values():
            ws = [w for w in set(words(t)) if self.idf(w) >= self.idf_gate]
            for a in ws:
                tdf[a] += 1
            for a in ws:
                cooc[a].update(ws)
        for qt, partners in cooc.items():
            np_ = tdf[qt]
            scored = [(dt, (c / np_) * self.idf(dt)) for dt, c in partners.items()
                      if dt != qt and c >= self.min_co]
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored:
                self.bridge[qt] = scored[:self.top_per_term]
        return self


def run(ds, lam):
    corpus, queries, train_q, test_q = load(ds)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive)
    cb = CorridorBridges(idx, N).learn(corpus)

    def lexical(q, k=10):
        s = idx._score(q)
        return sorted(s, key=s.get, reverse=True)[:k]

    def ev(fn):
        nd = rc = 0.0
        for qid in test_ids:
            r = fn(queries[qid])
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    b = ev(lexical)
    c = ev(lambda q: bridge_search(idx, cb, q, lam=lam))
    return b, c, len(cb.bridge)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    print(f"corridor-as-BRIDGES via the proven bridge_search mechanism ({ds}, zero-shot)\n")
    print(f"  {'lam':>5}{'baseline nDCG':>15}{'+corridor-bridge':>18}{'delta':>9}{'recall d':>10}")
    for lam in (0.10, 0.25, 0.5):
        b, c, nb = run(ds, lam)
        print(f"  {lam:>5}{b[0]:>15.4f}{c[0]:>18.4f}{c[0]-b[0]:>+9.4f}{c[1]-b[1]:>+10.4f}")
    print(f"\n  (corridor bridges built unsupervised from corpus co-occurrence, no qrels;")
    print(f"   same mechanism that gives the supervised bridges +6.5pp.)")


if __name__ == "__main__":
    main()
