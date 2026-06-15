#!/usr/bin/env python3
"""
Deep search step 4 - numpy-vectorized scorer: the last strictly-lossless win.

Same arithmetic as _score, but the per-term posting traversal becomes a
vectorized array op, so the Python loop runs ~30 times (query terms) instead of
millions (postings). float64 throughout to match Python's precision - the only
possible difference vs the dict path is float summation ORDER (cross-term),
which moves scores by ~1e-15 and can only reorder docs that are tied to ~15
figures. We MEASURE the top-10 overlap to prove it is lossless in practice.

Dense structure (built once, O(1)-appendable in principle - append rows to the
per-prime arrays):
  - doc_to_idx: alive doc -> dense index 0..N-1
  - denom_base[i] = A + B*dl_i   (the hoisted BM25 length norm, per doc)
  - per prime p: doc_idx[p] (int32), tf[p] (float64)
score: scores[doc_idx] += c * tf / (tf + denom_base[doc_idx])   (vectorized)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10, recall10


class DenseScorer:
    def __init__(self, idx):
        self.idx = idx
        N = max(1, len(idx.alive))
        self.N = N
        self.docs = list(idx.alive)
        self.doc_to_idx = {d: i for i, d in enumerate(self.docs)}
        k1, b = idx.k1, idx.b
        avgdl = idx._total_len / N
        self.k1p1 = k1 + 1
        A, Bc = k1 * (1 - b), k1 * b / avgdl
        self.denom_base = np.array([A + Bc * idx.doc_len[d] for d in self.docs],
                                   dtype=np.float64)
        self.tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
        self.df = idx.df
        self.token_prime = idx.token_prime
        self.p_doc, self.p_tf = {}, {}
        d2i = self.doc_to_idx
        for p, pl in idx.postings.items():
            di = np.fromiter((d2i[d] for d in pl if d in d2i), dtype=np.int32)
            if di.size:
                self.p_doc[p] = di
                self.p_tf[p] = np.fromiter((pl[d] for d in pl if d in d2i),
                                           dtype=np.float64, count=di.size)

    def score(self, query):
        idx, N = self.idx, self.N
        scores = np.zeros(N, dtype=np.float64)
        cap = self.tri_cap
        for tok, qwt in idx._multiview(query).items():
            p = self.token_prime.get(tok)
            if p is None:
                continue
            dfp = self.df[p]
            if dfp == 0:
                continue
            if cap is not None and tok[0] == "3" and dfp > cap:
                continue
            di = self.p_doc.get(p)
            if di is None:
                continue
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            c = qwt * idf * self.k1p1
            tfa = self.p_tf[p]
            scores[di] += c * tfa / (tfa + self.denom_base[di])
        return scores

    def search(self, query, k=10):
        scores = self.score(query)
        k = min(k, self.N)
        part = np.argpartition(scores, -k)[-k:]
        part = part[np.argsort(scores[part])[::-1]]
        return [self.docs[i] for i in part if scores[i] > 0.0]


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    print(f"\n{'='*60}\n{name}: {len(idx.alive):,} docs, {len(test_ids)} test q")

    t0 = time.perf_counter()
    dense = DenseScorer(idx)
    build_s = time.perf_counter() - t0
    mem = sum(a.nbytes for a in dense.p_doc.values()) \
        + sum(a.nbytes for a in dense.p_tf.values()) + dense.denom_base.nbytes
    print(f"  dense build: {build_s:.1f}s, arrays {mem/1e6:.0f} MB")

    def bench(search_fn):
        nd = rc = 0.0
        lat = []
        tops = []
        for qid in test_ids:
            t = time.perf_counter()
            ranked = search_fn(queries[qid])
            lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
            tops.append(ranked)
        n = len(test_ids)
        lat.sort()
        return nd / n, rc / n, sum(lat) / n, lat[int(n * 0.99)], tops

    nd0, rc0, ms0, p99_0, tops0 = bench(lambda q: idx.search(q, 10))
    nd1, rc1, ms1, p99_1, tops1 = bench(lambda q: dense.search(q, 10))
    overlap = sum(len(set(a) & set(b)) for a, b in zip(tops0, tops1)) \
        / (10 * len(test_ids))
    exact = sum(1 for a, b in zip(tops0, tops1) if a == b) / len(test_ids)

    print(f"  {'path':<14} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} {'p99':>7}")
    print(f"  {'dict (Python)':<14} {nd0:>7.4f} {rc0:>7.4f} {ms0:>7.2f} {p99_0:>7.2f}")
    print(f"  {'numpy dense':<14} {nd1:>7.4f} {rc1:>7.4f} {ms1:>7.2f} {p99_1:>7.2f}"
          f"   ({ms0/ms1:.1f}x)")
    print(f"  top-10 overlap: {overlap*100:.2f}%   exact-identical lists: {exact*100:.1f}%"
          f"   nDCG delta {nd1-nd0:+.5f}")
    return ms0 / ms1, overlap


def main():
    print("DEEP SEARCH step 4 - numpy-vectorized scorer (strictly lossless)")
    for ds in ("scifact", "nfcorpus"):
        run(ds)
    print("\n  Vectorized per-term scatter-add: Python loop is ~30 terms, not")
    print("  millions of postings. float64 => only FP summation order differs")
    print("  (~1e-15), so top-10 overlap should be ~100% = lossless speedup.")


if __name__ == "__main__":
    main()
