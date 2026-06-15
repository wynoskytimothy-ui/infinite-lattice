#!/usr/bin/env python3
"""
Deep search step 6 - wire the supervised bridges through the DENSE fast path.

best_search (the full stack) currently calls idx._score (the dict path, ~10ms)
for candidate generation, so the +bridges query never rode the 0.55ms dense
scorer. Here the bridge reranker uses:
  - idx.dense_scores(query)  -> full lexical score VECTOR (vectorized, ~0.5ms)
  - idx.dense_posting(prime) -> a bridge target word's dense posting, for a
    vectorized scatter-add expansion (relevance-synonym pool growth)
then the same conservative lex + lam*expand fusion.

Validates: top-10 must match the dict best_search (accuracy parity), and the
full-stack latency should drop from ~10ms to ~1ms. Held-out test queries.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges
from scripts.bench_active_learning import best_search   # dict-path baseline

LAM, N_EXPAND = 0.25, 20


def bridge_search_dense(idx, br, query, lam=LAM, n_expand=N_EXPAND):
    lex, docs = idx.dense_scores(query)               # full vector, dense indices
    nz = np.nonzero(lex)[0]
    if nz.size == 0:
        return []
    cand = nz[np.argpartition(lex[nz], -100)[-100:]] if nz.size > 100 else nz
    # bridge expansion via dense word postings (relevance-synonym reach)
    exp = np.zeros(len(docs), dtype=np.float64)
    for qt in set(words(query)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            post = idx.dense_posting(p)
            if post is None:
                continue
            di, tfa = post
            exp[di] += w * tfa / (tfa + 1.0)
    cset = set(int(i) for i in cand)
    enz = np.nonzero(exp)[0]
    extra = []
    if enz.size:
        order = enz[np.argsort(exp[enz])[::-1]]
        extra = [int(i) for i in order if int(i) not in cset][:n_expand]
    pool = [int(i) for i in cand] + extra
    lmax = max(lex[i] for i in pool) or 1.0
    emax = exp.max() or 1.0
    final = {i: lex[i] / lmax + lam * exp[i] / emax for i in pool}
    top = sorted(final, key=final.get, reverse=True)[:10]
    return [docs[i] for i in top]


def run(name, min_pairs):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    idx.finalize()
    print(f"\n{'='*62}\n{name}: {N:,} docs, {len(test_ids)} test q (min_pairs={min_pairs})")

    def bench(fn):
        nd = rc = 0.0
        lat = []
        tops = []
        for qid in test_ids:
            t = time.perf_counter()
            r = fn(queries[qid])
            lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
            tops.append(r)
        n = len(test_ids)
        lat.sort()
        return nd / n, rc / n, sum(lat) / n, lat[int(n * 0.99)], tops

    nd0, rc0, ms0, p99_0, t0 = bench(lambda q: best_search(idx, br, q))
    nd1, rc1, ms1, p99_1, t1 = bench(lambda q: bridge_search_dense(idx, br, q))
    ov = sum(len(set(a) & set(b)) for a, b in zip(t0, t1)) / (10 * len(test_ids))

    print(f"  {'path':<24} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} {'p99':>6}")
    print(f"  {'dict best_search':<24} {nd0:>7.4f} {rc0:>7.4f} {ms0:>7.2f} {p99_0:>6.1f}")
    print(f"  {'dense bridges':<24} {nd1:>7.4f} {rc1:>7.4f} {ms1:>7.2f} {p99_1:>6.1f}"
          f"   ({ms0/ms1:.1f}x)")
    print(f"  parity: top-10 overlap {ov*100:.2f}%   nDCG delta {nd1-nd0:+.5f}")
    return ms0, ms1


def main():
    print("DEEP SEARCH step 6 - bridges on the dense fast path (full-stack speed)")
    for name, mp in (("scifact", 1), ("nfcorpus", 2)):
        run(name, mp)
    print("\n  Bridges now generate candidates from the dense lexical vector and")
    print("  expand via dense word postings - the whole +bridges query rides the")
    print("  fast path. Accuracy must match the dict best_search (parity check).")


if __name__ == "__main__":
    main()
