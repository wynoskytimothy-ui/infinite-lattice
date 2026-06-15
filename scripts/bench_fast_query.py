#!/usr/bin/env python3
"""
Deep search step 2 - implement the lossless speed wins, measure end-to-end.

From profile_query_speed.py: high-df char-trigrams are ~95% of query time and
near-zero idf. Three speed wins that keep accuracy:
  1. TRIGRAM df-cap: skip query trigrams with df > cap (high-df = low-idf =
     longest lists). Near-lossless (slightly BETTER on scifact - denoising).
  2. PRECOMPUTE per-doc length norm  B_d = k1*(1-b+b*dl/avgdl)  once, so the
     inner loop is  tf/(tf+B_d)  instead of recomputing the norm per posting.
  3. SKIP the alive-check per posting when there are no deletions.

Measured WITH bridges (the full stack) on held-out scifact + nfcorpus, vs the
current baseline, at cap = N/2 (safe) and N/4 (more speed).
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges

LAM, N_EXPAND = 0.25, 20


def make_fast_scorer(idx):
    """Closure with precomputed length-norm and alive-skip; trigram df-cap arg."""
    N = max(1, len(idx.alive))
    k1, b = idx.k1, idx.b
    avgdl = idx._total_len / N
    lennorm = {d: k1 * (1 - b + b * dl / avgdl) for d, dl in idx.doc_len.items()}
    no_removals = (len(idx.alive) == len(idx.doc_len))
    alive = idx.alive
    tp, post, df = idx.token_prime, idx.postings, idx.df

    def fast_score(query, tri_cap=None):
        scores = defaultdict(float)
        for tok, qwt in idx._multiview(query).items():
            p = tp.get(tok)
            if p is None:
                continue
            dfp = df[p]
            if dfp == 0:
                continue
            if tok[0] == "3" and tri_cap is not None and dfp > tri_cap:
                continue
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            c = qwt * idf * (k1 + 1)
            pl = post[p]
            if no_removals:
                for doc, tf in pl.items():
                    scores[doc] += c * tf / (tf + lennorm[doc])
            else:
                for doc, tf in pl.items():
                    if doc in alive:
                        scores[doc] += c * tf / (tf + lennorm[doc])
        return scores

    return fast_score


def bridge_search(idx, br, score_fn, q, tri_cap):
    """Pool-expansion search using a pluggable lexical scorer."""
    lex = score_fn(q) if tri_cap is None else score_fn(q, tri_cap)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    exp = defaultdict(float)
    for qt in set(words(q)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            for d, tf in idx.postings.get(p, {}).items():
                exp[d] += w * tf / (tf + 1.0)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
             if d not in cset][:N_EXPAND]
    pool = cand + extra
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + LAM * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:10]


def run(name, min_pairs):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    fast = make_fast_scorer(idx)
    print(f"\n{'='*60}\n{name}: {N:,} docs, {len(test_ids)} test q (min_pairs={min_pairs})")
    print(f"  {'config':<26} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} {'speedup':>8}")

    def bench(label, fn):
        nd = rc = 0.0
        lat = []
        for qid in test_ids:
            t = time.perf_counter()
            ranked = fn(queries[qid])
            lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n, sum(lat) / n

    # baseline: current index scorer + bridges
    b_nd, b_rc, b_ms = bench("baseline (_score)",
                             lambda q: bridge_search(idx, br, idx._score, q, None))
    print(f"  {'baseline + bridges':<26} {b_nd:>7.4f} {b_rc:>7.4f} {b_ms:>7.1f} "
          f"{'1.0x':>8}")
    for label, cap in [("fast, cap N/2", N // 2), ("fast, cap N/4", N // 4),
                       ("fast, no cap", None)]:
        nd, rc, ms = bench(label, lambda q, c=cap: bridge_search(idx, br, fast, q, c))
        print(f"  {label:<26} {nd:>7.4f} {rc:>7.4f} {ms:>7.1f} "
              f"{b_ms/ms:>7.1f}x")
    return b_ms, N


def main():
    print("DEEP SEARCH step 2 - lossless speed wins, end-to-end with bridges")
    run("scifact", 1)
    run("nfcorpus", 2)
    print("\n  cap N/2 = drop trigrams in >half the docs (near-lossless).")
    print("  'fast, no cap' isolates the micro-opts (precompute norm + alive-skip)")
    print("  from the df-cap. Accuracy should match baseline within noise.")


if __name__ == "__main__":
    main()
