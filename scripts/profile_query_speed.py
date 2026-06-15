#!/usr/bin/env python3
"""
Deep search step 1 - profile the lexical query and find speed wins that keep
accuracy.

The hot loop in _score traverses every posting of every query token. Hypothesis:
char-trigrams dominate the cost (one word -> ~12 trigrams, each a common
substring with a long, low-idf posting list) while adding little accuracy.

Measures, on scifact + nfcorpus (held-out test, lexical only - bridges add a
flat ~1.3 ms on top):
  A. posting-length distribution per view (word / trigram / prefix) - where the
     traversal cost lives.
  B. per-view-subset cost vs accuracy: {word}, {word+prefix}, {word+tri},
     {all} - how much each view costs in ms and buys in nDCG/recall.
  C. trigram df-cap sweep: skip query trigrams whose df exceeds a cap (high-df =
     low-idf = near-zero contribution but longest lists). Speed vs accuracy.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def pctl(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else 0


def score_custom(idx, query, views=("w", "3", "p"), tri_df_cap=None,
                 skip_alive=True):
    """_score variant: filter views, cap trigram df, optionally skip alive-check."""
    N = max(1, len(idx.alive))
    avgdl = idx._total_len / N
    qbag = idx._multiview(query)
    scores = defaultdict(float)
    k1, b = idx.k1, idx.b
    for tok, qwt in qbag.items():
        if tok[0] not in views:
            continue
        p = idx.token_prime.get(tok)
        if p is None or idx.df[p] == 0:
            continue
        if tok[0] == "3" and tri_df_cap is not None and idx.df[p] > tri_df_cap:
            continue
        idf = idx._idf(p, N)
        c = qwt * idf * (k1 + 1)
        for doc, tf in idx.postings[p].items():
            denom = tf + k1 * (1 - b + b * idx.doc_len[doc] / avgdl)
            scores[doc] += c * tf / denom
    return scores


def search_custom(idx, query, **kw):
    s = score_custom(idx, query, **kw)
    return sorted(s, key=lambda d: s[d], reverse=True)[:10]


def evaluate(idx, queries, test_q, test_ids, **kw):
    nd = rc = 0.0
    lat = []
    for qid in test_ids:
        t = time.perf_counter()
        ranked = search_custom(idx, queries[qid], **kw)
        lat.append((time.perf_counter() - t) * 1000)
        nd += ndcg10(ranked, test_q[qid])
        rc += recall10(ranked, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n, sum(lat) / n, pctl(lat, 99)


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    print(f"\n{'='*66}\n{name}: {N:,} docs, {len(test_ids)} test q")

    # ---- A. posting-length distribution per view ----
    by_view = {"w": [], "3": [], "p": []}
    for (view, _tok), p in idx.token_prime.items():
        by_view.setdefault(view, []).append(len(idx.postings[p]))
    print("\n  A. posting lists per view")
    print(f"     {'view':<8} {'#primes':>8} {'mean len':>9} {'p99 len':>8} "
          f"{'max':>7} {'total post':>11}")
    for v, name_v in (("w", "word"), ("3", "trigram"), ("p", "prefix")):
        L = by_view.get(v, [])
        if L:
            print(f"     {name_v:<8} {len(L):>8,} {sum(L)/len(L):>9.1f} "
                  f"{pctl(L,99):>8,} {max(L):>7,} {sum(L):>11,}")

    # ---- B. per-view-subset cost vs accuracy ----
    print("\n  B. view subset: cost vs accuracy (lexical, held-out)")
    print(f"     {'views':<16} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} {'p99':>7}")
    for label, views in [("word", ("w",)), ("word+prefix", ("w", "p")),
                         ("word+tri", ("w", "3")), ("all (w+3+p)", ("w", "3", "p"))]:
        nd, rc, ms, p99 = evaluate(idx, queries, test_q, test_ids, views=views)
        print(f"     {label:<16} {nd:>7.4f} {rc:>7.4f} {ms:>7.1f} {p99:>7.1f}")

    # ---- C. trigram df-cap sweep (all views, cap only trigrams) ----
    print("\n  C. trigram df-cap (all views; skip query trigrams with df > cap)")
    print(f"     {'cap':<12} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} {'p99':>7}")
    caps = [None, int(N * 0.5), int(N * 0.25), int(N * 0.1), 2000, 1000, 500, 200]
    base = None
    for cap in caps:
        nd, rc, ms, p99 = evaluate(idx, queries, test_q, test_ids,
                                   views=("w", "3", "p"), tri_df_cap=cap)
        if base is None:
            base = (nd, ms)
        tag = ""
        if cap is not None and nd >= base[0] - 0.001 and ms < base[1]:
            tag = f"  speed x{base[1]/ms:.1f}, acc {nd-base[0]:+.4f}"
        capn = "none" if cap is None else f"{cap:,}"
        print(f"     {capn:<12} {nd:>7.4f} {rc:>7.4f} {ms:>7.1f} {p99:>7.1f}{tag}")


def main():
    print("DEEP SEARCH - profiling lexical query speed (keep accuracy)")
    for ds in ("scifact", "nfcorpus"):
        run(ds)
    print("\n  Read: if trigram lists dominate 'total post' (A) and df-cap (C)")
    print("  holds nDCG while cutting ms, the win is pruning high-df trigrams")
    print("  (low-idf, longest lists) - lossless speed. View subset (B) shows")
    print("  how much accuracy the trigram gear actually buys.")


if __name__ == "__main__":
    main()
