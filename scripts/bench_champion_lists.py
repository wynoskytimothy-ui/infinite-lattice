#!/usr/bin/env python3
"""
Deep search step 3 - champion lists: an O(1)-appendable speed structure native
to the lattice.

Three lattice properties combine into a new speed lever:
  - STABLE prime addresses: a term's posting-list identity never changes.
  - APPEND-ONLY: lists only grow at the tail.
  - ADDITIVE scoring: a doc's score is an independent sum over its term-primes.

=> For each term-prime keep a CHAMPION LIST: its top-M docs by length-normalised
impact tf/(tf+A+B*dl). On append you only push the new doc into the M-heaps of
its own terms (O(1) amortised) - no global reindex. At query time, the long
char-trigram / prefix lists are read only to depth M, while the short, high-idf
WORD gear stays FULL (exact discriminative signal). Trigrams are confirmatory
(a real match usually also hits the word), so champion-pruning them should be
near-lossless.

Honest test: measure nDCG / Recall / ms AND the top-10 exact-overlap vs the full
scorer, sweeping M, on held-out scifact + nfcorpus. Champion lists are an
approximation; they only "keep accuracy" if the top-10 barely moves.
"""

from __future__ import annotations

import heapq
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def build_champions(idx, M, views=("3", "p")):
    N = max(1, len(idx.alive))
    avgdl = idx._total_len / N
    k1, b = idx.k1, idx.b
    A, Bc = k1 * (1 - b), k1 * b / avgdl
    dl = idx.doc_len
    champ = {}
    n_entries = 0
    for (view, _tok), p in idx.token_prime.items():
        if view not in views:
            continue
        pl = idx.postings[p]
        if len(pl) <= M:
            items = list(pl.items())
        else:
            items = heapq.nlargest(M, pl.items(),
                                   key=lambda dt: dt[1] / (dt[1] + A + Bc * dl[dt[0]]))
        champ[p] = items
        n_entries += len(items)
    return champ, (A, Bc, N), n_entries


def score(idx, query, params, champ=None, champ_views=("3", "p"), tri_cap=None):
    A, Bc, N = params
    k1 = idx.k1
    k1p1 = k1 + 1
    dl = idx.doc_len
    tp, post, df = idx.token_prime, idx.postings, idx.df
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
        c = qwt * idf * k1p1
        if champ is not None and tok[0] in champ_views:
            items = champ.get(p, ())
        else:
            items = post[p].items()
        for doc, tf in items:
            scores[doc] += c * tf / (tf + A + Bc * dl[doc])
    return scores


def topk(scores, k=10):
    return heapq.nlargest(k, scores, key=scores.get)


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
    params = (idx.k1 * (1 - idx.b), idx.k1 * idx.b / (idx._total_len / N), N)
    total_post = sum(len(p) for p in idx.postings.values())
    print(f"\n{'='*64}\n{name}: {N:,} docs, {len(test_ids)} q, {total_post:,} postings")

    def evalfn(champ, champ_views, M=None):
        nd = rc = 0.0
        lat = []
        overlap = 0
        for qid in test_ids:
            t = time.perf_counter()
            sc = score(idx, queries[qid], params, champ, champ_views, tri_cap)
            ranked = topk(sc, 10)
            lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
            if champ is not None:
                full = set(topk(score(idx, queries[qid], params, None,
                                      tri_cap=tri_cap), 10))
                overlap += len(set(ranked) & full)
        n = len(test_ids)
        return nd / n, rc / n, sum(lat) / n, (overlap / (10 * n) if champ else 1.0)

    nd0, rc0, ms0, _ = evalfn(None, ())
    print(f"  {'config':<22} {'nDCG':>7} {'Recall':>7} {'ms/q':>7} "
          f"{'speedup':>8} {'top10':>7} {'extra MB':>8}")
    print(f"  {'full (current)':<22} {nd0:>7.4f} {rc0:>7.4f} {ms0:>7.2f} "
          f"{'1.0x':>8} {'100%':>7} {'-':>8}")
    for views, vlabel in [(("3",), "tri"), (("3", "p"), "tri+prefix")]:
        for M in (50, 100, 200):
            champ, _, nent = build_champions(idx, M, views)
            nd, rc, ms, ov = evalfn(champ, views, M)
            print(f"  champ M={M:<4}{vlabel:<11} {nd:>7.4f} {rc:>7.4f} {ms:>7.2f} "
                  f"{ms0/ms:>7.1f}x {ov*100:>6.1f}% {nent*5/1e6:>7.1f}")
    return nd0, ms0


def main():
    print("DEEP SEARCH step 3 - champion lists (O(1)-appendable, lattice-native)")
    print("WORD gear stays full (exact); long trigram/prefix lists read to depth M")
    for ds in ("scifact", "nfcorpus"):
        run(ds)
    print("\n  'top10' = mean overlap of champion top-10 with the full top-10.")
    print("  Near-100% overlap + speedup = lossless. Champions also bound query")
    print("  work to M/term and append in O(1) - they can REPLACE the trigram")
    print("  postings (store only the top-M), a footprint win too.")


if __name__ == "__main__":
    main()
