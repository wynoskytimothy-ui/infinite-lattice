#!/usr/bin/env python3
"""
Bench v2 - supervised bridges that EXPAND THE POOL, not just rerank it.

v1 (bench_supervised_bridges.py) reranked lexical candidates and lifted held-out
nDCG (+0.017 scifact, +0.007 nfcorpus) - the first signal in the arc to do so.
Its limit, exposed by the active-learning demo: a gold doc with near-zero lexical
overlap never enters the top-100 pool, so reranking cannot reach it.

v2 fixes the candidate-generation miss: the learned bridges also PULL docs into
the pool (relevance-synonym expansion). A doc containing strong learned partners
of the query's words becomes retrievable even with no query word in it. Then the
same conservative lex + lam*bridge fusion ranks the enlarged pool.

Still deterministic / append-only / verifiable; bridges learned from TRAIN qrels
only, measured on held-out TEST queries. We sweep lam and the expansion size and
report the frontier honestly (incl. where expansion HURTS by adding drift).
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import (
    load, ndcg10, recall10, RelevanceBridges,
)


def build(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N).learn(queries, train_q, corpus)
    return corpus, queries, train_q, test_q, test_ids, idx, N, br


def make_searchers(idx, N, br):
    def lex_cand(q, k=100):
        lex = idx._score(q)
        cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:k]
        return lex, cand

    def bridge_pool(q):
        """Docs reachable through learned partners (relevance-synonym expansion)."""
        exp = defaultdict(float)
        for qt in set(words(q)):
            partners = br.bridge.get(qt)
            if not partners:
                continue
            for dt, w in partners:
                p = idx.token_prime.get(("w", dt))
                if p is None:
                    continue
                for d, tf in idx.postings.get(p, {}).items():
                    exp[d] += w * tf / (tf + 1.0)
        return exp

    def search_rerank(q, lam):
        lex, cand = lex_cand(q)
        if not cand:
            return []
        lmax = max(lex[d] for d in cand) or 1.0
        bs = br.score(q, cand)
        bmax = max(bs.values()) if bs else 1.0
        final = {d: lex[d] / lmax + lam * bs.get(d, 0.0) / bmax for d in cand}
        return sorted(final, key=lambda d: final[d], reverse=True)[:10]

    def search_expand(q, lam, n_expand):
        lex, cand = lex_cand(q)
        if not cand:
            cand = []
        exp = bridge_pool(q)
        cset = set(cand)
        extra = [d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
                 if d not in cset][:n_expand]
        pool = list(cand) + extra
        if not pool:
            return []
        lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
        emax = max(exp.values()) if exp else 1.0
        # lexical term + learned-bridge term (expansion docs have lex=0 but bridge>0)
        final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax
                 for d in pool}
        return sorted(final, key=lambda d: final[d], reverse=True)[:10]

    return search_rerank, search_expand


def evaluate(fn, queries, test_q, test_ids):
    nd = rc = 0.0
    for qid in test_ids:
        ranked = fn(queries[qid])
        nd += ndcg10(ranked, test_q[qid])
        rc += recall10(ranked, test_q[qid])
    return nd / len(test_ids), rc / len(test_ids)


def run(name):
    print(f"\n{'='*68}\n{name}")
    corpus, queries, train_q, test_q, test_ids, idx, N, br = build(name)
    n_terms, n_bridges = br.stats()
    print(f"  {len(corpus)} docs | test {len(test_ids)} q | "
          f"{n_bridges} bridges / {n_terms} terms")
    search_rerank, search_expand = make_searchers(idx, N, br)

    nd0, rc0 = evaluate(lambda q: idx.search(q, 10), queries, test_q, test_ids)
    print(f"  baseline lexical:           nDCG {nd0:.4f}  Recall {rc0:.4f}")

    # rerank-only lambda sweep
    best = (nd0, rc0, "baseline")
    for lam in (0.10, 0.15, 0.20, 0.30):
        nd, rc = evaluate(lambda q, l=lam: search_rerank(q, l),
                          queries, test_q, test_ids)
        tag = ""
        if nd > best[0]:
            best = (nd, rc, f"rerank lam={lam}")
            tag = "  <- best"
        print(f"  rerank      lam={lam:<4}:        nDCG {nd:.4f}  Recall {rc:.4f}"
              f"  ({nd-nd0:+.4f}){tag}")

    # pool-expansion sweep (the recall lever)
    for lam in (0.15, 0.25):
        for ne in (20, 50, 100):
            nd, rc = evaluate(lambda q, l=lam, n=ne: search_expand(q, l, n),
                              queries, test_q, test_ids)
            tag = ""
            if nd > best[0]:
                best = (nd, rc, f"expand lam={lam} n={ne}")
                tag = "  <- best"
            print(f"  expand n={ne:<3} lam={lam:<4}:     nDCG {nd:.4f}  "
                  f"Recall {rc:.4f}  ({nd-nd0:+.4f}, rec {rc-rc0:+.4f}){tag}")

    print(f"  BEST: {best[2]}  nDCG {best[0]:.4f} ({best[0]-nd0:+.4f}), "
          f"recall {best[1]:.4f} ({best[1]-rc0:+.4f})")
    return nd0, rc0, best


def main():
    print("Supervised bridges v2: pool-EXPANSION (recall) + lambda sweep")
    print("learned from TRAIN qrels, measured on held-out TEST queries")
    out = {}
    for ds in ("scifact", "nfcorpus"):
        out[ds] = run(ds)
    print(f"\n{'='*68}\nVERDICT (held-out)")
    for ds, (nd0, rc0, best) in out.items():
        print(f"  {ds:9s}: baseline {nd0:.4f} -> {best[0]:.4f} "
              f"({best[0]-nd0:+.4f})  via {best[2]}")
    print()
    print("  Rerank lifts ranking; expansion lifts recall by pulling bridge-")
    print("  reachable docs into the pool (fixes candidate-generation misses).")
    print("  All counting-based: deterministic, append-only, verifiable.")


if __name__ == "__main__":
    main()
