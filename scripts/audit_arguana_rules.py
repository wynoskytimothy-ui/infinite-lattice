#!/usr/bin/env python3
"""Arguana goblin DEMOTION rules -- net effect test.

The goblin: rank-1 wrong doc is a near-restatement of the QUERY (high query-jaccard),
while the gold counter-argument has low query overlap. We test cheap O(pool) re-rankers
that penalize query-echo / low-diversity / over-length, and measure MRR@10 + how many
queries move gold to #1, across ALL test queries (the honest net metric, incl. collateral).
"""
import os, sys, math, re
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unified as U
from scripts.bench_supervised_bridges import load

WORD = U.WORD


def words_raw(s):
    return WORD.findall(s.lower())


def main():
    name = "arguana"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    mv = True
    eng = U.build_csr(corpus, mv)
    id2row = eng["id2row"]; cids = eng["cids"]

    # Precompute cheap per-doc features (O(pool) at query time: distinct-ratio, len, bag)
    docbag = {}
    docratio = np.zeros(eng["M"], np.float32)
    doclen_w = np.zeros(eng["M"], np.float32)
    for c, r in id2row.items():
        w = words_raw(corpus[c]); n = len(w); d = len(set(w))
        docratio[r] = (d / n) if n else 0.0
        doclen_w[r] = n
        docbag[r] = set(U.toks(corpus[c]))

    def pool_and_lex(q):
        lex = U.bm25(eng, U.doc_bag(q, mv))
        pool = np.argsort(-lex)[:100]
        return lex, pool

    def rank_of_gold(order, gold):
        for i, r in enumerate(order):
            if cids[r] in gold:
                return i
        return None

    def eval_rerank(scorer):
        """scorer(q, lex, pool) -> dict row->multiplier (>=, applied to lex). Returns MRR@10, #gold@1, mean gold-rank."""
        mrr = 0.0; at1 = 0; granks = []
        for q in test_ids:
            gold = {d for d, s in test_q[q].items() if s > 0 and d in id2row}
            if not gold:
                continue
            lex, pool = pool_and_lex(q)
            mult = scorer(q, lex, pool)
            sc = lex[pool] * mult
            order = pool[np.argsort(-sc)]
            gr = rank_of_gold(order, gold)
            if gr is not None:
                granks.append(gr)
                if gr < 10:
                    mrr += 1.0 / (gr + 1)
                if gr == 0:
                    at1 += 1
        n = len(test_ids)
        return mrr / n, at1, (np.mean(granks) if granks else float("nan"))

    # baseline: no rerank
    def base(q, lex, pool):
        return np.ones(len(pool), np.float32)

    # RULE A: query-echo demotion. penalty = (1 - alpha * jaccard(doc, query)).
    # jaccard computed ONLY over the pool (O(pool)), set-intersect with the query bag.
    def make_echo(alpha):
        def f(q, lex, pool):
            qb = set(U.toks(q))
            qn = len(qb)
            mult = np.ones(len(pool), np.float32)
            for i, r in enumerate(pool):
                db = docbag[r]
                inter = len(db & qb)
                jac = inter / (len(db) + qn - inter) if (len(db) + qn - inter) else 0.0
                mult[i] = 1.0 - alpha * jac
            return np.clip(mult, 0.05, 1.0)
        return f

    # RULE A': containment form -- fraction of the DOC's words present in the query.
    # near-restatements have almost every doc word in the query -> contain ~1.
    def make_contain(alpha):
        def f(q, lex, pool):
            qb = set(U.toks(q))
            mult = np.ones(len(pool), np.float32)
            for i, r in enumerate(pool):
                db = docbag[r]
                contain = (len(db & qb) / len(db)) if db else 0.0
                mult[i] = 1.0 - alpha * contain
            return np.clip(mult, 0.05, 1.0)
        return f

    # RULE B: distinct-ratio (anti-repetition), absolute cheap feature, no query needed.
    def make_ratio(alpha):
        def f(q, lex, pool):
            return (docratio[pool] ** alpha).astype(np.float32)
        return f

    print(f"arguana net re-rank test over {len(test_ids)} queries (MRR@10, gold@1, mean gold-rank in pool):\n")
    b = eval_rerank(base)
    print(f"  baseline (multi-view, no rule)         MRR {b[0]:.4f} | gold@1 {b[1]:4d} | mean-gold-rank {b[2]:.2f}")
    for a in (0.5, 1.0, 1.5, 2.0):
        r = eval_rerank(make_echo(a))
        print(f"  echo-demote  alpha={a:<4}                  MRR {r[0]:.4f} | gold@1 {r[1]:4d} | mean-gold-rank {r[2]:.2f}")
    for a in (1.0, 2.0, 3.0, 4.0):
        r = eval_rerank(make_contain(a))
        print(f"  contain-demote alpha={a:<4}                MRR {r[0]:.4f} | gold@1 {r[1]:4d} | mean-gold-rank {r[2]:.2f}")
    for a in (0.5, 1.0, 2.0):
        r = eval_rerank(make_ratio(a))
        print(f"  distinct-ratio^{a:<4}                     MRR {r[0]:.4f} | gold@1 {r[1]:4d} | mean-gold-rank {r[2]:.2f}")

    # combined best: containment demote * ratio
    def combo(q, lex, pool):
        qb = set(U.toks(q))
        mult = np.ones(len(pool), np.float32)
        for i, r in enumerate(pool):
            db = docbag[r]
            contain = (len(db & qb) / len(db)) if db else 0.0
            mult[i] = (1.0 - 0.9 * contain)
        return np.clip(mult, 0.05, 1.0) * (docratio[pool] ** 1.0)
    r = eval_rerank(combo)
    print(f"\n  COMBO contain(a=.9)*ratio^1            MRR {r[0]:.4f} | gold@1 {r[1]:4d} | mean-gold-rank {r[2]:.2f}")


if __name__ == "__main__":
    main()
