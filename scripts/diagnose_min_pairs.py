#!/usr/bin/env python3
"""
Diagnostic - does lowering the generalisation gate (min_pairs) recover the
0-dimensional biomaterials query, and what does it cost on held-out test?

The concept terms appear in exactly ONE train pair, so the min>=2 gate prunes
them. If train query 0's gold doc is the same nano doc that test query 1 needs,
then min_pairs=1 would form a memorised bridge that bridges the vocabulary gap.
We measure: (a) is train 0's gold the test 1 gold? (b) held-out nDCG/recall at
min_pairs in {1,2,3}, and (c) the rank of test 1's gold doc at each setting.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges
from scripts.bench_active_learning import best_search


def main(name="scifact"):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    # (a) train 0 vs test 1 gold docs
    t0_gold = [d for d, s in train_q.get("0", {}).items() if s > 0]
    t1_gold = [d for d, s in test_q.get("1", {}).items() if s > 0]
    print(f"train query 0: '{queries.get('0','')[:60]}'")
    print(f"   gold: {t0_gold}")
    print(f"test  query 1: '{queries.get('1','')[:60]}'")
    print(f"   gold: {t1_gold}")
    print(f"   SAME gold doc? {bool(set(t0_gold) & set(t1_gold))}\n")

    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)

    print(f"{'min_pairs':>9} {'bridges':>8} {'nDCG':>7} {'Recall':>7}   "
          f"{'test1 gold rank':>15}")
    for mp in (3, 2, 1):
        br = RelevanceBridges(idx, N, min_pairs=mp).learn(queries, train_q, corpus)
        _, nbridges = br.stats()
        nd = rc = 0.0
        t1_rank = None
        for qid in test_ids:
            ranked = best_search(idx, br, queries[qid])
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
            if qid == "1":
                # rank of the gold doc in a longer list
                long = best_search(idx, br, queries[qid])  # top-10
                # recompute a top-50 to locate gold if outside 10
                full_lex = idx._score(queries[qid])
                t1_rank = (long.index(t1_gold[0]) if t1_gold and t1_gold[0] in long
                           else None)
        nd /= len(test_ids); rc /= len(test_ids)
        # does the biomaterials bridge now exist?
        bm = br.bridge.get("biomaterials")
        bm_tgt = ", ".join(f"{dt}({w:.2f})" for dt, w in bm[:4]) if bm else "(pruned)"
        print(f"{mp:>9} {nbridges:>8} {nd:>7.4f} {rc:>7.4f}   "
              f"{str(t1_rank):>15}")
        print(f"            biomaterials -> {bm_tgt}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scifact")
