#!/usr/bin/env python3
"""
Per-corpus lever tuning - find the config that maximizes nDCG, glass-box.

    python scripts/tune_corpus.py <dataset>

Builds the index ONCE, then sweeps query-time levers (which gears contribute,
and the trigram df-cap) without rebuilding - the doc postings hold every view,
so skipping a view's query tokens = that gear off. Reports nDCG@10 per config so
we can SEE which lever helps which corpus (e.g. do char-trigrams help or add
noise on clean-lexical text?). Tuning that needs a rebuild (positional, gear
weights) is flagged separately.
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from scripts.profile_query_speed import score_custom


def evaluate(idx, queries, test_q, test_ids, views, tri_cap):
    nd = rc = 0.0
    for qid in test_ids:
        s = score_custom(idx, queries[qid], views=views, tri_df_cap=tri_cap)
        ranked = heapq.nlargest(10, s, key=s.get)
        nd += ndcg10(ranked, test_q[qid])
        rc += recall10(ranked, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    print(f"\n{'='*60}\n{name}: {N:,} docs | {len(test_ids)} test q")
    print(f"  {'config':<28} {'nDCG@10':>8} {'Recall@10':>10}")

    half, quarter = int(N * 0.5), int(N * 0.25)
    configs = [
        ("word only", ("w",), None),
        ("word + prefix", ("w", "p"), None),
        ("word + tri (cap N/2)", ("w", "3"), half),
        ("all (cap N/2) [default]", ("w", "3", "p"), half),
        ("all (cap N/4)", ("w", "3", "p"), quarter),
        ("all (no cap)", ("w", "3", "p"), None),
    ]
    base = None
    for label, views, cap in configs:
        nd, rc = evaluate(idx, queries, test_q, test_ids, views, cap)
        if base is None:
            base = nd
        tag = "  <- default" if "default" in label else (f"  ({nd-base:+.4f})" if nd != base else "")
        print(f"  {label:<28} {nd:>8.4f} {rc:>10.4f}{tag}")

    # ---- BM25 k1/b sweep (query-time, all views, no cap) ----
    bs = (0.3, 0.5, 0.75, 0.9)
    print(f"\n  BM25 k1/b sweep (all views, no cap):   " + "  ".join(f"b={b}" for b in bs))
    best = (0.0, None)
    for k1 in (0.6, 1.0, 1.2, 1.6, 2.0):
        row = []
        for b in bs:
            idx.k1, idx.b = k1, b
            nd, _ = evaluate(idx, queries, test_q, test_ids, ("w", "3", "p"), None)
            row.append(nd)
            if nd > best[0]:
                best = (nd, (k1, b))
        print(f"    k1={k1:<4}                           " + "  ".join(f"{nd:.4f}" for nd in row))
    idx.k1, idx.b = 1.2, 0.75
    print(f"  => best k1={best[1][0]} b={best[1][1]}: nDCG {best[0]:.4f}  "
          f"(default k1=1.2 b=0.75)")


def main():
    run(sys.argv[1] if len(sys.argv) > 1 else "scifact")


if __name__ == "__main__":
    main()
