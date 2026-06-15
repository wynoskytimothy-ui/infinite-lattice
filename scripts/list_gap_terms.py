#!/usr/bin/env python3
"""
List ALL rare terms in queries that currently miss gold@10 - the full set a
scifact knowledge glossary needs to define. Sorted by how many misses they touch.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)

    def idf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 0.0

    term_misses = Counter()
    n_miss = 0
    for qid in test_ids:
        r = bridge_search(idx, br, queries[qid], k=10)
        golds = {d for d, s in test_q[qid].items() if s > 0}
        if golds & set(r):
            continue
        n_miss += 1
        for w in set(words(queries[qid])):
            if idf(w) >= 3.5:                          # rare/technical
                term_misses[w] += 1
    print(f"{name}: {n_miss} queries miss gold@10; rare terms (idf>=3.5) in them:\n")
    for w, c in term_misses.most_common(60):
        print(f"     {c:2d}  {w:18s} idf {idf(w):.1f}")


if __name__ == "__main__":
    main()
