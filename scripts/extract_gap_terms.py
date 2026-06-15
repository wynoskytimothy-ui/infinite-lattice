#!/usr/bin/env python3
"""
Extract the DISCRIMINATIVE-GAP terms: rare query words whose gold doc does NOT
contain them (so no lexical/distributional method can bridge them). These are
the terms a KNOWLEDGE layer (real definitions) would need to explain.

Outputs (query, key term, gold-doc id) so we can source real definitions and
test whether telling the engine what the rare words mean recovers the golds.
We look only at query words + whether the gold contains them - NOT the gold's
content (so the definitions stay externally sourced, not reverse-engineered).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges
from scripts.bench_supervised_bridges import load
from scripts.diagnose_corpus import full_ranking


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

    gaps = []
    term_freq = Counter()
    for qid in test_ids:
        ranked, scored = full_ranking(idx, br, queries[qid])
        golds = {d for d, s in test_q[qid].items() if s > 0}
        if golds & set(ranked[:10]):
            continue                                   # already a hit
        bestdoc = min((d for d in golds if d in scored),
                      key=lambda d: ranked.index(d), default=None)
        if bestdoc is None:
            continue
        qwords = set(words(queries[qid]))
        gwords = set(words(corpus[bestdoc]))           # only to test membership
        top_w = max(qwords, key=idf, default="")
        if top_w and top_w not in gwords and idf(top_w) >= 3.0:   # rare key term absent
            gaps.append((qid, top_w, round(idf(top_w), 1), bestdoc))
            term_freq[top_w] += 1

    print(f"{name}: {len(gaps)} discriminative-gap misses (rare key term absent from gold)\n")
    print("  the rare terms a knowledge layer would need to define:")
    for qid, tw, tidf, gd in gaps:
        print(f"     q{qid:<5} key '{tw}' (idf {tidf})  gold {gd}  '{queries[qid][:46]}'")
    print(f"\n  distinct gap terms: {sorted(set(t for _, t, _, _ in gaps))}")


if __name__ == "__main__":
    main()
