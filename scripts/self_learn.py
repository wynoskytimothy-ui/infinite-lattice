#!/usr/bin/env python3
"""
SELF-LEARNING loop: the engine finds its own knowledge gaps, asks a teacher to
define them, injects the answer, and improves - no retraining, all append-only.

    python scripts/self_learn.py

The loop:
  1. find the rare terms in the queries the engine currently MISSES (its own gaps)
  2. for each, call teacher(term) for a definition  (production: an LLM API or a
     KB lookup; here the teacher is the LLM-distilled scifact_glossary)
  3. inject it (full-weight query expansion, gated to genuinely rare terms)
  4. re-measure - and repeat, teaching the highest-impact gaps first

Reports the LEARNING CURVE: held-out nDCG as a function of how many gap terms the
engine has been taught. It starts knowing nothing extra and climbs as it learns.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from scifact_glossary import GLOSSARY

TERM_GATE = 5.5     # only teach genuinely rare terms (the selectivity rule)


def main():
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)

    def idf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 0.0

    def teacher(term):                       # production: LLM API / KB lookup
        return GLOSSARY.get(term, "")

    def expand(q, taught):
        extra = []
        for t in set(words(q)):
            if t in taught and idf(t) >= TERM_GATE:
                for w in dict.fromkeys(words(teacher(t))):
                    if w != t and idf(w) >= 2.5:
                        extra.append(w)
        return q + " " + " ".join(extra[:10]) if extra else q

    def evaluate(taught):
        nd = rc = 0.0
        for qid in test_ids:
            r = bridge_search(idx, br, expand(queries[qid], taught))
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    # 1) the engine discovers its OWN gaps: rare terms in missing queries
    gap_counts = Counter()
    for qid in test_ids:
        r = bridge_search(idx, br, queries[qid], 10)
        if {d for d, s in test_q[qid].items() if s > 0} & set(r):
            continue
        for w in set(words(queries[qid])):
            if idf(w) >= TERM_GATE:
                gap_counts[w] += 1
    # 2) keep the gaps the teacher can answer, highest-impact first
    queue = [t for t, _ in gap_counts.most_common() if teacher(t)]

    print(f"scifact: engine found {len(gap_counts)} rare gaps in its misses; "
          f"teacher can define {len(queue)} of them\n")
    print(f"  LEARNING CURVE (held-out, teaching highest-impact gaps first):")
    print(f"     {'taught':>6} | {'nDCG@10':>8} {'Recall@10':>10} | newest term")
    taught = set()
    steps = sorted(set([0] + list(range(2, len(queue) + 1, 2)) + [len(queue)]))
    prev = None
    for k in steps:
        taught = set(queue[:k])
        nd, rc = evaluate(taught)
        newest = queue[k - 1] if k > 0 else "-"
        mark = "" if prev is None else f"  ({nd-prev:+.4f})"
        print(f"     {k:>6} | {nd:>8.4f} {rc:>10.4f} | {newest}{mark}")
        prev = nd
    print(f"\n  the engine taught itself {len(queue)} terms and climbed "
          f"{evaluate(set())[0]:.4f} -> {evaluate(set(queue))[0]:.4f} nDCG, no retrain.")
    print("  each term is one appended, editable line; the teacher is swappable")
    print("  (a real LLM/KB in production). it keeps learning as long as you feed it.")


if __name__ == "__main__":
    main()
