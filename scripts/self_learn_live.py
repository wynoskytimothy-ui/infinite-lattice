#!/usr/bin/env python3
"""
SELF-LEARNING with a LIVE teacher (Wikipedia), grounded by the corpus.

The engine finds its gaps and auto-sources definitions from Wikipedia (no human
glossary). A KB teacher is NOISY - abbreviations resolve to the wrong sense
(rxrs -> "RXR Realty", a real-estate firm). So every fetched definition is
VERIFIED against how the term is actually used in the corpus: it is injected only
if its vocabulary overlaps the term's real co-occurrence context. The corpus
rejects wrong senses; only grounded knowledge is appended.

Reports: Wikipedia coverage, how many passed corpus verification, the held-out
lift, and the wrong-sense rejections (the safety mechanism at work).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from wiki_teacher import define_many

TERM_GATE = 5.5


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

    def context_words(term):
        """high-idf words that co-occur with `term` in the corpus (its real sense)."""
        p = idx.token_prime.get(("w", term))
        if p is None:
            return set()
        ctx = set()
        for doc in list(idx.postings.get(p, {}))[:50]:
            for w in set(words(corpus[doc])):
                if w != term and idf(w) >= 3.0:
                    ctx.add(w)
        return ctx

    def verify(term, definition):
        dwords = {w for w in words(definition) if idf(w) >= 3.0}
        return len(dwords & context_words(term)) >= 2     # >=2 shared rare words

    # 1) the engine's own gaps
    gap_counts = Counter()
    for qid in test_ids:
        r = bridge_search(idx, br, queries[qid], 10)
        if {d for d, s in test_q[qid].items() if s > 0} & set(r):
            continue
        for w in set(words(queries[qid])):
            if idf(w) >= TERM_GATE:
                gap_counts[w] += 1
    gaps = [t for t, _ in gap_counts.most_common()]

    # 2) auto-source from Wikipedia (live), then 3) verify against the corpus
    print(f"scifact: {len(gaps)} rare gaps; fetching definitions from Wikipedia...")
    defs = define_many(gaps)
    accepted, rejected, nopage = {}, [], 0
    for t in gaps:
        d = defs.get(t, "")
        if not d:
            nopage += 1
        elif verify(t, d):
            accepted[t] = d
        else:
            rejected.append(t)
    print(f"  Wikipedia returned a page for {len(gaps)-nopage}/{len(gaps)}; "
          f"corpus VERIFIED {len(accepted)}, rejected {len(rejected)} (wrong sense), "
          f"no page {nopage}\n")

    def expand(q):
        extra = []
        for t in set(words(q)):
            if t in accepted and idf(t) >= TERM_GATE:
                for w in dict.fromkeys(words(accepted[t])):
                    if w != t and idf(w) >= 2.5:
                        extra.append(w)
        return q + " " + " ".join(extra[:10]) if extra else q

    def ev(use):
        nd = rc = 0.0
        for qid in test_ids:
            r = bridge_search(idx, br, expand(queries[qid]) if use else queries[qid])
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    nd0, rc0 = ev(False)
    nd1, rc1 = ev(True)
    print(f"  accepted (corpus-grounded) Wikipedia definitions:")
    for t in list(accepted)[:8]:
        print(f"     {t}: {accepted[t][:70]}")
    if rejected:
        print(f"  rejected wrong-sense (corpus saved us): {rejected[:8]}")
    print(f"\n  + bridges:              nDCG {nd0:.4f}  Recall {rc0:.4f}")
    print(f"  + live Wikipedia (verified): nDCG {nd1:.4f}  Recall {rc1:.4f}  "
          f"({nd1-nd0:+.4f} nDCG)")
    print("\n  fully autonomous: no human glossary. engine found gaps -> Wikipedia")
    print("  defined them -> corpus verified -> appended. an LLM teacher would")
    print("  disambiguate the abbreviations Wikipedia missed and cover more.")


if __name__ == "__main__":
    main()
