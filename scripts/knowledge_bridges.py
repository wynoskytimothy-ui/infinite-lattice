#!/usr/bin/env python3
"""
KNOWLEDGE injection: tell the engine what the rare/compound terms MEAN, and let
the definitions build bridges that distributional methods cannot.

A discriminative-gap query (gold doc lacks the query's rare key term, df=1, no
co-occurrence signal) is unreachable by counting. But a real DEFINITION of the
term carries the vocabulary the gold uses: define `lats1` as a Hippo-pathway
kinase that phosphorylates YAP downstream of NF2/Merlin, and the NF2 query now
bridges to its Hippo/YAP gold. Each knowledge bridge traces to its definition -
deterministic, append-only, verifiable.

The GLOSSARY below was written from general biomedical knowledge, NOT from the
gold documents (the test is whether real-world knowledge bridges the gap). We
measure: gold-rank recovery on the gap queries, and overall held-out nDCG.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10

from scifact_glossary import GLOSSARY  # comprehensive scifact knowledge base


def inject(br, glossary, idx, N, per_term=10, gate=2.0):
    """Add knowledge bridges from definitions (merged with learned bridges)."""
    added = 0
    for term, definition in glossary.items():
        defw = []
        for w in dict.fromkeys(words(definition)):
            if w == term:
                continue
            p = idx.token_prime.get(("w", w))
            if p is None:
                continue
            i = idx._idf(p, N)
            if i >= gate:
                defw.append((w, i))
        defw.sort(key=lambda x: -x[1])
        kb = defw[:per_term]
        if kb:
            existing = dict(br.bridge.get(term, []))
            for w, wt in kb:
                existing[w] = max(existing.get(w, 0.0), wt)   # merge, keep strongest
            br.bridge[term] = sorted(existing.items(), key=lambda x: -x[1])[:12]
            added += 1
    return added


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)

    def expand(q):
        """rewrite the query WITH the definitions of its glossary terms (full
        lexical weight - the strongest knowledge integration)."""
        extra = []
        for t in set(words(q)):
            if t in GLOSSARY:
                for w in dict.fromkeys(words(GLOSSARY[t])):
                    p = idx.token_prime.get(("w", w))
                    if w != t and p is not None and idx._idf(p, N) >= 2.5:
                        extra.append(w)
        return q + " " + " ".join(extra[:10]) if extra else q

    def ev(use_expand=False):
        nd = rc = 0.0
        for qid in test_ids:
            q = expand(queries[qid]) if use_expand else queries[qid]
            r = bridge_search(idx, br, q)
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    def gold_rank(qid, use_expand=False):
        q = expand(queries[qid]) if use_expand else queries[qid]
        full = bridge_search(idx, br, q, k=1000)
        golds = [d for d, s in test_q[qid].items() if s > 0]
        rs = [full.index(d) + 1 for d in golds if d in full]
        return min(rs) if rs else None

    # queries whose key term is in the glossary (the gap queries we target)
    targets = [qid for qid in test_ids
               if set(words(queries[qid])) & set(GLOSSARY)]
    before_ranks = {q: gold_rank(q, False) for q in targets}
    nd0, rc0 = ev(False)
    after_ranks = {q: gold_rank(q, True) for q in targets}      # query expansion w/ defs
    nd1, rc1 = ev(True)

    print(f"{name}: {len(GLOSSARY)} definitions; {len(targets)} targeted gap queries")
    print("  rewrite-query-with-definition (full lexical weight):\n")
    print("  gold-rank recovery on targeted gap queries:")
    recov = 0
    for q in targets:
        b, a = before_ranks[q], after_ranks[q]
        flag = ""
        if (b is None or b > 10) and a is not None and a <= 10:
            flag = "  <- RECOVERED into top-10"
            recov += 1
        print(f"     q{q:<5} gold rank {str(b):>6} -> {str(a):>6}{flag}")
    print(f"\n  overall held-out:")
    print(f"     bridges only:           nDCG {nd0:.4f}  Recall {rc0:.4f}")
    print(f"     + knowledge expansion:  nDCG {nd1:.4f}  Recall {rc1:.4f}  "
          f"({nd1-nd0:+.4f} nDCG)   {recov} of {len(targets)} recovered to top-10")


if __name__ == "__main__":
    main()
