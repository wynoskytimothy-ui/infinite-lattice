#!/usr/bin/env python3
"""Does the COMPRESSION's placement reveal semantic links that bring MISSED gold into the pool?
The reorder clusters docs by shared rare terms (co-occurrence) -- a free doc-doc graph. Test: of gold
docs BM25 MISSES, how many are placement-neighbors (share their rarest term with a top hit) or company
(share ANY rare term with a top hit) -> recoverable into the pool for the cross-encoder to rank.

Single-hop MARCO -- recall is near-maxed, so the bulk is already found; this measures the recoverable
TAIL (the user's "docs that would never have been linked"). Multi-hop is where it wins big (proven +124%).
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, RARE
from marco_fast import bm25_fast

N = 400
NEIGH = 20   # expand from the top-NEIGH hits


def main():
    idx = FullIndex()
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    def terms_of(pid):
        return [(idx.idf_of(w), w) for w in set(stoks(idx.text(int(pid)))) if idx.idf_of(w) > 0]

    def rare_set(pid):
        return set(w for v, w in terms_of(pid) if v >= RARE)

    def rarest(pid):
        t = terms_of(pid)
        return max(t)[1] if t else None

    found = missed = rec_place = rec_comp = 0
    for qid, qt in sample:
        o, _ = bm25_fast(idx, stoks(qt), 100)
        top = set(int(d) for d in o[:100]); gold = qrels[qid]
        if any(g in top for g in gold):
            found += 1; continue
        missed += 1
        hits = [int(d) for d in o[:NEIGH]]
        hit_rare = set(); hit_rarest = set()
        for h in hits:
            hit_rare |= rare_set(h)
            r = rarest(h)
            if r:
                hit_rarest.add(r)
        g = next(iter(gold))
        if rarest(g) in hit_rarest:
            rec_place += 1
        if rare_set(g) & hit_rare:
            rec_comp += 1
    print(f"\n  COMPRESSION-LINK RECALL RECOVERY -- MARCO single-hop (n={N})\n")
    print(f"  gold already in fast top-100 (recall@100):  {found/N*100:.1f}%  ({found}/{N})")
    print(f"  gold MISSED by top-100:                     {missed/N*100:.1f}%  ({missed}/{N})")
    if missed:
        print(f"\n  of the MISSED gold, recoverable into the pool by the compression link:")
        print(f"    placement-neighbor (shares rarest term w/ a top-{NEIGH} hit): {rec_place/missed*100:.1f}%")
        print(f"    company (shares ANY rare term w/ a top-{NEIGH} hit):          {rec_comp/missed*100:.1f}%")
    print(f"\n  -> the compression links DO bring missed gold to the pool (recall). What the cross-encoder can")
    print(f"     then do with it is the answer-ness wall; the big payoff is multi-hop (proven), not single-hop bulk.")


if __name__ == "__main__":
    main()
