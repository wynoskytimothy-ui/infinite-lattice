#!/usr/bin/env python3
"""
3-way meets as CONCEPT REGIONS (general meanings), not direct links.

The reframe: a 3-way intersection of rare anchors {a,b,c} defines a concept
REGION. A term is embedded by WHICH recurring rare meets it falls into; two
terms are related if they occupy the same regions - they share a general
meaning, even with no direct A~B edge. This is "the empty space between vectors"
made into coordinates: the recurring rare meets are the axes.

Compares 3-way concept-region neighbours to the 2-way direct-link neighbours,
glass-box, to SEE whether concept regions give cleaner / more general relations.
Keep iterating the rule until the relations are right.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_supervised_bridges import load
from aethos_append_index import words


def build(name, top_anchors=6, min_df=4, min_region_docs=3, max_inv=500):
    corpus, _, _, _ = load(name)
    df = Counter()
    docterms = []
    for txt in corpus.values():
        ts = set(words(txt))
        docterms.append(ts)
        for w in ts:
            df[w] += 1
    N = len(docterms)

    def idf(w):
        return math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))

    # pass 1: count recurring 3-way meets (rare anchor PAIRS = a region; the term
    # is the 3rd leg) -> keep only regions that recur (denoise)
    region_docs = Counter()
    doc_regions = []
    for ts in docterms:
        rare = heapq.nlargest(top_anchors, ts, key=idf)
        regs = [(a, b) for a, b in combinations(sorted(rare), 2)]
        doc_regions.append((ts, regs))
        for r in regs:
            region_docs[r] += 1
    good = {r for r, n in region_docs.items() if n >= min_region_docs}

    # pass 2: embed each term over the GOOD concept regions it falls into
    vec = defaultdict(Counter)            # term -> {region: count}
    for ts, regs in doc_regions:
        gr = [r for r in regs if r in good]
        for t in ts:
            if df[t] < min_df:
                continue
            vt = vec[t]
            for (a, b) in gr:
                if t != a and t != b:
                    vt[(a, b)] += 1
    inv = defaultdict(list)
    norm = {}
    for t, v in vec.items():
        ss = 0.0
        for r, c in v.items():
            a, b = r
            w = c * idf(a) * idf(b)       # region weight = rarity of its anchors
            ss += w * w
            inv[r].append(t)
        norm[t] = math.sqrt(ss) or 1.0

    def neighbours(term, k=6):
        if term not in vec or df[term] < min_df:
            return []
        cand = Counter()
        vt = vec[term]
        for r, c in vt.items():
            lst = inv[r]
            if len(lst) > max_inv:
                continue
            a, b = r
            wr = c * idf(a) * idf(b)
            wr2 = wr * idf(a) * idf(b)
            for t in lst:
                cand[t] += wr2 * vec[t][r]
        na = norm[term]
        scored = [(dot / (na * norm[t]), t) for t, dot in cand.items() if t != term]
        scored.sort(reverse=True)
        return [(t, s) for s, t in scored[:k] if s >= 0.4]

    return neighbours, idf, df, len(good)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    nbr, idf, df, n_regions = build(name)
    print(f"{name}: {n_regions:,} recurring concept regions (rare 3-way meets)\n")
    for term in ["copeptin", "biomaterials", "myocardial", "cholesterol",
                 "curcumin", "diabetes", "women", "tumor"]:
        o = nbr(term)
        if o:
            print(f"'{term}' (idf {idf(term):.1f}) ~ " +
                  ", ".join(f"{t}({s:.2f})" for t, s in o))


if __name__ == "__main__":
    main()
