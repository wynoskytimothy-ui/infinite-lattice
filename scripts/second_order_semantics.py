#!/usr/bin/env python3
"""
Second-order, rare-anchored term similarity - the proper semantic layer.

The idea: first-order co-occurrence gives ASSOCIATION (cholesterol~friedewald),
which drifts. SECOND-order gives SYNONYMY: two terms mean the same thing if they
keep the same RARE company - they co-occur with the same high-idf anchor terms,
even if they never co-occur with each other (synonyms rarely do: you say "car"
OR "automobile"). The rarest words mean the most, so anchors are idf-weighted and
shared rare anchors dominate (the lattice meet, weighted by rarity).

This script BUILDS it and INSPECTS the neighbours for sample terms, comparing
first-order (direct co-occurrence) vs second-order (shared rare context). The
make-or-break test: do the second-order neighbours read like SYNONYMS?
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_supervised_bridges import load
from aethos_append_index import words


def build(name, gate=2.0):
    corpus, queries, _, _ = load(name)
    df = Counter()
    docterms = []
    for d, txt in corpus.items():
        ts = set(words(txt))
        docterms.append(ts)
        for w in ts:
            df[w] += 1
    N = len(docterms)

    def idf(w):
        return math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))

    # co[t][a] = # docs where term t co-occurs with RARE anchor a (idf>=gate)
    co = defaultdict(Counter)
    for ts in docterms:
        rare = [w for w in ts if idf(w) >= gate]
        for t in ts:
            ct = co[t]
            for a in rare:
                if a != t:
                    ct[a] += 1
    # idf-weighted context vectors + norms + inverted index (anchor -> terms)
    inv = defaultdict(list)
    norm = {}
    for t, ctx in co.items():
        ss = 0.0
        for a, c in ctx.items():
            w = c * idf(a)
            ss += w * w
            inv[a].append(t)
        norm[t] = math.sqrt(ss) or 1.0
    return co, inv, norm, idf, df, N


def neighbors_2nd(term, co, inv, norm, idf, k=8):
    """second-order: terms whose rare-context vector is closest to `term`'s."""
    if term not in co:
        return []
    cand = Counter()
    for a, c in co[term].items():
        wa = c * idf(a)
        wa2 = wa * idf(a)                       # rare anchors weighted ^2 (meet weight)
        for t in inv[a]:
            cand[t] += wa2 * co[t][a]
    out = []
    na = norm[term]
    for t, dot in cand.items():
        if t == term:
            continue
        out.append((dot / (na * norm[t]), t))
    out.sort(reverse=True)
    return out[:k]


def neighbors_1st(term, co, idf, k=8):
    """first-order: the rare terms `term` directly co-occurs with most."""
    if term not in co:
        return []
    return sorted(((c * idf(a), a) for a, c in co[term].items()), reverse=True)[:k]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    co, inv, norm, idf, df, N = build(name)
    print(f"{name}: {N:,} docs, {len(co):,} terms\n")
    samples = ["copeptin", "biomaterials", "myocardial", "cholesterol",
               "tumour", "tumor", "women", "diabetes"]
    for term in samples:
        if term not in co:
            continue
        o1 = neighbors_1st(term, co, idf, 6)
        o2 = neighbors_2nd(term, co, inv, norm, idf, 6)
        print(f"'{term}' (idf {idf(term):.1f}, df {df[term]})")
        print("   1st-order (co-occurs with):  " + ", ".join(f"{a}" for _, a in o1))
        print("   2nd-order (same rare context): " + ", ".join(f"{t}({s:.2f})" for s, t in o2))
        print()


if __name__ == "__main__":
    main()
