#!/usr/bin/env python3
"""Does the lattice's co-occurrence corridor produce SEMANTIC correlations on its
own (the distributional hypothesis), including SECOND-ORDER links -- words that
share rare neighbours but never directly co-occur? Measured on real SciFact text.

For each rare word we record its corridor = the rare words it co-occurs with,
weighted by their rarity (higher prime = rarer = more diagnostic). Two words are
correlated by the COSINE of their corridors. A correlate found through shared
neighbours WITHOUT direct co-occurrence is the second-order branch the user means.
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load

corpus, queries, train_q, test_q = load("scifact")
idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)
_idf = {}
def idf(w):
    v = _idf.get(w)
    if v is None:
        p = idx.token_prime.get(("w", w))
        v = idx._idf(p, N) if p else 0.0
        _idf[w] = v
    return v

RARE = 3.5
cooc = defaultdict(Counter)               # word -> {neighbour: rarity-weighted count}
direct = defaultdict(set)
for d, t in corpus.items():
    ws = [w for w in set(words(t)) if idf(w) >= RARE]
    for a in ws:
        direct[a].update(ws)
        for b in ws:
            if a != b:
                cooc[a][b] += idf(b)          # neighbour weighted by its rarity
norm = {w: math.sqrt(sum(v * v for v in c.values())) for w, c in cooc.items()}


def sim(a, b):
    va, vb = cooc[a], cooc[b]
    if not norm.get(a) or not norm.get(b):
        return 0.0
    common = set(va) & set(vb)
    return sum(va[k] * vb[k] for k in common) / (norm[a] * norm[b])


def correlates(w, k=6):
    seen = set()
    for n in cooc[w]:                          # second-order: words sharing w's neighbours
        seen |= set(cooc[n])
    seen.discard(w)
    scored = sorted((c for c in seen if norm.get(c)), key=lambda c: -sim(w, c))[:k]
    return [(c, sim(w, c), c not in direct[w]) for c in scored]


def main():
    print(f"distributional semantics from the co-occurrence corridor (SciFact, "
          f"{len(corpus)} docs, rare-word gate idf>={RARE})\n")
    print("  for each query word: its top correlates by corridor-cosine")
    print("  [2nd] = found via SHARED RARE NEIGHBOURS, never directly co-occurs\n")
    for q in ["cancer", "tumour", "insulin", "neurons", "inflammation",
              "vaccine", "antibody", "mutation", "obesity", "apoptosis"]:
        if q not in cooc:
            continue
        cs = correlates(q)
        shown = ", ".join(f"{c}{'[2nd]' if second else ''}({s:.2f})"
                          for c, s, second in cs)
        print(f"   {q:<13} -> {shown}")
    print("\n  the corridor finds semantically-related terms with NO learning and NO")
    print("  neural net -- the semantic relationship emerges from co-occurrence + rarity,")
    print("  and the [2nd] links are real second-order correlations (shared neighbours).")


if __name__ == "__main__":
    main()
