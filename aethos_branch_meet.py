#!/usr/bin/env python3
"""Branch intersection = contextual disambiguation. Trigger a few rare words; each
lights up its co-occurrence branch (the company it keeps, common->rare). The MEET
of the branches -- words sharing the company of ALL triggers, weighted rarest-first
-- is the precise topic/sense. Adding a rare trigger NARROWS the meet. (SciFact)."""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load

corpus, *_ = load("scifact")
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

RARE = 3.0
branch = defaultdict(Counter)                      # word -> its company (co-occurring rare words)
for d, t in corpus.items():
    ws = [w for w in set(words(t)) if idf(w) >= RARE]
    for a in ws:
        for b in ws:
            if a != b:
                branch[a][b] += 1


def meet(query):
    """words in the company of the MOST triggers, weighted by edge strength x rarity."""
    present = Counter()
    strength = Counter()
    qs = set(query)
    for w in query:
        if w not in branch:
            continue
        for c, n in branch[w].items():
            if c in qs:
                continue
            present[c] += 1
            strength[c] += n * idf(c)              # rarest shared company counts most
    need = len(query) if len(query) > 1 else 1
    cand = [c for c in present if present[c] >= need]
    return sorted(cand, key=lambda c: -strength[c])[:8]


def main():
    print(f"branch meet = contextual narrowing (SciFact, {len(corpus)} docs)\n")
    progressions = [
        ["insulin"],
        ["insulin", "resistance"],
        ["insulin", "resistance", "obesity"],
        ["expression"],
        ["expression", "tumour"],
        ["expression", "tumour", "metastasis"],
    ]
    for q in progressions:
        m = meet(q)
        tag = "broad" if len(q) == 1 else f"{len(q)} triggers"
        print(f"   [{tag:<10}] {' + '.join(q):<38} -> meet: {', '.join(m)}")
        if len(q) == 3:
            print()
    print("  triggering more rare words narrows the meet to the precise topic; the")
    print("  shared rarest company IS the contextual sense -- no learning, deterministic.")
    print("  (this is bag-of-context narrowing -- topic/sense; word ORDER/composition")
    print("   is the one thing it can't see -- that's the Gamma-ODE's job.)")


if __name__ == "__main__":
    main()
