#!/usr/bin/env python3
"""Compound tokens via prime composites (FTA): P_a x P_b is a UNIQUE address you can
factor back -- so a word-pair is a free token (no new storage, computed on demand).
And the compound has its OWN corridor, so one polysemous word means different things
with different partners (sense disambiguation). Measured on SciFact.
"""
import sys
from collections import Counter
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

def prime(w):
    return idx.token_prime.get(("w", w))

def docs(w):
    p = prime(w)
    return set(idx.postings.get(p, {})) if p else set()

def corridor(docset, exclude):
    c = Counter()
    for d in docset:
        for w in set(words(corpus[d])):
            if idf(w) >= 3.0 and w not in exclude:
                c[w] += idf(w)
    return [w for w, _ in c.most_common(7)]


def main():
    w = "resistance"
    print(f"compound tokens via prime composites (SciFact) -- the word '{w}' is "
          f"polysemous\n")
    print(f"   '{w}' alone (prime {prime(w)}), all senses mixed:")
    print(f"      corridor: {', '.join(corridor(docs(w), {w}))}\n")
    print(f"   COMPOUND = P('{w}') x P(partner)  [unique by FTA, factors back, 0 new storage]:")
    for partner in ["insulin", "antibiotic", "antimicrobial", "chemotherapy", "drug"]:
        if prime(partner) is None:
            continue
        comp = docs(w) & docs(partner)
        if len(comp) < 4:
            continue
        addr = prime(w) * prime(partner)
        back = (addr // prime(w)) == prime(partner)          # factor back
        print(f"      {partner+' x '+w:<24} prime {prime(partner)}x{prime(w)}={addr} "
              f"(factors back: {back})")
        print(f"         own corridor: {', '.join(corridor(comp, {w, partner}))}")
    print(f"\n  the same word, different partners -> DIFFERENT corridors = different senses.")
    print(f"  the compound address is just the product (computed, not stored): millions of")
    print(f"  free compound/phrase tokens, no extra memory, each with its own corridor.")


if __name__ == "__main__":
    main()
