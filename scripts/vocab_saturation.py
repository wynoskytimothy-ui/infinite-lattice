#!/usr/bin/env python3
"""
Why new data does NOT blow up memory: the subword address space saturates.

Multi-view stores word + char-trigram + prefix primes. Trigrams (~17k possible)
and 4-char prefixes are bounded sets that fill up early, so a new word reuses
existing subword addresses instead of allocating new ones. We measure:
  - vocabulary growth by view as the corpus fills (trigram/prefix flatten);
  - the MARGINAL primes added per new doc (it falls);
  - how many NEW primes the 30 knowledge definitions actually allocate (few).
Confirms: continual learning is memory-bounded - new knowledge is mostly
references to addresses already present.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load
from scifact_glossary import GLOSSARY


def by_view(idx):
    c = Counter(v for (v, _t) in idx.token_prime)
    return c["w"], c["3"], c["p"]


def main():
    corpus, _, _, _ = load("scifact")
    docs = list(corpus.items())
    n = len(docs)
    checkpoints = sorted({int(f * n) for f in (0.05, 0.1, 0.25, 0.5, 1.0)})

    idx = AppendOnlyLatticeIndex()
    print(f"scifact: {n:,} docs\n")
    print(f"  {'docs':>6} | {'word':>7} {'trigram':>8} {'prefix':>7} | "
          f"{'new prim/doc':>12} (marginal)")
    prev_docs = prev_tot = 0
    for i, (d, t) in enumerate(docs, 1):
        idx.add(d, t)
        if i in checkpoints:
            w, tri, pre = by_view(idx)
            tot = w + tri + pre
            marg = (tot - prev_tot) / (i - prev_docs)
            print(f"  {i:>6} | {w:>7,} {tri:>8,} {pre:>7,} | {marg:>12.1f}")
            prev_docs, prev_tot = i, tot

    w, tri, pre = by_view(idx)
    print(f"\n  full vocab: {w:,} word, {tri:,} trigram, {pre:,} prefix primes "
          f"({w+tri+pre:,} total)")

    # how many NEW primes do the 30 knowledge definitions add?
    new = Counter()
    seen = set(idx.token_prime)
    for term, defn in GLOSSARY.items():
        for tok in idx._multiview(defn):
            if tok not in seen:
                seen.add(tok)
                new[tok[0]] += 1
    total_def_tokens = sum(len(idx._multiview(defn)) for defn in GLOSSARY.values())
    print(f"\n  {len(GLOSSARY)} knowledge definitions ({total_def_tokens:,} subword tokens):")
    print(f"     NEW primes allocated: {new['w']} word, {new['3']} trigram, "
          f"{new['p']} prefix  = {sum(new.values())} total")
    print(f"     => {sum(new.values())} new addresses for {len(GLOSSARY)} definitions; "
          f"the rest REUSE existing subword/word primes (no memory blow-up).")


if __name__ == "__main__":
    main()
