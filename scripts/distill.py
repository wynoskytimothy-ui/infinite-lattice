#!/usr/bin/env python3
"""
LLM -> lattice knowledge distiller, end to end.

    python scripts/distill.py [scifact|nfcorpus|fiqa]

The loop: find the corpus's gap terms (list_gap_terms.py), have the teacher LLM
define them (the {corpus}_glossary.py files - written by the LLM from general
knowledge, NOT the gold docs), inject the definitions as full-weight query
expansion, and measure how high it pushes nDCG. The LLM teaches once, offline;
the lattice serves the knowledge forever at sub-ms with no model in the loop.

Reports lexical -> +bridges -> +knowledge on the held-out test split.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10

MIN_PAIRS = {"scifact": 1}      # paraphrase corpus; others default 2


def load_glossary(name):
    try:
        return importlib.import_module(f"{name.replace('-', '_')}_glossary").GLOSSARY
    except ModuleNotFoundError:
        return {}


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    glossary = load_glossary(name)
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=MIN_PAIRS.get(name, 2)).learn(
        queries, train_q, corpus)

    def idf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 0.0

    def expand(q, term_gate=5.5):
        # only inject knowledge for GENUINELY RARE query terms (real gaps the
        # engine has no signal on). Common/polysemous terms (fiqa's etf, cap)
        # already have lexical signal - expanding them only drifts.
        extra = []
        for t in set(words(q)):
            if t in glossary and idf(t) >= term_gate:
                for w in dict.fromkeys(words(glossary[t])):
                    if w != t and idf(w) >= 2.5:
                        extra.append(w)
        return q + " " + " ".join(extra[:10]) if extra else q

    def ev(fn):
        nd = rc = 0.0
        for qid in test_ids:
            r = fn(queries[qid])
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    touched = sum(1 for q in test_ids if set(words(queries[q])) & set(glossary))
    nd_l, rc_l = ev(lambda q: idx.search(q, 10))
    nd_b, rc_b = ev(lambda q: bridge_search(idx, br, q))
    nd_k, rc_k = ev(lambda q: bridge_search(idx, br, expand(q)))

    print(f"\n{name}: {N:,} docs | {len(test_ids)} test q | "
          f"glossary {len(glossary)} terms (touches {touched} queries)")
    print(f"  lexical:       nDCG {nd_l:.4f}  Recall {rc_l:.4f}")
    print(f"  + bridges:     nDCG {nd_b:.4f}  Recall {rc_b:.4f}   ({nd_b-nd_l:+.4f})")
    print(f"  + knowledge:   nDCG {nd_k:.4f}  Recall {rc_k:.4f}   "
          f"({nd_k-nd_b:+.4f} on top of bridges, {nd_k-nd_l:+.4f} total)")
    return name, nd_l, nd_b, nd_k


def main():
    targets = [sys.argv[1]] if len(sys.argv) > 1 else ["scifact", "nfcorpus", "fiqa"]
    rows = [run(t) for t in targets]
    print(f"\n{'='*60}\nDISTILLER SUMMARY (LLM taught the lattice, held-out)")
    print(f"  {'corpus':<10} {'lexical':>8} {'+bridges':>9} {'+knowledge':>11}")
    for name, l, b, k in rows:
        print(f"  {name:<10} {l:>8.4f} {b:>9.4f} {k:>11.4f}")
    print("\n  knowledge is appended data (definitions), not a retrain; served at")
    print("  sub-ms with no LLM in the loop. add a term -> one editable line.")


if __name__ == "__main__":
    main()
