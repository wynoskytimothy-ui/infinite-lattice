#!/usr/bin/env python3
"""
Diagnostic - do bridges exist for 0-dimensional / nano / quantum concepts?

The user's question: the "0-dimensional biomaterials" query missed. If other
TRAIN queries talk about the same concepts, bridges should already be there.
This script checks the actual data:

  1. which TRAIN and TEST queries mention each concept (0-dimensional, nano,
     quantum, ...), and how many RELEVANT pairs each concept-term appears in;
  2. for the missed query, a term-by-term bridge audit: idf, #train relevant
     pairs containing it, whether a bridge survived the min-2-pairs gate, and
     where it points;
  3. the gold doc's vocabulary, to see the actual lexical gap.

Root cause is one of: concept only in TEST (nothing to learn from), concept in
exactly ONE train pair (pruned by the generalisation gate), or concept absent
from the gold doc's own words (a coverage gap no bridge can close).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, RelevanceBridges

CONCEPTS = ["0-dimensional", "zero-dimensional", "0d", "dimensional",
            "nano", "nanoparticle", "nanomaterial", "nanotube", "nanoscale",
            "quantum", "biomaterial", "biomaterials", "inductive"]


def concept_hits(queries, qrels, substr):
    out = []
    for qid in qrels:
        if qid in queries and substr in queries[qid].lower():
            out.append(qid)
    return out


def main(name="scifact"):
    corpus, queries, train_q, test_q = load(name)
    print(f"{name}: {len(train_q)} train q, {len(test_q)} test q\n")

    # ---- 1. concept presence in train vs test (raw substring) ----
    print("CONCEPT PRESENCE (substring in query text)")
    print(f"  {'concept':<18} {'train':>6} {'test':>6}   example query")
    for c in CONCEPTS:
        tr = concept_hits(queries, train_q, c)
        te = concept_hits(queries, test_q, c)
        ex = ""
        if tr:
            ex = f"[train {tr[0]}] {queries[tr[0]][:48]}"
        elif te:
            ex = f"[test {te[0]}] {queries[te[0]][:48]}"
        if tr or te:
            print(f"  {c:<18} {len(tr):>6} {len(te):>6}   {ex}")

    # ---- build index + bridges (full train) ----
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br = RelevanceBridges(idx, N).learn(queries, train_q, corpus)

    # how many TRAIN relevant-pairs contain each concept token?
    print("\nCONCEPT TOKEN in TRAIN relevant pairs (drives bridge formation)")
    print(f"  {'token':<16} {'#rel pairs':>10} {'idf':>6} {'bridge?':>8}  -> top targets")
    for c in CONCEPTS:
        toks = words(c)
        for t in set(toks):
            npairs = br.qt_pairs.get(t, 0)
            p = idx.token_prime.get(("w", t))
            idf = idx._idf(p, N) if p else 0.0
            has = t in br.bridge
            tgt = ", ".join(f"{dt}" for dt, _ in br.bridge.get(t, [])[:4]) if has else "-"
            if npairs or (p and idx.df.get(p, 0)):
                print(f"  {t:<16} {npairs:>10} {idf:>6.1f} {str(has):>8}  -> {tgt}")

    # ---- 2. audit the missed query ----
    target = None
    for qid in test_q:
        if qid in queries and "dimensional" in queries[qid].lower() \
                and "biomaterial" in queries[qid].lower():
            target = qid
            break
    if not target:
        # fall back to any 0-dimensional / nano / quantum test query
        for qid in test_q:
            if qid in queries and any(c in queries[qid].lower()
                                      for c in ("dimensional", "nano", "quantum")):
                target = qid
                break
    if not target:
        print("\nno 0-dimensional/nano/quantum test query found")
        return

    print(f"\nMISSED-QUERY AUDIT - test query {target}")
    print(f"  '{queries[target]}'")
    gold = [d for d, s in test_q[target].items() if s > 0]
    print(f"  gold doc(s): {gold}")
    print(f"  {'query term':<16} {'idf':>6} {'#trainpairs':>11} {'bridge':>7}  -> targets")
    qterms = [w for w in words(queries[target])]
    for t in dict.fromkeys(qterms):
        p = idx.token_prime.get(("w", t))
        idf = idx._idf(p, N) if p else 0.0
        npairs = br.qt_pairs.get(t, 0)
        has = t in br.bridge
        tgt = ", ".join(f"{dt}({w:.2f})" for dt, w in br.bridge.get(t, [])[:4]) if has else "-"
        print(f"  {t:<16} {idf:>6.1f} {npairs:>11} {str(has):>7}  -> {tgt}")

    # ---- 3. gold-doc vocabulary + the lexical gap ----
    if gold:
        g = gold[0]
        gwords = set(words(corpus.get(g, "")))
        qset = set(qterms)
        overlap = qset & gwords
        print(f"\n  gold doc {g} ({len(corpus.get(g,''))} chars): "
              f"{len(gwords)} distinct words")
        print(f"  query<->gold word overlap: {sorted(overlap) if overlap else 'NONE'}")
        # which of the query's bridge targets ARE in the gold doc?
        reach = []
        for t in qset:
            for dt, w in br.bridge.get(t, []):
                if dt in gwords:
                    reach.append((t, dt, round(w, 2)))
        print(f"  bridge targets that land in the gold doc: "
              f"{reach if reach else 'NONE (no learned bridge reaches it)'}")
        # show a slice of the gold doc so the vocabulary is visible
        snippet = corpus.get(g, "")[:240].replace("\n", " ")
        print(f"  gold doc opening: {snippet}...")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scifact")
