#!/usr/bin/env python3
"""Before/after gold-rank on glossary gap queries."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import words
from aethos_encyclopedia_teacher import load_glossary
from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load


def gold_rank(brain, name, qid, test_q, k=1000):
    res = brain.search(queries[qid], corpus=name, k=k)
    golds = [d for d, s in test_q[qid].items() if s > 0]
    ranks = [res.local_ids.index(g) + 1 for g in golds if g in res.local_ids]
    return min(ranks) if ranks else None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    global queries
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    glossary = load_glossary(name)

    brain = MultiCorpusBrain()
    brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)

    targets = [qid for qid in test_ids if set(words(queries[qid])) & set(glossary)]
    before = {q: gold_rank(brain, name, q, test_q) for q in targets}

    all_q = {**train_q, **test_q}
    report = brain.mine_gaps(name, queries, all_q, miss_only=True)
    brain.teach_from_encyclopedia(
        name, report, queries=queries, use_wiki=False,
        max_terms=40, max_pairs=20, max_triples=12,
    )

    after = {q: gold_rank(brain, name, q, test_q) for q in targets}
    recov = 0
    print(f"{name}: {len(targets)} glossary queries, {len(glossary)} defs taught via gaps\n")
    for qid in sorted(targets, key=lambda q: int(q) if q.isdigit() else q):
        b, a = before[qid], after[qid]
        keys = [w for w in words(queries[qid]) if w in glossary]
        flag = ""
        if (b is None or b > 10) and a is not None and a <= 10:
            flag = "  <- RECOVERED"
            recov += 1
        elif b is not None and a is not None and a < b:
            flag = f"  (up {b}->{a})"
        print(f"  q{qid:<5} {keys[0] if keys else '?':<12} {str(b):>6} -> {str(a):>6}{flag}")
    print(f"\n  {recov} recovered to top-10")


if __name__ == "__main__":
    main()
