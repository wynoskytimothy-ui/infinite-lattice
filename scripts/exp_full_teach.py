#!/usr/bin/env python3
"""
Full knowledge teach — entire glossary + Wikipedia for absent terms.

The big-win path: teach_bridge star topology + definitional query rewrite on
ALL glossary terms, then wiki-fill every query term absent from the corpus.

Run:  python scripts/exp_full_teach.py [scifact|nfcorpus|both] [--no-wiki]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import words
from aethos_encyclopedia_teacher import load_glossary
from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def eval_brain(brain, name, queries, test_q, test_ids):
    nd = r10 = r100 = 0.0
    for qid in test_ids:
        res = brain.search(queries[qid], corpus=name, k=100)
        rels = test_q[qid]
        nd += ndcg10(res.local_ids, rels)
        r10 += recall10(res.local_ids, rels)
        rel = {d for d, s in rels.items() if s > 0}
        r100 += len(set(res.local_ids[:100]) & rel) / len(rel) if rel else 0.0
    n = len(test_ids)
    return nd / n, r10 / n, r100 / n


def gap_recovery(brain, name, queries, test_q, test_ids, glossary):
    targets = [qid for qid in test_ids if set(words(queries[qid])) & set(glossary)]
    recov = 0
    for qid in targets:
        res = brain.search(queries[qid], corpus=name, k=1000)
        golds = [d for d, s in test_q[qid].items() if s > 0]
        rank = min(
            (res.local_ids.index(g) + 1 for g in golds if g in res.local_ids),
            default=None,
        )
        if rank is not None and rank <= 10:
            recov += 1
    return len(targets), recov


def run_one(name: str, use_wiki: bool, max_absent: int) -> None:
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    glossary = load_glossary(name)
    all_q = {**train_q, **test_q}

    print(f"\n{'='*60}")
    print(f"  {name.upper()} — full knowledge teach")
    print(f"{'='*60}\n")

    brain = MultiCorpusBrain()
    brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)

    nd0, r10_0, r100_0 = eval_brain(brain, name, queries, test_q, test_ids)
    print(f"BEFORE: nDCG@10={nd0:.4f}  R@10={r10_0:.4f}  R@100={r100_0:.4f}")

    print(f"\nTeaching: {len(glossary)} glossary + wiki absent (max {max_absent}) ...")
    report, result = brain.teach_full_knowledge(
        name, queries, all_q,
        use_wiki=use_wiki,
        max_absent_wiki=max_absent,
    )
    summ = report.summary()
    print(f"  glossary={result.glossary_hits}  wiki={result.wiki_fetched}  "
          f"terms={result.terms_taught}  skipped={result.skipped_empty}")
    print(f"  pairs={result.pairs_taught}  triples={result.triples_taught}  "
          f"subwords={result.subwords_taught}")
    print(f"  absent pool: {summ['absent_from_corpus']}  miss@10: {summ['n_miss']}")

    nd1, r10_1, r100_1 = eval_brain(brain, name, queries, test_q, test_ids)
    n_tgt, recov = gap_recovery(brain, name, queries, test_q, test_ids, glossary)
    branch = brain._corpora[name]
    n_edges = sum(len(v) for v in branch.teach.edges.values())
    n_defs = len(branch.teach.definitions)

    print(f"\nAFTER:  nDCG@10={nd1:.4f}  R@10={r10_1:.4f}  R@100={r100_1:.4f}")
    print(f"  DELTA: nDCG {nd1-nd0:+.4f}  R@10 {r10_1-r10_0:+.4f}  "
          f"R@100 {r100_1-r100_0:+.4f}")
    print(f"  {recov}/{n_tgt} glossary queries in top-10")
    print(f"  memory: {n_defs} defs, {n_edges} edges (~{n_edges * 8 / 1024:.1f} KB)")


def main():
    corpora = ["scifact", "nfcorpus"]
    use_wiki = True
    max_absent = 200

    for arg in sys.argv[1:]:
        if arg == "--no-wiki":
            use_wiki = False
        elif arg.startswith("--max-absent="):
            max_absent = int(arg.split("=", 1)[1])
        elif arg in ("scifact", "nfcorpus", "both"):
            corpora = ["scifact", "nfcorpus"] if arg == "both" else [arg]

    for name in corpora:
        run_one(name, use_wiki, max_absent)


if __name__ == "__main__":
    main()
