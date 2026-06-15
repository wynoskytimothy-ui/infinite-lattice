#!/usr/bin/env python3
"""
Gap-mined encyclopedia teaching — find missing query signal, teach correlations.

  1. Mine gaps: words / subwords / pairs / triples absent from corpus
  2. Teach from scifact|nfcorpus glossary + Wikipedia (cached)
  3. Measure held-out lift (never indexes encyclopedia text as documents)

Run:  python scripts/exp_teach_gaps.py [scifact|nfcorpus] [--no-wiki]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def main():
    name = "scifact"
    use_wiki = True
    for arg in sys.argv[1:]:
        if arg == "--no-wiki":
            use_wiki = False
        elif not arg.startswith("-"):
            name = arg

    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    print(f"=== {name}: gap-mined encyclopedia teaching ===\n")
    brain = MultiCorpusBrain()
    brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)

    nd0, r10_0, r100_0 = eval_brain(brain, name, queries, test_q, test_ids)
    print(f"BEFORE teach: nDCG@10={nd0:.4f} R@10={r10_0:.4f} R@100={r100_0:.4f}")

    all_q = {**train_q, **test_q}
    report = brain.mine_gaps(name, queries, all_q, miss_only=True)
    summ = report.summary()
    print(f"\nGap audit ({summ['n_miss']}/{summ['n_queries']} miss gold@10):")
    print(f"  missing words: {summ['missing_words']}  absent from corpus: {summ['absent_from_corpus']}")
    print(f"  missing subwords: {summ['missing_subwords']}")
    print(f"  missing pairs: {summ['missing_pairs']}  triples: {summ['missing_triples']}")
    print(f"  priority terms: {summ['priority_top12']}")
    gloss_targets = report.glossary_targets(
        __import__("aethos_encyclopedia_teacher", fromlist=["load_glossary"]).load_glossary(name),
        queries,
    )
    print(f"  glossary miss-query targets: {len(gloss_targets)}  e.g. {gloss_targets[:8]}")

    print(f"\nTeaching (glossary + {'Wikipedia' if use_wiki else 'glossary only'}) ...")
    result = brain.teach_from_encyclopedia(
        name, report,
        queries=queries,
        use_wiki=use_wiki,
        max_terms=35,
        max_pairs=20,
        max_triples=12,
    )
    print(f"  terms={result.terms_taught} pairs={result.pairs_taught} "
          f"triples={result.triples_taught} subwords={result.subwords_taught}")
    print(f"  glossary_hits={result.glossary_hits} wiki_fetched={result.wiki_fetched} "
          f"skipped={result.skipped_empty}")

    nd1, r10_1, r100_1 = eval_brain(brain, name, queries, test_q, test_ids)
    print(f"\nAFTER teach:  nDCG@10={nd1:.4f} R@10={r10_1:.4f} R@100={r100_1:.4f}")
    print(f"  delta: nDCG {nd1-nd0:+.4f}  R@10 {r10_1-r10_0:+.4f}  R@100 {r100_1-r100_0:+.4f}")

    branch = brain._corpora[name]
    n_edges = sum(len(v) for v in branch.teach.edges.values())
    print(f"\n  teach memory: {n_edges} edges (~{n_edges * 8 / 1024:.1f} KB)")
    print(f"  vocab unchanged: {brain.vocab_size} primes (correlations only)")


if __name__ == "__main__":
    main()
