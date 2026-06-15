#!/usr/bin/env python3
"""
Closed-loop saturation learning — mine → teach → re-eval until convergence.

Run:  python scripts/exp_learn_saturate.py [scifact|nfcorpus] [--no-wiki]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load


def main():
    name = "scifact"
    use_wiki = True
    self_teach = False
    for arg in sys.argv[1:]:
        if arg == "--no-wiki":
            use_wiki = False
        elif arg == "--self":
            self_teach = True
        elif not arg.startswith("-"):
            name = arg

    corpus, queries, train_q, test_q = load(name)
    print(f"=== {name}: learn_until_saturated "
          f"(wiki={use_wiki}, self_teach={self_teach}) ===\n")

    brain = MultiCorpusBrain()
    brain.stack_corpus(name, corpus, queries=queries, train_qrels=train_q)

    result = brain.learn_until_saturated(
        name,
        queries,
        train_q,
        eval_qrels=test_q,
        use_wiki=use_wiki,
        self_teach=self_teach,
        max_iterations=8,
        max_absent_wiki=40,
    )

    print(f"\nConverged: {result.converged} ({result.reason})")
    print(f"  test nDCG: {result.ndcg_before:.4f} -> {result.ndcg_after:.4f} "
          f"({result.ndcg_after - result.ndcg_before:+.4f})")
    print(f"  test R@10: {result.recall_before:.4f} -> {result.recall_after:.4f}")
    print(f"  total terms taught: {result.total_terms_taught}")


if __name__ == "__main__":
    main()
