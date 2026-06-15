#!/usr/bin/env python3
"""
Multi-corpus brain demo — BIT 3 κ dots, route labels, auto-route, fused search.

Run:  python scripts/exp_multi_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_multi_corpus import MultiCorpusBrain
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from scripts.exp_teach_no_forget import load_glossary


PROBE_QUERIES = [
    ("p53 phosphorylation suppresses tumor growth", "scifact"),
    ("Do cholesterol statin drugs cause breast cancer", "nfcorpus"),
    ("vitamin worms longevity autophagy", "nfcorpus"),
    ("CRISPR gene editing therapeutic application", "scifact"),
]

SCIFACT_LABELS = frozenset({
    "crispr", "phosphorylation", "knockout", "mrna", "genome", "chromatin",
    "receptor", "mutation", "transcription", "polymerase",
})


def eval_corpus(brain, name, queries, test_q, test_ids):
    nd = r10 = 0.0
    routed_ok = 0
    for qid in test_ids:
        q = queries[qid]
        res = brain.search(q, corpus=name, k=100)
        rels = test_q[qid]
        nd += ndcg10(res.local_ids, rels)
        r10 += recall10(res.local_ids, rels)
        auto, _ = brain.route_corpus(q)
        if auto == name:
            routed_ok += 1
    n = len(test_ids)
    return {
        "ndcg10": nd / n,
        "recall10": r10 / n,
        "auto_route_pct": routed_ok / n,
        "n": n,
    }


def main():
    print("MultiCorpusBrain: BIT 3 kappa + route labels + shared registry\n")

    brain = MultiCorpusBrain()
    vocab0 = brain.vocab_size

    for name in ("scifact", "nfcorpus"):
        corpus, queries, train_q, test_q = load(name)
        extra_labels = SCIFACT_LABELS if name == "scifact" else None
        print(f"stacking {name}: {len(corpus)} docs ...", flush=True)
        brain.stack_corpus(
            name,
            corpus,
            queries=queries,
            train_qrels=train_q,
            route_labels=extra_labels,
            finalize=True,
            build_kappa=True,
        )
        gloss = load_glossary(name)
        if gloss:
            brain.teach_glossary(name, gloss)
            print(f"  taught {len(gloss)} concept correlations")

    stats = brain.stats()
    print(f"\nshared vocab: {vocab0} -> {stats['vocab_primes']} primes")
    print(f"registry words: {stats['registry_words']}")
    for cname, cstats in stats["corpora"].items():
        kappa = cstats.get("kappa_buckets") or {}
        print(
            f"  {cname}: root={cstats['root_prime']} docs={cstats['n_docs']} "
            f"mode={cstats['expansion_mode']} labels={cstats['route_labels']} "
            f"kappa_buckets={kappa.get('buckets', 0)} "
            f"kappa_docs={kappa.get('docs', 0)} "
            f"teach={cstats['teach_edges']}"
        )

    print("\n--- auto-route probes ---")
    ok = 0
    for q, expected in PROBE_QUERIES:
        res = brain.search(q, k=5)
        picked = res.corpus
        hit = picked == expected
        ok += int(hit)
        tag = "OK" if hit else f"MISS (expected {expected})"
        print(f"  [{tag}] route={picked} kappa_cand={res.kappa_candidates} keys={res.kappa_keys}")
        print(f"       top={res.local_ids[:3]}")
        print(f"       scores={{{', '.join(f'{k}:{v:.1f}' for k,v in sorted(res.route_scores.items(), key=lambda x:-x[1]))}}}")
        print(f"       q={q[:55]!r}")
    print(f"  probes: {ok}/{len(PROBE_QUERIES)}")

    print("\n--- per-corpus eval (kappa + routed, first 80 test q) ---")
    for name in ("scifact", "nfcorpus"):
        corpus, queries, train_q, test_q = load(name)
        test_ids = [q for q in test_q if q in queries][:80]
        m = eval_corpus(brain, name, queries, test_q, test_ids)
        print(
            f"  {name}: nDCG@10={m['ndcg10']:.4f} R@10={m['recall10']:.4f} "
            f"auto-route={100*m['auto_route_pct']:.0f}%  (n={m['n']})"
        )

    print("\n--- cross-corpus isolation ---")
    a = brain.search("p53 tumor suppressor apoptosis", corpus="scifact", k=3)
    b = brain.search("p53 tumor suppressor apoptosis", corpus="nfcorpus", k=3)
    overlap = set(a.local_ids) & set(b.local_ids)
    print(f"  shared local ids across branches: {len(overlap)} (expect 0)")

    print("\nDone.")


if __name__ == "__main__":
    main()
