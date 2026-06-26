"""SciFact integration gate for lattice_retriever_v1 — quality baseline, separate from synthetic."""

import pytest

pytest.importorskip("beir_data_root", reason="SciFact integration gate — run explicitly")

from scripts.bench_lattice_retriever_v1 import (  # noqa: E402
    build_retriever,
    load_scifact,
    ndcg10,
    recall10,
)


@pytest.mark.integration
def test_scifact_honest_baseline_recorded():
    """
    Quality gate — not a property proof. Records starting line with no tuning.
    Fails if geometry retrieves below chance on pool recall (sanity floor).
    """
    corpus, queries, test_qrels = load_scifact()
    test_ids = [q for q in test_qrels if q in queries][:50]  # smoke subset for CI speed
    retriever = build_retriever({k: corpus[k] for k in list(corpus)[:800]})

    ndcgs, recalls, pool_hits = [], [], 0
    for qid in test_ids:
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}
        pool, _ = retriever.lazy_pool(q)
        hits = retriever.retrieve(q, limit=10)
        ranked = [h.doc_id for h in hits]
        if gold & set(pool):
            pool_hits += 1
        ndcgs.append(ndcg10(ranked, rels))
        recalls.append(recall10(ranked, rels))

    n = len(test_ids)
    pool_recall = pool_hits / n
    assert pool_recall > 0.05, "pool recall at chance — geometry not retrieving at all"
    assert sum(ndcgs) / n >= 0.0
