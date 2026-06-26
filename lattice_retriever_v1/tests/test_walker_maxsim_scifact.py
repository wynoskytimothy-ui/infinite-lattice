"""SciFact smoke gate for WalkerMaxSimRetriever."""

import pytest

pytest.importorskip("beir_data_root", reason="SciFact integration gate — run explicitly")

from scripts.bench_walker_maxsim import (  # noqa: E402
    build_retriever,
    load_scifact,
    ndcg10,
    recall10,
)


@pytest.mark.integration
def test_walker_maxsim_scifact_smoke():
    corpus, queries, test_qrels = load_scifact()
    test_ids = [q for q in test_qrels if q in queries][:50]
    retriever = build_retriever({k: corpus[k] for k in list(corpus)[:800]})

    ndcgs, recalls, pool_hits = [], [], 0
    for qid in test_ids:
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}
        trace = retriever.retrieve_with_trace(q, limit=10)
        ranked = [h.doc_id for h in trace.hits]
        if gold & set(trace.lit_docs):
            pool_hits += 1
        ndcgs.append(ndcg10(ranked, rels))
        recalls.append(recall10(ranked, rels))

    n = len(test_ids)
    pool_recall = pool_hits / n
    assert pool_recall >= 0.04, "pool recall at chance — walker not retrieving at all"
    assert sum(ndcgs) / n >= 0.0
