#!/usr/bin/env python3
"""Audit EQ-RAG soft expand on SciFact 300q — trace fires, terms, pool recall delta."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_lattice_retriever_v1 import load_scifact, ndcg10, recall10
from lattice_retriever_v1.hybrid_retriever import (
    HybridConfig,
    build_hybrid_retriever,
    resolve_eq_rag_config,
)


def _baseline_cfg() -> HybridConfig:
    cfg = HybridConfig(
        lexical_mode="append_index",
        enable_append_pool_union=True,
        append_pool_k=200,
        lam_lex=1.0,
        lam_l2=0.0,
        lam_walk=0.0,
        enable_eq_rag_expand=False,
        enable_corpus_lattice=False,
        enable_rccm=False,
    )
    return resolve_eq_rag_config(cfg)


def _eq_cfg(base: HybridConfig) -> HybridConfig:
    return resolve_eq_rag_config(replace(base, enable_eq_rag_expand=True))


def main() -> None:
    corpus, queries, test_qrels = load_scifact()
    test_ids = [q for q in test_qrels if q in queries]

    print(f"Building index ({len(corpus)} docs)...", flush=True)
    t0 = time.time()
    base_cfg = _baseline_cfg()
    # Build with corpus lattice skeleton (needed for EQ-RAG recovery)
    build_cfg = resolve_eq_rag_config(replace(base_cfg, enable_eq_rag_expand=True))
    retriever = build_hybrid_retriever(corpus, config=build_cfg)
    print(f"  build {time.time() - t0:.1f}s", flush=True)

    fired = 0
    pool_lift = 0
    gold_entered = 0
    ndcg_base_sum = 0.0
    ndcg_eq_sum = 0.0
    pool_base_miss_eq_hit = 0
    term_counts: dict[str, int] = {}
    examples: list[dict] = []

    for i, qid in enumerate(test_ids):
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}

        retriever.config = _baseline_cfg()
        trace_off = retriever.retrieve_with_trace(q, limit=10)
        pool_off = trace_off.pool_docs
        ndcg_off = ndcg10([h.doc_id for h in trace_off.hits], rels)

        retriever.config = _eq_cfg(base_cfg)
        trace_on = retriever.retrieve_with_trace(q, limit=10)
        pool_on = trace_on.pool_docs
        ndcg_on = ndcg10([h.doc_id for h in trace_on.hits], rels)

        eq_steps = [
            s for s in trace_on.filter_steps if s.get("step") == "eq_rag_expanded_terms"
        ]
        recovered: list[str] = []
        if eq_steps:
            fired += 1
            recovered = eq_steps[0].get("recovered_terms", [])
            for t in recovered:
                term_counts[t] = term_counts.get(t, 0) + 1

        new_docs = pool_on - pool_off
        gold_new = gold & new_docs
        if new_docs:
            pool_lift += 1
        if gold_new:
            gold_entered += 1
        if not (gold & pool_off) and (gold & pool_on):
            pool_base_miss_eq_hit += 1

        ndcg_base_sum += ndcg_off
        ndcg_eq_sum += ndcg_on

        if recovered and (gold_new or abs(ndcg_on - ndcg_off) > 0.01):
            examples.append(
                {
                    "qid": qid,
                    "query": q[:120],
                    "recovered": recovered,
                    "gold_new": sorted(gold_new),
                    "pool_delta": len(new_docs),
                    "ndcg_off": round(ndcg_off, 4),
                    "ndcg_on": round(ndcg_on, 4),
                }
            )

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_ids)} queries...", flush=True)

    n = len(test_ids)
    top_terms = sorted(term_counts.items(), key=lambda x: -x[1])[:30]
    report = {
        "n_queries": n,
        "eq_rag_fired": fired,
        "eq_rag_fire_rate": round(fired / n, 4),
        "queries_with_pool_delta": pool_lift,
        "queries_gold_entered_pool": gold_entered,
        "baseline_miss_eq_rescued": pool_base_miss_eq_hit,
        "ndcg_at_10_baseline": round(ndcg_base_sum / n, 4),
        "ndcg_at_10_eq_rag": round(ndcg_eq_sum / n, 4),
        "ndcg_delta": round((ndcg_eq_sum - ndcg_base_sum) / n, 4),
        "top_recovered_terms": top_terms,
        "notable_examples": examples[:25],
    }

    log_path = Path(__file__).resolve().parents[1] / "logs" / "eq_rag_audit_300q.json"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== EQ-RAG 300q Audit ===")
    for k, v in report.items():
        if k not in ("top_recovered_terms", "notable_examples"):
            print(f"  {k}: {v}")
    print(f"  top_recovered_terms (top 15): {top_terms[:15]}")
    print(f"  log: {log_path}")


if __name__ == "__main__":
    main()
