#!/usr/bin/env python3
"""
P4 diagnostic — quantify zero-BM25 queries and lattice-only retrieval paths.

Shows which candidate tier fired per query and how often the gold doc was
reachable only via MeetIndex (tiers 2–3), not lexical overlap.

Usage:
  python p4_diagnostic.py scifact quality
  python p4_diagnostic.py scifact quality --from-checkpoint brains/scifact_quality.eval.pkl
  python p4_diagnostic.py scifact scale --max-queries 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from beir_data_root import resolve_beir_root
from eval_beir import (
    build_meet_index,
    build_neighbor_weights,
    candidate_generation_tier,
    candidate_ids,
    doc_text,
    load_corpus,
    load_paths,
    load_qrels,
    load_queries,
    merge_qrels,
    make_pipeline,
    ingest_corpus,
    ndcg_at_k,
    recall_at_k,
)
from eval_checkpoint import checkpoint_path, load_checkpoint
from aethos_hub_signature import build_all_hub_signatures, build_query_profile, rank_with_hub_signatures
from aethos_iterative import build_multi_pass


def _zero_bm25_query(profile, gold_doc: str, doc_tokens: dict) -> bool:
    """Query shares no surface token with the gold document."""
    gold_tokens = doc_tokens.get(gold_doc, frozenset())
    return not (profile.word_set & gold_tokens)


def run_diagnostic(
    dataset: str,
    mode: str,
    *,
    from_ckpt: str | Path | None = None,
    max_queries: int | None = None,
) -> int:
    if from_ckpt:
        bundle = load_checkpoint(from_ckpt)
        pipe = bundle.pipe
        cidx = bundle.cidx
        meet_index = bundle.meet_index
        neighbor_map = bundle.neighbor_map
        hub_sigs = bundle.hub_sigs
        queries = bundle.queries
        qrels = bundle.qrels
        qids = bundle.qids
        print(f"Loaded checkpoint: {from_ckpt}", flush=True)
    else:
        ckpt_path = checkpoint_path(dataset, mode)
        if ckpt_path.exists():
            return run_diagnostic(
                dataset, mode, from_ckpt=ckpt_path, max_queries=max_queries,
            )

        root = Path(resolve_beir_root())
        paths = load_paths(root, dataset)
        print(f"Building {dataset} ({mode}) for P4 diagnostic...", flush=True)
        corpus = load_corpus(paths.corpus, max_docs=None)
        queries = load_queries(paths.queries)
        qrels_test = load_qrels(paths.qrels_test)
        qids = [q for q in qrels_test if q in queries]
        if max_queries:
            qids = qids[:max_queries]
        qrels = {q: qrels_test[q] for q in qids}

        pipe = make_pipeline(mode)
        _, cidx = ingest_corpus(pipe, corpus, mode=mode)
        try:
            pipe.flush()
        except Exception:
            pass
        corpus_texts = [doc_text(doc) for doc in corpus.values()]
        build_multi_pass(pipe, corpus_texts, cidx.doc_tokens, n_passes=1, verbose=False)
        hub_sigs = build_all_hub_signatures(
            cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12,
        )
        meet_index = build_meet_index(hub_sigs, pipe.registry)
        neighbor_map = build_neighbor_weights(pipe.registry)

    if max_queries:
        qids = qids[:max_queries]

    tier_counts: Counter[str] = Counter()
    zero_bm25_total = 0
    zero_bm25_tier3 = 0
    lattice_path_queries = 0
    lattice_gold_in_candidates = 0
    lattice_r10_positive = 0
    lattice_ndcg_positive = 0
    tier4_fallback = 0
    gold_missed = 0

    n_docs = len(cidx.doc_ids)

    print(f"\nP4 diagnostic: {dataset} / {mode}  ({len(qids)} test queries)", flush=True)
    print(f"  meet index: {len(meet_index)} pool-prime postings", flush=True)
    print("-" * 60, flush=True)

    for qi, qid in enumerate(qids):
        profile = build_query_profile(
            queries[qid],
            pipe.registry,
            neighbor_map=neighbor_map,
            doc_freq=cidx.doc_freq,
            n_docs=n_docs,
        )
        tier = candidate_generation_tier(
            profile.words,
            cidx.inv,
            neighbor_map,
            cidx.doc_ids,
            meet_index=meet_index,
            registry=pipe.registry,
        )
        tier_counts[tier] += 1

        cands = candidate_ids(
            profile.words,
            cidx.inv,
            neighbor_map,
            cidx.doc_ids,
            meet_index=meet_index,
            registry=pipe.registry,
        )
        ranked = rank_with_hub_signatures(
            profile,
            cands,
            hub_sigs,
            cidx.doc_ids,
            doc_tokens=cidx.doc_tokens,
            doc_tf=cidx.doc_tf,
            doc_len=cidx.doc_len,
            avg_dl=cidx.avg_dl,
            registry=pipe.registry,
            top_k=100,
        )

        rel = qrels[qid]
        gold_ids = set(rel.keys())
        cand_set = set(cands)
        r10 = recall_at_k(ranked, rel, 10)
        ndcg = ndcg_at_k(ranked, rel, 10)

        if tier == "tier4_full_corpus":
            tier4_fallback += 1

        any_zero_bm25_gold = False
        for gid in gold_ids:
            if _zero_bm25_query(profile, gid, cidx.doc_tokens):
                any_zero_bm25_gold = True
                break

        if any_zero_bm25_gold:
            zero_bm25_total += 1
            if tier in ("tier3_meet_pool", "tier3_meet_fuzzy"):
                zero_bm25_tier3 += 1

        if tier not in ("tier1_lexical", "tier4_full_corpus"):
            lattice_path_queries += 1
            if gold_ids & cand_set:
                lattice_gold_in_candidates += 1
                if r10 > 0:
                    lattice_r10_positive += 1
                if ndcg > 0:
                    lattice_ndcg_positive += 1

        if not (gold_ids & cand_set):
            gold_missed += 1

    print("\nCandidate tier distribution:", flush=True)
    for t, n in tier_counts.most_common():
        pct = 100.0 * n / max(len(qids), 1)
        print(f"  {t:22s}  {n:4d}  ({pct:5.1f}%)", flush=True)

    print("\nZero-BM25 vocabulary mismatch (query vs gold doc):", flush=True)
    print(f"  queries with zero surface overlap:  {zero_bm25_total}", flush=True)
    if zero_bm25_total:
        pct = 100.0 * zero_bm25_tier3 / zero_bm25_total
        print(
            f"  of those, tier-3 meet fired:        {zero_bm25_tier3}  ({pct:.1f}%)",
            flush=True,
        )

    print("\nLattice-only candidate path (tier 2/3, not lexical):", flush=True)
    print(f"  queries using meet tiers:           {lattice_path_queries}", flush=True)
    print(f"  gold in candidate set (meet path):  {lattice_gold_in_candidates}", flush=True)
    print(f"  R@10 > 0 on meet path:              {lattice_r10_positive}", flush=True)
    print(f"  NDCG@10 > 0 on meet path:           {lattice_ndcg_positive}", flush=True)

    print("\nFallback / misses:", flush=True)
    print(f"  tier-4 full corpus fallback:        {tier4_fallback}", flush=True)
    print(f"  gold never in candidate set:        {gold_missed}", flush=True)

    print("\n--- Eric metric ---", flush=True)
    print(
        "  Queries where lattice (tier 2/3) was the ONLY path to reach gold:",
        flush=True,
    )
    print(f"    meet-path queries with gold in cands: {lattice_gold_in_candidates}", flush=True)
    print(
        "  (Cross-check: zero-BM25 count above = BM25-structural blind spot.)",
        flush=True,
    )
    print("-" * 60, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 zero-BM25 / meet-index diagnostic")
    parser.add_argument("dataset", nargs="?", default="scifact")
    parser.add_argument("mode", nargs="?", default="quality", choices=("quality", "scale"))
    parser.add_argument("--from-checkpoint", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()
    return run_diagnostic(
        args.dataset,
        args.mode,
        from_ckpt=args.from_checkpoint,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    raise SystemExit(main())
