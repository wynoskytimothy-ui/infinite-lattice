#!/usr/bin/env python3
"""
eval_beir_lattice.py - benchmark LatticeRetriever on BEIR datasets.

Targets (per user):
  - nDCG@10 at or above BM25 reference for the dataset
  - <= 500 B/doc footprint
  - <= 50 ms p50 per-query latency

Usage:
  python eval_beir_lattice.py --dataset scifact
  python eval_beir_lattice.py --dataset nfcorpus --max-queries 200
  python eval_beir_lattice.py --dataset fiqa --max-docs 10000
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_lattice_retrieval import LatticeRetriever
from beir_data_root import resolve_beir_root
from eval_beir import (
    BM25_REF,
    doc_text,
    load_corpus,
    load_paths,
    load_qrels,
    load_queries,
    ndcg_at_k,
    recall_at_k,
)


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact",
                    choices=["scifact", "nfcorpus", "fiqa", "trec-covid", "webis-touche2020"])
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    if not paths.corpus.exists():
        print(f"FATAL: corpus not found at {paths.corpus}")
        sys.exit(1)

    print("=" * 78)
    print(f"BEIR LATTICE RETRIEVAL  dataset={args.dataset}  k={args.k}")
    print("=" * 78)

    # ---- load ----
    print("\nLoading...")
    t0 = time.time()
    corpus = load_corpus(paths.corpus, max_docs=args.max_docs)
    queries_all = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    if not qrels:
        qrels = load_qrels(paths.qrels_train)
    load_s = time.time() - t0
    print(f"  corpus  : {len(corpus):>6} docs  ({load_s:.1f}s)")
    print(f"  queries : {len(queries_all):>6} total; qrels for {len(qrels)}")

    # ---- build index ----
    n_docs = len(corpus)
    # Size pools generously; chain_primes() generates ~N primes
    token_pool = max(30000, n_docs * 6)
    doc_pool = max(n_docs * 2, 10000)
    print(f"\nBuilding LatticeRetriever (token_pool={token_pool}, doc_pool={doc_pool})...")

    t0 = time.time()
    retriever = LatticeRetriever(token_pool_size=token_pool, doc_pool_size=doc_pool)
    pool_alloc_ms = (time.time() - t0) * 1000
    print(f"  prime-pool allocation: {pool_alloc_ms:.0f} ms")

    t0 = time.time()
    corpus_text = {did: doc_text(d) for did, d in corpus.items()}
    retriever.build_from_corpus(corpus_text)
    build_ms = (time.time() - t0) * 1000
    print(f"  ingest               : {build_ms:.0f} ms  ({n_docs / (build_ms/1000):.0f} docs/sec)")

    fp = retriever.estimated_footprint()
    print(f"  index size           : {fp['total_bytes']/1024:.0f} KB total")
    print(f"  per-doc footprint    : {fp['bytes_per_doc']:.0f} B/doc")
    print(f"  per-token footprint  : {fp['bytes_per_token']:.0f} B/token "
          f"(avg {fp['avg_parents_per_token']:.1f} parents)")
    print(f"  vocab                : {fp['n_tokens']} unique tokens")

    # ---- queries ----
    # Restrict to queries that have qrels
    qids = [qid for qid in queries_all if qid in qrels]
    if args.max_queries:
        qids = qids[:args.max_queries]

    print(f"\nRunning {len(qids)} queries...")
    ndcg_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    latencies_ms: list[float] = []
    n_zero_results = 0

    progress_every = max(1, len(qids) // 10)
    for i, qid in enumerate(qids, start=1):
        q_text = queries_all[qid]
        rel = qrels.get(qid, {})
        if not rel:
            continue

        t0 = time.time()
        results = retriever.query(q_text, k=100)
        elapsed = (time.time() - t0) * 1000
        latencies_ms.append(elapsed)

        ranked = [d for d, _ in results]
        if not ranked:
            n_zero_results += 1

        ndcg_scores.append(ndcg_at_k(ranked, rel, k=args.k))
        recall_scores.append(recall_at_k(ranked, rel, k=args.k))
        mrr_scores.append(mrr_at_k(ranked, rel, k=args.k))

        if i % progress_every == 0:
            mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
            p50 = percentile(latencies_ms, 0.50)
            print(f"  [{i:>5}/{len(qids)}]  ndcg@{args.k}={mean_ndcg:.3f}  "
                  f"p50={p50:.1f} ms  empty={n_zero_results}")

    # ---- final metrics ----
    print()
    print("=" * 78)
    print("FINAL METRICS")
    print("=" * 78)
    n = max(len(ndcg_scores), 1)
    ndcg_mean = sum(ndcg_scores) / n
    recall_mean = sum(recall_scores) / n
    mrr_mean = sum(mrr_scores) / n
    p50 = percentile(latencies_ms, 0.50)
    p95 = percentile(latencies_ms, 0.95)
    p99 = percentile(latencies_ms, 0.99)
    mean_lat = sum(latencies_ms) / max(len(latencies_ms), 1)

    bm25 = BM25_REF.get(args.dataset)
    print(f"  dataset            : {args.dataset}  ({n_docs} docs)")
    print(f"  queries evaluated  : {n}")
    print(f"  queries with empty : {n_zero_results}")
    print()
    print(f"  nDCG@{args.k}             : {ndcg_mean:.4f}")
    print(f"  Recall@{args.k}           : {recall_mean:.4f}")
    print(f"  MRR@{args.k}              : {mrr_mean:.4f}")
    if bm25 is not None:
        delta = ndcg_mean - bm25
        print(f"  BM25 reference     : {bm25:.4f}  (delta {delta:+.4f})")
    print()
    print(f"  per-query latency  : mean {mean_lat:.2f} ms  "
          f"p50 {p50:.2f}  p95 {p95:.2f}  p99 {p99:.2f}")
    print(f"  per-doc footprint  : {fp['bytes_per_doc']:.0f} B/doc")
    print(f"  total index        : {fp['total_bytes']/1024:.0f} KB "
          f"({fp['total_bytes']/n_docs:.0f} B/doc all-in)")
    print()
    print(f"  TARGETS:")
    target_acc = bm25 if bm25 is not None else 0.5
    print(f"    nDCG@{args.k} >= {target_acc:.3f} (BM25)  : "
          f"{'PASS' if ndcg_mean >= target_acc else 'FAIL'} ({ndcg_mean:.4f})")
    print(f"    <= 500 B/doc                : "
          f"{'PASS' if fp['bytes_per_doc'] <= 500 else 'FAIL'} ({fp['bytes_per_doc']:.0f})")
    print(f"    <= 50 ms p50 latency        : "
          f"{'PASS' if p50 <= 50 else 'FAIL'} ({p50:.2f})")


if __name__ == "__main__":
    main()
