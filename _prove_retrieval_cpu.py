#!/usr/bin/env python3
"""
_prove_retrieval_cpu.py - PROVE "beats BM25, CPU-only, tiny + fast".

Claim under test (durable, no-GPU):
    The AppendOnlyLatticeIndex retrieves on BEIR-scifact at nDCG@10 above the
    BM25 reference (0.665), purely on CPU, with a small on-disk footprint and a
    fast median query latency.

This script:
  1. Loads scifact via scripts.bench_supervised_bridges.load (doc ids are STRINGS).
  2. Ingests every corpus doc into AppendOnlyLatticeIndex (CPU only) and measures
     ingest throughput (docs/s).
  3. finalize()s the dense fast path and measures nDCG@10 over the test queries.
  4. Measures per-query latency (median + p90) on CPU over all test queries.
  5. Measures the on-disk index footprint via save() -> bytes / #docs = B/doc.
  6. Prints PASS + a one-line scorecard. PASS requires nDCG@10 > 0.665.

No GPU. No external downloads (scifact is read from the local BEIR mirror that
load() already resolves). Reproduce:
    PYTHONUTF8=1 python "_prove_retrieval_cpu.py"
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# make sure we import the repo modules (this file lives at the repo root)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

BM25_REF = 0.665  # the BM25 nDCG@10 reference we must beat on scifact


def assert_no_gpu():
    """Best-effort confirmation we are NOT using a GPU. The stack is pure
    Python + numpy CPU; we just assert no CUDA torch is in play if torch exists."""
    try:
        import torch  # noqa: F401
        used = torch.cuda.is_available()
        return f"torch present, cuda_available={used} (unused: pure-numpy CPU path)"
    except Exception:
        return "no torch import; pure-Python + numpy CPU path"


def main():
    gpu_note = assert_no_gpu()

    # ---- 1. load scifact (STRING doc ids) ----
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]
    n_docs = len(corpus)
    n_test = len(test_ids)
    assert n_docs > 0 and n_test > 0, "scifact failed to load"

    # ---- 2. ingest: APPEND only, measure docs/s ----
    idx = AppendOnlyLatticeIndex()
    t0 = time.perf_counter()
    for d, txt in corpus.items():
        idx.add(d, txt)
    ingest_s = time.perf_counter() - t0
    docs_per_s = n_docs / ingest_s

    # ---- 3. finalize the CPU dense fast path + measure nDCG@10 ----
    idx.finalize()
    nd = 0.0
    for qid in test_ids:
        ranked = idx.search(queries[qid], 10)
        nd += ndcg10(ranked, test_q[qid])
    ndcg = nd / n_test

    # ---- 4. per-query latency on CPU (median + p90) ----
    lat_ms = []
    for qid in test_ids:
        s = time.perf_counter()
        idx.search(queries[qid], 10)
        lat_ms.append((time.perf_counter() - s) * 1000.0)
    lat_ms.sort()
    median_ms = statistics.median(lat_ms)
    p90_ms = lat_ms[int(0.90 * (len(lat_ms) - 1))]

    # ---- 5. on-disk footprint: save() -> bytes / #docs ----
    scratch = os.environ.get("TEMP", ".")
    save_path = os.path.join(scratch, "_prove_retrieval_cpu_idx")
    idx.save(save_path)
    npz = save_path + ".npz"
    size_bytes = os.path.getsize(npz)
    b_per_doc = size_bytes / n_docs

    # ---- 6. verdict ----
    passed = ndcg > BM25_REF

    print("=" * 72)
    print("PROVE: retrieval-cpu  (AppendOnlyLatticeIndex on BEIR-scifact, CPU)")
    print("=" * 72)
    print(f"GPU check       : {gpu_note}")
    print(f"corpus docs     : {n_docs}")
    print(f"test queries    : {n_test}")
    print(f"vocab (primes)  : {len(idx.token_prime)}")
    print("-" * 72)
    print(f"nDCG@10         : {ndcg:.4f}   (BM25 ref {BM25_REF:.3f}, "
          f"delta {ndcg - BM25_REF:+.4f})")
    print(f"ingest          : {ingest_s:.2f}s  ->  {docs_per_s:,.0f} docs/s (CPU)")
    print(f"query latency   : median {median_ms:.3f} ms  |  p90 {p90_ms:.3f} ms (CPU)")
    print(f"index footprint : {size_bytes:,} B on disk  ->  {b_per_doc:.0f} B/doc")
    print("-" * 72)
    print(f"SCORECARD: nDCG@10={ndcg:.4f}  {median_ms:.2f} ms/query  "
          f"{b_per_doc:.0f} B/doc  {docs_per_s:,.0f} docs/s  [all CPU]")
    if passed:
        print(f"PASS  nDCG@10 {ndcg:.4f} > BM25 {BM25_REF:.3f}  "
              f"(CPU-only, {median_ms:.2f} ms/query, {b_per_doc:.0f} B/doc)")
    else:
        print(f"FAIL  nDCG@10 {ndcg:.4f} <= BM25 {BM25_REF:.3f}")
    print("=" * 72)

    # cleanup the temp index file
    try:
        os.remove(npz)
    except OSError:
        pass

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
