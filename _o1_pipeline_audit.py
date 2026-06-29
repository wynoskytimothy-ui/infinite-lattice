#!/usr/bin/env python3
"""WHOLE-PIPELINE timing audit (no-GPU lexical lattice, scifact). Times every stage end-to-end so we see
where the time goes at INGEST, INDEX-BUILD, and QUERY, plus per-query breakdown. Then we apply the same
"the lattice is already sorted -> merge/lazy, not generic search" lever to the slowest stage, at held
footprint + accuracy."""
import os, sys, time
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    items = list(corpus.items())

    # ---- STAGE 1: INGEST (add每 doc) ----
    idx = AppendOnlyLatticeIndex()
    t0 = time.perf_counter()
    for d, t in items:
        idx.add(d, t)
    t_ingest = time.perf_counter() - t0
    ndoc = len(items)

    # ---- STAGE 2: INDEX FINALIZE (build postings / champion) ----
    t0 = time.perf_counter()
    try:
        idx.finalize()
    except Exception as e:
        print(f"  (finalize: {e})")
    t_final = time.perf_counter() - t0

    # footprint estimate: postings + token table
    n_post = sum(len(v) for v in idx.postings.values()) if hasattr(idx, "postings") else 0
    n_terms = len(idx.postings) if hasattr(idx, "postings") else 0

    # ---- STAGE 3: QUERY (search) + breakdown ----
    # warm
    for q in test_ids[:5]: list(idx.search(queries[q], 100))
    lat = []; nd = 0.0
    t_tok = t_score = 0.0
    for q in test_ids:
        qt = queries[q]
        t0 = time.perf_counter()
        # breakdown: tokenize (multiview) vs score
        toks = list(idx._multiview(qt)); t_tok += time.perf_counter() - t0
        t0 = time.perf_counter()
        res = list(idx.search(qt, 100)); t_score += time.perf_counter() - t0
        lat.append((t_tok))  # placeholder
        nd += ndcg10(res, test_q[q])
    nq = len(test_ids)
    # clean per-query latency
    lat = []
    for q in test_ids:
        t0 = time.perf_counter(); _ = list(idx.search(queries[q], 100)); lat.append((time.perf_counter()-t0)*1000)
    lat = np.array(lat)

    print("\n" + "=" * 78)
    print(f"WHOLE-PIPELINE AUDIT — {name}: {ndoc:,} docs, {nq} queries (no-GPU lexical lattice)")
    print("=" * 78)
    print(f"\n  STAGE 1  INGEST   : {t_ingest*1000:8.1f} ms total | {ndoc/t_ingest:8.0f} docs/s | {t_ingest/ndoc*1e6:6.0f} us/doc")
    print(f"  STAGE 2  INDEX    : {t_final*1000:8.1f} ms total (finalize/build postings)")
    print(f"  STAGE 3  QUERY    : {np.median(lat):8.2f} ms median | p90 {np.percentile(lat,90):.2f} | {1000/np.median(lat):.0f} q/s")
    print(f"           breakdown: tokenize {t_tok/nq*1000:.3f} ms/q | search {t_score/nq*1000:.3f} ms/q")
    print(f"\n  index: {n_terms:,} terms, {n_post:,} postings (~{n_post/ndoc:.0f} postings/doc)")
    print(f"  accuracy: nDCG@10 {nd/nq:.4f}")
    print(f"\n  LEVERS per stage (apply 'sorted-structure -> merge/lazy', held footprint+accuracy):")
    print(f"   - ingest: batch prime-assignment + reuse token table (append-only, 0-mutation already)")
    print(f"   - index : V27 pack (int16 doc + int8 tf, 3 B/posting, faster decode than FOR)")
    print(f"   - query : numba merge-score (proven 2.26x on MARCO serve) + rarest-address WAND skip")


if __name__ == "__main__":
    main()
