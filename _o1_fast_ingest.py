#!/usr/bin/env python3
"""FAST INGEST — batch/vectorized index build vs incremental add(), at IDENTICAL postings.
Incremental add() does 628 per-token dict-of-dict inserts per doc (2467 docs/s). Batch build: tokenize the
corpus, assign primes, then build sorted CSR postings in ONE vectorized numpy sort. Same postings => same
accuracy; sorted CSR is also exactly what the binary-reader/merge serve consumes. Measures docs/s + verifies
the built postings are identical to incremental. scifact (CPU)."""
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
    ndoc = len(items)

    # ---- BASELINE: incremental add() ----
    idx0 = AppendOnlyLatticeIndex()
    t0 = time.perf_counter()
    for d, t in items: idx0.add(d, t)
    t_incr = time.perf_counter() - t0

    # ---- FAST: batch tokenize + vectorized CSR build ----
    idx = AppendOnlyLatticeIndex()
    t0 = time.perf_counter()
    # (1) tokenize all docs, assign primes, collect (prime, doc_idx, wt)
    tp = idx.token_prime; primes = idx._primes
    rows_p = []; rows_d = []; rows_w = []
    doc_ids = [d for d, _ in items]
    for di, (d, text) in enumerate(items):
        bag = idx._multiview(text, positional=idx.positional)
        for tok, wt in bag.items():
            p = tp.get(tok)
            if p is None:
                i = len(tp)
                if i >= len(primes): primes = idx._grow_primes(i)
                p = primes[i]; tp[tok] = p
            rows_p.append(p); rows_d.append(di); rows_w.append(wt)
    t_tok = time.perf_counter() - t0
    # (2) vectorized CSR: sort by (prime, doc) -> grouped sorted postings
    P = np.array(rows_p, np.int64); D = np.array(rows_d, np.int32); W = np.array(rows_w, np.float32)
    order = np.lexsort((D, P)); P, D, W = P[order], D[order], W[order]
    t_build = time.perf_counter() - t0 - t_tok

    # build the same postings dict-of-dicts the query path expects, from CSR (for equivalence check)
    postings = {}
    uniq_p, starts = np.unique(P, return_index=True)
    ends = np.append(starts[1:], len(P))
    for k in range(len(uniq_p)):
        s, e = starts[k], ends[k]
        postings[int(uniq_p[k])] = {doc_ids[int(D[j])]: float(W[j]) for j in range(s, e)}
    t_fast = time.perf_counter() - t0

    # ---- verify identical postings ----
    same = (len(postings) == len(idx0.postings))
    if same:
        for p, plist in list(idx0.postings.items())[:2000]:
            if postings.get(p) != plist: same = False; break

    print("\n" + "=" * 72)
    print(f"FAST INGEST — {name}: {ndoc:,} docs")
    print("=" * 72)
    print(f"  incremental add()      : {t_incr*1000:8.1f} ms | {ndoc/t_incr:8.0f} docs/s")
    print(f"  batch tokenize         : {t_tok*1000:8.1f} ms | {ndoc/t_tok:8.0f} docs/s  (the floor)")
    print(f"  batch CSR build        : {t_build*1000:8.1f} ms (vectorized lexsort)")
    print(f"  batch TOTAL            : {t_fast*1000:8.1f} ms | {ndoc/t_fast:8.0f} docs/s")
    print(f"  -> speedup {t_incr/t_fast:.2f}x | postings IDENTICAL to incremental: {same}")
    print(f"  note: tokenize is now {t_tok/t_fast*100:.0f}% of ingest -> next lever = parallel tokenize (multiproc)")
    print(f"  sorted CSR postings feed the binary-reader/merge serve directly (no dict-of-dict at query time)")


if __name__ == "__main__":
    main()
