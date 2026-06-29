#!/usr/bin/env python3
"""EDGE footprint<->accuracy knee — find the smallest index that keeps accuracy.
Measures REAL on-disk bytes (save() -> .npz, delta+float16+zlib) and nDCG@10 for:
  1. full multiview (word+trigram+prefix)   -- the fat baseline
  2. word-only (kappa_primary)              -- drops the ~80%-of-postings trigram gear
  3. word-only + per-doc top-k pruning       -- the SPLADE trick: keep each doc's strongest terms
Then the curve so we pick the knee instead of guessing."""
import os, sys, time, gc, tempfile
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10


def build(corpus, **cfg):
    idx = AppendOnlyLatticeIndex(**cfg)
    for d, t in corpus.items(): idx.add(d, t)
    return idx


def prune_per_doc(idx, k):
    """Keep each doc's top-k highest-weight postings; rebuild postings/df/doc_len. Word-only assumed."""
    by_doc = defaultdict(list)
    for p, pl in idx.postings.items():
        for d, wt in pl.items(): by_doc[d].append((wt, p))
    postings = defaultdict(dict); df = defaultdict(int); doc_len = {}
    for d, lst in by_doc.items():
        lst.sort(reverse=True)
        keep = lst[:k]
        dl = 0.0
        for wt, p in keep:
            postings[p][d] = wt; df[p] += 1; dl += wt
        doc_len[d] = dl
    idx.postings = postings; idx.df = df; idx.doc_len = doc_len
    idx._total_len = sum(doc_len.values()); idx._dense_ready = False
    return idx


def measure(idx, corpus, queries, test_q, test_ids, label):
    gc.collect()
    fd, path = tempfile.mkstemp(suffix=".npz"); os.close(fd)
    idx.save(path[:-4]); nbytes = os.path.getsize(path); os.remove(path)
    bdoc = nbytes / len(corpus)
    idx.finalize()
    lat = []
    nd = []
    for q in test_ids:
        t0 = time.perf_counter(); res = idx.search(queries[q], 100); lat.append((time.perf_counter()-t0)*1000)
        nd.append(ndcg10(res, test_q[q]))
    npost = sum(len(v) for v in idx.postings.values())
    print(f"  {label:<34} {bdoc:>7.0f} B/doc  {nbytes/1e6:>6.1f} MB  nDCG {np.mean(nd):.4f}  "
          f"{np.median(lat):>5.2f} ms  {npost:>9,} post")
    return bdoc, float(np.mean(nd))


def main():
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]
    print("=" * 92)
    print(f"EDGE FOOTPRINT<->ACCURACY KNEE — scifact {len(corpus):,} docs (REAL save() bytes)")
    print("=" * 92)
    print(f"  {'config':<34} {'B/doc':>7}  {'total':>6}     {'acc':>4}      {'speed':>5}  {'postings':>9}")

    measure(build(corpus), corpus, queries, test_q, test_ids, "1. full multiview (word+tri+prefix)")
    measure(build(corpus, index_mode="kappa_primary"), corpus, queries, test_q, test_ids, "2. word-only (drop trigram gear)")
    for k in [60, 40, 25, 15, 10]:
        idx = prune_per_doc(build(corpus, index_mode="kappa_primary"), k)
        measure(idx, corpus, queries, test_q, test_ids, f"3. word-only + top-{k}/doc")

    print("\n  READ: trigram gear is ~80% of postings (typo-robustness); dropping it is the big footprint win.")
    print("  per-doc top-k then trims the tail. The knee = smallest B/doc whose nDCG still beats BM25 (~0.665).")


if __name__ == "__main__":
    main()
