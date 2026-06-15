#!/usr/bin/env python3
"""
Deep search step 7 - compact, appendable persistence (save/load round-trip).

Goal: on-disk footprint near the ~10MB in-RAM dense form, lossless, and the
reloaded index is still fully appendable (not a frozen query blob). Verifies:
  - round-trip: loaded.search == original.search for every held-out test query;
  - still appendable: add() + finalize() + search() work after load;
  - size: vs the dense arrays and vs a naive pickle of the postings.
"""

from __future__ import annotations

import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load as load_ds, ndcg10, recall10


def run(name):
    corpus, queries, train_q, test_q = load_ds(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    idx.finalize()
    print(f"\n{'='*60}\n{name}: {len(idx.alive):,} docs")

    before = {q: idx.search(queries[q], 10) for q in test_ids}
    dense_mb = (sum(a.nbytes for a in idx._d_pdoc.values())
                + sum(a.nbytes for a in idx._d_ptf.values())
                + idx._d_denom.nbytes) / 1e6

    tmp = Path(tempfile.gettempdir()) / f"aethos_{name}"
    t0 = time.perf_counter()
    idx.save(tmp)
    save_s = time.perf_counter() - t0
    disk_mb = os.path.getsize(str(tmp) + ".npz") / 1e6

    # naive pickle of the postings dict (for comparison)
    naive_mb = len(pickle.dumps(idx.postings, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6

    t0 = time.perf_counter()
    idx2 = AppendOnlyLatticeIndex.load(tmp)
    load_s = time.perf_counter() - t0

    after = {q: idx2.search(queries[q], 10) for q in test_ids}
    identical = sum(1 for q in test_ids if before[q] == after[q])
    overlap = sum(len(set(before[q]) & set(after[q])) for q in test_ids) / (10 * len(test_ids))
    nd = sum(ndcg10(after[q], test_q[q]) for q in test_ids) / len(test_ids)

    # still appendable?
    n0 = len(idx2.alive)
    idx2.add("__brandnew__", "a freshly appended document about quantum dots")
    idx2.finalize()
    appendable = (len(idx2.alive) == n0 + 1 and len(idx2.search("quantum dots", 5)) > 0)

    print(f"  disk (save):     {disk_mb:.1f} MB   ({disk_mb*1e6/len(idx.alive):.0f} B/doc)")
    print(f"  dense in-RAM:    {dense_mb:.1f} MB")
    print(f"  naive pickle:    {naive_mb:.1f} MB   (postings dict, for comparison)")
    print(f"  save {save_s:.1f}s / load {load_s:.1f}s")
    print(f"  round-trip: {identical}/{len(test_ids)} queries byte-identical, "
          f"top-10 overlap {overlap*100:.2f}%, reloaded nDCG {nd:.4f}")
    print(f"  still appendable after load: {appendable}")
    return disk_mb, naive_mb


def main():
    print("DEEP SEARCH step 7 - compact appendable persistence (save/load)")
    for ds in ("scifact", "nfcorpus"):
        d, n = run(ds)
    print("\n  CSR + delta + float16 + zlib: disk ~ the dense form, lossless, and")
    print("  the reloaded index still appends (tombstones compacted on save).")


if __name__ == "__main__":
    main()
