#!/usr/bin/env python3
"""LEVER 2 -- the user's PLACEMENT/ROTATION. Reassign doc-ids so docs sharing their rarest term get
adjacent numbers (a lattice-address ordering). Smaller doc-id gaps -> smaller compressed postings,
potentially BELOW the standard-compression entropy floor. Tests whether the lattice geometry buys
footprint that off-the-shelf IR ordering does not. Measures varbyte size natural vs reordered.
"""
import numpy as np
from marco_full_eval import FullIndex

CH = 25_000_000


def vb(g):
    b = np.ones(len(g), np.int64)
    for thr in (1 << 7, 1 << 14, 1 << 21, 1 << 28):
        b += (g >= thr)
    return int(b.sum())


def main():
    idx = FullIndex()
    ptr = idx.ptr; nterms = len(ptr) - 1; N = idx.N
    lens = np.diff(ptr).astype(np.int64)
    try:
        print("  rarest-term-per-doc (sort postings by doc asc, idf desc) ...", flush=True)
        term_post = np.repeat(np.arange(nterms, dtype=np.int32), lens)
        idf_milli = np.minimum((idx.idfa.astype(np.float64) * 1000), 65535).astype(np.int64)
        key = np.empty(len(idx.di), dtype=np.int64)
        for a in range(0, len(idx.di), CH):
            b = min(a + CH, len(idx.di))
            key[a:b] = (idx.di[a:b].astype(np.int64) << 16) | (65535 - idf_milli[term_post[a:b]])
        order = np.argsort(key); del key
        di_s = idx.di[order]; term_s = term_post[order]; del term_post, order
        uniq, first = np.unique(di_s, return_index=True); del di_s
        rarest = np.zeros(N, dtype=np.int32); rarest[uniq] = term_s[first]; del term_s
        print(f"  {len(uniq):,}/{N:,} docs have postings; assigning placement ...", flush=True)
        perm = np.argsort(rarest, kind="stable"); del rarest
        new_id = np.empty(N, dtype=np.int64); new_id[perm] = np.arange(N); del perm
    except MemoryError:
        print("  [MemoryError building the global reorder -- would need a doc-sample or more RAM]")
        return

    rng = np.random.default_rng(0)
    sample = rng.choice(nterms, size=8000, replace=False)
    nat = reo = 0; npost = 0
    for t in sample:
        s, e = int(ptr[t]), int(ptr[t + 1])
        if e - s < 2:
            continue
        p = idx.di[s:e]
        nat += vb(np.diff(p.astype(np.int64))) + 4
        q = np.sort(new_id[p])
        reo += vb(np.diff(q)) + 4
        npost += (e - s)
    print(f"\n  PLACEMENT/ROTATION -- lattice rarest-term doc reorder, varbyte on {len(sample)} terms ({npost:,} postings):")
    print(f"    natural order:   {nat/1e6:.2f} MB-equiv")
    print(f"    reordered:       {reo/1e6:.2f} MB-equiv   ({(1-reo/nat)*100:+.1f}% vs natural)")
    if reo < nat:
        print(f"  -> your placement shrinks postings {(1-reo/nat)*100:.1f}% BELOW natural order -- beats off-the-shelf compression")
    else:
        print(f"  -> rarest-term ordering did not help here; a stronger lattice ordering (2-key / BP) is the next try")


if __name__ == "__main__":
    main()
