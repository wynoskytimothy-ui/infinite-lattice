#!/usr/bin/env python3
"""LEVER 3 -- carry the fast meet into MULTI-HOP. A corpus-scale 2-hop traversal = hop-1 retrieve +
hop-2 bridge meet (the anchor's rare terms -> docs sharing them). Both hops are meets, so the 18x
single-hop speedup should carry. Times the 2-hop pattern fast (rarest-address meet) vs full-scan on
the 8.8M index. Speed demo -- HotpotQA was the accuracy result; MARCO has no bridge qrels.
"""
import time, random
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, RARE
from marco_fast import bm25_fast

N = 200


def main():
    idx = FullIndex()
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries.append(a[1])
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    def rare_terms_of(pid):
        return [w for w in set(stoks(idx.text(int(pid)))) if idx.idf_of(w) >= RARE][:8]

    for qt in sample[:5]:                          # warm
        o, _ = bm25_fast(idx, stoks(qt), 100)
        rt = rare_terms_of(o[0])
        if rt:
            bm25_fast(idx, rt, 100)

    tfast, tfull = [], []
    for qt in sample:
        qts = stoks(qt)
        t0 = time.perf_counter()
        o1, _ = bm25_fast(idx, qts, 100)
        rt = rare_terms_of(o1[0])
        if rt:
            bm25_fast(idx, rt, 100)
        tfast.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        of1, _ = idx.bm25_top(qts, 100)
        rt2 = rare_terms_of(of1[0])
        if rt2:
            idx.bm25_top(rt2, 100)
        tfull.append((time.perf_counter() - t0) * 1000)
    tfast = np.array(tfast); tfull = np.array(tfull)
    print(f"\n  CORPUS-SCALE 2-HOP TRAVERSAL (hop-1 retrieve + hop-2 bridge meet), 8.8M docs, n={N}\n")
    print(f"  {'pipeline':<22}{'median ms':>12}{'mean':>9}{'p90':>9}{'max':>9}")
    print(f"  {'full scan 2-hop':<22}{np.median(tfull):>12.1f}{tfull.mean():>9.1f}{np.percentile(tfull,90):>9.1f}{tfull.max():>9.0f}")
    print(f"  {'fast meet 2-hop':<22}{np.median(tfast):>12.1f}{tfast.mean():>9.1f}{np.percentile(tfast,90):>9.1f}{tfast.max():>9.0f}")
    print(f"\n  speedup: {np.median(tfull)/np.median(tfast):.1f}x median -- the single-hop meet speedup carries into multi-hop")


if __name__ == "__main__":
    main()
