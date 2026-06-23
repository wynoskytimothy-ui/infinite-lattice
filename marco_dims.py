#!/usr/bin/env python3
"""Test "billions of free dimensions": higher-order (3,4,5-way) intersections as extra dimensions/axes.
Free DIMENSIONS (formula-computed) -- yes. Free INFORMATION -- no: information is bounded by the corpus
co-occurrence statistics, not the dimension count (a formula injects zero new info). Measurable consequence:
as intersection order k rises, the meet SUPPORT collapses and gold RECALL collapses -- higher-order axes
become sparse SINGLETONS (one doc = a memorized key, not a generalizable dimension). Measured on MARCO.
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO

N = 800


def main():
    idx = FullIndex()
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    random.Random(0).shuffle(queries); sample = queries[:N]

    sup = {1: [], 2: [], 3: [], 4: []}; rec = {1: 0, 2: 0, 3: 0, 4: 0}
    cnt = {1: 0, 2: 0, 3: 0, 4: 0}; single = {2: 0, 3: 0, 4: 0}
    for qid, qt in sample:
        ts = [w for w in set(stoks(qt)) if w in idx.tid and idx.idf_of(w) >= 2.0]
        ts = sorted(ts, key=lambda w: -idx.idf_of(w))[:4]      # rarest-first, up to 4
        if not ts:
            continue
        gold = qrels[qid]; meet = None
        for k, w in enumerate(ts, 1):
            i = idx.tid[w]; s, e = int(idx.ptr[i]), int(idx.ptr[i + 1]); ps = idx.di[s:e]
            meet = ps if meet is None else np.intersect1d(meet, ps, assume_unique=True)
            sup[k].append(len(meet)); cnt[k] += 1
            if any(int(g) in meet for g in gold):
                rec[k] += 1
            if k >= 2 and len(meet) <= 1:
                single[k] += 1
    print(f"\n  k-WAY INTERSECTION -- dimensions are free, is INFORMATION? MARCO 8.8M, n={cnt[1]}\n")
    print(f"  {'order k':<9}{'#queries':>10}{'median support':>16}{'gold recall':>14}{'singletons':>13}")
    for k in (1, 2, 3, 4):
        if cnt[k] == 0:
            continue
        med = int(np.median(sup[k])); r = rec[k] / cnt[k] * 100
        sg = (single[k] / cnt[k] * 100) if k >= 2 else 0.0
        print(f"  {k:<9}{cnt[k]:>10}{med:>16,}{r:>13.1f}%{sg:>12.1f}%")
    print(f"\n  support collapses (each higher order = far fewer docs), gold recall collapses (the gold lacks all")
    print(f"  k terms), singletons rise (one-doc axes = memorized keys, no generalization). Free DIMENSIONS,")
    print(f"  but INFORMATION saturates at low order -- bounded by the data, not the dimension count.")


if __name__ == "__main__":
    main()
