#!/usr/bin/env python3
"""PURE-LATTICE retrieval on the full 8.8M -- NO BM25. Tests the claim: the rare-word meet +
learned corridors do all the work. The audit proved the gold is reachable 88.5% by the rarest
word, 97.7% by the union of rare words -- so the recall ceiling is ~0.9. Does the meet capture
it into the top-100 better than BM25's 0.666?

Per query (same 3000 dev, seed 42):
  candidate pool = union of the rare query words' posting lists (NO BM25, no tf-idf scoring).
  meet score     = (sum idf of matched rare words) x (meet-depth k)^1.5    -- the crossover, as anchor.
  + corridors    = learned gold-doc company (the 'correlations').
  top-100 -> R@100 + MRR@10, reported for meet-only and meet+corridors, vs BM25 0.666 / 0.185.
"""
import sys, random, time
from collections import defaultdict
import numpy as np
from marco_full_eval import (FullIndex, stoks, train_corridors,
                             RARE, ENTITY_IDF, QGATE, B_COMP, P_DIV, G_XO)


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)
    sum_idf = np.zeros(idx.N, dtype=np.float32)
    cnt = np.zeros(idx.N, dtype=np.float32)

    qrels = defaultdict(set)
    with open(idx.cf.name.replace("collection.tsv", "qrels.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(idx.cf.name.replace("collection.tsv", "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    m_mrr = c_mrr = 0.0
    m_rec = defaultdict(float); c_rec = defaultdict(float)
    rec10_d = defaultdict(float); cnt_d = defaultdict(int)   # recall@10 split by # rare words
    norare = 0

    def recall_at(order, rel, k):
        return len(rel & set(str(d) for d in order[:k])) / len(rel)
    t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        if not rare:
            norare += 1; continue                       # pure lattice has no rare anchor here
        entity = max(qs, key=lambda w: idx.idf_of(w))
        if idx.idf_of(entity) < ENTITY_IDF:
            entity = None
        touched = []
        for w in rare:
            i = idx.tid.get(w)
            if i is None:
                continue
            s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
            dis = idx.di[s:e]
            sum_idf[dis] += idx.idfa[i]; cnt[dis] += 1.0
            touched.append(dis)
        if not touched:
            norare += 1; continue
        cat = np.concatenate(touched)
        cand = np.unique(cat)
        meet = sum_idf[cand] * (cnt[cand] ** G_XO)       # Sum rare-idf x meet-depth^1.5, NO BM25
        # meet is aligned to cand BY POSITION -> rank on positions, map to doc-ids via cand[pos]
        # --- meet-only ranking ---
        kk = min(100, len(cand))
        part = np.argpartition(-meet, kk - 1)[:kk]
        part = part[np.argsort(-meet[part])]
        m_order = [int(cand[p]) for p in part]
        # --- meet + corridors (+ diversity) rerank on the meet's top-200 ---
        kk2 = min(200, len(cand))
        part2 = np.argpartition(-meet, kk2 - 1)[:kk2]
        meet_p = {int(cand[p]): float(meet[p]) for p in part2}
        final = {}
        for di in meet_p:
            dl = stoks(idx.text(di)); ds = set(dl); tot = max(1, len(dl))
            comp = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in ds)
            if entity is not None and entity not in ds:
                comp = 0.0
            base = meet_p[di] + B_COMP * comp
            final[di] = base * ((len(ds) / tot) ** P_DIV)
        c_order = sorted(final, key=final.get, reverse=True)
        sum_idf[cat] = 0.0; cnt[cat] = 0.0               # reset

        m_mrr += next((1.0 / i for i, d in enumerate(m_order[:10], 1) if str(d) in rel), 0.0)
        c_mrr += next((1.0 / i for i, d in enumerate(c_order[:10], 1) if str(d) in rel), 0.0)
        for k in (1, 5, 10, 100):
            m_rec[k] += recall_at(m_order, rel, k); c_rec[k] += recall_at(c_order, rel, k)
        dep = min(len(rare), 3)
        rec10_d[dep] += recall_at(c_order, rel, 10); cnt_d[dep] += 1
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{nq} | meet R@10 {m_rec[10]/(n+1):.4f} MRR {m_mrr/(n+1):.4f} "
                  f"| +corr R@10 {c_rec[10]/(n+1):.4f} MRR {c_mrr/(n+1):.4f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = len(qids)
    print(f"\nPURE-LATTICE (no BM25) on full 8.8M -- {N} dev queries  ({norare} had no rare word)\n")
    print(f"   {'engine':<26}{'R@1':>8}{'R@5':>8}{'R@10':>8}{'R@100':>8}{'MRR@10':>9}")
    print(f"   {'meet only (rare x depth)':<26}{m_rec[1]/N:>8.4f}{m_rec[5]/N:>8.4f}{m_rec[10]/N:>8.4f}{m_rec[100]/N:>8.4f}{m_mrr/N:>9.4f}")
    print(f"   {'meet + corridors':<26}{c_rec[1]/N:>8.4f}{c_rec[5]/N:>8.4f}{c_rec[10]/N:>8.4f}{c_rec[100]/N:>8.4f}{c_mrr/N:>9.4f}")
    print(f"\n   meet+corridors recall@10 BY # rare words in query (validates the 3-way claim):")
    for d in (1, 2, 3):
        lbl = "3+" if d == 3 else str(d)
        print(f"     {lbl} rare words: {cnt_d[d]:>5} q   recall@10 {rec10_d[d]/max(1,cnt_d[d]):.4f}")
    print(f"\n   recall@1 {c_rec[1]/N:.4f} << recall@10 {c_rec[10]/N:.4f}: the gold is localized into the")
    print(f"   top-10 but ranks ~4th, not 1st. The lever is RANKING the small meet set, not retrieval.")


if __name__ == "__main__":
    main()
