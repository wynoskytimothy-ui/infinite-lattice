#!/usr/bin/env python3
"""MS MARCO step 2b -- WHERE do the correlations help? The additive bridges hurt on the
easy pool (BM25 already right -> reranking only demotes). Diagnose honestly:
  1. INSPECT the learned bridges (do they narrow sensibly, or are they garbage?)
  2. RRF fusion (rank-based) instead of additive (scale-uncalibrated)
  3. SPLIT by BM25-hit vs BM25-miss -- bridges should earn their keep on the MISS set
     (relevant doc ranked 11-100 by BM25; can bridges promote it into the top-10?)
"""
import sys
from collections import defaultdict
from marco_baseline import tok, load_dev, build_pool, load_texts, BM25
from marco_bridges import search_scored, learn_bridges


def per_query_rows(bm, texts, query, bridges, cand=100):
    cands = search_scored(bm, query, cand)
    qbr = [w for w in set(tok(query)) if w in bridges]
    rows = []
    for di, base in cands:
        dts = set(tok(texts[bm.docids[di]]))
        bonus = sum(w for qt in qbr for dt, w in bridges[qt] if dt in dts)
        rows.append((di, base, bonus))
    return rows


def rr_at10(order_dis, bm, rel):
    for i, di in enumerate(order_dis[:10], 1):
        if bm.docids[di] in rel:
            return 1.0 / i
    return 0.0


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    mtq = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
    qrels, queries = load_dev()
    qids, pool, rel_pids = build_pool(qrels, queries, nq, nd)
    texts = load_texts(pool)
    qids = [q for q in qids if all(p in texts for p in qrels[q])]
    bm = BM25(); bm.index(list(texts.items()))
    bridges = learn_bridges(bm.idf, mtq)

    # 1. INSPECT -- what did it learn to narrow toward?
    print("\n1. LEARNED BRIDGES (query-term -> top narrowing doc-terms, by weight):")
    for t in ["biomaterials", "nanofiber", "nanowire", "extracellular", "insulin",
              "metformin", "diabetes", "vaccine", "quantum"]:
        b = bridges.get(t)
        if b:
            print(f"   {t:>14} -> " + ", ".join(f"{dt}({w:.1f})" for dt, w in b[:7]))
        else:
            print(f"   {t:>14} -> (no bridge learned)")

    # 2+3. FUSION x SPLIT
    K = 60
    agg = {m: {"hit": 0.0, "miss": 0.0, "all": 0.0} for m in ("bm25", "rrf", "add0.3")}
    nhit = nmiss = 0
    for q in qids:
        rel = qrels[q]
        rows = per_query_rows(bm, texts, queries[q], bridges)
        if not rows:
            continue
        bm_order = [r[0] for r in sorted(rows, key=lambda r: -r[1])]
        br_order = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
        # RRF over the two rankings
        rrf = defaultdict(float)
        for rank, di in enumerate(bm_order):
            rrf[di] += 1.0 / (K + rank + 1)
        for rank, di in enumerate(br_order):
            rrf[di] += 1.0 / (K + rank + 1)
        rrf_order = sorted(rrf, key=rrf.get, reverse=True)
        add_order = [r[0] for r in sorted(rows, key=lambda r: -(r[1] + 0.3 * r[2]))]

        rr_bm = rr_at10(bm_order, bm, rel)
        bucket = "hit" if rr_bm > 0 else "miss"
        if bucket == "hit":
            nhit += 1
        else:
            nmiss += 1
        for m, order in (("bm25", bm_order), ("rrf", rrf_order), ("add0.3", add_order)):
            rr = rr_at10(order, bm, rel)
            agg[m][bucket] += rr
            agg[m]["all"] += rr

    n = nhit + nmiss
    print(f"\n2/3. FUSION x SPLIT  (MRR@10; {nhit} BM25-hit, {nmiss} BM25-miss of {n})")
    print(f"   {'method':>10}{'ALL':>9}{'hit-set':>10}{'miss-set':>10}")
    for m in ("bm25", "rrf", "add0.3"):
        print(f"   {m:>10}{agg[m]['all']/n:>9.4f}{agg[m]['hit']/max(1,nhit):>10.4f}"
              f"{agg[m]['miss']/max(1,nmiss):>10.4f}")
    print(f"\n   miss-set = BM25 missed top-10 (relevant at rank 11-100). ANY method > 0 there")
    print(f"   = bridges RESCUED queries lexical alone failed -- the narrowing that matters.")


if __name__ == "__main__":
    main()
