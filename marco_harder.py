#!/usr/bin/env python3
"""MS MARCO step 2c -- does the correlation lift GROW as the pool hardens? On the easy
300k pool, calibrated bridges (lam=0.3) gave +0.0077 MRR (small, little headroom: 74%
already hit). The lever's value should rise with BM25's miss-rate. Same queries, same
(fixed) bridges, sweep the distractor count -> harder pools -> bigger miss-set -> test if
the bridge lift grows. Projects toward full-collection value.
"""
import sys, gc
from marco_baseline import load_dev, build_pool, load_texts, BM25
from marco_bridges import learn_bridges
from marco_bridges2 import per_query_rows, rr_at10


def eval_pool(bm, texts, qids, queries, qrels, bridges, lam=0.3):
    fmrr = bmrr = 0.0
    miss = 0
    for q in qids:
        rel = qrels[q]
        rows = per_query_rows(bm, texts, queries[q], bridges)
        bm_order = [r[0] for r in sorted(rows, key=lambda r: -r[1])]
        add_order = [r[0] for r in sorted(rows, key=lambda r: -(r[1] + lam * r[2]))]
        rb = rr_at10(bm_order, bm, rel)
        fmrr += rb
        bmrr += rr_at10(add_order, bm, rel)
        if rb == 0:
            miss += 1
    n = len(qids)
    return fmrr / n, bmrr / n, miss / n


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    mtq = int(sys.argv[2]) if len(sys.argv) > 2 else 150_000
    pools = [300_000, 1_000_000, 2_500_000]
    qrels, queries = load_dev()
    bridges = None
    print(f"MS MARCO -- does the correlation lift grow with pool hardness? "
          f"({nq} queries, fixed bridges, lam=0.3)\n", flush=True)
    print(f"   {'pool':>10}{'BM25 miss%':>12}{'BM25 MRR':>11}{'+bridges':>11}{'lift':>10}", flush=True)
    for nd in pools:
        qids, pool, rel = build_pool(qrels, queries, nq, nd)
        texts = load_texts(pool)
        qids = [q for q in qids if all(p in texts for p in qrels[q])]
        bm = BM25(); bm.index(list(texts.items()))
        if bridges is None:
            bridges = learn_bridges(bm.idf, mtq)
        floor, bridge, missrate = eval_pool(bm, texts, qids, queries, qrels, bridges)
        print(f"   {len(pool):>10}{missrate:>11.1%}{floor:>11.4f}{bridge:>11.4f}"
              f"{bridge-floor:>+10.4f}", flush=True)
        del bm, texts, pool
        gc.collect()
    print(f"\n   miss% rises with pool size (more distractors -> lexical fails more).")
    print(f"   if 'lift' grows with miss% -> correlations matter MORE at scale -> projects up"
          f" toward full-collection value. that is the whole thesis, measured.")


if __name__ == "__main__":
    main()
