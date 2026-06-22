#!/usr/bin/env python3
"""Decisive test of the user's thesis: does PRECISE compression-link pool expansion lift the HYBRID MRR?
Baseline: fast top-100 -> cross-encoder. Expanded: top-100 + precise placement-neighbors (docs sharing a
top hit's RARE rarest term, TIGHT clusters only, df<=DF_NEIGH) -> CE. If the recovered gold outranks the
added noise, MRR rises; if noise dominates, it falls. Either way it answers "does recall recovery help".
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, RARE
from marco_fast import bm25_fast

N = 200
NEIGH_HITS = 10      # expand from the top-10 hits
DF_NEIGH = 400       # only pull a hit's rarest-term cluster if it's tight (<=400 docs)
POOL_CAP = 220


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
    random.Random(0).shuffle(queries)
    sample = queries[:N]

    from sentence_transformers import CrossEncoder
    import torch
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256,
                      device="cuda" if torch.cuda.is_available() else "cpu")

    def rarest_tid(pid):
        best = None
        for w in set(stoks(idx.text(int(pid)))):
            i = idx.tid.get(w)
            if i is None:
                continue
            v = float(idx.idfa[i])
            if best is None or v > best[0]:
                best = (v, i)
        return best

    def neighbors(base):
        add = set()
        for h in base[:NEIGH_HITS]:
            b = rarest_tid(h)
            if b is None or b[0] < RARE:
                continue
            ti = b[1]; s, e = int(idx.ptr[ti]), int(idx.ptr[ti + 1])
            if e - s <= DF_NEIGH:                 # tight, precise cluster only
                add.update(int(d) for d in idx.di[s:e])
        return add

    def rerank(pids, qt):
        texts = [idx.text(p) for p in pids]
        sc = ce.predict([(qt, t) for t in texts], batch_size=128, show_progress_bar=False)
        return [pids[i] for i in np.argsort(-sc)]

    for qid, qt in sample[:5]:
        o, _ = bm25_fast(idx, stoks(qt), 100); rerank([int(d) for d in o[:100]], qt)

    mB = mE = 0.0; rec = 0; psz = []
    for qid, qt in sample:
        o, _ = bm25_fast(idx, stoks(qt), 100); base = [int(d) for d in o[:100]]; gold = qrels[qid]
        rb = rerank(base, qt)
        for r, d in enumerate(rb[:10]):
            if d in gold:
                mB += 1 / (r + 1); break
        add = neighbors(base) - set(base)
        exp = (base + list(add))[:POOL_CAP]
        psz.append(len(exp))
        if any(g in add for g in gold) and not any(g in base for g in gold):
            rec += 1
        re_ = rerank(exp, qt)
        for r, d in enumerate(re_[:10]):
            if d in gold:
                mE += 1 / (r + 1); break
    print(f"\n  PLACEMENT-EXPANDED HYBRID -- does precise compression-link recall help the CE? n={N}\n")
    print(f"  {'pipeline':<32}{'med pool':>9}{'MRR@10':>10}")
    print(f"  {'baseline (top-100 -> CE)':<32}{'100':>9}{mB/N:>10.4f}")
    print(f"  {'+ placement neighbors -> CE':<32}{int(np.median(psz)):>9}{mE/N:>10.4f}")
    print(f"\n  gold newly recovered into pool: {rec}/{N}   MRR delta: {(mE-mB)/N:+.4f}")
    print(f"  (recall recovery helps the hybrid only if recovered gold outranks the added noise = answer-ness wall)")


if __name__ == "__main__":
    main()
