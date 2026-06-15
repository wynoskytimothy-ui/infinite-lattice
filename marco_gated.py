#!/usr/bin/env python3
"""The SELECTIVE GATE -- the one wiring that structurally can't lose the hit-set. For each
query: anchor (exact stem BM25) ranks; measure anchor CONFIDENCE = top1 / query-idf-mass.
  confident query  -> anchor untouched (those 74% keep 0.7378)
  weak query       -> rerank anchor's top-200 with anchor-DOMINANT (beta 0.05) corridor
                      CONVERGENCE (your meet: #query-term reaches that land on the doc)
Sweep the gate threshold (0 = never rerank = floor; large = always). Fast: rerank computed
once/query, gate decisions are cheap. Goal: beat 0.5777 by rescuing miss WITHOUT touching hit.
"""
import re, time
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
IDF_GATE, TOP, MIN_CO, BETA = 4.0, 10, 3, 0.05


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)
cooc = defaultdict(Counter); freq = Counter()
for t in texts.values():
    rare = [w for w in set(stoks(t)) if idx.idf.get(w, 0) >= IDF_GATE]
    for a in rare:
        freq[a] += 1
    for a in rare:
        ca = cooc[a]
        for b in rare:
            if a != b:
                ca[b] += 1
corridor = {}
for a, cnt in cooc.items():
    fa = freq[a]
    sc = sorted(((b, (c / fa) * idx.idf.get(b, 0.0)) for b, c in cnt.items() if c >= MIN_CO),
                key=lambda x: -x[1])
    if sc:
        corridor[a] = sc[:TOP]
print(f"  index {len(idx.post)} terms, corridor {len(corridor)} anchors\n", flush=True)


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def rr10(ranked, rel):
    for i, pid in enumerate(ranked[:10], 1):
        if pid in rel:
            return 1.0 / i
    return 0.0


THRESH = (0.0, 0.15, 0.30, 0.50, 9.9)            # 0 = never rerank (floor); 9.9 ~ always
agg = {th: {"all": 0.0, "hit": 0.0, "miss": 0.0} for th in THRESH}
nhit = nmiss = 0
t0 = time.perf_counter()
for q in qids:
    rel = qrels[q]
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
    anchor = defaultdict(float)
    for w in qs:
        idfw = idx.idf[w]
        for di, c in idx.post[w]:
            anchor[di] += idfw * bm(di, c)
    cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
    anchor_rank = [idx.docids[di] for di in cands[:100]]
    rr_a = rr10(anchor_rank, rel)
    bucket = "hit" if rr_a > 0 else "miss"
    nhit += bucket == "hit"; nmiss += bucket == "miss"
    qmass = sum(idx.idf[w] for w in qs) or 1.0
    conf = (anchor[cands[0]] / qmass) if cands else 0.0
    qreach = [({w} | {ct for ct, _ in corridor.get(w, [])}) for w in qs]
    rer = {}
    for di in cands:
        dts = set(stoks(texts[idx.docids[di]]))
        conv = 0; cs = 0.0
        for s in qreach:
            hit = s & dts
            if hit:
                conv += 1
                cs += max(idx.idf.get(t, 0.0) for t in hit)
        rer[di] = anchor[di] + BETA * cs * conv
    rer_rank = [idx.docids[di] for di in sorted(rer, key=rer.get, reverse=True)[:100]]
    rr_r = rr10(rer_rank, rel)
    for th in THRESH:
        rr = rr_r if conf < th else rr_a              # gate: rerank only weak queries
        agg[th]["all"] += rr; agg[th][bucket] += rr
n = nhit + nmiss
print(f"  eval {n} queries in {time.perf_counter()-t0:.0f}s ({nhit} hit, {nmiss} miss)\n")
print(f"   {'gate<conf':>12}{'MRR@10':>9}{'hit-set':>10}{'miss-set':>10}{'reranked%':>11}")
for th in THRESH:
    tag = "  (floor)" if th == 0 else ("  (always)" if th == 9.9 else "")
    print(f"   {th:>12}{agg[th]['all']/n:>9.4f}{agg[th]['hit']/max(1,nhit):>10.4f}"
          f"{agg[th]['miss']/max(1,nmiss):>10.4f}{tag}")
print("\n  a threshold that beats 0.5777 with hit-set ~0.7378 held = selective gate WINS.")
