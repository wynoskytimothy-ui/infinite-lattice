#!/usr/bin/env python3
"""The query-level MEET: ALL query terms (stems) cascade together; a doc scores by how many
distinct query-term corridors CONVERGE on it (super-linear ^alpha). Convergence IS the
selectivity gate -- right doc is hit by every term (stays top, hit-set safe); a topical wrong
doc catches only one term's corridor (suppressed, no drift); hard query rescued where multiple
corridors still converge. Measured vs the anchor floor (0.5777) with the hit/miss split.
"""
import re, math, time
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
IDF_GATE, TOP, MIN_CO, BETA = 4.0, 10, 3, 0.3


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
    sc = [(b, (c / fa) * idx.idf.get(b, 0.0)) for b, c in cnt.items() if c >= MIN_CO]
    sc.sort(key=lambda x: -x[1])
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


ALPHAS = (0.0, 0.5, 1.0, 2.0)
agg = {a: {"all": 0.0, "hit": 0.0, "miss": 0.0} for a in ("anchor",) + ALPHAS}
nhit = nmiss = 0
t0 = time.perf_counter()
for q in qids:
    rel = qrels[q]
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
    base = defaultdict(float)
    reached = defaultdict(set)
    base_ex = defaultdict(float)
    for qi, w in enumerate(qs):
        idfw = idx.idf[w]
        for di, c in idx.post[w]:
            v = idfw * bm(di, c)
            base[di] += v; base_ex[di] += v; reached[di].add(qi)
        for ct, wt in corridor.get(w, []):
            if ct in idx.post:
                idfc = idx.idf.get(ct, 0.0)
                for di, c in idx.post[ct]:
                    base[di] += BETA * wt * idfc * bm(di, c); reached[di].add(qi)
    anchor_rank = [idx.docids[di] for di in sorted(base_ex, key=base_ex.get, reverse=True)[:100]]
    rr_a = rr10(anchor_rank, rel)
    bucket = "hit" if rr_a > 0 else "miss"
    nhit += bucket == "hit"; nmiss += bucket == "miss"
    agg["anchor"]["all"] += rr_a; agg["anchor"][bucket] += rr_a
    for a in ALPHAS:
        final = {di: base[di] * (len(reached[di]) ** a) for di in base}
        mrank = [idx.docids[di] for di in sorted(final, key=final.get, reverse=True)[:100]]
        rr = rr10(mrank, rel)
        agg[a]["all"] += rr; agg[a][bucket] += rr
n = nhit + nmiss
print(f"  eval {n} queries in {time.perf_counter()-t0:.0f}s ({nhit} hit, {nmiss} miss)\n")
print(f"   {'method':>18}{'MRR@10':>9}{'hit-set':>10}{'miss-set':>10}")
print(f"   {'anchor floor':>18}{agg['anchor']['all']/n:>9.4f}{agg['anchor']['hit']/max(1,nhit):>10.4f}{agg['anchor']['miss']/max(1,nmiss):>10.4f}")
for a in ALPHAS:
    tag = "  <- convergence" if a > 0 else "  (additive, no meet)"
    print(f"   {'meet a='+str(a):>18}{agg[a]['all']/n:>9.4f}{agg[a]['hit']/max(1,nhit):>10.4f}{agg[a]['miss']/max(1,nmiss):>10.4f}{tag}")
print("\n  meet a>0 beats anchor + holds hit-set + lifts miss = the query-level meet narrows it.")
