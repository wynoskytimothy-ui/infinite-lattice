#!/usr/bin/env python3
"""The integrated engine on VERIFIED-clean components: conservative-stem anchor (rung 2,
0.5777) + the audited corridor (rung 4), ANCHOR-DOMINANT. Anchor leads the ranking; the
corridor only expands candidates + nudges (small lam), so it rescues the miss-set without
demoting the easy wins. Measured vs the stem floor with the hit/miss split -- verify it
ADDS, doesn't drift (the honest re-run of the corridor, now on a clean base + dominant anchor).
"""
import re, time
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
IDF_GATE, TOP, MIN_CO = 4.0, 10, 3


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)
print(f"  stem anchor index: {len(idx.post)} terms ({idx.build_s:.0f}s)", flush=True)

t0 = time.perf_counter()
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
print(f"  corridor: {len(corridor)} anchors ({time.perf_counter()-t0:.0f}s)\n", flush=True)


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def search(query, lam, k=100, min_idf=0.3):
    qs = set(stoks(query))
    score = defaultdict(float)
    for w in qs:                                   # anchor: exact stem match (dominant)
        idf = idx.idf.get(w)
        if idf is None or idf < min_idf:
            continue
        for di, c in idx.post[w]:
            score[di] += idf * bm(di, c)
    if lam > 0:                                    # corridor: fill gaps (small weight)
        for w in qs:
            for ct, wt in corridor.get(w, []):
                idfc = idx.idf.get(ct, 0.0)
                if ct in idx.post:
                    for di, c in idx.post[ct]:
                        score[di] += lam * wt * idfc * bm(di, c)
    top = sorted(score, key=score.get, reverse=True)[:k]
    return [idx.docids[di] for di in top]


def rr10(ranked, rel):
    for i, pid in enumerate(ranked[:10], 1):
        if pid in rel:
            return 1.0 / i
    return 0.0


ref = None
print("INTEGRATED (stem anchor + audited corridor; floor stem = 0.5777):")
print(f"   {'lam':>8}{'MRR@10':>9}{'hit-set':>10}{'miss-set':>10}")
for lam in (0.0, 0.1, 0.3, 0.6):
    mrr = 0.0; rrs = {}
    for q in qids:
        r = qrels[q]
        rr = rr10(search(queries[q], lam), r)
        rrs[q] = rr; mrr += rr
    if ref is None:
        ref = rrs
    h = hs = m = ms = 0
    for q in qids:
        if ref[q] > 0:
            h += 1; hs += rrs[q]
        else:
            m += 1; ms += rrs[q]
    tag = "  (anchor only)" if lam == 0 else ""
    print(f"   {lam:>8}{mrr/len(qids):>9.4f}{hs/max(1,h):>10.4f}{ms/max(1,m):>10.4f}{tag}", flush=True)
print("\n  beat 0.5777 + miss-set up + hit-set held = the integrated engine works on clean parts.")
