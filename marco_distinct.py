#!/usr/bin/env python3
"""The diversity/anti-repetition rule -- the small demotion lever the pollutant profile found
(goblins are longer + more repetitive, distinct-vocab 67% vs the gold's 74%). Reward focused,
diverse docs over repetitive boilerplate: cascade_score x distinct_ratio^p. Measured vs 0.6004.
"""
import re, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
RARE, ENTITY_IDF = 4.0, 5.5


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)
qrels_tr = defaultdict(set)
with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) >= 4 and int(p[3]) > 0:
            qrels_tr[p[0]].add(p[2])
sel = list(qrels_tr); random.Random(42).shuffle(sel); sel = sel[:120_000]
sel_set = set(sel)
gtexts = load_texts({pid for q in sel for pid in qrels_tr[q]})
qtexts = {}
with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
    for line in f:
        a = line.rstrip("\n").split("\t", 1)
        if len(a) == 2 and a[0] in sel_set:
            qtexts[a[0]] = a[1]
cooc = defaultdict(Counter); npairs = Counter()
for q in sel:
    qts = [w for w in set(stoks(qtexts.get(q, ""))) if idx.idf.get(w, 0) >= 2.0]
    for pid in qrels_tr[q]:
        t = gtexts.get(pid)
        if not t:
            continue
        dts = [w for w in set(stoks(t)) if idx.idf.get(w, 0) >= 3.0]
        for qt in qts:
            npairs[qt] += 1; cooc[qt].update(dts)
gold = {}
for qt, cnt in cooc.items():
    n = npairs[qt]
    sc = sorted(((dt, (c / n) * idx.idf.get(dt, 0.0)) for dt, c in cnt.items()
                 if dt != qt and idx.idf.get(dt, 0.0) > 0 and c >= 2), key=lambda x: -x[1])
    if sc:
        gold[qt] = sc[:12]
distinct = {}
for di, pid in enumerate(idx.docids):
    d = stoks(texts[pid]); distinct[di] = len(set(d)) / max(1, len(d))


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def run(p):
    mrr = 0.0; hs = ms = 0.0; nh = nm = 0
    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
        if not qs:
            nm += 1; continue
        qrare = set(w for w in qs if idx.idf.get(w, 0) >= RARE)
        entity = max(qs, key=lambda w: idx.idf.get(w, 0))
        if idx.idf.get(entity, 0) < ENTITY_IDF:
            entity = None
        anchor = defaultdict(float)
        for w in qs:
            for di, c in idx.post[w]:
                anchor[di] += idx.idf[w] * bm(di, c)
        cands = sorted(anchor, key=anchor.get, reverse=True)[:100]
        final = {}
        for di in cands:
            dts = set(stoks(texts[idx.docids[di]]))
            k = sum(1 for w in qrare if w in dts)
            xo = sum(idx.idf[w] for w in qrare if w in dts) * (k ** 1.5 if k else 0)
            comp = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
            if entity is not None and entity not in dts:
                comp = 0.0
            base = anchor[di] + 0.5 * xo + 0.3 * comp
            final[di] = base * (distinct[di] ** p)
        ra = next((1.0 / i for i, di in enumerate(cands[:10], 1) if idx.docids[di] in rel), 0.0)
        v = next((1.0 / i for i, di in enumerate(sorted(final, key=final.get, reverse=True)[:10], 1)
                  if idx.docids[di] in rel), 0.0)
        mrr += v
        if ra > 0:
            nh += 1; hs += v
        else:
            nm += 1; ms += v
    n = len(qids)
    return mrr / n, hs / max(1, nh), ms / max(1, nm)


print("DIVERSITY rule (cascade x distinct_ratio^p) -- demote repetitive goblins -- baseline 0.6004\n")
print(f"   {'p':>6}{'MRR@10':>9}{'hit':>9}{'miss':>9}")
for p in (0.0, 0.25, 0.5, 1.0):
    r = run(p)
    tag = "  (=cascade baseline)" if p == 0 else ""
    print(f"   {p:>6}{r[0]:>9.4f}{r[1]:>9.4f}{r[2]:>9.4f}{tag}", flush=True)
print("\n  beat 0.6004 = the diversity penalty demotes the repetitive pollutants, as profiled.")
