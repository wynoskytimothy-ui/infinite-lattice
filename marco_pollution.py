#!/usr/bin/env python3
"""Profile the POLLUTANT docs -- the wrong rank-1 docs that beat the gold on failures -- vs the
gold docs, to find the goblin signature. Metrics: length, query-DENSITY (matches/length, focus),
DISTINCT ratio (list/glossary vs focused), REPETITION, entity presence. Where wrong systematically
differs from gold = a rule to demote the pollutant.
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


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def profile(pid, qstems, qrare, entity):
    d = stoks(texts.get(pid, "")); dl = max(1, len(d)); ds = set(d)
    return {"length": dl, "qmatch": len(qstems & ds), "density": len(qstems & ds) / dl * 100,
            "distinct": len(ds) / dl * 100, "repeat": (1 - len(ds) / dl) * 100,
            "entity": 100 if (entity is None or entity in ds) else 0, "rare": len(qrare & ds)}


def med(xs):
    s = sorted(xs); return s[len(s) // 2] if s else 0


METRICS = ["length", "qmatch", "density", "distinct", "repeat", "rare", "entity"]
W = defaultdict(list); G = defaultdict(list); nf = 0
for q in qids:
    rel = qrels[q]
    qstems = set(w for w in stoks(queries[q]) if idx.idf.get(w, 0) >= 0.3)
    qrare = set(w for w in qstems if idx.idf.get(w, 0) >= RARE)
    if not qstems:
        continue
    entity = max(qstems, key=lambda w: idx.idf.get(w, 0))
    if idx.idf.get(entity, 0) < ENTITY_IDF:
        entity = None
    anchor = defaultdict(float)
    for w in qstems:
        for di, c in idx.post[w]:
            anchor[di] += idx.idf[w] * bm(di, c)
    cands = sorted(anchor, key=anchor.get, reverse=True)[:100]
    final = {}
    for di in cands:
        dts = set(stoks(texts[idx.docids[di]]))
        k = sum(1 for w in qrare if w in dts)
        xo = sum(idx.idf[w] for w in qrare if w in dts) * (k ** 1.5 if k else 0)
        comp = sum(w for qt in qstems for dt, w in gold.get(qt, []) if dt in dts)
        if entity is not None and entity not in dts:
            comp = 0.0
        final[di] = anchor[di] + 0.5 * xo + 0.3 * comp
    ranked = [idx.docids[di] for di in sorted(final, key=final.get, reverse=True)]
    if any(p in rel for p in ranked[:10]):
        continue
    gp = next((p for p in rel if p in ranked), None)
    if gp is None:
        continue
    nf += 1
    wp = profile(ranked[0], qstems, qrare, entity)
    gpp = profile(gp, qstems, qrare, entity)
    for m in METRICS:
        W[m].append(wp[m]); G[m].append(gpp[m])

print(f"POLLUTANT PROFILE -- {nf} failures, wrong rank-1 doc vs gold doc\n")
print(f"   {'metric':>10}{'WRONG (median)':>16}{'GOLD (median)':>15}{'goblin?':>10}")
for m in METRICS:
    w, g = med(W[m]), med(G[m])
    flag = ""
    if m == "length" and w > g * 1.3:
        flag = "<- longer"
    if m == "density" and g > w * 1.3:
        flag = "<- gold denser"
    if m == "distinct" and w > g * 1.15:
        flag = "<- list-like"
    print(f"   {m:>10}{w:>16.1f}{g:>15.1f}{flag:>12}")
print("\n  wrong longer / lower-density / higher-distinct = pollutant matches by VOLUME not focus")
print("  -> a length/density penalty would demote the goblins and lift the focused gold.")
