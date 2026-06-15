#!/usr/bin/env python3
"""Zero-shot RULE #1: entity-as-constraint. The diagnostic showed the company firing on
wrong-entity docs (jupiter->moon lit a doc with 'moon' but not 'jupiter' -> a Pluto doc won).
Rule: the gold-doc company may only lift a doc that ALSO contains the query's rarest term
(the entity, idf>=5.5). Generic company can't substitute for the entity. General, no learning.
Measure held-out vs the no-gate gold-doc baseline (0.5822) + confirm Jupiter/Freon flip.
"""
import re, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
ENTITY_IDF = 5.5


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
sel = list(qrels_tr); random.Random(42).shuffle(sel); sel = sel[:150_000]
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


def full_rank(q, gate):
    """Full ranking for a single query (used for case checks)."""
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
    entity = max(qs, key=lambda w: idx.idf.get(w, 0), default=None)
    if entity is not None and idx.idf.get(entity, 0) < ENTITY_IDF:
        entity = None
    anchor = defaultdict(float)
    for w in qs:
        for di, c in idx.post[w]:
            anchor[di] += idx.idf[w] * bm(di, c)
    cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
    final = {}
    for di in cands:
        dts = set(stoks(texts[idx.docids[di]]))
        comp = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
        if gate and entity is not None and entity not in dts:
            comp = 0.0
        final[di] = anchor[di] + 0.3 * comp
    return [idx.docids[di] for di in sorted(final, key=final.get, reverse=True)]


def run():
    """ONE anchor computation per query; derive anchor-only, +company, +company-gated."""
    agg = {"base": {"all": 0.0, "hit": 0.0, "miss": 0.0},
           "gate": {"all": 0.0, "hit": 0.0, "miss": 0.0}}
    nh = nm = 0
    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
        if not qs:
            nm += 1; continue
        entity = max(qs, key=lambda w: idx.idf.get(w, 0))
        if idx.idf.get(entity, 0) < ENTITY_IDF:
            entity = None
        anchor = defaultdict(float)
        for w in qs:
            idfw = idx.idf[w]
            for di, c in idx.post[w]:
                anchor[di] += idfw * bm(di, c)
        cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
        comp = {}; comp_g = {}
        for di in cands:
            dts = set(stoks(texts[idx.docids[di]]))
            c = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
            comp[di] = c
            comp_g[di] = c if (entity is None or entity in dts) else 0.0

        def rr(scoref):
            for i, di in enumerate(sorted(cands, key=scoref, reverse=True)[:10], 1):
                if idx.docids[di] in rel:
                    return 1.0 / i
            return 0.0
        ra = rr(lambda di: anchor[di])
        bucket = "hit" if ra > 0 else "miss"
        if ra > 0:
            nh += 1
        else:
            nm += 1
        rb = rr(lambda di: anchor[di] + 0.3 * comp[di])
        rg = rr(lambda di: anchor[di] + 0.3 * comp_g[di])
        agg["base"]["all"] += rb; agg["base"][bucket] += rb
        agg["gate"]["all"] += rg; agg["gate"][bucket] += rg
        anchor.clear()
    return agg, nh, nm


print("ZERO-SHOT RULE: entity-as-constraint (company only on docs with the query's rarest term)\n")
agg, nh, nm = run()
n = len(qids)
print(f"   {'method':>22}{'MRR@10':>9}{'hit':>9}{'miss':>9}")
print(f"   {'gold-doc (no gate)':>22}{agg['base']['all']/n:>9.4f}{agg['base']['hit']/max(1,nh):>9.4f}{agg['base']['miss']/max(1,nm):>9.4f}")
print(f"   {'+ entity gate':>22}{agg['gate']['all']/n:>9.4f}{agg['gate']['hit']/max(1,nh):>9.4f}{agg['gate']['miss']/max(1,nm):>9.4f}   ({(agg['gate']['all']-agg['base']['all'])/n:+.4f})")

print("\n   case check (gold-doc rank, no-gate -> gated):")
for q in qids:
    if any(k in queries[q].lower() for k in ["jupiter to spin", "breathe freon", "gps satellites orbit"]):
        rel = qrels[q]; gp = next(iter(rel))
        r0 = full_rank(q, False); r1 = full_rank(q, True)
        rk0 = (r0.index(gp) + 1) if gp in r0 else 999
        rk1 = (r1.index(gp) + 1) if gp in r1 else 999
        print(f"      '{queries[q][:42]}'  rank {rk0} -> {rk1}")
