#!/usr/bin/env python3
"""Cascade reranker. The gold is in the top-100 ~91% of the time -- so rerank the pool by the
discrimination cascade: reward docs SUPER-LINEARLY for matching multiple RARE query terms
(1 rare->~76 docs, 2->~12, 3->~1), with the entity anchor + gold-doc company. The answer is
in the pool; pull it to the top. Measured vs the 0.5898 (anchor+company+entity) baseline.
"""
import re, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
RARE = 4.0
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


def run(alpha, gamma):
    mrr = 0.0; h = hs = m = ms = 0.0
    nh = nm = 0; r100 = 0
    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
        if not qs:
            nm += 1; continue
        rare = [w for w in qs if idx.idf.get(w, 0) >= RARE]
        entity = max(qs, key=lambda w: idx.idf.get(w, 0))
        if idx.idf.get(entity, 0) < ENTITY_IDF:
            entity = None
        anchor = defaultdict(float)
        for w in qs:
            for di, c in idx.post[w]:
                anchor[di] += idx.idf[w] * bm(di, c)
        cands = sorted(anchor, key=anchor.get, reverse=True)[:100]
        cand_pids = [idx.docids[di] for di in cands]
        if any(p in rel for p in cand_pids):
            r100 += 1
        final = {}
        for di in cands:
            dts = set(stoks(texts[idx.docids[di]]))
            k = sum(1 for w in rare if w in dts)
            xover = sum(idx.idf[w] for w in rare if w in dts) * (k ** gamma if k else 0)
            comp = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
            if entity is not None and entity not in dts:
                comp = 0.0
            final[di] = anchor[di] + alpha * xover + 0.3 * comp

        def rr():
            for i, di in enumerate(sorted(final, key=final.get, reverse=True)[:10], 1):
                if idx.docids[di] in rel:
                    return 1.0 / i
            return 0.0
        ra = 0.0
        for i, di in enumerate(cands[:10], 1):
            if idx.docids[di] in rel:
                ra = 1.0 / i; break
        v = rr(); mrr += v
        if ra > 0:
            nh += 1; hs += v
        else:
            nm += 1; ms += v
    n = len(qids)
    return mrr / n, hs / max(1, nh), ms / max(1, nm), r100 / n


print(f"CASCADE RERANK (rare crossover ^gamma + entity + company) -- baseline anchor+comp+entity 0.5898\n")
print(f"   {'alpha,gamma':>14}{'MRR@10':>9}{'hit':>9}{'miss':>9}{'R@100':>9}")
for a, g in [(0.0, 1.0), (0.5, 1.5), (1.0, 1.5), (1.0, 2.0), (2.0, 2.0)]:
    r = run(a, g)
    tag = "  (=anchor+comp+entity)" if a == 0 else ""
    print(f"   {f'{a},{g}':>14}{r[0]:>9.4f}{r[1]:>9.4f}{r[2]:>9.4f}{r[3]:>9.4f}{tag}", flush=True)
print("\n  beat 0.5898 = the rare-crossover cascade pulls the in-pool gold up. R@100 = the ceiling.")
