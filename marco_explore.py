#!/usr/bin/env python3
"""White-box exploration: on the CURRENT FAILURES (gold not in top-10), compare the gold doc
to the wrong doc we picked, across a battery of signals. The signal that most often favors the
gold over the wrong doc = the next rule with the most headroom. Tests: exact-count, rare-count,
PHRASE adjacency, PROXIMITY window, COMPANY, SUBWORD/prefix, PLURAL, ORDER. Found by data.
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


def raw(s):
    return WORD.findall(s.lower())


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


def sigvals(qstems, qseq, qrare, qcomp_terms, doc_pid):
    dseq = stoks(texts.get(doc_pid, ""))
    dset = set(dseq)
    draw = raw(texts.get(doc_pid, ""))
    dbigr = set(zip(dseq, dseq[1:]))
    exact = len(qstems & dset)
    rare = len(qrare & dset)
    phrase = sum(1 for bg in zip(qseq, qseq[1:]) if bg in dbigr)
    # proximity: smallest window containing the 2 rarest query stems
    pos = {w: [i for i, x in enumerate(dseq) if x == w] for w in qrare if w in dset}
    prox = 0.0
    rr = sorted(qrare, key=lambda w: -idx.idf.get(w, 0))[:2]
    if len(rr) == 2 and all(w in pos for w in rr):
        best = min(abs(a - b) for a in pos[rr[0]] for b in pos[rr[1]])
        prox = 1.0 / (1 + best)
    company = sum(w for dt, w in qcomp_terms if dt in dset)
    subword = sum(1 for w in qrare if len(w) >= 5 and any(len(x) >= 5 and x != w and x[:5] == w[:5] for x in dset))
    plural = sum(1 for w in qstems if (w + "s") in dset or (w[:-1] in dset if w.endswith("s") else False))
    order = 1.0 if any(" ".join(qseq[i:i+2]) in " ".join(draw) for i in range(len(qseq)-1)) else 0.0
    return {"exact": exact, "rare": rare, "phrase": phrase, "proximity": prox,
            "company": company, "subword": subword, "plural": plural, "order": order}


SIGS = ["exact", "rare", "phrase", "proximity", "company", "subword", "plural", "order"]
favor = Counter(); tie = Counter(); nfail = 0
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
        continue                                          # only FAILURES
    gp = next((p for p in rel if p in ranked), None)
    if gp is None:
        continue                                          # gold not even in pool -> encyclopedia case
    nfail += 1
    wrong = ranked[0]
    qseq = stoks(queries[q])
    qcomp = [(dt, w) for qt in qstems for dt, w in gold.get(qt, [])]
    gv = sigvals(qstems, qseq, qrare, qcomp, gp)
    wv = sigvals(qstems, qseq, qrare, qcomp, wrong)
    for s in SIGS:
        if gv[s] > wv[s]:
            favor[s] += 1
        elif gv[s] == wv[s]:
            tie[s] += 1

print(f"WHITE-BOX EXPLORATION -- {nfail} failures (gold in pool, not in top-10)\n")
print(f"  which signal favors the GOLD over the wrong doc we picked?\n")
print(f"   {'signal':>12}{'favors gold':>13}{'tie':>8}{'favors wrong':>14}")
for s in sorted(SIGS, key=lambda s: -favor[s]):
    fw = nfail - favor[s] - tie[s]
    print(f"   {s:>12}{favor[s]:>11}({favor[s]/nfail:.0%}){tie[s]:>8}{fw:>12}({fw/nfail:.0%})")
print("\n  high 'favors gold' = a rule on that signal would pull the gold over the wrong doc.")
