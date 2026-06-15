#!/usr/bin/env python3
"""Glass-box diagnostic: 10 questions answered RIGHT, 10 answered WRONG. For each --
trigger words (rare query stems) + their LEARNED correlations (gold-doc company) + which
correlations FIRED (the semantic match). For failures: the doc it brought up vs the doc it
should have, and which correlations fired on each.
"""
import re, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)

# learn gold-doc corridors from train
qrels_tr = defaultdict(set)
with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) >= 4 and int(p[3]) > 0:
            qrels_tr[p[0]].add(p[2])
sel = list(qrels_tr); random.Random(42).shuffle(sel); sel = sel[:150_000]
sel_set = set(sel)
gold_pids = {pid for q in sel for pid in qrels_tr[q]}
gtexts = load_texts(gold_pids)
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
        gold[qt] = sc[:10]
print(f"# learned correlations for {len(gold)} query-terms\n")


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def snip(pid, n=140):
    t = texts.get(pid, "")[:n].replace("\n", " ")
    return t.encode("ascii", "ignore").decode("ascii") + "..."


def fired_on(qs, pid):
    dts = set(stoks(texts.get(pid, "")))
    out = []
    for qt in qs:
        hits = [dt for dt, _ in gold.get(qt, []) if dt in dts]
        if hits:
            out.append(f"{qt}->{'/'.join(hits[:4])}")
    return out


def analyze(q):
    rel = qrels[q]
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
    anchor = defaultdict(float)
    for w in qs:
        for di, c in idx.post[w]:
            anchor[di] += idx.idf[w] * bm(di, c)
    cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
    comp = {}
    for di in cands:
        dts = set(stoks(texts[idx.docids[di]]))
        comp[di] = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
    ranked = [idx.docids[di] for di in sorted(cands, key=lambda di: -(anchor[di] + 0.3 * comp[di]))]
    gpid = min(rel, key=lambda p: ranked.index(p) if p in ranked else 999)
    grank = (ranked.index(gpid) + 1) if gpid in ranked else 999
    return qs, ranked, gpid, grank


def print_triggers(qs):
    trig = sorted([(w, idx.idf[w]) for w in qs if idx.idf.get(w, 0) >= 2.0], key=lambda x: -x[1])
    for w, i in trig[:4]:
        comp = gold.get(w)
        rels = ", ".join(t for t, _ in comp[:6]) if comp else "(none learned)"
        print(f"     trigger '{w}' (idf {i:.1f})  ->  {rels}")


wins, fails = [], []
order = qids[:]; random.Random(1).shuffle(order)
for q in order:
    qs, ranked, gpid, grank = analyze(q)
    if grank == 1 and len(wins) < 10:
        wins.append((q, qs, ranked, gpid, grank))
    elif grank > 10 and len(fails) < 10:
        fails.append((q, qs, ranked, gpid, grank))
    if len(wins) >= 10 and len(fails) >= 10:
        break

print("=" * 70)
print("10 ANSWERED RIGHT (gold doc at rank 1)")
print("=" * 70)
for i, (q, qs, ranked, gpid, grank) in enumerate(wins, 1):
    print(f"\n[{i}] Q: {queries[q]}")
    print_triggers(qs)
    print(f"     correlations that FIRED on the right doc: {fired_on(qs, gpid) or '(lexical only)'}")
    print(f"     gold doc #{gpid}: {snip(gpid)}")

print("\n" + "=" * 70)
print("10 ANSWERED WRONG (gold doc NOT in top-10)")
print("=" * 70)
for i, (q, qs, ranked, gpid, grank) in enumerate(fails, 1):
    top = ranked[0]
    print(f"\n[{i}] Q: {queries[q]}")
    print_triggers(qs)
    print(f"     BROUGHT UP (rank 1) #{top}: {snip(top)}")
    print(f"        fired here: {fired_on(qs, top) or '(lexical only)'}")
    print(f"     SHOULD HAVE (gold, rank {grank if grank<999 else '>200'}) #{gpid}: {snip(gpid)}")
    print(f"        fired there: {fired_on(qs, gpid) or '(nothing fired - gold doc lacks the trigger company)'}")
