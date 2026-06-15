#!/usr/bin/env python3
"""The continual-learning loop -- built the honest way, with the HELD-OUT scoreboard.

Deepen gold-doc correlation learning over ROUNDS (more train questions + wider company +
2-hop cascade). After each round, score the HELD-OUT dev pool -- queries, docs, and gold
labels NEVER used in training. The held-out curve is the verdict:
  held-out MRR climbs across rounds  -> the loop learns GENERAL meaning (smarter on unseen)
  held-out flat while it deepens     -> overfit, caught (the glass-box scoreboard working)

Correlations are glass-box (printed sample below) so each is auditable as a real-world fact.
"""
import re, time, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


# HELD-OUT dev pool + clean-stem anchor (the scoreboard -- never trained on)
qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)
print(f"  held-out anchor: {len(idx.post)} terms (floor 0.5777)\n", flush=True)

# load TRAIN data once (qrels.train gold docs + query texts)
qrels_tr = defaultdict(set)
with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) >= 4 and int(p[3]) > 0:
            qrels_tr[p[0]].add(p[2])
sel = list(qrels_tr); random.Random(42).shuffle(sel); sel = sel[:150_000]
sel_set = set(sel)
gold_pids = {pid for q in sel for pid in qrels_tr[q]}
gold_texts = load_texts(gold_pids)
qtexts = {}
with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
    for line in f:
        a = line.rstrip("\n").split("\t", 1)
        if len(a) == 2 and a[0] in sel_set:
            qtexts[a[0]] = a[1]
print(f"  train material: {len(sel)} questions, {len(gold_pids)} gold docs loaded\n", flush=True)


def learn(n_q, top_per, gate_q=2.0, gate_d=3.0):
    cooc = defaultdict(Counter); npairs = Counter()
    for q in sel[:n_q]:
        qts = [w for w in set(stoks(qtexts.get(q, ""))) if idx.idf.get(w, 0) >= gate_q]
        if not qts:
            continue
        for pid in qrels_tr[q]:
            t = gold_texts.get(pid)
            if not t:
                continue
            dts = [w for w in set(stoks(t)) if idx.idf.get(w, 0) >= gate_d]
            for qt in qts:
                npairs[qt] += 1; cooc[qt].update(dts)
    gold = {}
    for qt, cnt in cooc.items():
        n = npairs[qt]
        sc = sorted(((dt, (c / n) * idx.idf.get(dt, 0.0)) for dt, c in cnt.items()
                     if dt != qt and idx.idf.get(dt, 0.0) > 0 and c >= 2), key=lambda x: -x[1])
        if sc:
            gold[qt] = sc[:top_per]
    return gold


def deepen(gold, decay=0.4, top_per=14):
    """2-hop: a term's company also pulls in its company's company (rare x cross) -- 'train deeper'."""
    out = {}
    for qt, comp in gold.items():
        acc = defaultdict(float); paths = defaultdict(int)
        for dt, w in comp:
            acc[dt] += w; paths[dt] += 1
            for dt2, w2 in gold.get(dt, []):
                if dt2 != qt:
                    acc[dt2] += decay * w * w2; paths[dt2] += 1
        out[qt] = sorted(((t, acc[t] * paths[t]) for t in acc), key=lambda x: -x[1])[:top_per]
    return out


def bm(di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def heldout(gold, beta=0.3):
    mrr = 0.0; h = hs = m = ms = 0
    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
        anchor = defaultdict(float)
        for w in qs:
            idfw = idx.idf[w]
            for di, c in idx.post[w]:
                anchor[di] += idfw * bm(di, c)
        cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
        comp = {}
        for di in cands:
            dts = set(stoks(texts[idx.docids[di]]))
            comp[di] = sum(w for qt in qs for dt, w in gold.get(qt, []) if dt in dts)
        ranked = [idx.docids[di] for di in sorted(cands, key=lambda di: -(anchor[di] + beta * comp[di]))[:10]]
        a_rank = [idx.docids[di] for di in cands[:10]]
        rr = next((1.0 / i for i, pid in enumerate(ranked, 1) if pid in rel), 0.0)
        ra = next((1.0 / i for i, pid in enumerate(a_rank, 1) if pid in rel), 0.0)
        mrr += rr
        if ra > 0:
            h += 1; hs += rr
        else:
            m += 1; ms += rr
    n = len(qids)
    return mrr / n, hs / max(1, h), ms / max(1, m)


print("CONTINUAL-LEARNING LOOP -- HELD-OUT scoreboard (dev queries never trained on):\n")
print(f"   {'round (deepen on TRAIN)':>28}{'held-out MRR':>14}{'hit':>8}{'miss':>8}")
m0 = heldout({}, beta=0.0)
print(f"   {'R0  anchor floor':>28}{m0[0]:>14.4f}{m0[1]:>8.4f}{m0[2]:>8.4f}", flush=True)
last = None
for name, fn in [("R1  40k q, company 6", lambda: learn(40_000, 6)),
                 ("R2  80k q, company 12", lambda: learn(80_000, 12)),
                 ("R3  150k q, company 12", lambda: learn(150_000, 12)),
                 ("R4  150k +2-hop deepen", lambda: deepen(learn(150_000, 12)))]:
    t0 = time.perf_counter()
    g = fn()
    mr = heldout(g)
    arrow = "" if last is None else (" UP" if mr[0] > last + 1e-4 else (" flat" if abs(mr[0]-last) <= 1e-4 else " DOWN"))
    print(f"   {name:>28}{mr[0]:>14.4f}{mr[1]:>8.4f}{mr[2]:>8.4f}  ({time.perf_counter()-t0:.0f}s){arrow}", flush=True)
    last = mr[0]
    if name.startswith("R3"):
        print("\n   glass-box sample (auditable real-world facts learned):", flush=True)
        for probe in ["arthritis", "insulin", "vaccine"]:
            st = safe(probe); c = g.get(st)
            if c:
                print(f"      {probe:>10} -> " + ", ".join(t for t, _ in c[:6]), flush=True)
        print(flush=True)
print("\n  held-out climbing across rounds = generalizes to UNSEEN questions (real learning).")
print("  held-out flat while deepening = overfit, caught. dev never trained on -> honest curve.")


if __name__ == "__main__":
    pass
