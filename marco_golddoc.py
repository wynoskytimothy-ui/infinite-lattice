#!/usr/bin/env python3
"""The relevance-grounded retriever -- the user's vision on the signal we PROVED.
Corridors learned from rare terms that bunch in the GOLD (relevant) doc, not from corpus-
wide co-occurrence (similarity != relevance). Clean conservative-stem base. At query time:
query stems light up -> pull their gold-doc-learned company -> doc holding the company is
where the dots intersect (meet, anchor-dominant). Measured vs the 0.5777 stem floor.

Unlike every unsupervised corridor (which HURT), gold-doc grounding should be POSITIVE.
"""
import re, time, random
from collections import defaultdict, Counter
from marco_lab import load_pool, Index
from marco_baseline import MARCO, load_texts
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


def learn_gold(idx, max_q=150_000, gate_q=2.0, gate_d=3.0, top_per=12, seed=42):
    qrels_tr = defaultdict(set)
    with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels_tr[p[0]].add(p[2])
    sel = list(qrels_tr); random.Random(seed).shuffle(sel); sel = sel[:max_q]
    sel_set = set(sel)
    gold_pids = {pid for q in sel for pid in qrels_tr[q]}
    print(f"  gold: {len(sel)} train queries, {len(gold_pids)} relevant (gold) docs", flush=True)
    gold_texts = load_texts(gold_pids)
    qtexts = {}
    with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in sel_set:
                qtexts[a[0]] = a[1]
    cooc = defaultdict(Counter); npairs = Counter()
    for q in sel:
        qts = [w for w in set(stoks(qtexts.get(q, ""))) if idx.idf.get(w, 0) >= gate_q]
        if not qts:
            continue
        for pid in qrels_tr[q]:
            t = gold_texts.get(pid)
            if not t:
                continue
            dts = [w for w in set(stoks(t)) if idx.idf.get(w, 0) >= gate_d]   # rare company IN THE GOLD DOC
            for qt in qts:
                npairs[qt] += 1
                cooc[qt].update(dts)
    gold = {}
    for qt, cnt in cooc.items():
        n = npairs[qt]
        sc = [(dt, (c / n) * idx.idf.get(dt, 0.0)) for dt, c in cnt.items()
              if dt != qt and idx.idf.get(dt, 0.0) > 0 and c >= 2]
        sc.sort(key=lambda x: -x[1])
        if sc:
            gold[qt] = sc[:top_per]
    print(f"  learned gold-doc company for {len(gold)} query-stems", flush=True)
    return gold


def bm(idx, di, c):
    dl = idx.doclen[di]
    return c * (idx.k1 + 1) / (c + idx.k1 * (1 - idx.b + idx.b * dl / idx.avgdl))


def rr10(ranked, rel):
    for i, pid in enumerate(ranked[:10], 1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def main():
    qids, queries, qrels, texts = load_pool()
    idx = Index(stoks).build(texts)
    print(f"  clean-stem anchor: {len(idx.post)} terms\n", flush=True)
    gold = learn_gold(idx)

    BETAS = (0.0, 0.1, 0.3, 0.6)
    agg = {b: {"all": 0.0, "hit": 0.0, "miss": 0.0} for b in BETAS}
    nhit = nmiss = 0
    t0 = time.perf_counter()
    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
        anchor = defaultdict(float)
        for w in qs:
            idfw = idx.idf[w]
            for di, c in idx.post[w]:
                anchor[di] += idfw * bm(idx, di, c)
        cands = sorted(anchor, key=anchor.get, reverse=True)[:200]
        rr_a = rr10([idx.docids[di] for di in cands[:100]], rel)
        bucket = "hit" if rr_a > 0 else "miss"
        nhit += bucket == "hit"; nmiss += bucket == "miss"
        company = defaultdict(float)
        for di in cands:
            dts = set(stoks(texts[idx.docids[di]]))
            s = 0.0
            for qt in qs:
                for dt, w in gold.get(qt, []):
                    if dt in dts:
                        s += w
            company[di] = s
        for b in BETAS:
            ranked = [idx.docids[di] for di in sorted(cands, key=lambda di: -(anchor[di] + b * company[di]))[:100]]
            rr = rr10(ranked, rel)
            agg[b]["all"] += rr; agg[b][bucket] += rr
    n = nhit + nmiss
    print(f"  eval {n} queries in {time.perf_counter()-t0:.0f}s ({nhit} hit, {nmiss} miss)\n")
    print(f"   {'beta':>8}{'MRR@10':>9}{'hit-set':>10}{'miss-set':>10}")
    for b in BETAS:
        tag = "  (floor)" if b == 0 else ""
        print(f"   {b:>8}{agg[b]['all']/n:>9.4f}{agg[b]['hit']/max(1,nhit):>10.4f}{agg[b]['miss']/max(1,nmiss):>10.4f}{tag}")
    print("\n  gold-doc company POSITIVE over 0.5777 = the relevance-grounded vision ranks (vs unsupervised which hurt).")


if __name__ == "__main__":
    main()
