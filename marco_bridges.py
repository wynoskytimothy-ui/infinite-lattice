#!/usr/bin/env python3
"""MS MARCO step 2 -- the CORRELATIONS lever: supervised bridges from qrels.train.

The proven mechanism (+6.5pp on scifact): bridge[query_term] = relevant-doc-terms that
co-occur in (query, relevant-passage) pairs, weighted P(dt|qt)*idf(dt). Learn from
qrels.train (MARCO's own training signal), rerank BM25 top-100 on the SAME dev pool as
the floor, measure the MRR lift. lam=0 reproduces the BM25 floor exactly (comparability).

  python marco_bridges.py [n_queries] [n_distractors] [max_train_q]
"""
import sys, time, random
from collections import defaultdict, Counter
from marco_baseline import (MARCO, N_PASSAGES, tok, load_dev, build_pool,
                            load_texts, BM25)


def search_scored(bm, query, k, min_idf=0.3):
    """BM25 top-k as (doc_idx, score); skip near-zero-idf stopwords for speed."""
    sc = defaultdict(float)
    for w in set(tok(query)):
        idf = bm.idf.get(w)
        if idf is None or idf < min_idf:
            continue
        for di, c in bm.post[w]:
            dl = bm.doclen[di]
            sc[di] += idf * c * (bm.k1 + 1) / (c + bm.k1 * (1 - bm.b + bm.b * dl / bm.avgdl))
    top = sorted(sc, key=sc.get, reverse=True)[:k]
    return [(di, sc[di]) for di in top]


def learn_bridges(idf, max_q, gate_q=2.0, gate_d=3.0, top_per=12, seed=42):
    qrels_tr = defaultdict(set)
    with open(MARCO / "qrels.train.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels_tr[p[0]].add(p[2])
    sel = list(qrels_tr)
    random.Random(seed).shuffle(sel)
    sel = sel[:max_q]
    sel_set = set(sel)
    rel_pids = set(p for q in sel for p in qrels_tr[q])
    print(f"  bridges: {len(sel)} train queries, {len(rel_pids)} relevant passages", flush=True)
    rel_texts = load_texts(rel_pids)
    qtexts = {}
    with open(MARCO / "queries.train.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2 and parts[0] in sel_set:
                qtexts[parts[0]] = parts[1]

    cooc = defaultdict(Counter)
    npairs = Counter()
    t0 = time.perf_counter()
    for qid in sel:
        qts = [w for w in set(tok(qtexts.get(qid, ""))) if idf.get(w, 0.0) >= gate_q]
        if not qts:
            continue
        for pid in qrels_tr[qid]:
            t = rel_texts.get(pid)
            if not t:
                continue
            dts = [w for w in set(tok(t)) if idf.get(w, 0.0) >= gate_d]
            for qt in qts:
                npairs[qt] += 1
                cooc[qt].update(dts)
    bridges = {}
    for qt, cnt in cooc.items():
        n = npairs[qt]
        scored = [(dt, (c / n) * idf.get(dt, 0.0)) for dt, c in cnt.items()
                  if dt != qt and idf.get(dt, 0.0) > 0 and c >= 2]
        scored.sort(key=lambda x: -x[1])
        if scored:
            bridges[qt] = scored[:top_per]
    print(f"  learned bridges for {len(bridges)} query-terms ({time.perf_counter()-t0:.0f}s)", flush=True)
    return bridges


def evaluate(bm, texts, qids, queries, qrels, bridges, lam, cand=100):
    mrr = r10 = r100 = 0.0
    for q in qids:
        rel = qrels[q]
        cands = search_scored(bm, queries[q], cand)
        if lam > 0 and bridges:
            qbr = [w for w in set(tok(queries[q])) if w in bridges]
            if qbr:
                rescored = []
                for di, base in cands:
                    dts = set(tok(texts[bm.docids[di]]))
                    bonus = sum(w for qt in qbr for dt, w in bridges[qt] if dt in dts)
                    rescored.append((di, base + lam * bonus))
                cands = sorted(rescored, key=lambda x: -x[1])
        ranked = [bm.docids[di] for di, _ in cands]
        rr = 0.0
        for i, pid in enumerate(ranked[:10], 1):
            if pid in rel:
                rr = 1.0 / i
                break
        mrr += rr
        r10 += len(rel & set(ranked[:10])) / len(rel)
        r100 += len(rel & set(ranked[:100])) / len(rel)
    n = len(qids)
    return mrr / n, r10 / n, r100 / n


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    mtq = int(sys.argv[3]) if len(sys.argv) > 3 else 150_000
    print(f"MS MARCO correlations lever -- bridges from qrels.train, rerank BM25 top-100\n"
          f"({nq} dev queries, ~{nd} distractors, {mtq} train queries)\n", flush=True)
    qrels, queries = load_dev()
    qids, pool, rel_pids = build_pool(qrels, queries, nq, nd)
    texts = load_texts(pool)
    qids = [q for q in qids if all(p in texts for p in qrels[q])]
    bm = BM25(); bm.index(list(texts.items()))
    bridges = learn_bridges(bm.idf, mtq)

    print(f"\n  {'method':>22}{'MRR@10':>9}{'R@10':>8}{'R@100':>8}", flush=True)
    base = evaluate(bm, texts, qids, queries, qrels, bridges, lam=0.0)
    print(f"  {'BM25 floor (lam=0)':>22}{base[0]:>9.4f}{base[1]:>8.4f}{base[2]:>8.4f}", flush=True)
    for lam in (1.0, 2.0, 4.0):
        r = evaluate(bm, texts, qids, queries, qrels, bridges, lam=lam)
        d = r[0] - base[0]
        print(f"  {'+bridges lam='+str(lam):>22}{r[0]:>9.4f}{r[1]:>8.4f}{r[2]:>8.4f}"
              f"   ({d:+.4f} MRR)", flush=True)
    print(f"\n  same pool as the floor -> deltas are honest. positive = correlations lift retrieval.")


if __name__ == "__main__":
    main()
