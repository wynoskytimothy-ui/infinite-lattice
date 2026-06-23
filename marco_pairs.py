#!/usr/bin/env python3
"""Test the user's 2-way/3-way INTERSECTION ranking: score a doc by the rarity-weighted ENSEMBLE of query
term-PAIRS (and triples) it satisfies -- coordination/proximity -- not just 1-way BM25. Closed form: a doc
holding m of the query's rare terms (idf-sum S) gets 2-way score (m-1)*S, 3-way (m-1)(m-2)/2*S -- rewards
co-presence of many rare terms super-linearly ("the rarest intersections narrow to the exact doc"). Compare
BM25 vs 2-way vs 3-way vs fused, on MARCO MRR@10 + recall@100. Does using every intersection narrow better?
"""
import random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B, SCORE_FLOOR

N = 500
DF_CAP = 100_000
CAND_FLOOR = 2.0


def score(idx, qterms, val, k=100):
    terms = []
    for w in set(qterms):
        i = idx.tid.get(w)
        if i is None:
            continue
        wi = float(idx.idfa[i])
        if wi < SCORE_FLOOR:
            continue
        s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
        terms.append((wi, e - s, s, e))
    if not terms:
        return None
    disc = [(s, e) for (wi, df, s, e) in terms if df < DF_CAP] or [min(terms, key=lambda t: t[1])[2:]]
    cand = np.unique(np.concatenate([idx.di[s:e] for (s, e) in disc]))
    dlc = idx.doclen[cand]
    bm25 = np.zeros(len(cand), np.float32); m = np.zeros(len(cand), np.float32); S = np.zeros(len(cand), np.float32)
    for (wi, df, s, e) in terms:
        dis = idx.di[s:e]; val[dis] = idx.tf[s:e]
        tfc = val[cand].astype(np.float32); hit = tfc > 0
        bm25[hit] += (wi * tfc * (K1 + 1) / (tfc + K1 * (1 - B + B * dlc / idx.avgdl)))[hit]
        if wi >= CAND_FLOOR:
            m[hit] += 1; S[hit] += wi
        val[dis] = 0
    two = np.maximum(m - 1, 0) * S
    three = np.maximum(m - 1, 0) * np.maximum(m - 2, 0) / 2 * S
    return cand, bm25, two, three


def topk(cand, sc, k):
    sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
    return cand[sel[np.argsort(-sc[sel])]]


def main():
    idx = FullIndex(); val = np.zeros(idx.N, np.uint16)
    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(int(p[2]))
    queries = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2 and a[0] in qrels:
                queries.append((a[0], a[1]))
    random.Random(0).shuffle(queries); sample = queries[:N]

    names = ["bm25", "2way", "3way", "fused2", "fused23"]
    mrr = {k: 0.0 for k in names}; rec = {k: 0 for k in names}; nb = 0
    for qid, qt in sample:
        r = score(idx, stoks(qt), val, 100)
        if r is None:
            continue
        cand, bm25, two, three = r; gold = qrels[qid]; nb += 1
        scs = {"bm25": bm25, "2way": two, "3way": three,
               "fused2": bm25 + 0.5 * two, "fused23": bm25 + 0.5 * two + 0.1 * three}
        for name, sc in scs.items():
            o = topk(cand, sc, 100)
            for rk, d in enumerate(o[:10]):
                if int(d) in gold:
                    mrr[name] += 1.0 / (rk + 1); break
            if any(int(d) in gold for d in o[:100]):
                rec[name] += 1
    print(f"\n  2-WAY/3-WAY INTERSECTION RANKING vs BM25 -- MARCO 8.8M, n={nb}\n")
    print(f"  {'method':<28}{'MRR@10':>10}{'recall@100':>13}")
    label = {"bm25": "BM25 (1-way)", "2way": "2-way coord (m-1)*S", "3way": "3-way coord",
             "fused2": "BM25 + 0.5*2way", "fused23": "BM25 + 2way + 3way"}
    for name in names:
        print(f"  {label[name]:<28}{mrr[name]/nb:>10.4f}{rec[name]/nb*100:>12.1f}%")
    print(f"\n  if fused > bm25: the rarity-weighted intersection coordination sharpens ranking; if not, the")
    print(f"  2-way's real payoff is recall + the multi-hop bridge (already validated), not single-hop rank.")


if __name__ == "__main__":
    main()
