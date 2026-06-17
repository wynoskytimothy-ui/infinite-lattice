#!/usr/bin/env python3
"""HYBRID: lattice/BM25 first-stage recall + a cross-encoder answer-ness rerank (the honest
division of labor the diagnostics proved). The lattice does what it's best at -- recall -- and a
small MS MARCO cross-encoder (MiniLM, cached, GPU) does the answer-ness rerank the symbolic frame
can't. Tests whether the lattice's recall pool feeds the reranker as well as BM25's.

Full 8.8M, dev queries. Two first stages, each top-100 -> cross-encoder rerank -> MRR@10:
  BM25       : stemmed BM25 (recall@100 ~0.666)
  lattice    : bm25-rare (rare words + tf-sat) + corridors (recall@100 ~0.646, + semantic reach)
"""
import sys, time, random
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE, K1, B
from sentence_transformers import CrossEncoder


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    idx = FullIndex()
    gold = train_corridors(idx)
    sum_idf = np.zeros(idx.N, np.float32); bm = np.zeros(idx.N, np.float32); corr = np.zeros(idx.N, np.float32)
    t0 = time.perf_counter()
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device="cuda")
    print(f"  loaded cross-encoder on GPU ({time.perf_counter()-t0:.0f}s)", flush=True)

    qrels = defaultdict(set)
    with open(idx.cf.name.replace("collection.tsv", "qrels.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(idx.cf.name.replace("collection.tsv", "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    def lattice_top(qs, rare, k=100):
        cterms = defaultdict(float)
        for qt in qs:
            for dt, w in gold.get(qt, []):
                cterms[dt] += w
        touched = []
        for w in rare:
            i = idx.tid.get(w)
            if i is None:
                continue
            s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
            dis = idx.di[s:e]; tfs = idx.tf[s:e].astype(np.float32); dl = idx.doclen[dis]; wi = idx.idfa[i]
            bm[dis] += wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / idx.avgdl)); touched.append(dis)
        for dt, w in cterms.items():
            i = idx.tid.get(dt)
            if i is not None:
                d = idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]; corr[d] += w; touched.append(d)
        if not touched:
            return []
        cat = np.concatenate(touched); cand = np.unique(cat)
        sc = bm[cand] + 0.3 * corr[cand]
        sel = np.argpartition(-sc, k)[:k] if len(cand) > k else np.arange(len(cand))
        order = sel[np.argsort(-sc[sel])]
        res = [int(d) for d in cand[order]]
        bm[cat] = 0.0; corr[cat] = 0.0
        return res

    def mrr_recall(reranked, pids, rel):
        rr = next((1.0 / i for i, d in enumerate(reranked[:10], 1) if str(d) in rel), 0.0)
        rec = 1.0 if any(str(d) in rel for d in pids) else 0.0
        return rr, rec

    res = defaultdict(float); n_eval = 0
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        n_eval += 1
        qtext = queries[q]
        for tag, pids in (("bm25", [int(d) for d in idx.bm25_top(qs, 100)[0]]),
                          ("lattice", lattice_top(qs, rare, 100) if rare else [])):
            if not pids:
                continue
            psgs = [idx.text(p) for p in pids]
            scores = ce.predict([(qtext, p) for p in psgs], batch_size=128, show_progress_bar=False)
            reranked = [p for _, p in sorted(zip(scores, pids), key=lambda x: -x[0])]
            rr, rec = mrr_recall(reranked, pids, rel)
            res[(tag, "mrr")] += rr; res[(tag, "rec")] += rec
            res[(tag, "bm25mrr")] += next((1.0 / i for i, d in enumerate(pids[:10], 1) if str(d) in rel), 0.0)
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{nq} | bm25+CE {res[('bm25','mrr')]/n_eval:.4f} "
                  f"| lattice+CE {res[('lattice','mrr')]/n_eval:.4f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nHYBRID (first-stage recall + cross-encoder rerank) -- full 8.8M, {N} dev q\n")
    print(f"   {'first stage':<12}{'recall@100':>12}{'1st-stage MRR':>15}{'+CE rerank MRR':>16}")
    for tag in ("bm25", "lattice"):
        print(f"   {tag:<12}{res[(tag,'rec')]/N:>12.4f}{res[(tag,'bm25mrr')]/N:>15.4f}{res[(tag,'mrr')]/N:>16.4f}")
    print(f"\n   ref: BM25 alone ~0.185, full ladder ~0.195, MARCO dev SOTA ~0.38-0.40.")
    print(f"   the cross-encoder supplies the answer-ness the symbolic frame couldn't; the lattice")
    print(f"   supplies recall (+ a semantic-reach edge BM25 lacks). higher recall@100 -> higher CE ceiling.")


if __name__ == "__main__":
    main()
