#!/usr/bin/env python3
"""Does the LATTICE'S reach beat BM25 INSIDE the hybrid? The cross-encoder can only rank golds
that are in the pool. BM25's pool misses golds with no lexical overlap; the lattice's corridor
reach (99.7% membership) catches some of them. So feed the cross-encoder THREE pools and compare:

  BM25 top-100             -> CE
  lattice top-100          -> CE   (bm25-rare + tf + corridors)
  UNION (BM25 | lattice)   -> CE   (best recall -- lattice ADDS golds BM25 never sees)

If UNION > BM25, the lattice adds real value on its own turf (recall). We also count the queries
where the gold is in the lattice pool but NOT BM25's -- the lattice-only recall wins -- and check
whether the CE ranks those into the top-10. Full 8.8M, dev queries; one CE pass over the union.
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
    bm = np.zeros(idx.N, np.float32); corr = np.zeros(idx.N, np.float32)
    t0 = time.perf_counter()
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256, device="cuda")
    print(f"  cross-encoder on GPU ({time.perf_counter()-t0:.0f}s)", flush=True)

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
        res = [int(d) for d in cand[sel[np.argsort(-sc[sel])]]]
        bm[cat] = 0.0; corr[cat] = 0.0
        return res

    def rr(order_pids, rel):
        return next((1.0 / i for i, p in enumerate(order_pids[:10], 1) if str(p) in rel), 0.0)

    res = defaultdict(float); n_eval = 0
    lat_only_golds = 0; lat_only_ranked = 0
    rec = defaultdict(float)
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        n_eval += 1
        pb = [int(d) for d in idx.bm25_top(qs, 100)[0]]
        pl = lattice_top(qs, rare, 100) if rare else []
        sb, sl = set(str(x) for x in pb), set(str(x) for x in pl)
        union = list(dict.fromkeys(pb + pl))
        # recall of each pool
        rec["bm"] += 1.0 if (rel & sb) else 0.0
        rec["lat"] += 1.0 if (rel & sl) else 0.0
        rec["union"] += 1.0 if (rel & set(str(x) for x in union)) else 0.0
        # lattice-only gold (in lattice pool, NOT bm25 pool)
        lat_only = bool(rel & sl) and not bool(rel & sb)
        if lat_only:
            lat_only_golds += 1
        # one CE pass over the union
        psgs = [idx.text(p) for p in union]
        scores = ce.predict([(queries[q], p) for p in psgs], batch_size=128, show_progress_bar=False)
        cescore = {p: float(s) for p, s in zip(union, scores)}
        def rerank(pids):
            return sorted(pids, key=lambda p: -cescore[p])
        res["bm"] += rr(rerank(pb), rel)
        res["lat"] += rr(rerank(pl) if pl else [], rel)
        ur = rerank(union)
        res["union"] += rr(ur, rel)
        if lat_only and any(str(d) in rel for d in ur[:10]):
            lat_only_ranked += 1
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{nq} | bm->CE {res['bm']/n_eval:.4f} union->CE {res['union']/n_eval:.4f} "
                  f"| lat-only golds {lat_only_golds} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nLATTICE-REACH HYBRID -- full 8.8M, {N} dev q (cross-encoder rerank)\n")
    print(f"   {'pool -> CE':<22}{'recall@pool':>12}{'MRR@10':>9}")
    print(f"   {'BM25':<22}{rec['bm']/N:>12.4f}{res['bm']/N:>9.4f}")
    print(f"   {'lattice':<22}{rec['lat']/N:>12.4f}{res['lat']/N:>9.4f}")
    print(f"   {'UNION (bm25 | lattice)':<22}{rec['union']/N:>12.4f}{res['union']/N:>9.4f}")
    print(f"\n   lattice reaches a gold BM25 MISSES on {lat_only_golds}/{N} = {lat_only_golds/N:.1%} of queries;")
    print(f"   the CE then ranks {lat_only_ranked} of those into the top-10 = the lattice's net new wins.")
    print(f"   UNION recall - BM25 recall = {(rec['union']-rec['bm'])/N:+.4f} (the golds the lattice ADDS).")


if __name__ == "__main__":
    main()
