#!/usr/bin/env python3
"""Subword bridge as a RECALL rule (the lattice's char-gram view). Build a 4-gram -> term index
over the vocab, so each query rare word pulls in its morphological relatives (nanofiber ->
fiber/fibers/nanofibers) -- terms BM25 can't connect because they're different words. Add those
neighbors' postings to the candidate pool and measure the recall lift over the bm25-rare +
corridor recall engine. Full 8.8M, 3000 dev q.

  baseline   : bm25-rare (sum rare-idf x tf-sat) + corridor company   (= recall-max engine)
  + subword  : also pool/score the rare words' char-gram neighbors (idf>=4, >=2 shared 4-grams)
"""
import sys, random, time, pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE, K1, B

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
CG = 4


def build_chargrams(idx):
    cache = MARCO / "full_idx_chargram.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            g2t = pickle.load(f)
        print(f"  loaded char-gram index: {len(g2t):,} 4-grams", flush=True)
        return g2t
    t0 = time.perf_counter()
    g2t = defaultdict(list)
    for tid, term in enumerate(idx.vocab):
        if len(term) < CG or idx.idfa[tid] < 3.0:
            continue
        for g in set(term[i:i + CG] for i in range(len(term) - CG + 1)):
            g2t[g].append(tid)
    g2t = {g: np.array(t, dtype=np.int32) for g, t in g2t.items() if len(t) >= 2}
    with open(cache, "wb") as f:
        pickle.dump(g2t, f)
    print(f"  built char-gram index: {len(g2t):,} 4-grams ({time.perf_counter()-t0:.0f}s)", flush=True)
    return g2t


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)
    g2t = build_chargrams(idx)
    sum_idf = np.zeros(idx.N, np.float32); bm = np.zeros(idx.N, np.float32); corr = np.zeros(idx.N, np.float32)

    GRAM_CAP = 800   # skip 4-grams mapping to >800 terms: common, not morphologically distinctive
    def neighbors(r, topk=15):
        grams = [r[i:i + CG] for i in range(len(r) - CG + 1)]
        if not grams:
            return []
        cnt = Counter()
        for g in set(grams):
            arr = g2t.get(g)
            if arr is not None and len(arr) <= GRAM_CAP:
                cnt.update(int(t) for t in arr)
        need = max(2, (len(grams)) // 2)         # share >=half the distinctive grams
        out = [(tid, c) for tid, c in cnt.items()
               if c >= need and idx.idfa[tid] >= RARE and idx.vocab[tid] != r]
        return sorted(out, key=lambda x: -x[1])[:topk]

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    Ks = (100, 500, 1000)
    rec_base = defaultdict(float); rec_sw = defaultdict(float)
    n_eval = 0; t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = set(int(p) for p in qrels[q])
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        if not rare:
            continue
        n_eval += 1
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
            sum_idf[dis] += wi
            bm[dis] += wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / idx.avgdl))
            touched.append(dis)
        for dt, w in cterms.items():
            i = idx.tid.get(dt)
            if i is not None:
                idis = idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]; corr[idis] += w; touched.append(idis)
        base_cat = np.concatenate(touched) if touched else np.empty(0, np.uint32)
        base_cand = np.unique(base_cat)
        # --- subword neighbors: add their postings, scored like a (weaker) rare match ---
        sw_touched = []
        for w in rare:
            for tid, ov in neighbors(w, topk=6):
                s, e = int(idx.ptr[tid]), int(idx.ptr[tid + 1])
                dis = idx.di[s:e]; tfs = idx.tf[s:e].astype(np.float32); dl = idx.doclen[dis]
                wi = idx.idfa[tid] * (ov / max(2, len(w) - CG + 1))   # discount by char-overlap fraction
                bm[dis] += wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / idx.avgdl))
                sw_touched.append(dis)
        sw_cat = np.concatenate(touched + sw_touched) if (touched or sw_touched) else base_cat
        sw_cand = np.unique(sw_cat)

        def score(cand):
            return bm[cand] + 0.3 * corr[cand]

        def topset(cand, K):
            sc = score(cand)
            if len(cand) <= K:
                return set(int(d) for d in cand)
            return set(int(d) for d in cand[np.argpartition(-sc, K)[:K]])
        for K in Ks:
            rec_base[K] += len(rel & topset(base_cand, K)) / len(rel)
        # subword score uses the SAME bm (now incl. neighbor contributions) over the expanded pool
        for K in Ks:
            rec_sw[K] += len(rel & topset(sw_cand, K)) / len(rel)
        sum_idf[base_cat] = 0.0; corr[base_cat] = 0.0
        bm[sw_cat] = 0.0
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{nq} | base R@1000 {rec_base[1000]/n_eval:.3f} "
                  f"+subword {rec_sw[1000]/n_eval:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nSUBWORD-EXPANDED RECALL -- full 8.8M, {N} dev q (>=1 rare word)\n")
    print(f"   {'engine':<20}{'R@100':>9}{'R@500':>9}{'R@1000':>9}")
    print(f"   {'bm25-rare + corr':<20}{rec_base[100]/N:>9.4f}{rec_base[500]/N:>9.4f}{rec_base[1000]/N:>9.4f}")
    print(f"   {'+ subword bridge':<20}{rec_sw[100]/N:>9.4f}{rec_sw[500]/N:>9.4f}{rec_sw[1000]/N:>9.4f}")
    d100 = (rec_sw[100] - rec_base[100]) / N; d1000 = (rec_sw[1000] - rec_base[1000]) / N
    print(f"   {'subword lift':<20}{d100:>+9.4f}{'':>9}{d1000:>+9.4f}")
    print(f"\n   subword bridges = the lattice's char-gram view (nanofiber~fiber) -- recall BM25 can't reach.")


if __name__ == "__main__":
    main()
