#!/usr/bin/env python3
"""LATTICE-INSIDE-LATTICE: an unsupervised rare<->rare co-occurrence sub-lattice built at INGEST
(no qrels) -- the user's "each rare prime branches its own lattice; measure how often rare terms
co-occur in the same doc; separate from the supervised 0.7." All three design agents converged
on this. Tests the decisive question first (membership): does this zero-shot layer reach golds the
rare-word union + supervised corridor MISS? Then recall@K with the SOFT MEET (expand the pool via
each query rare word's learned co-occurring siblings, instead of a literal 2nd query word).

  build: doc-level rare-rare co-occurrence, idf>=RARE & df<=cap, PPMI-weighted, top-K/term (cached).
  test:  pool membership (rare / +supervised / +unsup / unsup-NOVEL) + recall@100/500/1000.
"""
import sys, time, math, pickle, random
from collections import defaultdict, Counter
from itertools import combinations
from pathlib import Path
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE, G_XO

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
SUB_DFCAP = 20000     # sub-lattice over idf>=RARE terms with df<=this (bounds cost; rarest words drive the cascade)
MAXDOC = 25           # cap rare terms scored per passage
MIN_COOC = 3          # prune pair counts below this
TOPK = 12             # co-occurring siblings kept per rare term


def build_cooc(idx):
    cache = MARCO / "full_idx_sublattice.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            sub = pickle.load(f)
        print(f"  loaded sub-lattice: {len(sub):,} terms", flush=True)
        return sub
    t0 = time.perf_counter()
    df = (idx.ptr[1:] - idx.ptr[:-1]).astype(np.int64)
    rare_ok = np.zeros(len(idx.vocab), dtype=bool)
    rare_ok[(idx.idfa >= RARE) & (df >= 3) & (df <= SUB_DFCAP)] = True
    print(f"  sub-lattice rare-term band (idf>={RARE}, df in [3,{SUB_DFCAP}]): {int(rare_ok.sum()):,} terms", flush=True)
    cooc = defaultdict(Counter)
    with open(MARCO / "collection.tsv", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f):
            tab = line.find("\t")
            if tab < 0:
                continue
            rs = [idx.tid[w] for w in set(stoks(line[tab + 1:])) if w in idx.tid and rare_ok[idx.tid[w]]]
            if len(rs) < 2:
                continue
            if len(rs) > MAXDOC:
                rs = sorted(rs, key=lambda t: df[t])[:MAXDOC]
            rs.sort()
            for a, b in combinations(rs, 2):
                cooc[a][b] += 1
            if (ln + 1) % 2_000_000 == 0:
                print(f"    {ln+1:,} docs, {len(cooc):,} src terms, {time.perf_counter()-t0:.0f}s", flush=True)
    N = idx.N
    nbr = defaultdict(list)
    for a, ctr in cooc.items():
        for b, c in ctr.items():
            if c < MIN_COOC:
                continue
            ppmi = math.log2(c * N / (df[a] * df[b]))
            if ppmi > 0:
                nbr[a].append((b, ppmi)); nbr[b].append((a, ppmi))
    sub = {a: sorted(v, key=lambda x: -x[1])[:TOPK] for a, v in nbr.items()}
    with open(cache, "wb") as f:
        pickle.dump(sub, f)
    edges = sum(len(v) for v in sub.values())
    print(f"  built sub-lattice: {len(sub):,} terms, {edges:,} edges (~{edges*8/1e6:.0f}MB) ({time.perf_counter()-t0:.0f}s)", flush=True)
    return sub


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)
    sub = build_cooc(idx)

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

    sum_idf = np.zeros(idx.N, np.float32); cnt = np.zeros(idx.N, np.float32); subsc = np.zeros(idx.N, np.float32)
    Ks = (100, 500, 1000)
    memR = memC = memU = memUnovel = 0.0
    recB = defaultdict(float); recS = defaultdict(float); n_eval = 0; t0 = time.perf_counter()

    def reach(term_ids, rel_arr):
        for ti in term_ids:
            dis = idx.di[int(idx.ptr[ti]):int(idx.ptr[ti + 1])]
            if len(dis):
                j = np.clip(np.searchsorted(dis, rel_arr), 0, len(dis) - 1)
                if np.any(dis[j] == rel_arr):
                    return True
        return False

    for n, q in enumerate(qids):
        rel = qrels[q]; rel_arr = np.fromiter((int(p) for p in rel), dtype=np.int64)
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = [w for w in qs if idx.idf_of(w) >= RARE]
        if not rare:
            continue
        n_eval += 1
        rare_tids = [idx.tid[w] for w in rare if w in idx.tid]
        sup_tids = [idx.tid[dt] for qt in qs for dt, _ in gold.get(qt, []) if dt in idx.tid]
        sub_nbr = {}                                  # unsupervised siblings of the query's rare words
        for ti in rare_tids:
            for b, w in sub.get(ti, []):
                sub_nbr[b] = max(sub_nbr.get(b, 0.0), w)
        sub_tids = list(sub_nbr)
        # --- membership audit (the decisive numbers) ---
        in_rare = reach(rare_tids, rel_arr)
        in_sup = in_rare or reach(sup_tids, rel_arr)
        in_unsup = reach(sub_tids, rel_arr)
        memR += 1.0 if in_rare else 0.0
        memC += 1.0 if in_sup else 0.0
        memU += 1.0 if (in_sup or in_unsup) else 0.0
        if in_unsup and not in_sup:
            memUnovel += 1.0                          # gold reachable ONLY via the unsupervised sub-lattice
        # --- recall@K: baseline (rare union, meet) vs + sub-lattice SOFT MEET ---
        touched = []
        for ti in rare_tids:
            s, e = int(idx.ptr[ti]), int(idx.ptr[ti + 1])
            dis = idx.di[s:e]; sum_idf[dis] += idx.idfa[ti]; cnt[dis] += 1.0; touched.append(dis)
        base_cat = np.concatenate(touched); base_cand = np.unique(base_cat)
        meetB = sum_idf[base_cand] * (cnt[base_cand] ** G_XO)
        # sub-lattice: add siblings' docs (gated to docs already carrying a query rare word) + reward agreement
        sub_touch = []
        for b, w in sub_nbr.items():
            dis = idx.di[int(idx.ptr[b]):int(idx.ptr[b + 1])]
            subsc[dis] += w; sub_touch.append(dis)
        # soft-meet pool = rare union (gated expansion already implicit: sub agreement only lifts union docs)
        scoreS = meetB + 0.5 * subsc[base_cand]
        for K in Ks:
            recB[K] += len(rel & set(str(int(d)) for d in (base_cand if len(base_cand) <= K else base_cand[np.argpartition(-meetB, K)[:K]]))) / len(rel)
            recS[K] += len(rel & set(str(int(d)) for d in (base_cand if len(base_cand) <= K else base_cand[np.argpartition(-scoreS, K)[:K]]))) / len(rel)
        sum_idf[base_cat] = 0.0; cnt[base_cat] = 0.0
        if sub_touch:
            subsc[np.concatenate(sub_touch)] = 0.0
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{nq} | memC {memC/n_eval:.3f} memU {memU/n_eval:.3f} novel {memUnovel/n_eval:.3f} "
                  f"| R@1000 base {recB[1000]/n_eval:.3f} +sub {recS[1000]/n_eval:.3f} | {time.perf_counter()-t0:.0f}s", flush=True)

    N = n_eval
    print(f"\nSUB-LATTICE (unsupervised rare-rare co-occurrence) -- full 8.8M, {N} dev q\n")
    print(f"   POOL MEMBERSHIP (gold reachable, any rank):")
    print(f"     rare-word union:            {memR/N:.4f}")
    print(f"     + supervised corridor:      {memC/N:.4f}")
    print(f"     + unsupervised sub-lattice: {memU/N:.4f}   (total)")
    print(f"     NOVEL (unsup reaches, supervised does NOT): {memUnovel/N:.4f}  <- the decisive additive number")
    print(f"\n   RANKED RECALL (rare pool, soft meet):")
    print(f"   {'engine':<20}{'R@100':>9}{'R@500':>9}{'R@1000':>9}")
    print(f"   {'meet (baseline)':<20}{recB[100]/N:>9.4f}{recB[500]/N:>9.4f}{recB[1000]/N:>9.4f}")
    print(f"   {'meet + sub-lattice':<20}{recS[100]/N:>9.4f}{recS[500]/N:>9.4f}{recS[1000]/N:>9.4f}")
    print(f"\n   NOVEL>0 = the lattice-inside-lattice reaches golds the supervised 0.7 can't, zero-shot.")


if __name__ == "__main__":
    main()
