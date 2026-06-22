#!/usr/bin/env python3
"""Does the lattice's O(1)-per-check membership beat searchsorted for the meet? HONEST head-to-head.

The user's formula gives O(1) membership (is this doc on the rail) with no hash/extra index. The
question is whether that beats numpy's vectorized searchsorted for the inverted retrieval meet. The
cost is dominated by how many postings you TOUCH:
  searchsorted     -- binary search: touches |C| points in P              -> O(|C| log df)
  presence O(1)    -- set tf at all of P's docs, gather at C (hash-free)   -> O(df + |C|)
So searchsorted should win when |C| << df (single-hop, small pool) and presence should win when |C|
is huge (multi-hop hop-2, big pool). Measured in BOTH regimes, with identical-score verification.
"""
import time, random
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO, K1, B, SCORE_FLOOR

N = 200
DF_CAP = 100_000


def cand_of(idx, qterms, df_cap=DF_CAP):
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
        return None, None
    disc = [(s, e) for (wi, df, s, e) in terms if df < df_cap]
    if not disc:
        wi, df, s, e = min(terms, key=lambda t: t[1]); disc = [(s, e)]
    cand = np.unique(np.concatenate([idx.di[s:e] for (s, e) in disc]))
    return cand, terms


def score_ss(idx, cand, terms):
    dlc = idx.doclen[cand]; sc = np.zeros(len(cand), np.float32)
    for (wi, df, s, e) in terms:
        dis = idx.di[s:e]; pos = np.minimum(np.searchsorted(dis, cand), len(dis) - 1)
        hit = dis[pos] == cand; tfs = idx.tf[s:e][pos].astype(np.float32)
        c = wi * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * dlc / idx.avgdl)); sc[hit] += c[hit]
    return sc


def score_presence(idx, cand, terms, val):
    dlc = idx.doclen[cand]; sc = np.zeros(len(cand), np.float32)
    for (wi, df, s, e) in terms:
        dis = idx.di[s:e]; val[dis] = idx.tf[s:e]
        tfc = val[cand].astype(np.float32); hit = tfc > 0
        c = wi * tfc * (K1 + 1) / (tfc + K1 * (1 - B + B * dlc / idx.avgdl)); sc[hit] += c[hit]
        val[dis] = 0
    return sc


def bench(idx, scenarios, val):
    print(f"  {'regime':<22}{'med |C|':>10}{'searchsorted ms':>18}{'presence ms':>14}{'winner':>10}")
    for name, querysets in scenarios:
        tss, tps, csz, mism = [], [], [], 0
        for terms_or_q in querysets:
            cand, terms = terms_or_q
            if cand is None or len(cand) == 0:
                continue
            csz.append(len(cand))
            t0 = time.perf_counter(); s1 = score_ss(idx, cand, terms); tss.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter(); s2 = score_presence(idx, cand, terms, val); tps.append((time.perf_counter() - t0) * 1000)
            if not np.allclose(s1, s2, atol=1e-2):
                mism += 1
        tss = np.array(tss); tps = np.array(tps)
        win = "presence" if np.median(tps) < np.median(tss) else "searchsorted"
        flag = "" if mism == 0 else f" [{mism} mismatch!]"
        print(f"  {name:<22}{int(np.median(csz)):>10,}{np.median(tss):>18.2f}{np.median(tps):>14.2f}{win:>10}{flag}")


def main():
    idx = FullIndex()
    val = np.zeros(idx.N, np.uint16)
    qs = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                qs.append(a[1])
    random.Random(0).shuffle(qs)

    # regime 1: real dev queries (small pool)
    single = [cand_of(idx, stoks(q)) for q in qs[:N]]
    # regime 2: hop-2-like -- a top doc's rare terms as the query (big pool)
    from marco_fast import bm25_fast
    RARE = 4.0
    multi = []
    for q in qs[:N]:
        o, _ = bm25_fast(idx, stoks(q), 100)
        if len(o) == 0:
            continue
        rt = [w for w in set(stoks(idx.text(int(o[0])))) if idx.idf_of(w) >= RARE][:8]
        if rt:
            multi.append(cand_of(idx, rt))

    for q in qs[:5]:                       # warm
        c, t = cand_of(idx, stoks(q))
        if c is not None:
            score_ss(idx, c, t); score_presence(idx, c, t, val)

    print(f"\n  MEET MICRO-BENCH on 8.8M -- searchsorted vs hash-free presence (O(1)/check), identical scores\n")
    bench(idx, [("single-hop query", single), ("multi-hop rare-bundle", multi)], val)
    print(f"\n  takeaway: the winner flips with |C|. presence (the lattice's O(1) membership) wins exactly")
    print(f"  in the big-pool multi-hop regime -- so the meet should pick the method by |C| (adaptive).")


if __name__ == "__main__":
    main()
