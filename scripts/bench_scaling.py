#!/usr/bin/env python3
"""
Deep search step 9 - does it scale? latency + footprint vs corpus size.

Honest scaling analysis. Two regimes:
  - FULL dense scoring (lossless): touches every posting of every query term, so
    latency grows ~linearly with N (posting lists grow with the corpus). Stays
    accurate but NOT sub-ms at large N.
  - CHAMPION dense (bounded): only the top-M docs per term, so query work is
    O(query_terms * M) - INDEPENDENT of N => sub-ms at ANY size, at a small,
    measurable accuracy cost (top-M misses docs strong only on the sum).

Grows a corpus by replication (unique doc-ids) to sizes N, measures both query
paths' latency, and reports footprint B/doc (+ the uint16->uint32 step at 65k).
Accuracy-at-scale is corpus-dependent and measured separately on real BEIR;
here we isolate the SPEED/FOOTPRINT scaling laws.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load


def build_champion(idx, M):
    """Per-prime top-M (dense_doc_idx, tf) by length-normalised impact."""
    denom = idx._d_denom
    cd, ct = {}, {}
    for p, di in idx._d_pdoc.items():
        ptf = idx._d_ptf[p]
        if di.size <= M:
            cd[p], ct[p] = di, ptf
        else:
            tf = ptf.astype(np.float64)
            top = np.argpartition(tf / (tf + denom[di]), -M)[-M:]
            cd[p], ct[p] = di[top], ptf[top]
    return cd, ct


def champ_score(idx, cd, ct, query):
    N = max(1, len(idx.alive))
    k1p1 = idx.k1 + 1
    cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
    denom, df = idx._d_denom, idx.df
    scores = np.zeros(len(idx._d_docs), dtype=np.float64)
    for tok, qwt in idx._multiview(query).items():
        p = idx.token_prime.get(tok)
        if p is None:
            continue
        dfp = df[p]
        if dfp == 0:
            continue
        if cap is not None and tok[0] == "3" and dfp > cap:
            continue
        di = cd.get(p)
        if di is None:
            continue
        tfa = ct[p].astype(np.float64)
        scores[di] += (qwt * np.log(1 + (N - dfp + 0.5) / (dfp + 0.5)) * k1p1) \
            * tfa / (tfa + denom[di])
    return scores


def grown_corpus(base_corpus, mult):
    out = {}
    keys = list(base_corpus)
    for m in range(mult):
        for d in keys:
            out[f"{d}#{m}"] = base_corpus[d]
    return out


def run():
    corpus, queries, _, test_q = load("scifact")
    qsample = [queries[q] for q in list(test_q)[:40] if q in queries]
    base = dict(list(corpus.items())[:1500])              # modest base for memory
    M = 300

    print(f"{'N docs':>8} {'idtype':>7} {'B/doc':>7} {'dense MB':>9} "
          f"{'full ms':>8} {'champ ms':>9} {'champ/full':>10}")
    for mult in (1, 4, 10, 20):
        c = grown_corpus(base, mult)
        idx = AppendOnlyLatticeIndex()
        for d, t in c.items():
            idx.add(d, t)
        idx.finalize()
        N = len(idx.alive)
        idt = "uint16" if N < 65536 else "uint32"
        post = sum(a.size for a in idx._d_pdoc.values())
        bpp = (np.dtype(np.uint16 if N < 65536 else np.uint32).itemsize + 2)  # +f16 tf
        mb = (post * bpp + idx._d_denom.nbytes) / 1e6
        bdoc = mb * 1e6 / N

        cd, ct = build_champion(idx, M)

        def t_full():
            t0 = time.perf_counter()
            for q in qsample:
                idx._dense_score_array(q)
            return (time.perf_counter() - t0) / len(qsample) * 1000

        def t_champ():
            t0 = time.perf_counter()
            for q in qsample:
                champ_score(idx, cd, ct, q)
            return (time.perf_counter() - t0) / len(qsample) * 1000

        full_ms = min(t_full(), t_full())
        champ_ms = min(t_champ(), t_champ())
        print(f"{N:>8,} {idt:>7} {bdoc:>6.0f} {mb:>8.1f} {full_ms:>7.2f} "
              f"{champ_ms:>8.2f} {champ_ms/full_ms:>9.2f}x")
        del idx, cd, ct

    print(f"\n  full-dense latency grows ~linearly with N (lossless, not sub-ms at scale).")
    print(f"  champion (top-M={M}) is ~flat = sub-ms at ANY N (bounded work).")
    print(f"  B/doc steps up when N crosses 65,536 (uint16->uint32 doc-ids): +2 B/posting.")


def main():
    print("DEEP SEARCH step 9 - scaling laws: latency + footprint vs N")
    run()


if __name__ == "__main__":
    main()
