#!/usr/bin/env python3
"""Where is the 0.68 GB footprint, and how much can the free-token/composite idea shave?
The composite-as-prime model (stage05_free_token): a pair/phrase address = P*Q, computed never stored,
FTA-factorable back to primes -> no vocab row for composites. This measures the actual headroom on the
QUANTIZED index: term/posting/vocab distribution by df, the droppable/derivable rare tail vs the common-
prime postings bulk (a frequent atomic word is a PRIME, not a composite, so it can't be derived away).
Also: do queries rely on rare terms (the cost of dropping them)?
"""
import random
import numpy as np
from marco_full_eval import FullIndex, stoks, MARCO


def main():
    idx = FullIndex()
    df = np.diff(idx.ptr.astype(np.int64))
    nterms = len(df); npost = int(df.sum())
    vlen = np.fromiter((len(t) for t in idx.vocab), dtype=np.int64, count=nterms)
    vbytes = int(vlen.sum())
    di_per = 0.450e9 / npost                      # measured varbyte bytes per posting
    print(f"  {nterms:,} terms, {npost:,} postings, vocab strings ~{vbytes/1e6:.0f} MB")
    print(f"  quantized 0.68 GB = di 0.45 (varbyte) + tf4 0.18 + vocab/ptr ~0.05\n")
    print(f"  {'df bucket':<10}{'#terms':>13}{'%terms':>8}{'postings':>15}{'%post':>8}{'vocab MB':>10}")
    for lo, hi, lab in [(1, 1, '1'), (2, 9, '2-9'), (10, 99, '10-99'), (100, 999, '100-999'), (1000, 10**15, '1000+')]:
        m = (df >= lo) & (df <= hi); nt = int(m.sum()); pp = int(df[m].sum()); vb = int(vlen[m].sum())
        print(f"  {lab:<10}{nt:>13,}{nt/nterms*100:>7.1f}%{pp:>15,}{pp/npost*100:>7.2f}%{vb/1e6:>9.1f}")

    print()
    for K in (2, 10, 100):
        m = df < K; nt = int(m.sum()); pp = int(df[m].sum()); vb = int(vlen[m].sum())
        saved = pp * di_per + pp * 0.5 + vb + nt * 8        # di + tf4 + vocab + ptr(8B/term)
        print(f"  drop/derive df<{K}: {nt:,} terms ({nt/nterms*100:.0f}% of vocab), {pp:,} postings "
              f"({pp/npost*100:.1f}%) -> save ~{saved/1e6:.0f} MB = {saved/0.68e9*100:.1f}% of the index")

    qs = []
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                qs.append(a[1])
    random.Random(0).shuffle(qs)
    nq = 2000; rely2 = rely10 = 0; used = 0
    for q in qs[:nq]:
        ws = [w for w in set(stoks(q)) if w in idx.tid]
        if not ws:
            continue
        used += 1
        mindf = min(int(idx.ptr[idx.tid[w] + 1] - idx.ptr[idx.tid[w]]) for w in ws)
        if mindf < 2:
            rely2 += 1
        if mindf < 10:
            rely10 += 1
    print(f"\n  query reliance (n={used}): rarest query term has df<2 in {rely2/used*100:.1f}% of queries, "
          f"df<10 in {rely10/used*100:.1f}%")
    print(f"  -> dropping df<K hurts exactly those queries (the rare term IS the address). composite/free-token")
    print(f"     wins on VOCAB + pair-corridors (we don't store those anyway); the postings BULK is common")
    print(f"     atomic primes -- not composites -- so it is bounded there. subword-derive keeps rare terms but")
    print(f"     trades stored bytes for noisy/slower meets at query time.")


if __name__ == "__main__":
    main()
