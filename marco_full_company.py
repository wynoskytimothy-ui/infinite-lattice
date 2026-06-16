#!/usr/bin/env python3
"""Can the COMPANY the 2 rarest words keep (their corridor intersection / union) recover the
gold docs the LEXICAL 2-way meet misses? The meet on MEANING instead of letters.

The lexical 2-way meet (gold contains both r1 AND r2) keeps the gold ~65.6% -- loses ~34%.
This tests: for those misses, does the gold contain the COMPANY of both rare words (>=1 corridor
term of r1 AND >=1 of r2), even without the literal words? And is that gold-SPECIFIC (vs a
random pool doc)? Full 8.8M, 3000 dev q, queries with >=2 rare words (idf>=4) that have corridors.

  GOLD vs RANDOM pool doc:  lexical-2way / company-of-both / shared-company coverage
  RECOVERY = golds with company-of-both but NOT in the lexical 2-way meet.
"""
import sys, random as rnd
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    idx = FullIndex()
    gold = train_corridors(idx)
    rng = rnd.Random(42)

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
    rng.shuffle(qids); qids = qids[:nq]

    n = 0
    g_lex2 = g_compboth = g_shared = 0          # gold coverage
    r_lex2 = r_compboth = r_shared = 0          # random pool-doc coverage
    recov = 0                                   # gold: company-of-both but NOT lexical 2-way
    only_company = 0                            # gold: company-of-both and lacks at least one rare word

    def posting(term):
        i = idx.tid.get(term)
        if i is None:
            return None
        return idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]

    for q in qids:
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = sorted([w for w in qs if idx.idf_of(w) >= RARE], key=lambda w: -idx.idf_of(w))
        if len(rare) < 2:
            continue
        r1, r2 = rare[0], rare[1]
        C1 = set(dt for dt, _ in gold.get(r1, []))
        C2 = set(dt for dt, _ in gold.get(r2, []))
        if not C1 or not C2:                    # need corridors for both rare words
            continue
        shared = C1 & C2
        n += 1
        # a random doc from the candidate pool (contains r1 or r2) -- the discrimination baseline
        p1 = posting(r1); p2 = posting(r2)
        pool = np.union1d(p1, p2) if (p1 is not None and p2 is not None) else (p1 if p1 is not None else p2)
        rdoc = int(pool[rng.randrange(len(pool))])
        rt = set(stoks(idx.text(rdoc)))
        if (r1 in rt) and (r2 in rt): r_lex2 += 1
        if (rt & C1) and (rt & C2): r_compboth += 1
        if rt & shared: r_shared += 1
        # the gold doc(s)
        g_l = g_c = g_s = g_only = 0
        for gp in qrels[q]:
            gt = set(stoks(idx.text(int(gp))))
            lex2 = (r1 in gt) and (r2 in gt)
            compboth = bool(gt & C1) and bool(gt & C2)
            g_l = max(g_l, lex2); g_c = max(g_c, compboth); g_s = max(g_s, bool(gt & shared))
            if compboth and not lex2:
                g_only = 1
        g_lex2 += g_l; g_compboth += g_c; g_shared += g_s
        if g_c and not g_l:
            recov += 1

    print(f"\nRARE-PAIR COMPANY recovery -- full 8.8M, {n} queries (>=2 rare words w/ corridors)\n")
    print(f"   {'signal':<26}{'GOLD':>9}{'random pool doc':>18}")
    print(f"   {'lexical 2-way (both words)':<26}{g_lex2/n:>9.3f}{r_lex2/n:>18.3f}")
    print(f"   {'company of BOTH words':<26}{g_compboth/n:>9.3f}{r_compboth/n:>18.3f}")
    print(f"   {'shared company (C1 & C2)':<26}{g_shared/n:>9.3f}{r_shared/n:>18.3f}")
    print(f"\n   RECOVERY: golds with company-of-both but NOT in the lexical 2-way meet: "
          f"{recov}/{n} = {recov/n:.3f}")
    print(f"   combined (lexical 2-way OR company-of-both): {(g_lex2 + recov)/n:.3f}")
    print(f"\n   gold vs random gap = discrimination. If company-of-both is ~as common in random")
    print(f"   docs, it recovers recall but not RANKING (matches everything); gap = real signal.")


if __name__ == "__main__":
    main()
