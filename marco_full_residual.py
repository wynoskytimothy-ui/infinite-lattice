#!/usr/bin/env python3
"""Glass-box battery on the RESIDUAL golds -- the ~19% that are in the rare-word pool but the
lexical 2-way meet AND the rare-pair company both miss (so: only one rare word, no corridor
company). What third bridge recovers them? Tests the user's hypotheses, each vs a random pool
doc (the discrimination baseline) so we see what recovers gold SPECIFICALLY:

  1 subword bridge   : a query rare word shares a 4+ char substring with a gold term
                       (morphology / compounds the stemmer missed: nanofiber~fiber, biomaterial~material)
  2 single company   : gold has the corridor company of EITHER rare word (looser than 'both')
  3 second-order     : gold has a term 2 hops out -- in the corridor of a corridor term
  4 any-term company : gold has the company of ANY query term (idf>=2), not just the 2 rarest
  5 medium bridge    : a medium query term (2<=idf<4) is present (extra lexical anchor)

Full 8.8M, 3000 dev q. Reports, on the residual golds: gold-coverage vs random, and the extra
recovery the best signal buys beyond the 0.815 the meet+company already reached.
"""
import sys, random as rnd
from collections import defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, train_corridors, RARE, QGATE


def shares_subword(rares, terms, k=4):
    """does any rare query word share a k+ char contiguous substring with any doc term?"""
    for r in rares:
        if len(r) < k:
            continue
        grams = {r[i:i + k] for i in range(len(r) - k + 1)}
        for t in terms:
            if t == r or len(t) < k:
                continue
            for i in range(len(t) - k + 1):
                if t[i:i + k] in grams:
                    return True
    return False


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

    def posting(term):
        i = idx.tid.get(term)
        return None if i is None else idx.di[int(idx.ptr[i]):int(idx.ptr[i + 1])]

    SIGS = ["subword", "single_co", "second_ord", "anyterm_co", "medium"]
    gcov = defaultdict(int); rcov = defaultdict(int)   # gold / random coverage on residual
    n_resid = 0; n_q = 0; recovered = defaultdict(int)

    for q in qids:
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        rare = sorted([w for w in qs if idx.idf_of(w) >= RARE], key=lambda w: -idx.idf_of(w))
        if len(rare) < 2:
            continue
        r1, r2 = rare[0], rare[1]
        C1 = set(dt for dt, _ in gold.get(r1, []))
        C2 = set(dt for dt, _ in gold.get(r2, []))
        if not C1 or not C2:
            continue
        n_q += 1
        medium = [w for w in qs if 2.0 <= idx.idf_of(w) < RARE]
        anyco = set()
        for qt in qs:
            anyco |= set(dt for dt, _ in gold.get(qt, []))
        second = set()                                   # 2-hop: corridor of corridor terms
        for ct in (C1 | C2):
            second |= set(dt for dt, _ in gold.get(ct, []))
        second -= (C1 | C2)

        def signals(toks):
            return {
                "subword": shares_subword(rare, toks),
                "single_co": bool(toks & C1) or bool(toks & C2),
                "second_ord": bool(toks & second),
                "anyterm_co": bool(toks & anyco),
                "medium": any(m in toks for m in medium),
            }

        # random pool doc (has a rare word) -- discrimination baseline
        p1, p2 = posting(r1), posting(r2)
        pool = np.union1d(p1, p2) if (p1 is not None and p2 is not None) else (p1 if p1 is not None else p2)
        rt = set(stoks(idx.text(int(pool[rng.randrange(len(pool))]))))
        rsig = signals(rt)

        for gp in qrels[q]:
            gt = set(stoks(idx.text(int(gp))))
            lex2 = (r1 in gt) and (r2 in gt)
            compboth = bool(gt & C1) and bool(gt & C2)
            in_pool = (r1 in gt) or (r2 in gt) or bool(gt & set(rare))
            if (lex2 or compboth) or not in_pool:
                continue                                  # not residual (caught, or unreachable)
            n_resid += 1
            gsig = signals(gt)
            for s in SIGS:
                if gsig[s]:
                    gcov[s] += 1
                if rsig[s]:
                    rcov[s] += 1
            # a "recovered" residual gold = the signal fires on gold but not the random doc
            for s in SIGS:
                if gsig[s] and not rsig[s]:
                    recovered[s] += 1
            break   # one residual gold per query is enough for the profile

    print(f"\nRESIDUAL GLASS-BOX -- full 8.8M, {n_q} queries (>=2 rare+corridors), "
          f"{n_resid} residual golds (in pool, missed by meet+company)\n")
    print(f"   {'signal':<14}{'GOLD':>8}{'random':>9}{'gold/rand':>11}")
    for s in SIGS:
        g = gcov[s] / max(1, n_resid); r = rcov[s] / max(1, n_resid)
        ratio = g / r if r > 0 else float('inf')
        print(f"   {s:<14}{g:>8.3f}{r:>9.3f}{('inf' if r==0 else f'{ratio:.1f}x'):>11}")
    print(f"\n   recovered = residual golds where the signal fires on GOLD but not the random doc:")
    for s in SIGS:
        print(f"     {s:<14}{recovered[s]/max(1,n_resid):>7.3f}  ({recovered[s]}/{n_resid})")
    print(f"\n   high gold + low random + high recovered = a rule that recovers the residual cleanly.")


if __name__ == "__main__":
    main()
