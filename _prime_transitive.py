#!/usr/bin/env python3
"""Frequency-weighted TRANSITIVE correlation (Timothy: primes hold frequency = weight; rare connector =
reliable link). Build at ingest: rare-term co-occurrence -> PMI edges. Transitive closure a~b~x kept ONLY
through RARE connectors b (df_b <= CONN_DF), weighted by the connector's rarity (weakest-link). Expand a
query's rare terms with their direct + transitive correlates, weight = corr_strength x idf(correlate) so
common correlates can't drift it. RRF-fuse with lexical, measure nDCG@10. The rare-connector filter IS the
credit-assignment the prime frequency provides."""
import os, sys, math
os.environ.setdefault("PYTHONUTF8", "1")
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

DFMIN = 2; DFCAPFRAC = 0.05      # rare terms
CONN_DF = 40                      # connector must be THIS rare to carry a transitive link (specific)
MIN_COOC = 3; PMI_MIN = 2.5
NEIGH = 8; TOPK = 200


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive); cap = DFCAPFRAC * N

    def rares(text):
        s = set()
        for tok in idx._multiview(text):
            if tok[0] != "w":
                continue
            p = idx.token_prime.get(tok)
            if p is not None and DFMIN <= idx.df[p] <= cap:
                s.add(p)
        return s
    drare = {d: rares(t) for d, t in corpus.items()}

    def idf(p):
        return math.log(1 + (N - idx.df[p] + 0.5) / (idx.df[p] + 0.5))

    # co-occurrence + PMI (frequency-aware by construction)
    cooc = Counter()
    for s in drare.values():
        ss = sorted(s)
        for i in range(len(ss)):
            for j in range(i + 1, len(ss)):
                cooc[(ss[i], ss[j])] += 1
    pmi = {}
    nbr = defaultdict(list)               # direct correlates
    for (a, b), c in cooc.items():
        if c < MIN_COOC:
            continue
        v = math.log((c * N) / (idx.df[a] * idx.df[b]))
        if v >= PMI_MIN:
            pmi[(a, b)] = v; pmi[(b, a)] = v
            nbr[a].append(b); nbr[b].append(a)

    # transitive correlates a~x via a RARE connector b only (the frequency filter)
    def correlates(a):
        scores = defaultdict(float)
        for b in nbr.get(a, ()):                      # direct
            scores[b] = max(scores[b], pmi[(a, b)])
        for b in nbr.get(a, ()):                      # 2-hop through rare connector b
            if idx.df[b] > CONN_DF:
                continue
            link = pmi[(a, b)]
            for x in nbr.get(b, ()):
                if x == a:
                    continue
                scores[x] = max(scores[x], min(link, pmi[(b, x)]) * 0.7)   # weakest-link, hop-discounted
        # weight each correlate by its OWN rarity (idf): common correlates -> low weight
        return sorted(((x, sc * idf(x)) for x, sc in scores.items()), key=lambda t: -t[1])[:NEIGH]

    corr_cache = {}
    def expand(qr):
        ex = defaultdict(float)
        for a in qr:
            if a not in corr_cache:
                corr_cache[a] = correlates(a)
            for x, w in corr_cache[a]:
                if x not in qr:
                    ex[x] = max(ex[x], w)
        return ex

    def evalrun(beta, krrf=60):
        nd = 0.0
        for q in test_ids:
            lex = list(idx.search(queries[q], TOPK))
            if not lex:
                continue
            qr = rares(queries[q]); ex = expand(qr)
            lex_rank = {d: i for i, d in enumerate(lex)}
            bsc = [(d, sum(w for x, w in ex.items() if x in drare.get(d, ()))) for d in lex]
            br = [d for d, s in sorted(bsc, key=lambda t: -t[1])]
            br_rank = {d: i for i, d in enumerate(br)}
            fused = sorted(lex, key=lambda d: -(1.0 / (krrf + lex_rank[d]) + beta / (krrf + br_rank[d])))
            nd += ndcg10(fused, test_q[q])
        return nd / len(test_ids)

    base = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])
    print(f"\n{name}: {N:,} docs | frequency-weighted transitive corr (rare connector df<={CONN_DF})", flush=True)
    print(f"  lexical:  nDCG@10 {base:.4f}", flush=True)
    best = (base, 0.0)
    for beta in (0.05, 0.1, 0.2, 0.35):
        nd = evalrun(beta)
        if nd > best[0]:
            best = (nd, beta)
        print(f"  + transitive (RRF b={beta:<4}): nDCG@10 {nd:.4f}  ({nd-base:+.4f})", flush=True)
    print(f"  BEST: {best[0]:.4f} at b={best[1]}  ({best[0]-base:+.4f})", flush=True)


if __name__ == "__main__":
    main()
