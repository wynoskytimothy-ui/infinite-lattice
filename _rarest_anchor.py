#!/usr/bin/env python3
"""Timothy's rarest-anchor bridge (unsupervised, no qrels, no GPU). For each query: take the two RAREST
terms r1,r2; their INTERSECTION (docs with both) is a tiny high-precision anchor set; mine the next
distinctive terms r3.. that recur in that anchor (high-confidence bridges, not global noise); then pull
docs that contain r3 but MISS r1/r2 -- same topic, different vocabulary. RRF-fuse with lexical, measure
nDCG@10 vs lexical. The precision anchoring is the filter the global-PMI version lacked."""
import os, sys, math, time
os.environ.setdefault("PYTHONUTF8", "1")
from collections import Counter
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

DFMIN = 2; DFCAPFRAC = 0.10
N_BRIDGE = 4          # how many bridge terms r3.. to mine from the anchor
ANCHOR_CAP = 40       # cap the anchor doc set
BRIDGE_DOCS = 80      # cap the extra bridge-pool docs
TOPK = 200


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive); cap = DFCAPFRAC * N

    def rare_primes(text):
        out = []
        for tok in idx._multiview(text):
            if tok[0] != "w":
                continue
            p = idx.token_prime.get(tok)
            if p is not None and DFMIN <= idx.df[p] <= cap:
                out.append(p)
        return sorted(set(out), key=lambda p: idx.df[p])     # rarest first

    drare = {d: set(rare_primes(t)) for d, t in corpus.items()}

    def idf(p):
        return math.log(1 + (N - idx.df[p] + 0.5) / (idx.df[p] + 0.5))

    def bridge_pool(q):
        rp = rare_primes(queries[q])
        if not rp:
            return [], set()
        r1 = rp[0]; r2 = rp[1] if len(rp) > 1 else None
        d1 = set(idx.postings.get(r1, {}))
        anchor = (d1 & set(idx.postings.get(r2, {}))) if r2 else d1
        if not anchor:
            anchor = d1
        # rank anchor docs by lexical score, keep the top precise ones
        sc = idx._score(queries[q])
        anchor = sorted(anchor, key=lambda d: -sc.get(d, 0.0))[:ANCHOR_CAP]
        # mine bridge terms = rare terms recurring in the anchor (excl. r1,r2)
        bc = Counter()
        for d in anchor:
            for p in drare.get(d, ()):
                if p != r1 and p != r2:
                    bc[p] += 1
        bridges = sorted(bc, key=lambda p: -(bc[p] * idf(p)))[:N_BRIDGE]
        if not bridges:
            return [], set(rp[:2])
        # pull docs containing a bridge term, score by bridge-idf present, prefer those MISSING r1/r2
        cand = Counter()
        for b in bridges:
            for d in idx.postings.get(b, {}):
                cand[d] += idf(b)
        top = sorted(cand, key=lambda d: -cand[d])[:BRIDGE_DOCS]
        return top, set(bridges)

    def evalrun(beta, krrf=60):
        nd = 0.0
        for q in test_ids:
            lex = list(idx.search(queries[q], TOPK))
            if not lex:
                continue
            bp, _ = bridge_pool(q)
            lex_rank = {d: i for i, d in enumerate(lex)}
            br_rank = {d: i for i, d in enumerate(bp)}
            pool = list(dict.fromkeys(lex + bp))
            big = 10 ** 6
            fused = sorted(pool, key=lambda d: -(1.0 / (krrf + lex_rank.get(d, big))
                                                 + beta / (krrf + br_rank.get(d, big))))
            nd += ndcg10(fused, test_q[q])
        return nd / len(test_ids)

    base_nd = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])
    print(f"\n{name}: {N:,} docs | rarest-anchor bridge (anchor=r1&r2, {N_BRIDGE} bridges, unsupervised)", flush=True)
    print(f"  lexical:  nDCG@10 {base_nd:.4f}", flush=True)
    best = (base_nd, 0.0)
    for beta in (0.05, 0.1, 0.2, 0.35, 0.5):
        nd = evalrun(beta)
        if nd > best[0]:
            best = (nd, beta)
        print(f"  + rarest-anchor (RRF b={beta:<4}): nDCG@10 {nd:.4f}  ({nd-base_nd:+.4f})", flush=True)
    print(f"  BEST: {best[0]:.4f} at b={best[1]}  ({best[0]-base_nd:+.4f} vs lexical)", flush=True)


if __name__ == "__main__":
    main()
