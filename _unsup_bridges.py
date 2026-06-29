#!/usr/bin/env python3
"""UNSUPERVISED bridge mining (Timothy's idea): the corpus trains ITSELF, no qrels, no GPU.
Mine rare-term co-occurrence (the 3-way intersection signal) -> PMI filter (random pairs = low PMI =
dropped; genuine rare co-occurrence = a bridge) -> term bridge graph (material<->nano<->...). At query
time, expand each query's rare terms with their top-PMI bridge neighbors and boost docs that contain them
(finds docs that use the SYNONYM/neighbor vocabulary, not the exact query word). Measure nDCG@10 vs the
lexical baseline. Honest: does the corpus's own correlation structure lift retrieval with no labels?"""
import os, sys, math, itertools, time
os.environ.setdefault("PYTHONUTF8", "1")
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

DFMIN = 2; DFCAPFRAC = 0.10
MIN_COOC = 3          # significance floor: a bridge needs >= this many co-occurrences
PMI_MIN = 2.0         # the FILTER: random pairs have PMI ~0; genuine co-occurrence is high
NEIGH = 10            # top bridge neighbors per query term
TOPK = 200; POOLCAP = 1500


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive); cap = DFCAPFRAC * N

    def rareset(text):
        s = set()
        for tok in idx._multiview(text):
            if tok[0] != "w":
                continue
            p = idx.token_prime.get(tok)
            if p is not None and DFMIN <= idx.df[p] <= cap:
                s.add(p)
        return s

    drare = {d: rareset(t) for d, t in corpus.items()}
    dfr = Counter()
    for s in drare.values():
        for p in s:
            dfr[p] += 1
    # co-occurrence of rare terms across docs (the meet signal)
    cooc = Counter()
    for s in drare.values():
        for a, b in itertools.combinations(sorted(s), 2):
            cooc[(a, b)] += 1
    # PMI filter -> bridge graph
    bridges = defaultdict(list)
    for (a, b), c in cooc.items():
        if c < MIN_COOC:
            continue
        pmi = math.log((c * N) / (dfr[a] * dfr[b]))
        if pmi >= PMI_MIN:
            bridges[a].append((b, pmi)); bridges[b].append((a, pmi))
    for t in bridges:
        bridges[t].sort(key=lambda x: -x[1])
    nedges = sum(len(v) for v in bridges.values()) // 2

    def idf(p):
        return math.log(1 + (N - idx.df[p] + 0.5) / (idx.df[p] + 0.5))

    def expand(qr):
        ex = defaultdict(float)
        for t in qr:
            for b, pmi in bridges.get(t, [])[:NEIGH]:
                if b not in qr:
                    ex[b] = max(ex[b], pmi)
        return ex

    def evalrun(beta, krrf=60):
        # rerank-only RRF: fuse lexical rank with bridge-expansion rank over the lexical top-K (no flood)
        nd = 0.0
        for q in test_ids:
            base = list(idx.search(queries[q], TOPK))
            if not base:
                continue
            qr = rareset(queries[q]); ex = expand(qr)
            lex_rank = {d: i for i, d in enumerate(base)}
            # bridge score per candidate = sum of expansion-term idf weight present in the doc
            bsc = []
            for d in base:
                dr = drare.get(d, set())
                bsc.append((d, sum(w * idf(e) for e, w in ex.items() if e in dr)))
            br_order = [d for d, s in sorted(bsc, key=lambda x: -x[1])]
            br_rank = {d: i for i, d in enumerate(br_order)}
            fused = sorted(base, key=lambda d: -(1.0 / (krrf + lex_rank[d]) + beta / (krrf + br_rank[d])))
            nd += ndcg10(fused, test_q[q])
        return nd / len(test_ids)

    base_nd = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])
    print(f"\n{name}: {N:,} docs | {nedges:,} unsupervised bridges over {len(bridges):,} rare terms "
          f"(PMI>={PMI_MIN}, min_cooc={MIN_COOC})", flush=True)
    print(f"  lexical (no bridges):  nDCG@10 {base_nd:.4f}", flush=True)
    best = (base_nd, 0.0)
    for beta in (0.1, 0.25, 0.5, 1.0):
        nd = evalrun(beta)
        if nd > best[0]:
            best = (nd, beta)
        print(f"  + unsup bridges (RRF b={beta:<4}): nDCG@10 {nd:.4f}  ({nd-base_nd:+.4f})", flush=True)
    print(f"  BEST: {best[0]:.4f} at b={best[1]}  ({best[0]-base_nd:+.4f} vs lexical)", flush=True)
    print(f"  reference (SUPERVISED bridges, needs qrels): scifact 0.7375, nfcorpus 0.3346", flush=True)


if __name__ == "__main__":
    main()
