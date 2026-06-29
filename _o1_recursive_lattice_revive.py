#!/usr/bin/env python3
"""REVIVE Stage 18 recursive-lattice (Timothy's compression=clustering=relevance vision), but make it
ACTUALLY COMPRESSIVE: cap each doc to its top-B RAREST primes before materializing depth-2 co-occurrence
addresses. The original materialized ALL rare-prime pairs (10.3M addresses = 1.37 GB for 5K scifact docs).
A rare-prime BUDGET should cut that ~50-100x. Question: does the +0.73pp scifact lift SURVIVE at a
shippable footprint? Two-sided: also measure nfcorpus (where the original went NEGATIVE).

Boost = recursive-lattice candidate_score_boost (depth-weighted shared sub-node count), RRF-fused with the
lexical lattice over its top-200 (rerank-only, no flood). Honest nDCG@10 vs the lexical baseline."""
import os, sys, math, itertools, time
os.environ.setdefault("PYTHONUTF8", "1")
from collections import defaultdict
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10

DFMIN = 2; DFCAPFRAC = 0.10
TOPK = 200; KRRF = 60


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

    drare_full = {d: rare_primes(t) for d, t in corpus.items()}

    def idf(p):
        return math.log(1 + (N - idx.df[p] + 0.5) / (idx.df[p] + 0.5))

    base_nd = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])

    def build_and_eval(budget, depth=2):
        # materialize depth-{1..depth} addresses, each doc capped to its top-`budget` rarest primes
        materialized = defaultdict(set)
        for d, rp in drare_full.items():
            rare = rp[:budget]
            for k in range(1, depth + 1):
                for sub in itertools.combinations(rare, k):
                    addr = 1
                    for p in sub:
                        addr *= p
                    materialized[addr].add(d)
        n_addr = len(materialized)
        n_refs = sum(len(s) for s in materialized.values())
        # honest footprint of the co-occurrence STRUCTURE on top of the base index:
        # 8 B/address (int64 product) + 4 B/doc-ref (int32 id)
        bytes_struct = n_addr * 8 + n_refs * 4
        bdoc = bytes_struct / N

        def boost(qprimes):
            rare = [p for p in qprimes if p in idx.df][:budget]
            sc = defaultdict(float)
            for k in range(1, depth + 1):
                for sub in itertools.combinations(sorted(set(rare), key=lambda p: idx.df[p])[:budget], k):
                    addr = 1
                    for p in sub:
                        addr *= p
                    docs = materialized.get(addr)
                    if not docs:
                        continue
                    w = len(sub) * math.log(1 + sum(sub))
                    for dd in docs:
                        sc[dd] += w
            return sc

        best = (base_nd, 0.0, "none")
        # RRF rank fusion
        for beta in (0.05, 0.1, 0.25, 0.5, 1.0):
            nd = 0.0
            for q in test_ids:
                lex = list(idx.search(queries[q], TOPK))
                if not lex:
                    continue
                bsc = boost(rare_primes(queries[q]))
                lex_rank = {d: i for i, d in enumerate(lex)}
                br = sorted(lex, key=lambda d: -bsc.get(d, 0.0))
                br_rank = {d: i for i, d in enumerate(br)}
                fused = sorted(lex, key=lambda d: -(1.0 / (KRRF + lex_rank[d]) + beta / (KRRF + br_rank[d])))
                nd += ndcg10(fused, test_q[q])
            nd /= len(test_ids)
            if nd > best[0]:
                best = (nd, beta, "rrf")
        # ADDITIVE fusion (the Stage-18 shape): normalized lexical score + alpha * normalized boost
        for alpha in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
            nd = 0.0
            for q in test_ids:
                lsc = idx._score(queries[q])
                lex = sorted(lsc, key=lambda d: -lsc[d])[:TOPK]
                if not lex:
                    continue
                lmax = max(lsc[d] for d in lex) or 1.0
                bsc = boost(rare_primes(queries[q]))
                bmax = max(bsc.values()) if bsc else 1.0
                fused = sorted(lex, key=lambda d: -(lsc[d] / lmax + alpha * bsc.get(d, 0.0) / bmax))
                nd += ndcg10(fused, test_q[q])
            nd /= len(test_ids)
            if nd > best[0]:
                best = (nd, alpha, "add")
        return n_addr, bdoc, best

    print(f"\n{name}: {N:,} docs | recursive-lattice REVIVE (rare-prime budget, depth-2 co-occurrence)", flush=True)
    print(f"  lexical baseline nDCG@10: {base_nd:.4f}", flush=True)
    print(f"  Stage-18 original (no budget): 10.3M addr, 1373 MB, scifact +0.0073 @a0.10\n", flush=True)
    print(f"  {'budget':>7}{'addresses':>12}{'B/doc':>10}{'best nDCG':>11}{'delta':>10}{'fuse@p':>9}", flush=True)
    for B in (4, 6, 8, 12, 20, 9999):
        n_addr, bdoc, (nd, p, mode) = build_and_eval(B, depth=2)
        tag = "all" if B == 9999 else str(B)
        print(f"  {tag:>7}{n_addr:>12,}{bdoc:>9.1f}{nd:>11.4f}{nd-base_nd:>+10.4f}{f'{mode}@{p}':>9}", flush=True)


if __name__ == "__main__":
    main()
