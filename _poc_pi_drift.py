#!/usr/bin/env python3
"""π-DRIFT proof-of-concept (Timothy's #1 grand-vision idea, on the measurement stand).

The claim: a 3-way intersection of words is a DIMENSION; a word DRIFTS toward it; the constructive-π
geometry (the 32-quadrant phase) MEASURES how far it drifted; that drift is a free relevance signal.

This isolates the claim with a 3-ARM test so we learn exactly what π buys:
  ARM 1  lexical (BM25-on-lattice)                         -- the baseline
  ARM 2  lexical + plain co-occurrence boost (NO π)        -- the known co-occurrence channel
  ARM 3  lexical + π-DRIFT boost (co-occurrence × π-phase-closeness to the query's 3-way anchor)
If ARM3 > ARM2 > ARM1, the π geometry adds non-redundant signal. If ARM3 ≈ ARM2, π is decorative on top of
co-occurrence. If both ≈ ARM1, the channel is redundant with lexical. Honest, two-sided, measured.

Drift geometry uses Timothy's CONSTRUCTIVE π (aethos_fused_meet.pi_phases, no math.pi) + the 3-way meet cell
(aethos_semantic_lattice.triple_cell). θ(prime)=pi_phases(32)[prime%32]; anchor phase = phase of the query's
rarest-3 triple_cell; closeness=(1+cos(θ_w−θ_A))/2."""
import os, sys, math, itertools
os.environ.setdefault("PYTHONUTF8", "1")
from collections import defaultdict
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10
from aethos_fused_meet import pi_phases
from aethos_semantic_lattice import triple_cell

DFMIN = 2; DFCAPFRAC = 0.10; TOPK = 200; KRRF = 60
PHASES32 = pi_phases(32)   # constructive-π angles of the 32 sub-quadrants (no math.pi)


def theta(prime):
    return PHASES32[prime % 32]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive); cap = DFCAPFRAC * N

    def rares(text):
        out = []
        for tok in idx._multiview(text):
            if tok[0] != "w":
                continue
            p = idx.token_prime.get(tok)
            if p is not None and DFMIN <= idx.df[p] <= cap:
                out.append(p)
        return sorted(set(out), key=lambda p: idx.df[p])   # rarest first

    drare = {d: set(rares(t)) for d, t in corpus.items()}

    def idf(p):
        return math.log(1 + (N - idx.df[p] + 0.5) / (idx.df[p] + 0.5))

    def anchor_phase(qr):
        """The query's 3-way anchor (3 rarest primes) -> triple_cell coord -> a π-phase angle."""
        if len(qr) < 3:
            base = qr if qr else [2]
            s = sum(base)
        else:
            a, p, q = qr[0], qr[1], qr[2]
            cell = triple_cell(a, p, q)        # (sum, interior, sum)
            s = cell[0]                        # the locked zeta = sum
        return PHASES32[int(s) % 32]

    def boosts(q):
        """Return (cooc, drift) per candidate doc for query q.
        cooc  = Σ idf(shared rare prime)                          (no π)
        drift = Σ idf(shared) × closeness(θ(shared), θ_anchor)    (π-phase weighted)"""
        qr = rares(queries[q])
        th_a = anchor_phase(qr)
        qset = set(qr)
        cooc = defaultdict(float); drift = defaultdict(float)
        for p in qr:
            w = idf(p)
            close = 0.5 * (1.0 + math.cos(theta(p) - th_a))   # ∈ [0,1], π-phase closeness to anchor
            for d in idx.postings.get(p, {}):
                cooc[d] += w
                drift[d] += w * close
        return cooc, drift

    def evalrun(channel, beta):
        nd = 0.0
        for q in test_ids:
            lex = list(idx.search(queries[q], TOPK))
            if not lex:
                continue
            cooc, drift = boosts(q)
            sc = drift if channel == "drift" else cooc
            lex_rank = {d: i for i, d in enumerate(lex)}
            br = sorted(lex, key=lambda d: -sc.get(d, 0.0))
            br_rank = {d: i for i, d in enumerate(br)}
            fused = sorted(lex, key=lambda d: -(1.0 / (KRRF + lex_rank[d]) + beta / (KRRF + br_rank[d])))
            nd += ndcg10(fused, test_q[q])
        return nd / len(test_ids)

    base = np.mean([ndcg10(list(idx.search(queries[q], TOPK)), test_q[q]) for q in test_ids])
    print(f"\nπ-DRIFT PoC — {name}: {N:,} docs, {len(test_ids)} test queries", flush=True)
    print(f"  ARM 1  lexical baseline:            nDCG@10 {base:.4f}", flush=True)
    for ch, label in (("cooc", "ARM 2  + co-occurrence (no π)"), ("drift", "ARM 3  + π-DRIFT (phase-weighted)")):
        best = (base, 0.0)
        for beta in (0.05, 0.1, 0.25, 0.5, 1.0):
            nd = evalrun(ch, beta)
            if nd > best[0]:
                best = (nd, beta)
        print(f"  {label:<34} nDCG@10 {best[0]:.4f}  ({best[0]-base:+.4f})  @beta={best[1]}", flush=True)
    print(f"\n  read: ARM3>ARM2 ⇒ π adds signal; ARM3≈ARM2 ⇒ π decorative; both≈ARM1 ⇒ redundant w/ lexical", flush=True)


if __name__ == "__main__":
    main()
