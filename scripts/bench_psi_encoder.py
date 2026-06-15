#!/usr/bin/env python3
"""
Bench - the psi-encoder geodesic rerank: does the formula encoding add semantic
signal BM25 lacks?

Builds a 24-D psi vector per doc from the wing formula (8 wings x (Re z, Im z,
zeta) on each word's letter-prime chain, idf-weighted doc mean), then reranks
BM25 candidates by cosine and by geodesic distance in psi-space, blended with
BM25 (RRF). Measures on real scifact vs BM25 (0.700) and the v10 (0.78).

Honest question: the v10's 0.78 uses a 24-D Aethos encoder + geodesic. Is the
psi encoding SEMANTIC (lifts beyond lexical) or surface (letter-based, behaves
like the lexical attempts)? Measured, not assumed.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind
from core.primes import chain_primes

LP = {chr(ord("a") + i): chain_primes(26)[i] for i in range(26)}


def find_ds(name):
    for c in [os.environ.get("BEIR_DATA_DIR", ""), "beir_datasets",
              r"C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets"]:
        if c and (Path(c) / name / "corpus.jsonl").exists():
            return Path(c) / name
    sys.exit("dataset not found")


_wordvec = {}


def word_psi(w):
    """24-D psi vector for a word: 8 wings x (Re z, Im z, zeta) on its
    distinct-letter prime chain (n=7, VA1)."""
    v = _wordvec.get(w)
    if v is not None:
        return v
    chain = tuple(sorted({LP[c] for c in w if c in LP}))
    if len(chain) < 1:
        v = np.zeros(24, dtype=np.float32)
    else:
        feats = []
        for wing in range(1, 9):
            psi = wing_transform(BranchKind.VA1, chain, 7, wing)
            x, y, z = psi.coord
            feats.extend((x, y, z))
        v = np.array(feats, dtype=np.float32)
    _wordvec[w] = v
    return v


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    root = find_ds(ds)
    corpus, queries, qrels = {}, {}, {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        corpus[o["_id"]] = o.get("title", "") + " " + o.get("text", "")
    for line in open(root / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]
    r = csv.reader(open(root / "qrels" / "test.tsv", encoding="utf-8"), delimiter="\t")
    next(r)
    for qid, cid, sc in r:
        qrels.setdefault(qid, {})[cid] = int(sc)
    test_q = [q for q in qrels if q in queries]
    print(f"{ds}: {len(corpus)} docs, {len(test_q)} queries")

    idx = AppendOnlyLatticeIndex()
    for d in corpus:
        idx.add(d, corpus[d])
    N = len(idx.alive)

    # idf for word weighting in the psi mean
    def widf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 1.0

    # encode every doc into a normalized 24-D psi vector
    doc_vec = {}
    for d, txt in corpus.items():
        acc = np.zeros(24, dtype=np.float32)
        wsum = 0.0
        for w in words(txt):
            iw = widf(w)
            acc += iw * word_psi(w)
            wsum += iw
        if wsum > 0:
            acc /= wsum
        nrm = np.linalg.norm(acc)
        doc_vec[d] = acc / nrm if nrm > 0 else acc

    def query_vec(q):
        acc = np.zeros(24, dtype=np.float32)
        wsum = 0.0
        for w in words(q):
            iw = widf(w)
            acc += iw * word_psi(w)
            wsum += iw
        if wsum > 0:
            acc /= wsum
        nrm = np.linalg.norm(acc)
        return acc / nrm if nrm > 0 else acc

    def ndcg10(ranked, rels):
        dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels.values(), reverse=True)[:10]))
        return dcg / idcg if idcg else 0.0

    def recall10(ranked, rels):
        rel = {d for d, s in rels.items() if s > 0}
        return len(set(ranked[:10]) & rel) / len(rel) if rel else 0.0

    def rrf(rankings, k=60):
        sc = defaultdict(float)
        for rk in rankings:
            for i, d in enumerate(rk):
                sc[d] += 1.0 / (k + i + 1)
        return sc

    def evaluate(fn):
        nd = rc = 0.0
        for qid in test_q:
            ranked = fn(queries[qid])
            nd += ndcg10(ranked, qrels[qid])
            rc += recall10(ranked, qrels[qid])
        return nd / len(test_q), rc / len(test_q)

    def bm25_then_psi(query, blend):
        base = idx.search(query, 50)
        if not base:
            return base
        qv = query_vec(query)
        cos = {d: float(np.dot(qv, doc_vec[d])) for d in base}
        psi_rank = sorted(base, key=lambda d: cos[d], reverse=True)
        if blend == "cos_rerank":
            return psi_rank[:10]
        if blend == "rrf":
            fused = rrf([base, psi_rank])
            return sorted(fused, key=lambda d: fused[d], reverse=True)[:10]
        return base[:10]

    print("\n  method                          | nDCG@10 | Recall")
    print("  " + "-" * 50)
    nd, rc = evaluate(lambda q: idx.search(q, 10))
    print(f"  {'BM25+positional':<31} | {nd:>7.4f} | {rc:>6.4f}")
    bm = nd
    nd, rc = evaluate(lambda q: bm25_then_psi(q, "cos_rerank"))
    print(f"  {'psi-cosine rerank (pure)':<31} | {nd:>7.4f} | {rc:>6.4f}")
    nd, rc = evaluate(lambda q: bm25_then_psi(q, "rrf"))
    print(f"  {'BM25 (+) psi-cosine RRF':<31} | {nd:>7.4f} | {rc:>6.4f}")

    print(f"\n  references: BM25+positional {bm:.4f}, v10 UltraFast 0.781")
    print(f"  (psi here is letter-prime based -> surface geometry; semantic")
    print(f"  lift would require a co-occurrence/correlation encoding, not")
    print(f"  letters. This isolates whether the FORMULA encoding alone is")
    print(f"  semantic.)")


if __name__ == "__main__":
    main()
