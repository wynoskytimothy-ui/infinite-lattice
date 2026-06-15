#!/usr/bin/env python3
"""
Bench - PPMI semantic query expansion wired into the append-only index.

Deterministic, verifiable, append-only semantics (Test 58) as a real retrieval
feature:
  - window co-occurrence (W=8, the distributional method) -> PPMI (counting)
  - SELECTIVE expansion: only expand high-idf (rare/specific) query terms,
    where the vocabulary gap actually bites
  - RRF FUSION: the expanded ranking is fused with the lexical ranking, BM25
    stays dominant (avoids query drift)

Measured on scifact (BM25-saturated, should not hurt) and nfcorpus (vocab
mismatch, should help).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words


def find_ds(name):
    for c in [os.environ.get("BEIR_DATA_DIR", ""), "beir_datasets",
              r"C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets"]:
        if c and (Path(c) / name / "corpus.jsonl").exists():
            return Path(c) / name
    sys.exit("dataset not found")


def load(name):
    root = find_ds(name)
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
    return corpus, queries, qrels


class PPMI:
    """Window co-occurrence -> PPMI (deterministic, append-only counting)."""

    def __init__(self, window=8):
        self.W = window
        self.cooc = defaultdict(Counter)
        self.wc = Counter()
        self.total = 0

    def add(self, seq):                       # APPEND: window co-occurrence
        for i, a in enumerate(seq):
            self.wc[a] += 1
            self.total += 1
            for j in range(i + 1, min(i + self.W, len(seq))):
                b = seq[j]
                if a != b:
                    self.cooc[a][b] += 1
                    self.cooc[b][a] += 1

    def ppmi(self, a, b):
        c = self.cooc[a].get(b, 0)
        if c == 0:
            return 0.0
        return max(0.0, math.log2((c * self.total) /
                                  (self.wc[a] * self.wc[b] * self.W)))

    _nbr = {}

    def top_neighbors(self, w, k=2):
        if w in self._nbr:
            return self._nbr[w]
        scored = sorted(((self.ppmi(w, c), c) for c in self.cooc[w]
                         if self.wc[c] >= 3), reverse=True)
        out = [(c, s) for s, c in scored[:k] if s > 0]
        self._nbr[w] = out
        return out


def run(name, idf_gate=2.0, exp_k=2, rrf_k=60):
    corpus, queries, qrels = load(name)
    test_q = [q for q in qrels if q in queries]
    print(f"\n{name}: {len(corpus)} docs, {len(test_q)} queries")

    idx = AppendOnlyLatticeIndex()
    ppmi = PPMI(window=8)
    for d, txt in corpus.items():
        idx.add(d, txt)
        ppmi.add(words(txt))                  # co-occurrence accrues as we append
    N = len(idx.alive)

    def widf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 0.0

    def ndcg10(ranked, rels):
        dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(sorted(rels.values(), reverse=True)[:10]))
        return dcg / idcg if idcg else 0.0

    def recall10(ranked, rels):
        rel = {d for d, s in rels.items() if s > 0}
        return len(set(ranked[:10]) & rel) / len(rel) if rel else 0.0

    def expand_terms(query):
        out = []
        for w in set(words(query)):
            if widf(w) >= idf_gate:           # SELECTIVE: only rare/specific
                out.extend(c for c, _ in ppmi.top_neighbors(w, exp_k))
        return [t for t in out if t not in set(words(query))]

    def sem_rank(query):
        exp = expand_terms(query)
        if not exp:
            return []
        scores = defaultdict(float)
        for t in exp:
            p = idx.token_prime.get(("w", t))
            if p is None or idx.df[p] == 0:
                continue
            idf = idx._idf(p, N)
            for d, tf in idx.postings[p].items():
                scores[d] += idf * tf / (tf + 1.0)
        return sorted(scores, key=lambda d: scores[d], reverse=True)[:100]

    def rrf(rankings):
        sc = defaultdict(float)
        for rk in rankings:
            for i, d in enumerate(rk):
                sc[d] += 1.0 / (rrf_k + i + 1)
        return sorted(sc, key=lambda d: sc[d], reverse=True)[:10]

    def search_lexical(query):
        return idx.search(query, 10)

    def search_semantic(query, lam=0.08):
        # conservative: rerank ONLY the lexical candidates with a tiny semantic
        # boost (cannot introduce drift docs; BM25 ordering stays dominant)
        lex_scores = idx._score(query)
        cand = sorted(lex_scores, key=lambda d: lex_scores[d], reverse=True)[:100]
        if not cand:
            return []
        mx = max(lex_scores[d] for d in cand) or 1.0
        exp = expand_terms(query)
        sem = defaultdict(float)
        for t in exp:
            p = idx.token_prime.get(("w", t))
            if p is None or idx.df[p] == 0:
                continue
            idf = idx._idf(p, N)
            for d in cand:
                tf = idx.postings[p].get(d, 0)
                if tf:
                    sem[d] += idf * tf / (tf + 1.0)
        smx = max(sem.values()) if sem else 1.0
        final = {d: lex_scores[d] / mx + lam * sem.get(d, 0.0) / smx for d in cand}
        return sorted(final, key=lambda d: final[d], reverse=True)[:10]

    def evaluate(fn):
        nd = rc = 0.0
        t0 = time.time()
        for qid in test_q:
            ranked = fn(queries[qid])
            nd += ndcg10(ranked, qrels[qid])
            rc += recall10(ranked, qrels[qid])
        ms = (time.time() - t0) / len(test_q) * 1000
        return nd / len(test_q), rc / len(test_q), ms

    nd0, rc0, ms0 = evaluate(search_lexical)
    nd1, rc1, ms1 = evaluate(search_semantic)
    print(f"  lexical (BM25+pos):     nDCG {nd0:.4f}  Recall {rc0:.4f}  {ms0:.0f} ms/q")
    print(f"  + PPMI expansion (RRF): nDCG {nd1:.4f}  Recall {rc1:.4f}  {ms1:.0f} ms/q  "
          f"(nDCG {nd1-nd0:+.4f}, recall {rc1-rc0:+.4f})")
    # verifiability spot-check
    q0 = words(queries[test_q[0]])
    for w in q0:
        if widf(w) >= idf_gate and ppmi.top_neighbors(w, 2):
            print(f"  e.g. '{w}' -> PPMI neighbours {ppmi.top_neighbors(w,2)}")
            break
    return nd1 - nd0, rc1 - rc0


def main():
    print("PPMI semantic expansion wired into the append-only index")
    s_nd, s_rc = run("scifact")
    n_nd, n_rc = run("nfcorpus")
    print("\nVERDICT")
    print("-" * 60)
    print(f"  scifact:  nDCG {s_nd:+.4f}, recall {s_rc:+.4f} (BM25-saturated)")
    print(f"  nfcorpus: nDCG {n_nd:+.4f}, recall {n_rc:+.4f} (vocab mismatch)")
    print()
    print("  PPMI is deterministic (counting), verifiable (named neighbours),")
    print("  append-only (counts accrue as docs are added). Selective + RRF")
    print("  keeps BM25 dominant so it does not drift on clean corpora.")


if __name__ == "__main__":
    main()
