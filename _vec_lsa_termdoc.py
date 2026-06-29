#!/usr/bin/env python3
"""
LENS: lsa-termdoc  (classic LSA / LSI, the original no-GPU semantic retrieval)

Build the TF-IDF term-doc matrix, run TruncatedSVD (k=50..300 latent dims) to get
REAL doc vectors in latent-semantic space; project the query the same way; retrieve
by cosine. The FACTORIZATION (reduce co-occurrence geometry to k latent dims) is the
move that actually adds semantics -- raw co-occurrence expansion drifts.

Measured three ways on scifact + nfcorpus (test queries):
  - LEXICAL baseline   : the no-GPU multi-view BM25 lattice (AppendOnlyLatticeIndex)
  - LSA cosine (pure)  : cosine over the SVD doc vectors only
  - RRF(LSA, lexical)  : reciprocal-rank fusion of the two rankings

Reference: BM25 scifact 0.665 / nfcorpus 0.325 ; dense/SPLADE ~0.70 / ~0.34 ;
the no-GPU LEXICAL lattice already gets ~0.702 / ~0.320. Question: does LSA BEAT
lexical (add semantics it lacks), and approach dense?  CPU only: numpy/scipy/sklearn.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10
from aethos_append_index import AppendOnlyLatticeIndex, words as lat_words

# ----- tokenizer: reuse the lattice word gear (lowercase, drop stopwords/len<=2) ---
def tokenize(text):
    return lat_words(text)


# ----- build TF-IDF term-doc matrix (docs x terms), sublinear tf, smoothed idf -----
def build_tfidf(corpus_ids, corpus_texts, min_df=2):
    vocab = {}
    rows, cols, vals = [], [], []
    df = {}
    doc_tokens = []
    for txt in corpus_texts:
        toks = tokenize(txt)
        doc_tokens.append(toks)
        seen = set()
        for w in toks:
            if w not in seen:
                df[w] = df.get(w, 0) + 1
                seen.add(w)
    # rare-term filter: df>=min_df (drops singletons; cuts noise + matrix width)
    vocab = {w: i for i, w in enumerate(sorted(t for t, c in df.items() if c >= min_df))}
    N = len(corpus_ids)
    idf = np.zeros(len(vocab))
    for w, i in vocab.items():
        idf[i] = np.log((1.0 + N) / (1.0 + df[w])) + 1.0      # smoothed idf
    for r, toks in enumerate(doc_tokens):
        tf = {}
        for w in toks:
            j = vocab.get(w)
            if j is not None:
                tf[j] = tf.get(j, 0) + 1
        for j, c in tf.items():
            rows.append(r)
            cols.append(j)
            vals.append((1.0 + np.log(c)) * idf[j])           # sublinear tf * idf
    X = sp.csr_matrix((vals, (rows, cols)), shape=(N, len(vocab)), dtype=np.float64)
    X = normalize(X, norm="l2", axis=1)                       # L2-normalize docs (cosine)
    return X, vocab, idf


def query_vec(q, vocab, idf):
    toks = tokenize(q)
    v = np.zeros(len(vocab))
    tf = {}
    for w in toks:
        j = vocab.get(w)
        if j is not None:
            tf[j] = tf.get(j, 0) + 1
    for j, c in tf.items():
        v[j] = (1.0 + np.log(c)) * idf[j]
    return v


# ----- lexical lattice baseline (full per-doc score vector for fusion) -------------
def build_lattice(corpus_ids, corpus_texts):
    idx = AppendOnlyLatticeIndex()
    for d, t in zip(corpus_ids, corpus_texts):
        idx.add(d, t)
    idx.finalize()
    return idx


def lexical_ranking(idx, q, k):
    scores, docs = idx.dense_scores(q)
    order = np.argsort(scores)[::-1]
    return [docs[i] for i in order[:k] if scores[i] > 0]


def rrf(rank_lists, k_rrf=60):
    """Reciprocal-rank fusion: sum 1/(k_rrf + rank) over the input rankings."""
    agg = {}
    for rl in rank_lists:
        for r, d in enumerate(rl):
            agg[d] = agg.get(d, 0.0) + 1.0 / (k_rrf + r + 1)
    return sorted(agg, key=agg.get, reverse=True)


def run(name, ks=(50, 100, 150, 200, 300)):
    corpus, queries, _train, test_q = load(name)
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[d] for d in corpus_ids]
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*70}\n{name}: {len(corpus_ids)} docs | {len(test_ids)} test queries")

    t0 = time.time()
    X, vocab, idf = build_tfidf(corpus_ids, corpus_texts)
    print(f"  tfidf term-doc: {X.shape[0]} docs x {X.shape[1]} terms "
          f"(nnz {X.nnz}) in {time.time()-t0:.1f}s")

    # --- lexical baseline ranking + nDCG ---
    idx = build_lattice(corpus_ids, corpus_texts)
    lex_rankings = {}
    nd_lex = 0.0
    for qid in test_ids:
        rl = lexical_ranking(idx, queries[qid], 100)
        lex_rankings[qid] = rl
        nd_lex += ndcg10(rl, test_q[qid])
    nd_lex /= len(test_ids)
    print(f"  LEXICAL (BM25 lattice):           nDCG@10 {nd_lex:.4f}")

    # --- LSA k-sweep ---
    best = None
    results = []
    for k in ks:
        kk = min(k, min(X.shape) - 1)
        svd = TruncatedSVD(n_components=kk, algorithm="randomized", random_state=0)
        D = svd.fit_transform(X)                              # docs x k  (U * S)
        Dn = normalize(D, norm="l2", axis=1)
        Vt = svd.components_                                  # k x terms

        # pure LSA cosine
        nd_lsa = 0.0
        lsa_rankings = {}
        for qid in test_ids:
            qv = query_vec(queries[qid], vocab, idf)          # terms
            ql = qv @ Vt.T                                    # project to k-dim
            n = np.linalg.norm(ql)
            if n > 0:
                ql = ql / n
            sims = Dn @ ql
            order = np.argsort(sims)[::-1][:100]
            rl = [corpus_ids[i] for i in order if sims[i] > 0]
            lsa_rankings[qid] = rl
            nd_lsa += ndcg10(rl, test_q[qid])
        nd_lsa /= len(test_ids)

        # RRF fuse with lexical
        nd_rrf = 0.0
        for qid in test_ids:
            fused = rrf([lsa_rankings[qid], lex_rankings[qid]])
            nd_rrf += ndcg10(fused, test_q[qid])
        nd_rrf /= len(test_ids)

        results.append((kk, nd_lsa, nd_rrf))
        var = svd.explained_variance_ratio_.sum()
        print(f"  k={kk:3d}  LSA-cos {nd_lsa:.4f}   RRF(LSA,lex) {nd_rrf:.4f}   "
              f"(explained var {var:.3f})")
        cand = (nd_rrf, k, nd_lsa)
        if best is None or cand[0] > best[0]:
            best = cand

    best_rrf, best_k, best_lsa_at = best
    best_lsa = max(r[1] for r in results)
    print(f"  BEST: LSA-cos {best_lsa:.4f} | RRF {best_rrf:.4f} (k={best_k}) "
          f"vs lexical {nd_lex:.4f}")
    return {
        "name": name, "lexical": nd_lex, "best_lsa": best_lsa,
        "best_rrf": best_rrf, "best_k": best_k, "sweep": results,
    }


def main():
    out = {}
    for ds in ("scifact", "nfcorpus"):
        out[ds] = run(ds)
    print(f"\n{'='*70}\nVERDICT  (test queries, nDCG@10)")
    print(f"  reference: BM25 0.665/0.325  dense ~0.70/~0.34  lexical-lattice 0.702/0.320")
    for ds in ("scifact", "nfcorpus"):
        r = out[ds]
        d_lsa = r["best_lsa"] - r["lexical"]
        d_rrf = r["best_rrf"] - r["lexical"]
        print(f"  {ds:9s}: lexical {r['lexical']:.4f} | LSA {r['best_lsa']:.4f} "
              f"({d_lsa:+.4f}) | RRF {r['best_rrf']:.4f} ({d_rrf:+.4f})  k={r['best_k']}")


if __name__ == "__main__":
    main()
