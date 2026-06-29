#!/usr/bin/env python3
"""
LENS: random / structured projection  (Johnson-Lindenstrauss = "free dimensions")

The claim under test: JL projection gives you a compact dense embedding for FREE
(no training, no downloads) that preserves cosine. So the question splits cleanly:

  (A) PROJECTION ALONE (no semantics): project the sparse TF-IDF doc/query vectors
      to k dims. JL preserves inner products, so cosine in k-dim ~= cosine in full
      sparse space. Expectation: this should TIE the dense-TFIDF-cosine baseline,
      NOT the BM25 lexical lattice, and add ZERO semantics -- it's just a lossy
      compression of the exact same lexical signal. If it ties cosine-TFIDF it
      validates JL; it will NOT beat the lattice because there's no new signal.

  (B) PROJECTION OF A SEMANTIC GEOMETRY: first build PPMI (positive pointwise
      mutual information) term-term association, fold it into the doc vectors
      (doc -> sum of PPMI-smoothed term rows), THEN project. PPMI is the thing
      LSA/GloVe/word2vec implicitly factorize -- so projecting the PPMI geometry
      is where any *semantic* lift (matching synonyms the lexical index misses)
      would have to come from. We also do an explicit SVD of PPMI (LSA) as the
      "factorize, don't expand" reference point, since the memory note says raw
      co-occurrence EXPANSION drifts but FACTORIZING works.

Variants built (all CPU, no GPU, no downloads, no neural training):
  P0  cosine over raw TF-IDF (dense reference - the signal JL is compressing)
  P1  SparseRandomProjection(TF-IDF) -> k dims, cosine          (JL, sklearn)
  P2  prime-seeded sign (Rademacher) projection of TF-IDF, cosine (lattice JL)
  P3  PPMI doc embedding (doc = sum of PPMI term rows), cosine
  P4  SparseRandomProjection(PPMI doc embedding) -> k dims, cosine
  P5  LSA: TruncatedSVD(PPMI term-term) -> term vecs -> doc embed, cosine
  RRF fusion of each dense ranking with the BM25 lexical lattice ranking.

Retrieval: cosine over doc vectors; RRF-fuse with BM25 lexical lattice.
Measure: nDCG@10 on TEST queries for scifact AND nfcorpus.
Reference: BM25 0.665/0.325 ; lexical lattice 0.7023/0.3204 ; dense ~0.70/~0.34.
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from scripts.bench_supervised_bridges import load, ndcg10
from aethos_append_index import words
from core.primes import chain_primes


# -----------------------------------------------------------------------------
# tokenization / matrices
# -----------------------------------------------------------------------------
def build_vocab(corpus, min_df=2):
    df = defaultdict(int)
    toks = {}
    for d, txt in corpus.items():
        ws = set(words(txt))
        toks[d] = ws
        for w in ws:
            df[w] += 1
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= min_df))}
    return vocab, df, toks


def tf_matrix(corpus, vocab):
    """raw term-frequency sparse doc-term matrix (rows=docs)."""
    doc_ids = list(corpus.keys())
    row, col, data = [], [], []
    for r, d in enumerate(doc_ids):
        tf = defaultdict(int)
        for w in words(corpus[d]):
            j = vocab.get(w)
            if j is not None:
                tf[j] += 1
        for j, c in tf.items():
            row.append(r); col.append(j); data.append(c)
    M = sp.csr_matrix((data, (row, col)), shape=(len(doc_ids), len(vocab)), dtype=np.float64)
    return M, doc_ids


def tfidf_from_tf(tf_doc, idf):
    """csr (n x V) raw tf -> tf-idf with log tf and idf weighting, L2-normalized."""
    M = tf_doc.copy().astype(np.float64)
    M.data = (1.0 + np.log(M.data)) * idf[M.indices]
    return normalize(M, norm="l2", axis=1)


def query_tf_vec(q, vocab, V):
    tf = defaultdict(int)
    for w in words(q):
        j = vocab.get(w)
        if j is not None:
            tf[j] += 1
    if not tf:
        return sp.csr_matrix((1, V), dtype=np.float64)
    idx = np.array(list(tf.keys())); val = np.array(list(tf.values()), dtype=np.float64)
    return sp.csr_matrix((val, (np.zeros_like(idx), idx)), shape=(1, V), dtype=np.float64)


# -----------------------------------------------------------------------------
# BM25 lexical lattice ranking (baseline + fusion partner)
# -----------------------------------------------------------------------------
def bm25_index(tf_doc, k1=1.2, b=0.75):
    n, V = tf_doc.shape
    dl = np.asarray(tf_doc.sum(axis=1)).ravel()
    avgdl = dl.mean() if dl.mean() else 1.0
    df = np.asarray((tf_doc > 0).sum(axis=0)).ravel()
    idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
    return dict(tf=tf_doc.tocsc(), dl=dl, avgdl=avgdl, idf=idf, k1=k1, b=b, n=n)


def bm25_score(bm, q_terms):
    sc = np.zeros(bm["n"])
    k1, b, avgdl = bm["k1"], bm["b"], bm["avgdl"]
    denom_norm = k1 * (1 - b + b * bm["dl"] / avgdl)
    for j in q_terms:
        col = bm["tf"].getcol(j)
        docs = col.indices; tf = col.data
        s = bm["idf"][j] * (tf * (k1 + 1)) / (tf + denom_norm[docs])
        sc[docs] += s
    return sc


# -----------------------------------------------------------------------------
# PPMI term-term association (the semantic geometry to factorize/project)
# -----------------------------------------------------------------------------
def ppmi_termterm(tf_doc, shift=1.0):
    """term-term co-occurrence via binary doc incidence -> PPMI matrix (V x V)."""
    B = (tf_doc > 0).astype(np.float64)          # n x V binary incidence
    C = (B.T @ B).tocsr()                         # V x V co-occurrence counts
    total = C.sum()
    rowsum = np.asarray(C.sum(axis=1)).ravel()
    C = C.tocoo()
    # PMI = log( count * total / (rowsum_i * rowsum_j) ); PPMI = max(0, PMI - log shift)
    pmi = np.log(C.data * total / (rowsum[C.row] * rowsum[C.col]) + 1e-12) - math.log(shift)
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape, dtype=np.float64)
    return P


# -----------------------------------------------------------------------------
# prime-seeded Rademacher (sign) projection -- "lattice JL"
# -----------------------------------------------------------------------------
def prime_sign_projection(M, k, seed_primes):
    """
    Deterministic +-1 projection where each entry's sign is derived from the
    lattice's own primes: sign(feature j, dim t) = +1 if (p_j * (t+1)) is a QR
    mod a small prime, else -1. Sparse-friendly, fully reproducible from the
    prime address book (no RNG, no download). Equivalent JL guarantee to a
    Rademacher matrix when the sign bits are ~balanced.
    """
    V = M.shape[1]
    primes = np.array(seed_primes[:V], dtype=np.float64)
    R = np.empty((V, k), dtype=np.float64)
    mod = 1000003  # a prime modulus for the QR test
    for t in range(k):
        x = (primes.astype(np.int64) * (t + 1)) % mod
        # quadratic-residue sign bit via Euler's criterion proxy (parity of x*(t+3))
        sign = np.where(((x * (t * 7 + 3)) % mod) % 2 == 0, 1.0, -1.0)
        R[:, t] = sign
    R /= math.sqrt(k)
    return np.asarray(M @ R)


# -----------------------------------------------------------------------------
# retrieval + eval helpers
# -----------------------------------------------------------------------------
def cosine_rank(doc_vecs, q_vec, doc_ids, topk=100):
    sims = doc_vecs @ q_vec
    if topk < len(doc_ids):
        idx = np.argpartition(-sims, topk)[:topk]
        idx = idx[np.argsort(-sims[idx])]
    else:
        idx = np.argsort(-sims)
    return [doc_ids[i] for i in idx], sims


def score_rank(sc, doc_ids, topk=100):
    idx = np.argpartition(-sc, min(topk, len(doc_ids) - 1))[:topk]
    idx = idx[np.argsort(-sc[idx])]
    return [doc_ids[i] for i in idx]


def rrf(rank_a, rank_b, k=60, topk=10):
    score = defaultdict(float)
    for r, d in enumerate(rank_a):
        score[d] += 1.0 / (k + r + 1)
    for r, d in enumerate(rank_b):
        score[d] += 1.0 / (k + r + 1)
    return sorted(score, key=lambda d: score[d], reverse=True)[:topk]


def eval_dense(doc_vecs, qvec_fn, test_q, queries, doc_ids):
    nd = 0.0
    n = 0
    for qid in test_q:
        if qid not in queries:
            continue
        qv = qvec_fn(queries[qid])
        if qv is None:
            ranked = []
        else:
            ranked, _ = cosine_rank(doc_vecs, qv, doc_ids, topk=10)
        nd += ndcg10(ranked, test_q[qid]); n += 1
    return nd / n if n else 0.0


def eval_fused(doc_vecs, qvec_fn, bm, vocab, test_q, queries, doc_ids):
    nd = 0.0
    n = 0
    for qid in test_q:
        if qid not in queries:
            continue
        # dense ranking
        qv = qvec_fn(queries[qid])
        dense_rank = cosine_rank(doc_vecs, qv, doc_ids, topk=100)[0] if qv is not None else []
        # bm25 ranking
        qt = [vocab[w] for w in set(words(queries[qid])) if w in vocab]
        bsc = bm25_score(bm, qt)
        bm_rank = score_rank(bsc, doc_ids, topk=100)
        ranked = rrf(dense_rank, bm_rank, topk=10)
        nd += ndcg10(ranked, test_q[qid]); n += 1
    return nd / n if n else 0.0


def eval_bm25(bm, vocab, test_q, queries, doc_ids):
    nd = 0.0
    n = 0
    for qid in test_q:
        if qid not in queries:
            continue
        qt = [vocab[w] for w in set(words(queries[qid])) if w in vocab]
        bsc = bm25_score(bm, qt)
        ranked = score_rank(bsc, doc_ids, topk=10)
        nd += ndcg10(ranked, test_q[qid]); n += 1
    return nd / n if n else 0.0


# -----------------------------------------------------------------------------
def run(name, K=512, ppmi_dim=300):
    t0 = time.time()
    corpus, queries, train_q, test_q = load(name)
    vocab, df, toks = build_vocab(corpus, min_df=2)
    V = len(vocab)
    tf_doc, doc_ids = tf_matrix(corpus, vocab)
    n = len(doc_ids)
    print(f"\n{'='*70}\n{name}: {n} docs | vocab {V} | test {sum(1 for q in test_q if q in queries)} q | k={K}")

    idf = np.log(n / (1.0 + np.asarray((tf_doc > 0).sum(axis=0)).ravel()))
    tfidf = tfidf_from_tf(tf_doc, idf)

    bm = bm25_index(tf_doc)

    def qvec_tfidf(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        return np.asarray(v.todense()).ravel()

    results = {}

    # --- BM25 lexical baseline ---
    bm25_nd = eval_bm25(bm, vocab, test_q, queries, doc_ids)
    results["BM25 lexical"] = (bm25_nd, None)

    # --- P0: raw TF-IDF cosine (dense reference signal JL compresses) ---
    tfidf_dense = np.asarray(tfidf.todense())
    nd = eval_dense(tfidf_dense, qvec_tfidf, test_q, queries, doc_ids)
    fz = eval_fused(tfidf_dense, qvec_tfidf, bm, vocab, test_q, queries, doc_ids)
    results["P0 TF-IDF cosine (full)"] = (nd, fz)
    del tfidf_dense

    # --- P1: SparseRandomProjection of TF-IDF (sklearn JL) ---
    srp = SparseRandomProjection(n_components=K, random_state=0, dense_output=True)
    proj_doc = normalize(srp.fit_transform(tfidf), norm="l2", axis=1)

    def qvec_srp(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        pv = srp.transform(v)
        pv = normalize(pv, norm="l2", axis=1)
        return np.asarray(pv).ravel()

    nd = eval_dense(proj_doc, qvec_srp, test_q, queries, doc_ids)
    fz = eval_fused(proj_doc, qvec_srp, bm, vocab, test_q, queries, doc_ids)
    results[f"P1 SRP(TF-IDF) k={K}"] = (nd, fz)

    # --- P2: prime-seeded sign projection of TF-IDF (lattice JL) ---
    primes = chain_primes(V)
    psd_doc = normalize(prime_sign_projection(tfidf, K, primes), norm="l2", axis=1)
    R_cache = None  # rebuild for query via same fn (deterministic)

    def qvec_psd(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        pv = prime_sign_projection(v, K, primes)
        pv = normalize(pv, norm="l2", axis=1)
        return np.asarray(pv).ravel()

    nd = eval_dense(psd_doc, qvec_psd, test_q, queries, doc_ids)
    fz = eval_fused(psd_doc, qvec_psd, bm, vocab, test_q, queries, doc_ids)
    results[f"P2 prime-sign(TF-IDF) k={K}"] = (nd, fz)

    # --- PPMI geometry (the semantic part) ---
    P = ppmi_termterm(tf_doc, shift=1.0)        # V x V PPMI
    Pn = normalize(P, norm="l2", axis=1)         # smoothed term rows

    # doc embedding = tf-idf-weighted sum of PPMI term rows (folds synonyms in)
    # doc_ppmi[d] = (tfidf row d) @ Pn  -> a doc vector smeared over related terms
    doc_ppmi = normalize(tfidf @ Pn, norm="l2", axis=1)
    doc_ppmi_dense = np.asarray(doc_ppmi.todense())

    def qvec_ppmi(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        pv = normalize(v @ Pn, norm="l2", axis=1)
        return np.asarray(pv.todense()).ravel()

    # --- P3: PPMI doc embedding cosine (full V-dim, the semantic geometry) ---
    nd = eval_dense(doc_ppmi_dense, qvec_ppmi, test_q, queries, doc_ids)
    fz = eval_fused(doc_ppmi_dense, qvec_ppmi, bm, vocab, test_q, queries, doc_ids)
    results["P3 PPMI doc-embed (full)"] = (nd, fz)

    # --- P4: SparseRandomProjection of PPMI doc embedding (JL of semantics) ---
    srp2 = SparseRandomProjection(n_components=K, random_state=1, dense_output=True)
    proj_ppmi = normalize(srp2.fit_transform(doc_ppmi), norm="l2", axis=1)

    def qvec_ppmi_srp(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        pe = normalize(v @ Pn, norm="l2", axis=1)
        pv = normalize(srp2.transform(pe), norm="l2", axis=1)
        return np.asarray(pv).ravel()

    nd = eval_dense(proj_ppmi, qvec_ppmi_srp, test_q, queries, doc_ids)
    fz = eval_fused(proj_ppmi, qvec_ppmi_srp, bm, vocab, test_q, queries, doc_ids)
    results[f"P4 SRP(PPMI) k={K}"] = (nd, fz)

    # --- P5: LSA -- TruncatedSVD of PPMI term-term -> term vecs -> doc embed ---
    d = min(ppmi_dim, V - 1)
    # svds on symmetric-ish PPMI; use the left singular vectors as term embeddings
    U, S, Vt = svds(P.asfptype(), k=d)
    term_emb = U * S                              # V x d term embeddings
    term_emb = normalize(term_emb, norm="l2", axis=1)
    doc_lsa = normalize(np.asarray(tfidf @ term_emb), norm="l2", axis=1)

    def qvec_lsa(q):
        v = query_tf_vec(q, vocab, V)
        if v.nnz == 0:
            return None
        v.data = (1.0 + np.log(v.data)) * idf[v.indices]
        v = normalize(v, norm="l2", axis=1)
        qe = normalize(np.asarray(v @ term_emb), norm="l2", axis=1)
        return qe.ravel()

    nd = eval_dense(doc_lsa, qvec_lsa, test_q, queries, doc_ids)
    fz = eval_fused(doc_lsa, qvec_lsa, bm, vocab, test_q, queries, doc_ids)
    results[f"P5 LSA-SVD(PPMI) d={d}"] = (nd, fz)

    # --- report ---
    print(f"  built in {time.time()-t0:.1f}s")
    print(f"  {'method':32s} {'cosine':>8s} {'+RRF/BM25':>10s}")
    for k, (cos, fz) in results.items():
        fzs = f"{fz:.4f}" if fz is not None else "    -   "
        print(f"  {k:32s} {cos:8.4f} {fzs:>10s}")
    return results


if __name__ == "__main__":
    print("RANDOM / STRUCTURED PROJECTION lens (JL = free dimensions)")
    print("ref: BM25 0.665/0.325 | lexical lattice 0.7023/0.3204 | dense ~0.70/~0.34")
    allres = {}
    for ds in ("scifact", "nfcorpus"):
        allres[ds] = run(ds, K=512, ppmi_dim=300)

    print(f"\n{'='*70}\nSUMMARY (nDCG@10, cosine / RRF-fused with BM25)")
    methods = list(next(iter(allres.values())).keys())
    print(f"  {'method':32s} {'scifact':>16s} {'nfcorpus':>16s}")
    for m in methods:
        s_cos, s_fz = allres["scifact"][m]
        n_cos, n_fz = allres["nfcorpus"][m]
        sf = f"{s_cos:.4f}/{s_fz:.4f}" if s_fz is not None else f"{s_cos:.4f}/  -   "
        nf = f"{n_cos:.4f}/{n_fz:.4f}" if n_fz is not None else f"{n_cos:.4f}/  -   "
        print(f"  {m:32s} {sf:>16s} {nf:>16s}")
