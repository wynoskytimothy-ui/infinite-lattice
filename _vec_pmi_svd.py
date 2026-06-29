#!/usr/bin/env python3
"""
LENS: PMI-SVD word embeddings (word2vec-as-matrix-factorization, Levy & Goldberg 2014).

Build the term-term co-occurrence matrix from the corpus (sliding window),
convert to PPMI (positive pointwise mutual information), truncated-SVD to k dims.
Doc vector = IDF-weighted average of its word embeddings (SIF-style).
Retrieve by cosine, and RRF-fuse with a BM25 lexical baseline.

Question: does the latent PMI geometry add SEMANTICS the lexical lattice lacks?
Pure CPU: numpy + scipy.sparse + scipy.sparse.linalg.svds. No GPU, no training, no downloads.
"""
from __future__ import annotations
import math, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10

_TOK = re.compile(r"[a-z]+")
_STOP = set(("the a an and or to of in is are was were be been being it its this that these those "
             "for on with as at by from he she they we you i my your our their them not no can may also "
             "has have had but if than then so such into over under more most some any all "
             "which who what when where how why").split())

def toks(text):
    return [w for w in _TOK.findall(text.lower()) if w not in _STOP and len(w) > 2]


def build_bm25(corpus):
    docs = {d: toks(t) for d, t in corpus.items()}
    N = len(docs)
    df = Counter()
    for ws in docs.values():
        df.update(set(ws))
    idf = {w: math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in df}
    dl = {d: len(ws) for d, ws in docs.items()}
    avgdl = sum(dl.values()) / max(N, 1)
    tf = {d: Counter(ws) for d, ws in docs.items()}
    post = defaultdict(list)
    for d, c in tf.items():
        for w, f in c.items():
            post[w].append(d)
    k1, b = 1.5, 0.75
    def rank(qws, topn=1000):
        sc = defaultdict(float)
        for w in set(qws):
            if w not in idf:
                continue
            iw = idf[w]
            for d in post[w]:
                f = tf[d][w]
                sc[d] += iw * f * (k1 + 1) / (f + k1 * (1 - b + b * dl[d] / avgdl))
        return sorted(sc, key=sc.get, reverse=True)[:topn]
    return docs, idf, rank


def build_ppmi_svd(docs, idf, k=200, window=5, min_count=2, neg=5):
    # neg = Levy-Goldberg negative-sampling shift: PMI - log(neg), clip>0 (SPPMI).
    # Tuned best on scifact/nfcorpus: k=300, window=10, neg=5.
    # vocab: keep terms appearing >= min_count, drop hyper-common (df-based via idf>0)
    cnt = Counter()
    for ws in docs.values():
        cnt.update(ws)
    vocab = [w for w, c in cnt.items() if c >= min_count]
    vid = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    print(f"  vocab={V} (min_count={min_count}), window={window}, k={k}")

    # term-term co-occurrence via sliding window (symmetric)
    pair = defaultdict(float)
    wcount = np.zeros(V)
    for ws in docs.values():
        ids = [vid[w] for w in ws if w in vid]
        n = len(ids)
        for i, wi in enumerate(ids):
            wcount[wi] += 1
            lo, hi = max(0, i - window), min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                wj = ids[j]
                pair[(wi, wj)] += 1.0
    if not pair:
        return None, vid, V
    rows = np.fromiter((p[0] for p in pair), dtype=np.int32, count=len(pair))
    cols = np.fromiter((p[1] for p in pair), dtype=np.int32, count=len(pair))
    data = np.fromiter(pair.values(), dtype=np.float64, count=len(pair))

    total = data.sum()
    wsum = np.asarray(wcount, dtype=np.float64)  # context-marginal proxy
    wsum_total = wsum.sum()
    # PPMI: log( p(i,j) * total / (p_i * p_j) ), shifted (k_neg=1 => no shift) then clip>0
    pi = wsum / wsum_total
    pmi = np.log((data / total) / (pi[rows] * pi[cols])) - math.log(neg)
    ppmi = np.maximum(pmi, 0.0)
    keep = ppmi > 0
    M = sp.csr_matrix((ppmi[keep], (rows[keep], cols[keep])), shape=(V, V))
    print(f"  PPMI nnz={M.nnz}")

    # truncated SVD -> word embeddings (Levy-Goldberg: U * sqrt(S))
    k = min(k, V - 1)
    U, S, Vt = svds(M, k=k)
    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    W = U * np.sqrt(S)  # V x k word vectors
    return W, vid, V


def doc_vectors(docs, W, vid, idf):
    k = W.shape[1]
    D = list(docs.keys())
    mat = np.zeros((len(D), k), dtype=np.float64)
    for r, d in enumerate(D):
        acc = np.zeros(k)
        wt = 0.0
        for w in docs[d]:
            i = vid.get(w)
            if i is None:
                continue
            iw = idf.get(w, 0.0)
            acc += iw * W[i]
            wt += iw
        if wt > 0:
            acc /= wt
        mat[r] = acc
    nrm = np.linalg.norm(mat, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    mat /= nrm
    return D, mat


def query_vec(qws, W, vid, idf):
    k = W.shape[1]
    acc = np.zeros(k)
    wt = 0.0
    for w in qws:
        i = vid.get(w)
        if i is None:
            continue
        iw = idf.get(w, 0.0)
        acc += iw * W[i]
        wt += iw
    if wt > 0:
        acc /= wt
    n = np.linalg.norm(acc)
    return acc / n if n > 0 else acc


def rrf_fuse(lex_rank, dense_rank, k=60, topn=10):
    sc = defaultdict(float)
    for r, d in enumerate(lex_rank):
        sc[d] += 1.0 / (k + r + 1)
    for r, d in enumerate(dense_rank):
        sc[d] += 1.0 / (k + r + 1)
    return sorted(sc, key=sc.get, reverse=True)[:topn]


def run(name, k=200, window=5):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*64}\n{name}: {len(corpus)} docs | test {len(test_ids)} q")

    t0 = time.time()
    docs, idf, bm25 = build_bm25(corpus)
    W, vid, V = build_ppmi_svd(docs, idf, k=k, window=window)
    if W is None:
        print("  no co-occurrence")
        return
    D, dmat = doc_vectors(docs, W, vid, idf)
    didx = {d: i for i, d in enumerate(D)}
    print(f"  built in {time.time()-t0:.1f}s")

    # eval
    n_lex = n_dense = n_rrf = 0.0
    for qid in test_ids:
        rels = test_q[qid]
        qws = toks(queries[qid])
        lex = bm25(qws, topn=1000)
        n_lex += ndcg10(lex, rels)

        qv = query_vec(qws, W, vid, idf)
        # dense cosine over full corpus
        sims = dmat @ qv
        dense = [D[i] for i in np.argsort(-sims)[:1000]]
        n_dense += ndcg10(dense, rels)

        # RRF fuse top-100 of each
        fused = rrf_fuse(lex[:100], dense[:100])
        n_rrf += ndcg10(fused, rels)

    nq = len(test_ids)
    print(f"  nDCG@10  LEX(BM25)={n_lex/nq:.4f}  DENSE(PMI-SVD)={n_dense/nq:.4f}  RRF={n_rrf/nq:.4f}")
    return n_lex/nq, n_dense/nq, n_rrf/nq


if __name__ == "__main__":
    # tuned best config (see _vec_pmi_svd_tune.py sweep): k=300, window=10, SPPMI neg=5
    res = {}
    for name in ["scifact", "nfcorpus"]:
        res[name] = run(name, k=300, window=10)
    print("\n" + "="*64)
    print("SUMMARY (nDCG@10):  LEX / PMI-SVD-dense / RRF")
    for n, (l, d, r) in res.items():
        print(f"  {n:10s}  {l:.4f} / {d:.4f} / {r:.4f}")
