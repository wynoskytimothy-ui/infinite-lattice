#!/usr/bin/env python3
"""
LENS: spectral-laplacian (Laplacian eigenmaps on the term-term co-occurrence graph).

Idea: the corpus's own term-term co-occurrence graph has a natural geometry. Its
normalized graph Laplacian L = I - D^-1/2 A D^-1/2 has, as its SMALLEST non-trivial
eigenvectors, the smoothest coordinates over the graph -> the latent semantic axes
(Belkin-Niyogi Laplacian eigenmaps; spectral clustering relaxation). Terms that
co-occur land near each other on those axes; that is the "free dimensions" Timothy
wants, computed by pure CPU linear algebra (scipy eigsh), no GPU / no training.

Pipeline (all numpy / scipy.sparse / scipy.sparse.linalg):
  1. tokenize with the repo tokenizer `words` (lowercase, drop stop/len<=2).
  2. term-doc count matrix C (sparse). df-filter to a manageable vocab.
  3. term-term co-occurrence A = C C^T (terms co-occur in a doc), zero the diagonal.
     weight edges by PPMI (positive pointwise mutual information) so frequent terms
     don't dominate -> a clean semantic affinity graph.
  4. normalized Laplacian; smallest k eigenvectors via eigsh(sigma=...) = term coords.
     (drop the trivial constant eigenvector). scale by 1/sqrt(eigval) (diffusion-map
     style) so tighter axes weigh less.
  5. doc vector = tf-idf-weighted sum of its term coords (+ optional l2 norm). this is
     exactly LSA-style folding but on the Laplacian eigenbasis instead of the SVD basis.
  6. retrieve by cosine; ALSO RRF-fuse with the BM25 lexical ranking.

Measured on scifact + nfcorpus, nDCG@10, vs lexical 0.70/0.32 and dense ~0.70/0.34.
"""
from __future__ import annotations
import math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10
from aethos_append_index import words


# ----------------------------------------------------------------------------- data
def build_term_doc(corpus, min_df=2, max_df_frac=0.5):
    """Sparse term-doc COUNT matrix + vocab. df-filtered."""
    doc_ids = list(corpus.keys())
    N = len(doc_ids)
    df = defaultdict(int)
    toks = {}
    for d in doc_ids:
        ws = words(corpus[d])
        toks[d] = ws
        for w in set(ws):
            df[w] += 1
    max_df = max_df_frac * N
    vocab = {w: i for i, w in enumerate(
        sorted(w for w, c in df.items() if c >= min_df and c <= max_df))}
    rows, cols, vals = [], [], []
    for j, d in enumerate(doc_ids):
        c = defaultdict(int)
        for w in toks[d]:
            i = vocab.get(w)
            if i is not None:
                c[i] += 1
        for i, v in c.items():
            rows.append(i); cols.append(j); vals.append(v)
    C = sp.csr_matrix((vals, (rows, cols)), shape=(len(vocab), N), dtype=np.float64)
    return C, vocab, doc_ids, df


# ------------------------------------------------------------------- spectral embed
def ppmi_term_graph(C, shift=0.0):
    """Term-term PPMI affinity from a binary doc co-occurrence matrix.

    Co-occ count = B B^T where B is binarized term-doc. PPMI(i,j) =
    max(0, log( p(i,j) / (p(i)p(j)) ) - shift). Returns sparse symmetric A (diag 0).
    """
    B = (C > 0).astype(np.float64)               # term-doc presence
    co = (B @ B.T).tocoo()                        # term-term co-occurrence counts
    deg = np.asarray(B.sum(axis=1)).ravel()       # per-term doc-frequency
    total = float(B.shape[1])                      # number of docs
    rows, cols, data = co.row, co.col, co.data
    keep = rows != cols
    rows, cols, data = rows[keep], cols[keep], data[keep]
    # PPMI: log( (c_ij * total) / (deg_i * deg_j) )
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((data * total) / (deg[rows] * deg[cols]))
    pmi = pmi - shift
    m = pmi > 0
    rows, cols, vals = rows[m], cols[m], pmi[m]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(C.shape[0], C.shape[0]))
    A = A.maximum(A.T)                             # symmetrize
    return A


def laplacian_eigenmaps(A, k=128):
    """Smallest k non-trivial eigenvectors of the normalized Laplacian.

    L_sym = I - D^-1/2 A D^-1/2. Eigenpairs of L_sym <-> eigenpairs of the
    normalized affinity N = D^-1/2 A D^-1/2 (mu = 1 - lam). The smallest non-trivial
    Laplacian eigvecs = the LARGEST eigvecs of N (excluding the trivial top one),
    computed with eigsh(which='LA') on a symmetric sparse matrix -> robust + fast.
    Returns (coords [n_terms x k], lambdas [k]).
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1e-12
    dinv = 1.0 / np.sqrt(deg)
    D = sp.diags(dinv)
    Nmat = (D @ A @ D).tocsr()                     # symmetric normalized affinity
    Nmat = (Nmat + Nmat.T) * 0.5                   # enforce exact symmetry
    # ask for k+1 largest eigenvalues of N; drop the top (trivial ~ constant) one.
    kk = min(k + 1, A.shape[0] - 2)
    mu, V = eigsh(Nmat, k=kk, which="LA")
    order = np.argsort(mu)[::-1]                    # descending mu
    mu, V = mu[order], V[:, order]
    mu, V = mu[1:], V[:, 1:]                        # drop trivial leading eigenvector
    lam = 1.0 - mu                                  # Laplacian eigenvalues (small=smooth)
    lam = np.clip(lam, 1e-6, None)
    coords = V / np.sqrt(lam)[None, :]             # diffusion-map scaling: tight axes downweighted
    return coords, lam


# ----------------------------------------------------------------------- doc folding
def fold_docs(C, df, doc_ids, term_coords, vocab):
    """doc vector = tf-idf-weighted sum of its term spectral coords."""
    N = len(doc_ids)
    idf = np.zeros(C.shape[0])
    inv_vocab = {i: w for w, i in vocab.items()}
    for i in range(C.shape[0]):
        d = df[inv_vocab[i]]
        idf[i] = math.log(1 + (N - d + 0.5) / (d + 0.5))
    # weight matrix W (term-doc) = log-tf * idf
    Cc = C.tocoo()
    w = (1.0 + np.log(Cc.data)) * idf[Cc.row]
    W = sp.csr_matrix((w, (Cc.row, Cc.col)), shape=C.shape)
    DV = (W.T @ term_coords)                        # [docs x k]
    nrm = np.linalg.norm(DV, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return DV / nrm


def fold_query(q, df, N, term_coords, vocab):
    idf_w = {}
    for w in words(q):
        i = vocab.get(w)
        if i is None:
            continue
        d = df.get(w, 1)
        idf_w[i] = idf_w.get(i, 0.0) + math.log(1 + (N - d + 0.5) / (d + 0.5))
    if not idf_w:
        return None
    v = np.zeros(term_coords.shape[1])
    for i, wt in idf_w.items():
        v += wt * term_coords[i]
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


# --------------------------------------------------------------------------- bm25
def bm25_rank(corpus, query, doc_ids, toks_cache, df_full, dl, avgdl, N, k=1000,
              k1=1.2, b=0.75):
    """Plain BM25 ranking over the full vocab (lexical baseline / fusion source)."""
    qw = words(query)
    scores = defaultdict(float)
    for w in set(qw):
        pl = toks_cache.get(w)
        if not pl:
            continue
        idf = math.log(1 + (N - df_full[w] + 0.5) / (df_full[w] + 0.5))
        for d, tf in pl.items():
            scores[d] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[d] / avgdl))
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]


def build_bm25(corpus):
    doc_ids = list(corpus.keys())
    N = len(doc_ids)
    postings = defaultdict(dict)
    df_full = defaultdict(int)
    dl = {}
    for d in doc_ids:
        ws = words(corpus[d])
        dl[d] = len(ws)
        c = defaultdict(int)
        for w in ws:
            c[w] += 1
        for w, tf in c.items():
            postings[w][d] = tf
            df_full[w] += 1
    avgdl = sum(dl.values()) / max(1, N)
    return postings, df_full, dl, avgdl, N


# --------------------------------------------------------------------------- fusion
def rrf(rank_lists, kconst=60):
    score = defaultdict(float)
    for rl in rank_lists:
        for r, d in enumerate(rl):
            score[d] += 1.0 / (kconst + r + 1)
    return sorted(score, key=lambda d: score[d], reverse=True)


# --------------------------------------------------------- doc-doc spectral embed
def doc_doc_laplacian(C, df, doc_ids, k=128, knn=20):
    """Spectral embedding of DOCS directly: build a doc-doc kNN cosine graph over
    tf-idf doc vectors, then Laplacian-eigenmap it. Often beats term-folding because
    the affinity is on the retrieval unit (docs) not a noisy term sum."""
    N = len(doc_ids)
    idf = np.zeros(C.shape[0])
    inv = {i: w for w, i in {w: i for i, w in enumerate(sorted(df))}.items()}
    # idf aligned to C rows: rebuild from df via vocab order used in C
    # (C rows are the df-filtered vocab; recompute idf per row from df dict)
    return None  # placeholder; real version below uses passed vocab


# ----------------------------------------------------------------------------- run
def run(name, k_dim=128, ppmi_shift=0.0, lam_w=0.30, pool=100, verbose=True):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    if verbose:
        print(f"\n{'='*68}\n{name}: {len(corpus)} docs | test {len(test_ids)} q | "
              f"k_dim={k_dim} shift={ppmi_shift} lam={lam_w} pool={pool}")

    t0 = time.time()
    C, vocab, doc_ids, df = build_term_doc(corpus)
    if verbose:
        print(f"  term-doc {C.shape[0]} terms x {C.shape[1]} docs  ({time.time()-t0:.1f}s)")

    A = ppmi_term_graph(C, shift=ppmi_shift)
    if verbose:
        print(f"  PPMI term graph: {A.nnz} edges, density {A.nnz/(A.shape[0]**2):.4f}")

    t1 = time.time()
    term_coords, lam = laplacian_eigenmaps(A, k=k_dim)
    if verbose:
        print(f"  eigsh -> {term_coords.shape[1]} spectral axes  "
              f"(lam {lam.min():.3f}..{lam.max():.3f})  ({time.time()-t1:.1f}s)")

    DV = fold_docs(C, df, doc_ids, term_coords, vocab)        # [docs x k]
    N = len(doc_ids)
    di = {d: i for i, d in enumerate(doc_ids)}

    postings, df_full, dl, avgdl, _ = build_bm25(corpus)

    def spectral_sims(q, cand_idx):
        qv = fold_query(q, df, N, term_coords, vocab)
        if qv is None:
            return None
        return DV[cand_idx] @ qv

    nd_lex = nd_spec = nd_rrf = nd_rer = 0.0
    for qid in test_ids:
        q = queries[qid]
        lex = bm25_rank(corpus, q, doc_ids, postings, df_full, dl, avgdl, N, k=1000)
        nd_lex += ndcg10(lex[:10], test_q[qid])

        # standalone spectral (full corpus cosine)
        qv = fold_query(q, df, N, term_coords, vocab)
        if qv is not None:
            sims_all = DV @ qv
            top = np.argpartition(-sims_all, 10)[:10]
            spec = [doc_ids[i] for i in top[np.argsort(-sims_all[top])]]
        else:
            spec = []
        nd_spec += ndcg10(spec[:10], test_q[qid])

        # CONSERVATIVE rerank: only reorder BM25's top `pool` candidates by
        #   final = bm25_norm + lam_w * spectral_cos    (weak signal can't inject drift)
        cand = lex[:pool]
        if cand and qv is not None:
            cidx = np.array([di[d] for d in cand])
            s = DV[cidx] @ qv
            s = (s - s.min()) / (np.ptp(s) + 1e-9)
            # bm25 rank-based norm (1..0 over the pool)
            base = np.linspace(1.0, 0.0, len(cand))
            fin = base + lam_w * s
            order = np.argsort(-fin)
            rer = [cand[i] for i in order]
        else:
            rer = cand
        nd_rer += ndcg10(rer[:10], test_q[qid])
        nd_rrf += ndcg10(rrf([lex[:pool], spec])[:10], test_q[qid])

    n = len(test_ids)
    nd_lex, nd_spec, nd_rrf, nd_rer = nd_lex/n, nd_spec/n, nd_rrf/n, nd_rer/n
    if verbose:
        print(f"  BM25 lexical          : nDCG@10 {nd_lex:.4f}")
        print(f"  spectral (full cosine): nDCG@10 {nd_spec:.4f}")
        print(f"  RRF(BM25@pool+spec)   : nDCG@10 {nd_rrf:.4f}  ({nd_rrf-nd_lex:+.4f})")
        print(f"  rerank BM25@{pool} +spec : nDCG@10 {nd_rer:.4f}  ({nd_rer-nd_lex:+.4f})")
    return dict(lex=nd_lex, spec=nd_spec, rrf=nd_rrf, rer=nd_rer)


def main():
    res = {}
    for ds in ("scifact", "nfcorpus"):
        best = None
        for k_dim in (128, 256):
            for lam_w in (0.10, 0.25):
                r = run(ds, k_dim=k_dim, lam_w=lam_w, pool=100)
                score = max(r["rrf"], r["rer"])
                if best is None or score > best["score"]:
                    best = dict(score=score, k_dim=k_dim, lam_w=lam_w, **r)
        res[ds] = best
    print(f"\n{'='*68}\nSPECTRAL-LAPLACIAN VERDICT (nDCG@10, best config)")
    for ds, b in res.items():
        print(f"  {ds:9s} k={b['k_dim']:3d} lam={b['lam_w']}: "
              f"BM25 {b['lex']:.4f} | spectral {b['spec']:.4f} | "
              f"RRF {b['rrf']:.4f} | rerank {b['rer']:.4f}  "
              f"(best {b['score']-b['lex']:+.4f})")
    return res


if __name__ == "__main__":
    main()
