#!/usr/bin/env python3
"""
LENS: tensor-3way  (Timothy's 3-way meet as real linear algebra)

Goal: get dense-model-quality semantic retrieval with NO GPU / NO neural net /
NO downloads. Build REAL word vectors by FACTORIZING the corpus co-occurrence
geometry (the move that works: LSA / GloVe / word2vec = implicit PMI matrix
factorization). This agent's lens: go ONE order higher -- factorize the
term-term-term 3rd-order co-occurrence TENSOR via CP/PARAFAC (custom ALS, no
tensorly), giving each word a factor vector that encodes triadic context, then
compare it head-to-head against the 2-way PMI-SVD (ordinary LSA).

Pipeline per corpus (scifact, nfcorpus):
  1. tokenize -> term-doc counts (own tokenizer; rare-term vocab for the tensor)
  2. BM25 lexical ranking (the no-GPU baseline to beat / fuse with)
  3a. 2-way: shifted-PPMI term-term matrix -> truncated SVD -> word vectors  (LSA)
  3b. 3-way: sparse PPMI term-term-term tensor -> CP-ALS -> word vectors     (THIS LENS)
  4. doc vector = idf-weighted mean of its word vectors (both methods)
  5. retrieve by cosine; also RRF-fuse each with BM25
  6. nDCG@10 on test queries

HONEST EXPECTATION: 3rd-order co-occurrence on a TINY corpus (scifact ~5k docs)
is extremely sparse, so CP factors are noisy. We measure whether 3-way actually
beats 2-way PMI-SVD, or whether the extra order just adds noise. Report the real
numbers either way.
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10

_TOK = re.compile(r"[a-z]+")
_STOP = set(
    "the a an and or but of to in on at for with by from as is are was were be been "
    "being this that these those it its they them their we our you your he she his her "
    "i me my not no nor so than then thus also can will would should could may might "
    "have has had do does did done which who whom whose what when where why how all any "
    "each few more most other some such only own same too very s t".split()
)


def tok(text):
    return [w for w in _TOK.findall(text.lower()) if len(w) > 2 and w not in _STOP]


# ----------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, corpus, k1=0.9, b=0.4):
        self.docids = list(corpus)
        self.k1, self.b = k1, b
        self.df = defaultdict(int)
        self.tf = []          # list of Counter per doc
        self.dl = []
        for d in self.docids:
            c = Counter(tok(corpus[d]))
            self.tf.append(c)
            self.dl.append(sum(c.values()))
            for w in c:
                self.df[w] += 1
        self.N = len(self.docids)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.idf = {w: math.log(1 + (self.N - df + 0.5) / (df + 0.5)) for w, df in self.df.items()}
        # inverted index for fast scoring
        self.inv = defaultdict(list)   # w -> [(doc_idx, tf)]
        for i, c in enumerate(self.tf):
            for w, f in c.items():
                self.inv[w].append((i, f))

    def search(self, q, topn=100):
        qt = tok(q)
        scores = defaultdict(float)
        for w in qt:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.inv[w]:
                dl = self.dl[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores, key=lambda i: scores[i], reverse=True)[:topn]
        return [self.docids[i] for i in ranked]

    def score_all(self, q):
        qt = tok(q)
        scores = defaultdict(float)
        for w in qt:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.inv[w]:
                dl = self.dl[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores  # doc_idx -> score


# ----------------------------------------------------------------------------- vocab / co-occ
def build_vocab(corpus, min_df=3, max_df_frac=0.5, max_vocab=20000):
    """word -> id, restricted to a sane mid-frequency band (the band that carries
    semantics; ultra-rare = noise, ultra-common = stop)."""
    df = defaultdict(int)
    N = len(corpus)
    for d, txt in corpus.items():
        for w in set(tok(txt)):
            df[w] += 1
    max_df = max_df_frac * N
    cand = [(w, df[w]) for w in df if min_df <= df[w] <= max_df]
    cand.sort(key=lambda x: -x[1])      # keep most frequent within band
    cand = cand[:max_vocab]
    vocab = {w: i for i, (w, _) in enumerate(cand)}
    idf = {w: math.log(N / df[w]) for w, _ in cand}
    return vocab, idf


def doc_token_lists(corpus, vocab):
    """per-doc list of vocab-ids (tokens kept in vocab, with repeats)."""
    out = {}
    for d, txt in corpus.items():
        ids = [vocab[w] for w in tok(txt) if w in vocab]
        out[d] = ids
    return out


# ----------------------------------------------------------------------------- 2-way PMI-SVD (LSA reference)
def two_way_vectors(corpus, vocab, doc_ids, k=200, shift=1.0):
    """Shifted-PPMI term-term co-occurrence matrix -> truncated SVD word vectors.
    This is the standard word2vec-as-implicit-PMI-factorization baseline."""
    V = len(vocab)
    # term-term co-occurrence at document level (whole-doc window): C[i,j] = #docs
    # where both occur, weighted by min count. Use term-doc binary then C = B B^T.
    rows, cols = [], []
    for d, ids in doc_ids.items():
        s = set(ids)
        for i in s:
            rows.append(i)
            cols.append(d if isinstance(d, int) else 0)
    # Build term-doc binary sparse directly:
    dlist = list(doc_ids)
    dpos = {d: j for j, d in enumerate(dlist)}
    r, c = [], []
    for d, ids in doc_ids.items():
        for i in set(ids):
            r.append(i)
            c.append(dpos[d])
    B = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(V, len(dlist)))
    C = (B @ B.T).tocoo()            # term-term co-occurrence counts
    # PPMI
    total = C.data.sum()
    rowsum = np.asarray(C.sum(axis=1)).ravel() + 1e-9
    data = []
    rr, cc = [], []
    for i, j, v in zip(C.row, C.col, C.data):
        pmi = math.log((v * total) / (rowsum[i] * rowsum[j]) + 1e-12) - math.log(shift)
        if pmi > 0:
            rr.append(i); cc.append(j); data.append(pmi)
    M = sp.csr_matrix((np.asarray(data, dtype=np.float32), (rr, cc)), shape=(V, V))
    k = min(k, V - 1)
    U, S, Vt = svds(M, k=k)
    W = U * np.sqrt(S)               # word vectors
    return W


# ----------------------------------------------------------------------------- 3-way CP/PARAFAC (THE LENS)
def build_3way_ppmi_tensor(doc_ids, V, max_terms=2500, df_band=None,
                           min_count=2, shift=1.0, window=None):
    """Sparse symmetric 3rd-order co-occurrence: count triples (i,j,k) that
    co-occur in a SLIDING WINDOW (default whole doc), restricted to the
    most-frequent `max_terms` terms (these are the ones that actually co-occur
    3-at-a-time often enough to give a non-degenerate tensor). Returns COO index
    arrays + shifted-PPMI values over a remapped sub-vocab.

    PMI3 in a CONSISTENT probability space (all probs = fraction of windows):
        p(a,b,c) = (#windows with a,b,c) / W
        p(a)     = (#windows with a)     / W
        pmi3 = log p(a,b,c) - log p(a) - log p(b) - log p(c)  - log(shift)
    Keep only pmi3 > 0 (positive PMI) -> sparse, but now actually populated.
    """
    df = np.zeros(V, dtype=np.int64)
    for d, ids in doc_ids.items():
        for i in set(ids):
            df[i] += 1
    # keep the MOST frequent terms (they form triples); drop a tiny ultra-common
    # head only if it is truly saturating (>80% of docs).
    order = np.argsort(-df)
    keep = [int(t) for t in order if df[t] < 0.8 * len(doc_ids)][:max_terms]
    sub = {t: r for r, t in enumerate(keep)}               # orig id -> sub id
    S = len(sub)

    trip = Counter()          # (a,b,c) sorted sub-ids -> #windows
    single = np.zeros(S, dtype=np.float64)
    Wcount = 0                # number of windows
    for d, ids in doc_ids.items():
        sub_ids = [sub[i] for i in ids if i in sub]        # keep order for windows
        if window is None:
            wins = [sorted(set(sub_ids))]
        else:
            wins = []
            for s in range(0, max(1, len(sub_ids) - window + 1)):
                wins.append(sorted(set(sub_ids[s:s + window])))
        for present in wins:
            if len(present) < 3:
                continue
            Wcount += 1
            for a in present:
                single[a] += 1
            # cap per-window combinatorics
            if len(present) > 50:
                present = sorted(present, key=lambda x: single[x])[:50]
                present.sort()
            L = len(present)
            for ai in range(L):
                a = present[ai]
                for bi in range(ai + 1, L):
                    b = present[bi]
                    for ci in range(bi + 1, L):
                        trip[(a, b, present[ci])] += 1
    logW = math.log(max(Wcount, 1))
    logshift = math.log(shift)
    logsingle = np.log(single + 1e-9)                       # log #windows with a
    idx_a, idx_b, idx_c, vals = [], [], [], []
    for (a, b, c), cnt in trip.items():
        if cnt < min_count:
            continue
        # log p(a,b,c) - sum log p(a) = log cnt - logW - (logsingle - logW)*3
        pmi = (math.log(cnt) - logW) - (logsingle[a] + logsingle[b] + logsingle[c] - 3 * logW) - logshift
        if pmi > 0:
            idx_a.append(a); idx_b.append(b); idx_c.append(c); vals.append(pmi)
    inv_sub = {r: o for o, r in sub.items()}
    return (np.asarray(idx_a), np.asarray(idx_b), np.asarray(idx_c),
            np.asarray(vals, dtype=np.float64), S, inv_sub)


def cp_als_symmetric(ia, ib, ic, vals, S, rank=100, n_iter=12, reg=1e-2, seed=0,
                     max_nnz=1_500_000, batch=400_000):
    """Custom CP/PARAFAC ALS on a sparse SYMMETRIC 3rd-order tensor, fitting a
    single shared factor matrix A (S x rank) with X_abc ~ sum_r A[a,r]A[b,r]A[c,r].

    Memory-efficient symmetric MTTKRP: each canonical sorted triple (a,b,c)
    contributes to ALL THREE target modes (a gets v*A[b]*A[c], b gets v*A[a]*A[c],
    c gets v*A[a]*A[b]). We scatter-add these in BATCHES (no 6x copy, no 30GB
    Khatri-Rao blow-up). Update:  A <- MTTKRP_sym(A) @ (A^T A * A^T A + reg I)^-1
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((S, rank)).astype(np.float64) * 0.1

    # cap to the highest-PMI triples (top signal) to bound cost
    if len(vals) > max_nnz:
        keep = np.argpartition(-vals, max_nnz)[:max_nnz]
        ia, ib, ic, vals = ia[keep], ib[keep], ic[keep], vals[keep]
    nnz = len(vals)
    eye = reg * np.eye(rank)
    lam = np.ones(rank)

    for it in range(n_iter):
        AtA = A.T @ A
        G = (AtA * AtA) + eye
        Ginv = np.linalg.pinv(G)
        mttkrp = np.zeros((S, rank))
        for s in range(0, nnz, batch):
            e = min(s + batch, nnz)
            ba, bb, bc = ia[s:e], ib[s:e], ic[s:e]
            v = vals[s:e][:, None]
            Aa, Ab, Ac = A[ba], A[bb], A[bc]
            np.add.at(mttkrp, ba, v * (Ab * Ac))   # target mode a
            np.add.at(mttkrp, bb, v * (Aa * Ac))   # target mode b
            np.add.at(mttkrp, bc, v * (Aa * Ab))   # target mode c
        A_new = mttkrp @ Ginv
        # normalize columns to unit norm and stash the magnitude in lambda (the
        # standard CP scaling) -- do NOT flatten all norms to the mean, which
        # rank-collapses the model.
        norms = np.linalg.norm(A_new, axis=0) + 1e-12
        A_new = A_new / norms
        change = np.linalg.norm(A_new - A) / (np.linalg.norm(A) + 1e-9)
        A = A_new
        lam = norms
        if change < 1e-4:
            break
    # fold cube-root of lambda back into A so X_abc ~ sum (a)(b)(c) reproduces scale
    A = A * (lam ** (1.0 / 3.0))
    return A


def three_way_vectors(corpus, vocab, doc_ids, rank=100, max_terms=2500,
                      n_iter=12, window=None, min_count=2):
    V = len(vocab)
    ia, ib, ic, vals, S, inv_sub = build_3way_ppmi_tensor(
        doc_ids, V, max_terms=max_terms, window=window, min_count=min_count)
    info = f"tensor nnz={len(vals)} subvocab={S}"
    if len(vals) < rank * 5:
        return None, info + " (too sparse)"
    A = cp_als_symmetric(ia, ib, ic, vals, S, rank=min(rank, S - 1), n_iter=n_iter)
    # scatter sub-vocab vectors back into full V (terms outside sub get zero)
    W = np.zeros((V, A.shape[1]), dtype=np.float64)
    for sub_id in range(S):
        W[inv_sub[sub_id]] = A[sub_id]
    return W, info


# ----------------------------------------------------------------------------- doc vectors + retrieval
def doc_vectors(doc_ids, W, vocab, idf):
    """idf-weighted mean of word vectors per doc, L2-normalized."""
    V, k = W.shape
    inv_vocab = {i: w for w, i in vocab.items()}
    idf_arr = np.zeros(V)
    for w, i in vocab.items():
        idf_arr[i] = idf.get(w, 0.0)
    dlist = list(doc_ids)
    D = np.zeros((len(dlist), k), dtype=np.float64)
    for j, d in enumerate(dlist):
        ids = doc_ids[d]
        if not ids:
            continue
        c = Counter(ids)
        vec = np.zeros(k)
        wsum = 0.0
        for i, f in c.items():
            wt = idf_arr[i] * math.log(1 + f)
            vec += wt * W[i]
            wsum += wt
        if wsum > 0:
            vec /= wsum
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        D[j] = vec
    return D, dlist


def query_vector(q, W, vocab, idf):
    V, k = W.shape
    vec = np.zeros(k)
    wsum = 0.0
    for w in tok(q):
        if w in vocab:
            i = vocab[w]
            wt = idf.get(w, 0.0)
            vec += wt * W[i]
            wsum += wt
    if wsum > 0:
        vec /= wsum
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n
    return vec


def cosine_search(qvec, D, dlist, topn=100):
    if np.linalg.norm(qvec) == 0:
        return []
    sims = D @ qvec
    order = np.argsort(-sims)[:topn]
    return [dlist[i] for i in order], sims, order


def rrf_fuse(bm25_ranked, dense_ranked, k=60, topn=10):
    """Reciprocal-rank fusion of two ranked id lists."""
    score = defaultdict(float)
    for r, d in enumerate(bm25_ranked):
        score[d] += 1.0 / (k + r + 1)
    for r, d in enumerate(dense_ranked):
        score[d] += 1.0 / (k + r + 1)
    return sorted(score, key=lambda d: score[d], reverse=True)[:topn]


# ----------------------------------------------------------------------------- evaluation
def evaluate_corpus(name, rank=100, k_svd=200, max_terms=2500):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*70}\n{name}: {len(corpus)} docs | test {len(test_ids)} q")

    t0 = time.time()
    bm = BM25(corpus)
    print(f"  BM25 built in {time.time()-t0:.1f}s")

    vocab, idf = build_vocab(corpus)
    doc_ids = doc_token_lists(corpus, vocab)
    print(f"  vocab={len(vocab)}")

    # 2-way PMI-SVD
    t0 = time.time()
    W2 = two_way_vectors(corpus, vocab, doc_ids, k=k_svd)
    D2, dlist = doc_vectors(doc_ids, W2, vocab, idf)
    print(f"  2-way PMI-SVD vectors (k={W2.shape[1]}) in {time.time()-t0:.1f}s")

    # 3-way CP
    t0 = time.time()
    W3, info3 = three_way_vectors(corpus, vocab, doc_ids, rank=rank,
                                  max_terms=max_terms, n_iter=15)
    print(f"  3-way CP-ALS: {info3} in {time.time()-t0:.1f}s")
    if W3 is not None:
        D3, _ = doc_vectors(doc_ids, W3, vocab, idf)
    else:
        D3 = None

    # cache bm25 rankings
    bm_rank = {qid: bm.search(queries[qid], topn=100) for qid in test_ids}

    def eval_lex():
        return np.mean([ndcg10(bm_rank[q], test_q[q]) for q in test_ids])

    def eval_dense(D):
        nd = []
        for qid in test_ids:
            qv = query_vector(queries[qid], W2 if D is D2 else W3, vocab, idf)
            res = cosine_search(qv, D, dlist, topn=10)
            ranked = res[0] if res else []
            nd.append(ndcg10(ranked, test_q[qid]))
        return np.mean(nd)

    def eval_hybrid(D, W):
        nd = []
        for qid in test_ids:
            qv = query_vector(queries[qid], W, vocab, idf)
            res = cosine_search(qv, D, dlist, topn=100)
            dense_ranked = res[0] if res else []
            fused = rrf_fuse(bm_rank[qid], dense_ranked, topn=10)
            nd.append(ndcg10(fused, test_q[qid]))
        return np.mean(nd)

    lex = eval_lex()
    d2 = eval_dense(D2)
    h2 = eval_hybrid(D2, W2)
    print(f"  --- lexical BM25:          nDCG@10 {lex:.4f}")
    print(f"  2-way PMI-SVD cosine:      nDCG@10 {d2:.4f}")
    print(f"  2-way + BM25 (RRF):        nDCG@10 {h2:.4f}")
    if D3 is not None:
        d3 = eval_dense(D3)
        h3 = eval_hybrid(D3, W3)
        print(f"  3-way CP cosine:           nDCG@10 {d3:.4f}")
        print(f"  3-way + BM25 (RRF):        nDCG@10 {h3:.4f}")
    else:
        d3 = h3 = float("nan")
        print("  3-way CP: tensor too sparse, skipped")

    return {"lex": lex, "d2": d2, "h2": h2, "d3": d3, "h3": h3}


def main():
    print("LENS: tensor-3way (CP/PARAFAC of the term-term-term PPMI tensor)")
    print("compare 3-way word vectors vs 2-way PMI-SVD (LSA), cosine + RRF-BM25")
    res = {}
    for ds in ("scifact", "nfcorpus"):
        res[ds] = evaluate_corpus(ds, rank=150)

    print(f"\n{'='*70}\nSUMMARY (nDCG@10)")
    print(f"  {'corpus':10s} {'BM25':>8s} {'2wSVD':>8s} {'2w+RRF':>8s} {'3wCP':>8s} {'3w+RRF':>8s}")
    for ds, r in res.items():
        print(f"  {ds:10s} {r['lex']:8.4f} {r['d2']:8.4f} {r['h2']:8.4f} {r['d3']:8.4f} {r['h3']:8.4f}")
    print("\nReference: lexical lattice 0.7023/0.3204, dense ~0.70/0.34")
    return res


if __name__ == "__main__":
    main()
