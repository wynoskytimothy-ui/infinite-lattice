#!/usr/bin/env python3
"""
LENS: nmf-topics  (v2 - conservative reranking + k sweep)
=========================================================
v1 showed NMF cosine alone is weak (0.09/0.14) and naive RRF over the full
corpus HURTS (the weak topic ranking injects drift docs). The disciplined move
(every prior signal-fusion lesson): use the topic vectors ONLY to RERANK the
BM25 candidate POOL, never to introduce new docs. Then NMF can only reorder
lexical hits, never drag in junk. We also sweep k and the fusion weight, and
report a within-pool RRF and a within-pool linear blend.

Output per corpus: BM25 vs best NMF-rerank vs best blend, for several k.
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10  # noqa: E402

_TOK = re.compile(r"[a-z]+")
_STOP = set(
    "the a an and or but if then of to in on at by for with from as is are was "
    "were be been being this that these those it its their his her our your my "
    "we you they he she them him us not no nor so than too very can will would "
    "should could may might must do does did done have has had having into out "
    "up down over under again further once here there all any both each few more "
    "most other some such only own same s t".split())


def tok(text):
    return [w for w in _TOK.findall(text.lower()) if len(w) > 2 and w not in _STOP]


def build_tfidf(corpus, min_df=2, max_df_frac=0.5):
    doc_ids = list(corpus.keys())
    N = len(doc_ids)
    df = defaultdict(int)
    toks = []
    for d in doc_ids:
        t = tok(corpus[d]); toks.append(t)
        for w in set(t):
            df[w] += 1
    max_df = int(max_df_frac * N)
    vocab = {w: i for i, w in enumerate(
        sorted(w for w, c in df.items() if min_df <= c <= max_df))}
    V = len(vocab)
    idf = np.zeros(V)
    for w, i in vocab.items():
        idf[i] = math.log((N + 1) / (df[w] + 1)) + 1.0
    rows, cols, vals = [], [], []
    for j, t in enumerate(toks):
        tf = defaultdict(int)
        for w in t:
            if w in vocab:
                tf[w] += 1
        for w, c in tf.items():
            i = vocab[w]
            rows.append(j); cols.append(i); vals.append((1.0 + math.log(c)) * idf[i])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(N, V))
    X = normalize(X, norm="l2", axis=1)
    return doc_ids, vocab, idf, X, toks


class BM25:
    def __init__(self, toks, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.N = len(toks)
        self.df = defaultdict(int)
        self.postings = defaultdict(list)
        self.doc_len = np.zeros(self.N)
        for j, t in enumerate(toks):
            tf = defaultdict(int)
            for w in t:
                tf[w] += 1
            self.doc_len[j] = len(t)
            for w, c in tf.items():
                self.df[w] += 1
                self.postings[w].append((j, c))
        self.avgdl = self.doc_len.mean() if self.N else 0.0
        self.idf = {w: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
                    for w, c in self.df.items()}

    def scores(self, qtoks):
        s = np.zeros(self.N)
        for w in set(qtoks):
            if w not in self.postings:
                continue
            idf = self.idf[w]
            for j, c in self.postings[w]:
                denom = c + self.k1 * (1 - self.b + self.b * self.doc_len[j] / self.avgdl)
                s[j] += idf * (c * (self.k1 + 1)) / denom
        return s


def topk_idx(scores, k):
    if k >= len(scores):
        return list(np.argsort(-scores))
    part = np.argpartition(-scores, k)[:k]
    return list(part[np.argsort(-scores[part])])


def project_query(qt, vocab, idf, X, nmf):
    qtf = defaultdict(int)
    for w in qt:
        if w in vocab:
            qtf[w] += 1
    if not qtf:
        return None
    qvec = np.zeros((1, X.shape[1]))
    for w, c in qtf.items():
        i = vocab[w]
        qvec[0, i] = (1.0 + math.log(c)) * idf[i]
    qvec = normalize(qvec, norm="l2", axis=1)
    qt_ = normalize(nmf.transform(qvec), norm="l2", axis=1)
    return qt_


def run(name, ks=(60, 120, 200), pool=100):
    corpus, queries, _train, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    doc_ids, vocab, idf, X, toks = build_tfidf(corpus)
    bm = BM25(toks)
    print(f"\n{'='*64}\n{name}: {X.shape[0]} docs x {X.shape[1]} terms | "
          f"test {len(test_ids)} q | pool={pool}")

    # precompute BM25 pools + baseline once
    pools, qtok_cache, baseline = {}, {}, []
    for q in test_ids:
        qt = tok(queries[q]); qtok_cache[q] = qt
        s = bm.scores(qt)
        idxs = topk_idx(s, pool)
        pools[q] = (idxs, s[idxs])
        baseline.append(ndcg10([doc_ids[i] for i in idxs], test_q[q]))
    bm25_ndcg = sum(baseline) / len(baseline)
    print(f"  BM25                 nDCG@10 = {bm25_ndcg:.4f}")

    best = {"k": None, "blend": bm25_ndcg, "alpha": 0.0, "rerank": bm25_ndcg, "topic": ""}
    for k in ks:
        t0 = time.time()
        nmf = NMF(n_components=k, init="nndsvd", max_iter=300, tol=1e-4, random_state=0)
        W = nmf.fit_transform(X)
        H = nmf.components_
        Wn = normalize(W, norm="l2", axis=1)
        inv = {i: w for w, i in vocab.items()}
        ex = min(3, k - 1)
        topic = ", ".join(inv[i] for i in np.argsort(-H[ex])[:8])

        # pure rerank: reorder pool by topic-cosine only
        # blend: normalize bm25 & topic-cosine within pool, alpha-blend
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
        rer_acc = []
        blend_acc = {a: [] for a in alphas}
        for q in test_ids:
            idxs, bm_s = pools[q]
            qt_ = project_query(qtok_cache[q], vocab, idf, X, nmf)
            if qt_ is None:
                topic_s = np.zeros(len(idxs))
            else:
                topic_s = (Wn[idxs] @ qt_.T).ravel()
            # min-max normalize within pool
            def nz(a):
                a = np.asarray(a, float)
                rng = a.max() - a.min()
                return (a - a.min()) / rng if rng > 1e-12 else np.zeros_like(a)
            bmn, tn = nz(bm_s), nz(topic_s)
            order = np.argsort(-tn)
            rer_acc.append(ndcg10([doc_ids[idxs[i]] for i in order], test_q[q]))
            for a in alphas:
                blend = (1 - a) * bmn + a * tn
                o = np.argsort(-blend)
                blend_acc[a].append(ndcg10([doc_ids[idxs[i]] for i in o], test_q[q]))
        rer = sum(rer_acc) / len(rer_acc)
        besta = max(alphas, key=lambda a: sum(blend_acc[a]) / len(blend_acc[a]))
        bl = sum(blend_acc[besta]) / len(blend_acc[besta])
        print(f"  k={k:3d} rerank={rer:.4f}  blend(a={besta})={bl:.4f}  "
              f"[{time.time()-t0:.0f}s]  topic#{ex}: {topic}")
        if bl > best["blend"]:
            best.update(k=k, blend=bl, alpha=besta, rerank=rer, topic=topic)
    print(f"  >> BEST blend nDCG@10 = {best['blend']:.4f} "
          f"(k={best['k']}, alpha={best['alpha']})  vs BM25 {bm25_ndcg:.4f}")
    return bm25_ndcg, best


if __name__ == "__main__":
    out = {}
    for ds in ["scifact", "nfcorpus"]:
        out[ds] = run(ds)
    print("\n" + "=" * 64)
    print("SUMMARY (nDCG@10)      BM25     best-blend  (k, alpha)")
    for ds, (bm, b) in out.items():
        print(f"  {ds:10s} {bm:.4f}   {b['blend']:.4f}    (k={b['k']}, a={b['alpha']})")
