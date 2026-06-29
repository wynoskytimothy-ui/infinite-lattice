#!/usr/bin/env python3
"""
LENS: holographic-vsa
=====================
Holographic / Vector-Symbolic-Architecture doc vectors, NO training, NO GPU.

Each WORD gets a fixed high-dim bipolar hypervector (+/-1), seeded
deterministically from its prime/token address. A document hypervector is built
by BINDING and BUNDLING those word vectors:

  * bundle      : sum of (idf-weighted) word hypervectors  -> bag-of-words sketch
                  (this is literally a sign-random / Johnson-Lindenstrauss sketch
                   of the idf term-doc matrix -> a random-feature LSA)
  * pos-bind    : bind word with a permuted "position" role before bundling
                  (rho^k . w_token) -> sequence-sensitive
  * bigram-bind : bind adjacent word pairs by circular convolution (w_a (*) w_b)
                  and bundle those -> the part VSA genuinely adds over a bag:
                  composite ngram atoms that a lexical index never stores.

Queries are encoded the same way; retrieve by cosine over the D-dim doc vectors.
We also RRF-fuse the VSA cosine ranking with the BM25 lexical ranking.

Question under test: does fixed-random structured binding carry semantic
retrieval signal WITHOUT any learning, and does it beat the lexical lattice
baseline (scifact 0.7023 / nfcorpus 0.3204) or approach dense (~0.70 / ~0.34)?

CPU only: numpy. Measures nDCG@10 on scifact and nfcorpus.
"""
from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10

TOKEN_RE = re.compile(r"[a-z]+")


def toks(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if len(w) > 2]


# ----------------------------------------------------------------------------
# Fixed hypervector codebook: each word -> deterministic bipolar {+1,-1}^D
# seeded from a stable hash of the word (its "prime address" analogue).
# ----------------------------------------------------------------------------
class Codebook:
    def __init__(self, D, seed=12345):
        self.D = D
        self.seed = seed
        self._cache = {}
        # one fixed permutation used as the position "rotation" role rho
        rng = np.random.default_rng(seed ^ 0xABCDEF)
        self.perm = rng.permutation(D)

    def _seed_of(self, word):
        # stable 64-bit hash of the word string (token/prime address analogue)
        h = 1469598103934665603
        for ch in word.encode("utf-8"):
            h ^= ch
            h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        return (h ^ self.seed) & 0xFFFFFFFFFFFFFFFF

    def vec(self, word):
        v = self._cache.get(word)
        if v is None:
            rng = np.random.default_rng(self._seed_of(word))
            # bipolar +/-1 hypervector
            v = (rng.integers(0, 2, self.D, dtype=np.int8) * 2 - 1).astype(np.float32)
            self._cache[word] = v
        return v

    def permute(self, v, k):
        """rho^k applied to v (cheap sequence role binding)."""
        out = v
        for _ in range(k % 7):  # cap so far-apart positions still differ
            out = out[self.perm]
        return out


def circ_conv(a, b):
    """Circular convolution binding via FFT (holographic reduced representation)."""
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=a.shape[0])


# ----------------------------------------------------------------------------
def build(corpus, D, mode, idf):
    """Return docids, doc matrix (n,D) L2-normalized, and the codebook."""
    cb = Codebook(D)
    docids = list(corpus.keys())
    M = np.zeros((len(docids), D), dtype=np.float32)
    for i, d in enumerate(docids):
        tk = toks(corpus[d])
        if not tk:
            continue
        acc = np.zeros(D, dtype=np.float32)
        if mode == "bundle":
            for w in tk:
                acc += idf.get(w, 0.0) * cb.vec(w)
        elif mode == "pos":
            for k, w in enumerate(tk):
                acc += idf.get(w, 0.0) * cb.permute(cb.vec(w), k)
        elif mode == "bigram":
            # unigrams + bound adjacent bigrams (the VSA-native composite atoms)
            for w in tk:
                acc += idf.get(w, 0.0) * cb.vec(w)
            for a, b in zip(tk, tk[1:]):
                wt = math.sqrt(max(idf.get(a, 0.0), 0.0) * max(idf.get(b, 0.0), 0.0))
                if wt > 0:
                    acc += wt * circ_conv(cb.vec(a), cb.vec(b)).astype(np.float32)
        n = np.linalg.norm(acc)
        if n > 0:
            M[i] = acc / n
    return docids, M, cb


def encode_query(q, cb, D, mode, idf):
    tk = toks(q)
    acc = np.zeros(D, dtype=np.float32)
    if not tk:
        return acc
    if mode == "bundle":
        for w in tk:
            acc += idf.get(w, 0.0) * cb.vec(w)
    elif mode == "pos":
        for k, w in enumerate(tk):
            acc += idf.get(w, 0.0) * cb.permute(cb.vec(w), k)
    elif mode == "bigram":
        for w in tk:
            acc += idf.get(w, 0.0) * cb.vec(w)
        for a, b in zip(tk, tk[1:]):
            wt = math.sqrt(max(idf.get(a, 0.0), 0.0) * max(idf.get(b, 0.0), 0.0))
            if wt > 0:
                acc += wt * circ_conv(cb.vec(a), cb.vec(b)).astype(np.float32)
    n = np.linalg.norm(acc)
    return acc / n if n > 0 else acc


# ----------------------------------------------------------------------------
# BM25 lexical baseline (for fair comparison + RRF fusion)
# ----------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus, k1=0.9, b=0.4):
        self.docids = list(corpus.keys())
        self.k1, self.b = k1, b
        self.tf = []
        df = Counter()
        tot = 0
        for d in self.docids:
            c = Counter(toks(corpus[d]))
            self.tf.append(c)
            tot += sum(c.values())
            for w in c:
                df[w] += 1
        self.N = len(self.docids)
        self.avgdl = tot / max(self.N, 1)
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            dl = sum(c.values())
            for w, f in c.items():
                self.post[w].append((i, f, dl))

    def search(self, q, topk=100):
        sc = defaultdict(float)
        for w in set(toks(q)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f, dl in self.post[w]:
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                sc[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(sc, key=lambda i: sc[i], reverse=True)[:topk]
        return ranked, sc


def rrf(rank_lists, k=60):
    """Reciprocal-rank fusion of several ranked id-lists -> fused score dict."""
    fused = defaultdict(float)
    for rl in rank_lists:
        for r, i in enumerate(rl):
            fused[i] += 1.0 / (k + r + 1)
    return fused


# ----------------------------------------------------------------------------
def run(name, D=4096):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*70}\n{name}: {len(corpus)} docs | test {len(test_ids)} q | D={D}")

    # idf over corpus
    df = Counter()
    for d in corpus:
        for w in set(toks(corpus[d])):
            df[w] += 1
    N = len(corpus)
    idf = {w: math.log((N + 1) / (n + 0.5)) for w, n in df.items()}

    bm = BM25(corpus)
    id2pos = {d: i for i, d in enumerate(bm.docids)}

    results = {}

    # ---- BM25 lexical baseline ----
    nd = 0.0
    for qid in test_ids:
        ranked_i, _ = bm.search(queries[qid], 10)
        ranked = [bm.docids[i] for i in ranked_i]
        nd += ndcg10(ranked, test_q[qid])
    results["bm25"] = nd / len(test_ids)
    print(f"  BM25 lexical          nDCG@10 = {results['bm25']:.4f}")

    for mode in ("bundle", "pos", "bigram"):
        t0 = time.time()
        docids, M, cb = build(corpus, D, mode, idf)
        pos2id = {i: docids[i] for i in range(len(docids))}
        # precompute query encodings + cosine
        nd_pure = 0.0
        nd_rrf = 0.0
        for qid in test_ids:
            qv = encode_query(queries[qid], cb, D, mode, idf)
            sims = M @ qv  # cosine (M and qv are L2-normalized)
            top_vsa_pos = np.argpartition(-sims, min(100, len(sims) - 1))[:100]
            top_vsa_pos = top_vsa_pos[np.argsort(-sims[top_vsa_pos])]
            vsa_ranked_pos = list(top_vsa_pos)
            # pure VSA nDCG
            ranked_pure = [pos2id[i] for i in vsa_ranked_pos[:10]]
            nd_pure += ndcg10(ranked_pure, test_q[qid])
            # RRF with BM25 (convert both to a common id space)
            bm_i, _ = bm.search(queries[qid], 100)
            bm_ranked_ids = [bm.docids[i] for i in bm_i]
            vsa_ranked_ids = [pos2id[i] for i in vsa_ranked_pos]
            fused = rrf([bm_ranked_ids, vsa_ranked_ids])
            fused_ranked = sorted(fused, key=lambda d: fused[d], reverse=True)[:10]
            nd_rrf += ndcg10(fused_ranked, test_q[qid])
        nd_pure /= len(test_ids)
        nd_rrf /= len(test_ids)
        results[f"{mode}_pure"] = nd_pure
        results[f"{mode}_rrf"] = nd_rrf
        print(f"  VSA[{mode:6s}] pure    nDCG@10 = {nd_pure:.4f}   "
              f"RRF+BM25 = {nd_rrf:.4f}   ({time.time()-t0:.1f}s)")

    return results


def main():
    all_res = {}
    for ds in ("scifact", "nfcorpus"):
        all_res[ds] = run(ds, D=4096)
    print(f"\n{'='*70}\nSUMMARY (nDCG@10)")
    print(f"  reference: BM25 scifact 0.665 / nfcorpus 0.325 ; "
          f"lexical-lattice 0.7023 / 0.3204 ; dense ~0.70 / ~0.34")
    for ds, r in all_res.items():
        print(f"\n  {ds}:")
        for k, v in r.items():
            print(f"     {k:16s} {v:.4f}")


if __name__ == "__main__":
    main()
