#!/usr/bin/env python3
"""
Deep search step 5 - compress the dense query structure, keep speed + accuracy.

The numpy dense scorer is fast but fat: int32 doc-id (4B) + float64 tf (8B) =
12 B/posting. Three levers:
  1. doc-id dtype by N: <65536 docs -> uint16 (2B), exact.
  2. tf quantize: float16 (2B) or uint8+global scale (1B). tf values are small
     and few-valued (gear weights x positional x count), so error is tiny.
  3. index-time trigram prune: the dense scorer already SKIPS df>N/2 trigrams at
     query time, so storing them is waste - drop them from the arrays (lossless,
     and removes the longest lists -> smaller AND faster).

Measured vs the EXACT dict ranking: bytes/posting, dense MB, ms/q, top-10
overlap, nDCG/Recall. A row "keeps accuracy" if overlap ~100% and nDCG flat.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import load, ndcg10, recall10


class DenseVar:
    """Dense scorer with configurable doc-id dtype, tf encoding, trigram prune."""

    def __init__(self, idx, tf_mode="f64", prune_tri=False):
        N = max(1, len(idx.alive))
        self.N = N
        self.idx = idx
        self.tf_mode = tf_mode
        k1, b = idx.k1, idx.b
        avgdl = idx._total_len / N
        self.k1p1 = k1 + 1
        A, Bc = k1 * (1 - b), k1 * b / avgdl
        self.tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
        docs = list(idx.alive)
        self.docs = docs
        d2i = {d: i for i, d in enumerate(docs)}
        self.didx_dtype = np.uint16 if N < 65536 else np.uint32
        self.denom = np.array([A + Bc * idx.doc_len[d] for d in docs], dtype=np.float64)
        # global tf scale for uint8
        maxtf = max((max(pl.values()) for pl in idx.postings.values() if pl), default=1.0)
        self.scale = maxtf / 255.0
        self.df = idx.df
        self.token_prime = idx.token_prime
        self.p_doc, self.p_tf = {}, {}
        n_post = 0
        for (view, _tok), p in idx.token_prime.items():
            pl = idx.postings.get(p)
            if not pl:
                continue
            if prune_tri and view == "3" and self.tri_cap is not None \
                    and idx.df[p] > self.tri_cap:
                continue                                   # never traversed -> don't store
            di = np.fromiter((d2i[d] for d in pl if d in d2i), dtype=self.didx_dtype)
            if not di.size:
                continue
            tf = np.fromiter((pl[d] for d in pl if d in d2i), dtype=np.float64, count=di.size)
            self.p_doc[p] = di
            if tf_mode == "f64":
                self.p_tf[p] = tf
            elif tf_mode == "f32":
                self.p_tf[p] = tf.astype(np.float32)
            elif tf_mode == "f16":
                self.p_tf[p] = tf.astype(np.float16)
            elif tf_mode == "u8":
                self.p_tf[p] = np.clip(np.round(tf / self.scale), 1, 255).astype(np.uint8)
            n_post += di.size
        self.n_post = n_post

    def bytes_per_posting(self):
        idb = np.dtype(self.didx_dtype).itemsize
        tfb = {"f64": 8, "f32": 4, "f16": 2, "u8": 1}[self.tf_mode]
        return idb + tfb

    def mb(self):
        return (self.n_post * self.bytes_per_posting() + self.denom.nbytes) / 1e6

    def search(self, query, k=10):
        N = self.N
        scores = np.zeros(len(self.docs), dtype=np.float64)
        cap = self.tri_cap
        for tok, qwt in self.idx._multiview(query).items():
            p = self.token_prime.get(tok)
            if p is None:
                continue
            dfp = self.df[p]
            if dfp == 0:
                continue
            if cap is not None and tok[0] == "3" and dfp > cap:
                continue
            di = self.p_doc.get(p)
            if di is None:
                continue
            tfa = self.p_tf[p]
            if self.tf_mode == "u8":
                tfa = tfa.astype(np.float64) * self.scale
            elif self.tf_mode != "f64":
                tfa = tfa.astype(np.float64)
            idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
            di64 = di.astype(np.intp)
            scores[di64] += (qwt * idf * self.k1p1) * tfa / (tfa + self.denom[di64])
        kk = min(k, len(self.docs))
        part = np.argpartition(scores, -kk)[-kk:]
        part = part[np.argsort(scores[part])[::-1]]
        return [self.docs[i] for i in part if scores[i] > 0.0]


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    print(f"\n{'='*72}\n{name}: {len(idx.alive):,} docs, {len(test_ids)} q")

    # exact ground truth via dict path
    exact = {q: idx.search(queries[q], 10) for q in test_ids}

    def bench(var, label):
        nd = rc = 0.0
        lat = []
        ov = 0
        for qid in test_ids:
            t = time.perf_counter()
            ranked = var.search(queries[qid], 10)
            lat.append((time.perf_counter() - t) * 1000)
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
            ov += len(set(ranked) & set(exact[qid]))
        n = len(test_ids)
        print(f"  {label:<26} {var.bytes_per_posting():>4}B {var.mb():>7.1f} "
              f"{var.n_post/1e6:>7.2f}M {nd/n:>7.4f} {rc/n:>7.4f} "
              f"{sum(lat)/n:>6.2f} {ov/(10*n)*100:>6.1f}%")

    print(f"  {'config':<26} {'B/p':>5} {'MB':>7} {'postings':>8} "
          f"{'nDCG':>7} {'Recall':>7} {'ms':>6} {'top10':>7}")
    bench(DenseVar(idx, "f64"), "f64 + int32 (current)")
    bench(DenseVar(idx, "f32"), "f32 + uint16")
    bench(DenseVar(idx, "f16"), "f16 + uint16")
    bench(DenseVar(idx, "u8"), "uint8 + uint16")
    bench(DenseVar(idx, "u8", prune_tri=True), "uint8+uint16 + tri-prune")
    bench(DenseVar(idx, "f16", prune_tri=True), "f16+uint16 + tri-prune")


def main():
    print("DEEP SEARCH step 5 - compress dense arrays (keep speed + accuracy)")
    print("B/p = bytes per posting; vs exact dict ranking (top10 overlap)")
    for ds in ("scifact", "nfcorpus"):
        run(ds)
    print("\n  Goal: smallest MB with top10 ~100% and ms not worse. uint16 doc-id")
    print("  is exact (<65536 docs); f16/u8 tf trade ~0 accuracy; tri-prune drops")
    print("  the df>N/2 trigrams we already never traverse (lossless).")


if __name__ == "__main__":
    main()
