#!/usr/bin/env python3
"""RUNG 0 -- the shared measurement lab for the ground-up build.

Every rung swaps ONE thing (the tokenizer / representation) and is measured on the SAME
cached pool with the SAME engine (inverted index + idf), so comparisons are honest. The
engine is fixed; the rung defines `tok(text) -> [terms]`. Climb only when a rung is proven.

  from marco_lab import load_pool, Index, evaluate
  qids, queries, qrels, texts = load_pool()
  idx = Index(my_tokenizer).build(texts)
  evaluate(idx, qids, queries, qrels, label="rung-N variant")
"""
import os, math, pickle, time
from collections import defaultdict
from marco_baseline import load_dev, build_pool, load_texts

POOL_CACHE = "marco_pool.pkl"


def load_pool(nq=3000, nd=300_000, rebuild=False):
    """Fixed, cached pool so every rung sees identical data (reproducible + fast reload)."""
    if os.path.exists(POOL_CACHE) and not rebuild:
        with open(POOL_CACHE, "rb") as f:
            d = pickle.load(f)
        print(f"  [lab] loaded cached pool: {len(d[0])} queries, {len(d[3])} passages")
        return d
    qrels, queries = load_dev()
    qids, pool, _ = build_pool(qrels, queries, nq, nd)
    texts = load_texts(pool)
    qids = [q for q in qids if all(p in texts for p in qrels[q])]
    d = (qids, {q: queries[q] for q in qids}, {q: qrels[q] for q in qids}, texts)
    with open(POOL_CACHE, "wb") as f:
        pickle.dump(d, f)
    print(f"  [lab] built + cached pool: {len(qids)} queries, {len(texts)} passages")
    return d


class Index:
    """Fixed BM25 engine; the ONLY thing that varies across rungs is `tok`."""
    def __init__(self, tok, k1=0.9, b=0.4):
        self.tok, self.k1, self.b = tok, k1, b

    def build(self, texts):
        self.post = defaultdict(list)
        self.df = defaultdict(int)
        self.doclen, self.docids = [], []
        t0 = time.perf_counter()
        for pid, text in texts.items():
            di = len(self.docids)
            self.docids.append(pid)
            tf = defaultdict(int)
            for w in self.tok(text):
                tf[w] += 1
            self.doclen.append(sum(tf.values()) or 1)
            for w, c in tf.items():
                self.post[w].append((di, c))
                self.df[w] += 1
        self.N = len(self.docids)
        self.avgdl = sum(self.doclen) / max(1, self.N)
        self.idf = {w: math.log((self.N - df + 0.5) / (df + 0.5) + 1.0) for w, df in self.df.items()}
        self.postings_bytes = sum(len(v) for v in self.post.values()) * 8
        self.build_s = time.perf_counter() - t0
        return self

    def search(self, query, k=100, min_idf=0.3):
        sc = defaultdict(float)
        for w in set(self.tok(query)):
            idf = self.idf.get(w)
            if idf is None or idf < min_idf:
                continue
            for di, c in self.post[w]:
                dl = self.doclen[di]
                sc[di] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        top = sorted(sc, key=sc.get, reverse=True)[:k]
        return [self.docids[di] for di in top]


def evaluate(index, qids, queries, qrels, ref_rr=None, label=""):
    """MRR@10 / R@10 / R@100 on the fixed pool; optional hit/miss split vs a reference rr dict."""
    mrr = r10 = r100 = 0.0
    rrs = {}
    t0 = time.perf_counter()
    for q in qids:
        rel = qrels[q]
        ranked = index.search(queries[q], 100)
        rr = 0.0
        for i, pid in enumerate(ranked[:10], 1):
            if pid in rel:
                rr = 1.0 / i
                break
        rrs[q] = rr
        mrr += rr
        r10 += len(rel & set(ranked[:10])) / len(rel)
        r100 += len(rel & set(ranked[:100])) / len(rel)
    n = len(qids)
    split = ""
    if ref_rr is not None:
        hs = h = ms = m = 0
        for q in qids:
            if ref_rr.get(q, 0) > 0:
                h += 1; hs += rrs[q]
            else:
                m += 1; ms += rrs[q]
        split = f"   [hit {hs/max(1,h):.4f}  miss {ms/max(1,m):.4f}]"
    proj_gb = index.postings_bytes / index.N * 8_841_823 / 1e9
    print(f"  {label:>24}  MRR {mrr/n:.4f}  R@10 {r10/n:.4f}  R@100 {r100/n:.4f}  "
          f"({len(index.post)} terms, ~{proj_gb:.1f}GB full){split}")
    return {"mrr": mrr / n, "r10": r10 / n, "r100": r100 / n, "rrs": rrs}


if __name__ == "__main__":
    import re
    qids, queries, qrels, texts = load_pool()
    word = lambda s, _r=re.compile(r"[a-z0-9]+"): _r.findall(s.lower())
    idx = Index(word).build(texts)
    print(f"  [lab] rung-1 index built in {idx.build_s:.0f}s")
    evaluate(idx, qids, queries, qrels, label="rung1: words (floor)")
