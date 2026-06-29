#!/usr/bin/env python3
"""
LENS: lattice-coord-features
============================
Timothy's literal 3D-plane geometry AS the embedding feature map.

For each doc we map its terms through the REAL AETHOS formula
(aethos_complex_plane.wing_transform: 4 branches x 8 wings = 32 chambers,
each yielding (X, Y, zeta) coords) and POOL them into a fixed-dim vector, then
cosine-retrieve.  The honest question: does the lattice geometry carry SEMANTIC
similarity, or only value / positional structure?

To give the geometry a fair shot at semantics we test THREE ways of turning a
term into a lattice chain A = (a_1,...,a_k) (the geometry's input):

  (A) IDENTITY chain  : a = token-rank only.  Pure value/positional geometry,
      no corpus statistics.  (Expected: carries NOTHING semantic -> baseline.)

  (B) COOCC chain     : a term's chain = ranks of its top co-occurring partner
      terms (PPMI).  Now two terms that share neighbours get NEARBY chains, so
      the geometry pools real corpus structure.  This is the geometry acting as
      a non-linear feature map over a co-occurrence signature.

  (C) SVD-quantized   : factorize the PPMI co-occurrence (LSA), then QUANTIZE the
      top latent coords into a chain and run them through the geometry.  Tests
      whether the 3D-plane map adds anything on top of plain LSA.

Each term -> 32 chambers x (X,Y,zeta) = 96 raw numbers.  A doc vector pools its
terms' chamber coords with TF-IDF weights, using both MEAN (centroid) and a
chamber HISTOGRAM (which chamber each term's modulus is largest in) -> a fixed
feature vector.  Retrieve by cosine, and also RRF-fuse with the BM25 lexical
ranking.

CPU only: numpy, scipy.sparse, sklearn TruncatedSVD.  No torch / GPU / downloads.
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
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from scripts.bench_supervised_bridges import load, ndcg10  # noqa: E402
from aethos_complex_plane import wing_transform              # noqa: E402
from aethos_lattice import BranchKind                        # noqa: E402

TOKEN_RE = re.compile(r"[a-z]+")


def tok(text):
    return [w for w in TOKEN_RE.findall(text.lower()) if len(w) > 2]


# ---------------------------------------------------------------------------
# The 32-chamber lattice geometry: chain -> 96-dim coord block (X,Y,zeta x 32)
# ---------------------------------------------------------------------------
_BRANCHES = list(BranchKind)


def chain_to_chambers(chain, n):
    """Run a chain through ALL 32 chambers (4 branches x 8 wings) at transgressor
    n; return a (32,3) array of (X, Y, zeta).  This is the literal AETHOS map."""
    out = np.empty((32, 3), dtype=np.float64)
    i = 0
    for b in _BRANCHES:
        for w in range(1, 9):
            psi = wing_transform(b, chain, n, w)
            out[i, 0] = psi.z.real
            out[i, 1] = psi.z.imag
            out[i, 2] = psi.zeta
            i += 1
    return out


# ---------------------------------------------------------------------------
# BM25 lexical baseline (for the lexical reference + RRF fusion)
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, docs, k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.doc_ids = list(docs)
        self.df = defaultdict(int)
        self.tf = []
        self.dl = []
        for d in self.doc_ids:
            c = defaultdict(int)
            toks = tok(docs[d])
            for w in toks:
                c[w] += 1
            self.tf.append(c)
            self.dl.append(len(toks))
            for w in c:
                self.df[w] += 1
        self.N = len(self.doc_ids)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.idf = {w: math.log(1 + (self.N - f + 0.5) / (f + 0.5)) for w, f in self.df.items()}
        self.postings = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w, f in c.items():
                self.postings[w].append((i, f))

    def search(self, q, topk=100):
        scores = defaultdict(float)
        for w in set(tok(q)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                dl = self.dl[i]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(scores, key=scores.get, reverse=True)[:topk]
        return [(self.doc_ids[i], scores[i]) for i in ranked]


# ---------------------------------------------------------------------------
# Build the term universe + TF-IDF + PPMI co-occurrence
# ---------------------------------------------------------------------------
def build_vocab(corpus, min_df=2, max_terms=20000):
    df = defaultdict(int)
    doc_toks = {}
    for d, txt in corpus.items():
        ts = tok(txt)
        doc_toks[d] = ts
        for w in set(ts):
            df[w] += 1
    vocab = [w for w, f in df.items() if f >= min_df]
    # keep most informative (mid-df) terms first to bound cost
    vocab.sort(key=lambda w: df[w])
    vocab = vocab[:max_terms]
    vmap = {w: i for i, w in enumerate(vocab)}
    N = len(corpus)
    idf = np.array([math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in vocab])
    return vocab, vmap, doc_toks, idf, df


def ppmi_cooc(doc_toks, vmap, top_partners=3):
    """Term-term PPMI co-occurrence (doc-level), return for each term the ranks of
    its top co-occurring partners -> the term's lattice chain."""
    V = len(vmap)
    rows, cols, data = [], [], []
    for ts in doc_toks.values():
        present = sorted({vmap[w] for w in ts if w in vmap})
        for i in present:
            rows.append(i)
            cols.append(0)  # placeholder; we use term-doc then T@T
    # term-doc matrix
    rows, cols, data = [], [], []
    doc_index = {d: j for j, d in enumerate(doc_toks)}
    for d, ts in doc_toks.items():
        j = doc_index[d]
        for w in set(ts):
            i = vmap.get(w)
            if i is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)
    TD = sp.csr_matrix((data, (rows, cols)), shape=(V, len(doc_index)))
    CO = (TD @ TD.T).tocsr()  # term-term co-occurrence counts
    total = CO.sum()
    term_freq = np.asarray(CO.diagonal()).ravel() + 1e-9
    partners = []
    for i in range(V):
        start, end = CO.indptr[i], CO.indptr[i + 1]
        js = CO.indices[start:end]
        vs = CO.data[start:end]
        # PPMI
        ppmi = np.maximum(0.0, np.log((vs * total) / (term_freq[i] * term_freq[js] + 1e-9) + 1e-9))
        order = np.argsort(-ppmi)
        top = [int(js[k]) for k in order if js[k] != i][:top_partners]
        partners.append(top)
    return partners, TD


# ---------------------------------------------------------------------------
# Term -> chain, for each variant
# ---------------------------------------------------------------------------
def chains_identity(V):
    # token-rank only: chain = (i+1, i+3, i+5) — pure value geometry
    return [tuple(sorted({i + 1, i + 3, i + 5})) for i in range(V)]


def chains_cooc(partners, V):
    chains = []
    for i in range(V):
        ps = partners[i]
        base = sorted({i + 1} | {p + 1 for p in ps})
        if len(base) < 2:
            base = sorted({i + 1, i + 2})
        chains.append(tuple(base))
    return chains


def chains_svd(TD, V, dim=24):
    """LSA: factorize term-doc (TF) with TruncatedSVD, quantize top latent coords
    into a small strictly-increasing chain to feed the geometry."""
    svd = TruncatedSVD(n_components=dim, random_state=0)
    # term embeddings = U * S  (rows = terms)
    U = svd.fit_transform(TD)  # (V, dim)
    # rank within each latent dim -> integers; pick top-|coef| dims per term -> chain
    chains = []
    # global quantile bins per dim
    for i in range(V):
        v = U[i]
        idx = np.argsort(-np.abs(v))[:4]
        # map (dim_index, sign) into a positive integer anchor
        anchors = set()
        for d in idx:
            a = int(d) * 2 + (1 if v[d] >= 0 else 2)
            anchors.add(a + 1)
        base = sorted(anchors)
        if len(base) < 2:
            base = sorted({1, 2})
        chains.append(tuple(base))
    return chains


# ---------------------------------------------------------------------------
# Precompute per-term chamber feature block, then pool docs
# ---------------------------------------------------------------------------
def term_features(chains, n=None):
    """For each term chain, compute its 32-chamber coord block and reduce to a
    compact, scale-normalized per-term feature vector:
      - 96 raw coords (X,Y,zeta x 32 chambers), each normalized
      - chamber-modulus histogram (which chamber has the largest |z|): 32 dims
    Returns (term_raw (V,96), term_hist (V,32)) L2-normalized per term.
    """
    V = len(chains)
    raw = np.zeros((V, 96), dtype=np.float64)
    hist = np.zeros((V, 32), dtype=np.float64)
    for i, ch in enumerate(chains):
        nn = (max(ch) + 1) if n is None else n
        block = chain_to_chambers(ch, nn)  # (32,3)
        raw[i] = block.reshape(-1)
        mod = block[:, 0] ** 2 + block[:, 1] ** 2  # |z|^2 per chamber
        hist[i, np.argmax(mod)] = 1.0
    # standardize raw columns (z-score) so no single coord dominates
    mu = raw.mean(0)
    sd = raw.std(0) + 1e-9
    raw = (raw - mu) / sd
    return raw, hist


def doc_vectors(doc_toks, vmap, idf, term_raw, term_hist):
    """Pool each doc's terms' chamber features with TF-IDF weights (mean centroid
    over raw coords) and a TF-IDF-weighted chamber histogram. Concatenate."""
    V = term_raw.shape[0]
    doc_ids = list(doc_toks)
    D = len(doc_ids)
    F = term_raw.shape[1] + term_hist.shape[1]
    out = np.zeros((D, F), dtype=np.float64)
    for di, d in enumerate(doc_ids):
        ts = doc_toks[d]
        tfw = defaultdict(float)
        for w in ts:
            i = vmap.get(w)
            if i is not None:
                tfw[i] += 1.0
        if not tfw:
            continue
        wsum = 0.0
        acc_raw = np.zeros(term_raw.shape[1])
        acc_hist = np.zeros(term_hist.shape[1])
        for i, c in tfw.items():
            w = (1.0 + math.log(c)) * idf[i]
            acc_raw += w * term_raw[i]
            acc_hist += w * term_hist[i]
            wsum += w
        if wsum > 0:
            acc_raw /= wsum
        out[di, : term_raw.shape[1]] = acc_raw
        out[di, term_raw.shape[1]:] = acc_hist
    # L2 normalize doc vectors
    nrm = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    out = out / nrm
    return doc_ids, out


def query_vector(q, vmap, idf, term_raw, term_hist):
    ts = tok(q)
    tfw = defaultdict(float)
    for w in ts:
        i = vmap.get(w)
        if i is not None:
            tfw[i] += 1.0
    F = term_raw.shape[1] + term_hist.shape[1]
    v = np.zeros(F)
    if not tfw:
        return v
    wsum = 0.0
    for i, c in tfw.items():
        w = (1.0 + math.log(c)) * idf[i]
        v[: term_raw.shape[1]] += w * term_raw[i]
        v[term_raw.shape[1]:] += w * term_hist[i]
        wsum += w
    if wsum > 0:
        v[: term_raw.shape[1]] /= wsum
    nrm = np.linalg.norm(v) + 1e-12
    return v / nrm


# ---------------------------------------------------------------------------
# Retrieval + eval
# ---------------------------------------------------------------------------
def rrf(rank_lists, k=60):
    score = defaultdict(float)
    for rl in rank_lists:
        for r, d in enumerate(rl):
            score[d] += 1.0 / (k + r + 1)
    return sorted(score, key=score.get, reverse=True)


def eval_lens(name, variant, top_partners=3, svd_dim=24, fuse=True):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    vocab, vmap, doc_toks, idf, df = build_vocab(corpus)
    V = len(vocab)

    t0 = time.time()
    partners, TD = ppmi_cooc(doc_toks, vmap, top_partners=top_partners)
    if variant == "identity":
        chains = chains_identity(V)
    elif variant == "cooc":
        chains = chains_cooc(partners, V)
    elif variant == "svd":
        chains = chains_svd(TD, V, dim=svd_dim)
    else:
        raise ValueError(variant)

    term_raw, term_hist = term_features(chains)
    doc_ids, DV = doc_vectors(doc_toks, vmap, idf, term_raw, term_hist)
    docpos = {d: i for i, d in enumerate(doc_ids)}
    build_s = time.time() - t0

    bm = BM25(corpus)

    nd_lat = nd_bm = nd_fuse = 0.0
    for qid in test_ids:
        q = queries[qid]
        qv = query_vector(q, vmap, idf, term_raw, term_hist)
        sims = DV @ qv
        lat_rank = [doc_ids[i] for i in np.argsort(-sims)[:100]]
        bm_rank = [d for d, _ in bm.search(q, 100)]
        nd_lat += ndcg10(lat_rank, test_q[qid])
        nd_bm += ndcg10(bm_rank, test_q[qid])
        if fuse:
            fused = rrf([bm_rank, lat_rank])[:10]
            nd_fuse += ndcg10(fused, test_q[qid])
    n = len(test_ids)
    return {
        "lat": nd_lat / n,
        "bm": nd_bm / n,
        "fuse": (nd_fuse / n) if fuse else None,
        "V": V,
        "build_s": build_s,
        "nq": n,
    }


def main():
    print("=" * 74)
    print("LENS: lattice-coord-features  — AETHOS 3D-plane geometry AS embedding")
    print("=" * 74)
    summary = {}
    for ds in ("scifact", "nfcorpus"):
        print(f"\n##### {ds} #####")
        for variant in ("identity", "cooc", "svd"):
            r = eval_lens(ds, variant)
            tag = f"{ds}/{variant}"
            summary[tag] = r
            fuse_s = f"{r['fuse']:.4f}" if r["fuse"] is not None else "  -  "
            print(f"  [{variant:8s}] V={r['V']:5d} nq={r['nq']:3d} build={r['build_s']:4.1f}s | "
                  f"lattice nDCG {r['lat']:.4f} | BM25 {r['bm']:.4f} | RRF-fuse {fuse_s}")
    print("\n" + "=" * 74)
    print("REFERENCE: BM25 scifact 0.665 / nfcorpus 0.325 ; lexical-lattice 0.7023/0.3204 ;")
    print("           dense/SPLADE ~0.70 / ~0.34")
    print("=" * 74)
    return summary


if __name__ == "__main__":
    main()
