#!/usr/bin/env python3
"""The CONTINUOUS co-occurrence embedding (LSA) as a reranker -- the fair test of the user's
vector-geometry thesis. My earlier contextual reranker was BINARY (in-window or not); BERT's
edge is CONTINUOUS graded similarity. This builds a continuous semantic space from CO-OCCURRENCE
(the signal we proved is real semantics) -- NOT prime structure (proven surface, psi-encoder 0.10):
a term-document matrix over a corpus sample, truncated-SVD -> dense vectors where co-occurring
terms land close. Then rerank the same BM25 top-100 by GRADED cosine.

  does continuous (vs my binary 0.20) close the gap to the cross-encoder (0.41)?
Yardsticks (same BM25 pool, 500 dev q): BM25 0.189, binary-contextual 0.200, cross-encoder 0.407.
"""
import sys, time, random
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from marco_full_eval import FullIndex, stoks, QGATE

MARCO_COL = r"C:\Users\wynos\trng\marco_data\collection.tsv"
VOCAB = 50000      # discriminative vocabulary (cols)
NDOC = 500000      # corpus sample for co-occurrence
KDIM = 256         # embedding dimension


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    idx = FullIndex()

    # 1. vocabulary: terms with df in [50, 0.3N], top VOCAB by df
    dfs = (idx.ptr[1:] - idx.ptr[:-1]).astype(np.int64)
    keep = np.where((dfs >= 50) & (dfs <= int(0.3 * idx.N)))[0]
    order = keep[np.argsort(-dfs[keep])][:VOCAB]
    col = {idx.vocab[int(t)]: j for j, t in enumerate(order)}
    idfcol = np.array([idx.idfa[int(order[j])] for j in range(len(order))], dtype=np.float32)
    Vn = len(col)
    print(f"  vocab {Vn} terms; building term-doc over {NDOC:,} docs...", flush=True)

    # 2. doc x term tf-idf matrix over a corpus sample
    t0 = time.perf_counter()
    import array
    ri = array.array('i'); cj = array.array('i'); vv = array.array('f')
    with open(MARCO_COL, encoding="utf-8", errors="replace") as f:
        for di, line in enumerate(f):
            if di >= NDOC:
                break
            tab = line.find("\t")
            if tab < 0:
                continue
            tf = {}
            for w in stoks(line[tab + 1:]):
                j = col.get(w)
                if j is not None:
                    tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                ri.append(di); cj.append(j); vv.append(np.log1p(c) * idfcol[j])
            if (di + 1) % 200000 == 0:
                print(f"    {di+1:,} docs, {len(vv):,} nnz, {time.perf_counter()-t0:.0f}s", flush=True)
    M = csr_matrix((np.frombuffer(vv, dtype=np.float32),
                    (np.frombuffer(ri, dtype=np.int32), np.frombuffer(cj, dtype=np.int32))),
                   shape=(NDOC, Vn))
    print(f"  term-doc {M.shape}, {M.nnz:,} nnz ({time.perf_counter()-t0:.0f}s); SVD k={KDIM}...", flush=True)
    svd = TruncatedSVD(n_components=KDIM, random_state=42)
    svd.fit(M)
    print(f"  SVD done ({time.perf_counter()-t0:.0f}s, explained var {svd.explained_variance_ratio_.sum():.3f})", flush=True)

    def embed(tok_lists):
        a_ri = array.array('i'); a_cj = array.array('i'); a_vv = array.array('f')
        for r, toks in enumerate(tok_lists):
            tf = {}
            for t in toks:
                j = col.get(t)
                if j is not None:
                    tf[j] = tf.get(j, 0) + 1
            for j, c in tf.items():
                a_ri.append(r); a_cj.append(j); a_vv.append(np.log1p(c) * idfcol[j])
        Mq = csr_matrix((np.frombuffer(a_vv, dtype=np.float32),
                         (np.frombuffer(a_ri, dtype=np.int32), np.frombuffer(a_cj, dtype=np.int32))),
                        shape=(len(tok_lists), Vn))
        E = svd.transform(Mq).astype(np.float32)
        n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1.0
        return E / n

    qrels = defaultdict(set)
    with open(idx.cf.name.replace("collection.tsv", "qrels.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(idx.cf.name.replace("collection.tsv", "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    def rr(order_pids, rel):
        return next((1.0 / i for i, p in enumerate(order_pids[:10], 1) if str(p) in rel), 0.0)

    mrr = defaultdict(float); n_eval = 0; t1 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        n_eval += 1
        order, _ = idx.bm25_top(qs, k=100)
        pids = [int(d) for d in order]
        dtoks = [stoks(idx.text(p)) for p in pids]
        qE = embed([stoks(queries[q])])[0]
        dE = embed(dtoks)
        sims = dE @ qE
        mrr["bm25"] += rr(pids, rel)
        emb_order = [p for _, p in sorted(zip(sims, pids), key=lambda x: -x[0])]
        mrr["embed"] += rr(emb_order, rel)
        # RRF fusion of BM25 rank and embed rank
        bm_rank = {p: i for i, p in enumerate(pids)}
        em_rank = {p: i for i, p in enumerate(emb_order)}
        fused = sorted(pids, key=lambda p: -(1.0 / (60 + bm_rank[p]) + 1.0 / (60 + em_rank[p])))
        mrr["fused"] += rr(fused, rel)
        if (n + 1) % 100 == 0:
            print(f"    {n+1}/{nq} | bm25 {mrr['bm25']/n_eval:.3f} embed {mrr['embed']/n_eval:.3f} "
                  f"fused {mrr['fused']/n_eval:.3f} | {time.perf_counter()-t1:.0f}s", flush=True)

    N = n_eval
    print(f"\nCONTINUOUS CO-OCCURRENCE EMBEDDING (LSA) rerank -- full 8.8M, {N} dev q, k={KDIM}\n")
    print(f"   {'reranker':<24}{'MRR@10':>9}")
    print(f"   {'BM25 (first stage)':<24}{mrr['bm25']/N:>9.4f}")
    print(f"   {'binary contextual (ref)':<24}{0.2000:>9.4f}")
    print(f"   {'LSA embed (pure cosine)':<24}{mrr['embed']/N:>9.4f}")
    print(f"   {'LSA + BM25 (RRF)':<24}{mrr['fused']/N:>9.4f}")
    print(f"   {'cross-encoder (ref)':<24}{0.4065:>9.4f}")
    print(f"\n   continuous (graded) vs binary 0.20: does co-occurrence-as-vectors close the gap to 0.41?")


if __name__ == "__main__":
    main()
