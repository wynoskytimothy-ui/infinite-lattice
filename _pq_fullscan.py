#!/usr/bin/env python3
"""PQ FULL-SCAN — the STANDALONE index number that _pq_apex and _pq_sweet do not measure.

Both existing PQ scripts rerank a 200-doc pool produced by another tier ("apex ranks 0.3% of corpus"), so their
bytes ADD to that tier's bytes and their R@100 is capped by the pool. Their "dense_fp32" reference is itself
pool-capped (fiqa 0.3950, not the true full-scan 0.4432) -- so "ties dense" there means "ties pool-capped dense".

This measures PQ as the WHOLE index: ADC scan over every doc, no pool, no other tier. That is the number that can
be compared to a standalone lexical engine's bytes/doc, because nothing else is stored.

Honest byte accounting: codes (M*nb/8) + the codebook (2^nb * d * 4 bytes, FIXED regardless of N) / N.
Uses cached bge-large embeddings, so no GPU and no re-encode.
usage: python _pq_fullscan.py <corpus>
"""
import os, sys, json, math, collections, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beir_data_root import resolve_beir_root
from sklearn.cluster import MiniBatchKMeans

ROOT = resolve_beir_root(); CORPUS = sys.argv[1] if len(sys.argv) > 1 else "fiqa"
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = [(128, 8), (128, 6), (128, 5), (64, 8), (64, 6), (32, 8)]

def load_qrels(name):
    d = os.path.join(ROOT, name)
    qr = collections.defaultdict(dict)
    r = csv.reader(open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8"), delimiter="\t"); next(r, None)
    for row in r:
        if len(row) >= 3 and int(row[2]) > 0: qr[row[0]][row[1]] = int(row[2])
    return qr

def ndcg10(r, rels):
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(r[:10]))
    idc = sum(x / math.log2(i + 2) for i, x in enumerate(sorted(rels.values(), reverse=True)[:10]))
    return dcg / idc if idc else 0.0
def rec(r, rels, k):
    g = {d for d, s in rels.items() if s > 0}; return len(set(r[:k]) & g) / len(g) if g else 0.0

print("=" * 96); print(f"PQ FULL-SCAN (standalone index, no pool) | {CORPUS}"); print("=" * 96, flush=True)
qrels = load_qrels(CORPUS)
z = np.load(os.path.join(HERE, f"_sr_bge-large_{CORPUS}.npz"), allow_pickle=True)
doc_ids = [str(d) for d in z["doc_ids"]]
E = z["doc_emb"].astype(np.float32); E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
qids = [str(q) for q in z["qids"]]
Q = z["q_emb"].astype(np.float32); Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
N, dim = E.shape
keep = [i for i, q in enumerate(qids) if q in qrels and any(qrels[q].get(d, 0) > 0 for d in qrels[q])]
qids = [qids[i] for i in keep]; Q = Q[keep]
print(f"  {N:,} docs x {dim}-dim | {len(qids)} test queries | fp32 = {dim*4} B/doc", flush=True)

def evaluate(scores_fn, label, bpd):
    nd = []; rc = []
    for i, q in enumerate(qids):
        s = scores_fn(Q[i])
        top = np.argpartition(-s, 100)[:100]; top = top[np.argsort(-s[top])]
        r = [doc_ids[j] for j in top]
        nd.append(ndcg10(r, qrels[q])); rc.append(rec(r, qrels[q], 100))
    a, b = float(np.mean(nd)), float(np.mean(rc))
    print(f"  {label:24s} {bpd:>9.1f} {dim*4/bpd:>7.1f}x {a:>9.4f} {b:>8.4f}", flush=True)
    return {"bytes_per_doc": round(bpd, 1), "ratio": round(dim * 4 / bpd, 1), "nDCG@10": round(a, 4), "R@100": round(b, 4)}

print(f"\n  {'arm':24s} {'B/doc':>9} {'ratio':>8} {'nDCG@10':>9} {'R@100':>8}")
res = {}
res["dense_fp32"] = evaluate(lambda v: E @ v, "dense fp32 (FULL SCAN)", dim * 4)

for M, nb in CONFIGS:
    ds = dim // M; K = 2 ** nb
    if dim % M: continue
    cb = np.zeros((M, K, ds), np.float32); codes = np.zeros((N, M), np.int32)
    for m in range(M):
        sub = E[:, m * ds:(m + 1) * ds]
        km = MiniBatchKMeans(n_clusters=K, n_init=3, max_iter=60, batch_size=4096,
                             random_state=0, verbose=0).fit(sub)
        cb[m] = km.cluster_centers_.astype(np.float32); codes[:, m] = km.labels_
    code_b = M * nb / 8.0
    book_b = (K * dim * 4) / N                      # FIXED cost, amortized over N -- must be counted
    bpd = code_b + book_b
    def sf(v, cb=cb, codes=codes, M=M, ds=ds):
        lut = np.einsum('mkd,md->mk', cb, v.reshape(M, ds))    # ADC: per-subspace inner products
        return lut[np.arange(M), codes].sum(1)
    res[f"M{M} nb{nb}"] = evaluate(sf, f"PQ M={M} nb={nb} (scan)", bpd)
    res[f"M{M} nb{nb}"]["code_bytes"] = round(code_b, 1)
    res[f"M{M} nb{nb}"]["codebook_bytes_per_doc"] = round(book_b, 1)

d0 = res["dense_fp32"]
print(f"\n  dense full-scan bar: nDCG {d0['nDCG@10']:.4f}  R@100 {d0['R@100']:.4f}  @ {dim*4} B/doc")
for k, v in res.items():
    if k == "dense_fp32": continue
    print(f"  {k:12s} {v['bytes_per_doc']:>7.1f} B/doc  (codes {v['code_bytes']} + codebook {v['codebook_bytes_per_doc']})"
          f"  nDCG {v['nDCG@10']-d0['nDCG@10']:+.4f}  R@100 {v['R@100']-d0['R@100']:+.4f}")
out = {"corpus": CORPUS, "N": N, "dim": dim, "full_scan": True, "arms": res}
json.dump(out, open(os.path.join(HERE, f"_pq_fullscan_{CORPUS}.json"), "w"), indent=2)
print("RESULTS_JSON=" + json.dumps(out))
