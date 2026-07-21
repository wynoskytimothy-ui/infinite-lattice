#!/usr/bin/env python3
"""FULL LADDER on any corpus — for the four that have never had anything but BM25 run on them.

Every other corpus got the full ladder (lexical -> bridged -> distilled -> apex -> dense -> PQ). arguana got BM25
alone, and even that was crippled by the MAXQT=10 truncation bug (queries average 87 terms; uncapped BM25 goes
0.1359 -> 0.2844 nDCG, 0.4467 -> 0.9133 R@100).

**R@100 is already 0.9133 with no cap.** The right documents are in the pool. Nothing has ever reranked them.
That is precisely where the apex teacher gave its largest lift elsewhere (+0.15 on fiqa), so it is the biggest
untested opportunity we have.

ARMS (all on uncapped queries, so the truncation bug cannot reappear):
  BM25 uncapped                 the corrected floor
  dense bge-large full scan     the ceiling reference
  apex: BM25 pool -> dense rerank   the cheap production shape (no-GPU retrieval, teacher on ~200 docs)
  PQ apex                       same, with the teacher's vectors quantised to ~150 B/doc
Also reports R@100 for each, since a reranker cannot exceed the pool it is given.
usage: python _arguana_stack.py
"""
import os, sys, json, math, collections, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from beir_data_root import resolve_beir_root

ROOT = resolve_beir_root(); HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1] if len(sys.argv) > 1 else "arguana"
POOL = 200; NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MAXDOC = int(sys.argv[3]) if len(sys.argv) > 3 else 400000
STOP = set("a an the of and or in on for to with is are was were be been by as at from that this it we our you your "
           "what how why when where which who i me my do does did can could should would will if there their they "
           "them then than so such about into over under between after before have has had not no yes any all".split())
def stem(w, k=6): return w[:k]
def tok(s):
    return [stem(w) for w in ''.join(c.lower() if c.isalnum() else ' ' for c in s).split()
            if len(w) > 2 and w not in STOP]
def nd10(rk, rel):
    dcg = sum(rel.get(x, 0) / math.log2(i + 2) for i, x in enumerate(rk[:10]))
    idc = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(rel.values(), reverse=True)[:10]))
    return dcg / idc if idc else 0.0
def rec(rk, rel, k=100):
    g = {x for x, v in rel.items() if v > 0}
    return len(set(rk[:k]) & g) / len(g) if g else 0.0

d = os.path.join(ROOT, CORPUS); corpus = {}; queries = {}
for line in open(os.path.join(d, "corpus.jsonl"), encoding="utf-8"):
    o = json.loads(line); corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()
    if len(corpus) >= MAXDOC: break
for line in open(os.path.join(d, "queries.jsonl"), encoding="utf-8"):
    o = json.loads(line); queries[o["_id"]] = o["text"]
qr = collections.defaultdict(dict)
r = csv.reader(open(os.path.join(d, "qrels", "test.tsv"), encoding="utf-8"), delimiter="\t"); next(r, None)
for row in r:
    if len(row) >= 3 and int(row[2]) > 0: qr[row[0]][row[1]] = int(row[2])
doc_ids = list(corpus); N = len(doc_ids); d2i = {x: i for i, x in enumerate(doc_ids)}
toks = {x: set(tok(t)) for x, t in corpus.items()}
post = collections.defaultdict(set)
for x, ts in toks.items():
    for t in ts: post[t].add(d2i[x])
idf = {t: math.log(N / len(p)) for t, p in post.items()}
tfd = {x: collections.Counter(tok(corpus[x])) for x in doc_ids}
dl = {x: sum(tfd[x].values()) for x in doc_ids}; avgdl = float(np.mean(list(dl.values())))
ids = [q for q in qr if q in queries and any(qr[q].get(x, 0) > 0 for x in qr[q])]
import random as _r; _r.Random(0).shuffle(ids); ids = ids[:NQ]
gpq = np.mean([len(qr[q]) for q in ids]) if ids else 0
print("=" * 96); print(f"FULL LADDER | {CORPUS} | {N:,} docs | {len(ids)} queries | UNCAPPED queries"); print(f"  gold/query {gpq:.1f}" + ("  <-- R@100 is arithmetically capped at %.1f%%; read nDCG only" % (100*min(1,100/gpq)) if gpq > 20 else ""))
print("=" * 96, flush=True)

def bm25(qt):
    sc = collections.defaultdict(float)
    for t in qt:
        if t not in post: continue
        w = idf[t]
        for di in post[t]:
            x = doc_ids[di]; f = tfd[x].get(t, 0)
            sc[di] += w * (f * 1.9) / (f + 0.9 * (0.25 + 0.75 * dl[x] / avgdl))
    return sc

cache = os.path.join(HERE, f"_sr_bge-large_{CORPUS}.npz")
if os.path.exists(cache):
    z = np.load(cache, allow_pickle=True)
    E = z["doc_emb"].astype(np.float32); QE = z["q_emb"].astype(np.float32)
    dz = {str(x): i for i, x in enumerate(z["doc_ids"])}
    QV = {str(q): QE[i] for i, q in enumerate(z["qids"])}
    print("  loaded cached bge-large embeddings", flush=True)
else:
    from sentence_transformers import SentenceTransformer
    import torch
    m = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    E = m.encode([corpus[x] for x in doc_ids], batch_size=64, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
    qs = [queries[q] for q in ids]
    QEa = m.encode(qs, batch_size=64, normalize_embeddings=True, show_progress_bar=False,
                   convert_to_numpy=True).astype(np.float32)
    print(f"  encoded {N:,} docs + {len(ids)} queries in {time.time()-t0:.0f}s", flush=True)
    dz = {x: i for i, x in enumerate(doc_ids)}
    QV = {q: QEa[i] for i, q in enumerate(ids)}
    np.savez_compressed(cache, doc_emb=E, doc_ids=np.array(doc_ids, dtype=object),
                        q_emb=QEa, qids=np.array(ids, dtype=object))
    del m
    try:
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception: pass
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

def pq(E, M=128, nb=8):
    from sklearn.cluster import MiniBatchKMeans
    dim = E.shape[1]; ds = dim // M; K = 2 ** nb
    cb = np.zeros((M, K, ds), np.float32); codes = np.zeros((E.shape[0], M), np.int32)
    for m_ in range(M):
        km = MiniBatchKMeans(n_clusters=K, n_init=3, max_iter=60, batch_size=4096,
                             random_state=0).fit(E[:, m_*ds:(m_+1)*ds])
        cb[m_] = km.cluster_centers_.astype(np.float32); codes[:, m_] = km.labels_
    bpd = M*nb/8 + (K*dim*4)/E.shape[0]
    return cb, codes, ds, M, bpd
cb, codes, ds, M, bpd = pq(E)
print(f"  PQ M=128 nb=8 -> {bpd:.1f} B/doc (codes {M*8//8} + codebook {(256*1024*4)/N:.1f})", flush=True)

res = {}
def run(label, fn):
    nd = []; rc = []
    for q in ids:
        rk = fn(q)
        nd.append(nd10(rk, qr[q])); rc.append(rec(rk, qr[q]))
    a, b = float(np.mean(nd)), float(np.mean(rc))
    print(f"  {label:36s} {a:>9.4f} {b:>8.4f}", flush=True)
    res[label] = {"nDCG@10": round(a, 4), "R@100": round(b, 4)}
    return a

def qterms(q): return [t for t in dict.fromkeys(tok(queries[q])) if t in post]
def f_bm25(q):
    sc = bm25(qterms(q))
    return [doc_ids[i] for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:100]]
def f_dense(q):
    s = E @ QV[q]; top = np.argpartition(-s, 100)[:100]; top = top[np.argsort(-s[top])]
    return [doc_ids[i] for i in top]
def f_apex(q, quant=False):
    sc = bm25(qterms(q))
    pool = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:POOL]]
    if not pool: return []
    v = QV[q]
    if quant:
        lut = np.einsum('mkd,md->mk', cb, v.reshape(M, ds))
        s = np.array([lut[np.arange(M), codes[i]].sum() for i in pool])
    else:
        s = E[pool] @ v
    o = np.argsort(-s)
    return [doc_ids[pool[i]] for i in o[:100]]

print(f"\n  {'arm':36s} {'nDCG@10':>9} {'R@100':>8}")
run("BM25 uncapped (corrected floor)", f_bm25)
run("dense bge-large (full scan)", f_dense)
run(f"apex: BM25 pool{POOL} -> dense rerank", lambda q: f_apex(q, False))
run(f"apex: BM25 pool{POOL} -> PQ rerank", lambda q: f_apex(q, True))
b = res["BM25 uncapped (corrected floor)"]["nDCG@10"]
dn = res["dense bge-large (full scan)"]["nDCG@10"]
ap = res[f"apex: BM25 pool{POOL} -> dense rerank"]["nDCG@10"]
pq_ = res[f"apex: BM25 pool{POOL} -> PQ rerank"]["nDCG@10"]
print(f"\n  apex vs BM25 floor: {ap-b:+.4f}   apex vs dense: {ap-dn:+.4f}   PQ apex vs dense apex: {pq_-ap:+.4f}")
print(f"  PQ apex footprint: {bpd:.1f} B/doc for the teacher, on top of the lexical index")
json.dump({"corpus": CORPUS, "n_queries": len(ids), "pq_bytes_per_doc": round(bpd, 1), "arms": res},
          open(os.path.join(HERE, f"_stack_{CORPUS}.json"), "w"), indent=2)
print("\nARGUANA_STACK_JSON written")
