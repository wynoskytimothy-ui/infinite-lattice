#!/usr/bin/env python3
"""Raw (UNSTEMMED) BM25 floor on the full 8.8M collection -- the missing baseline.

The pool ladder's +11.3% included stemming (+6.6% of it). This builds an unstemmed full
index (cached full_idxraw_*) and runs BM25-only on the SAME 3000 dev queries (seed 42, same
gates) as marco_full_eval.py, so we can state the COMPLETE ladder on the full collection:
raw BM25 -> stem -> + rerank rules, comparable to the pool's 0.5419 -> 0.5777 -> 0.6030.
"""
import re, time, math, pickle, random
from array import array
from collections import defaultdict
from pathlib import Path
import numpy as np

MARCO = Path(r"C:\Users\wynos\trng\marco_data")
TOK = re.compile(r"[a-z0-9]+")
K1, B = 0.9, 0.4
QGATE, SCORE_FLOOR, CAND_FLOOR = 0.3, 1.0, 2.0
PRE = "full_idxraw_"


def build_raw():
    if (MARCO / (PRE + "meta.pkl")).exists():
        return
    post_di, post_tf = {}, {}
    doclen = []; total = 0; t0 = time.perf_counter()
    print("building UNSTEMMED full index...", flush=True)
    with open(MARCO / "collection.tsv", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f):
            tab = line.find("\t")
            if tab < 0:
                doclen.append(0); continue
            tf = {}
            for w in TOK.findall(line[tab + 1:].lower()):
                tf[w] = tf.get(w, 0) + 1
            dl = 0
            for s, c in tf.items():
                pd = post_di.get(s)
                if pd is None:
                    pd = array('I'); post_di[s] = pd; post_tf[s] = array('H')
                post_di[s].append(ln); post_tf[s].append(c if c < 65536 else 65535); dl += c
            doclen.append(dl); total += dl
            if (ln + 1) % 2_000_000 == 0:
                print(f"  {ln+1:,} docs, {len(post_di):,} terms, {time.perf_counter()-t0:.0f}s", flush=True)
    N = len(doclen)
    vocab = list(post_di.keys()); V = len(vocab)
    df = np.fromiter((len(post_di[t]) for t in vocab), dtype=np.uint64, count=V)
    P = int(df.sum()); ptr = np.zeros(V + 1, dtype=np.uint64); np.cumsum(df, out=ptr[1:])
    di_all = np.empty(P, dtype=np.uint32); tf_all = np.empty(P, dtype=np.uint16)
    for tid, t in enumerate(vocab):
        s, e = int(ptr[tid]), int(ptr[tid + 1])
        di_all[s:e] = np.frombuffer(post_di[t], dtype=np.uint32)
        tf_all[s:e] = np.frombuffer(post_tf[t], dtype=np.uint16)
    idf = np.log((N - df.astype(np.float64) + 0.5) / (df.astype(np.float64) + 0.5) + 1.0).astype(np.float32)
    np.save(MARCO / (PRE + "di.npy"), di_all); np.save(MARCO / (PRE + "tf.npy"), tf_all)
    np.save(MARCO / (PRE + "ptr.npy"), ptr); np.save(MARCO / (PRE + "doclen.npy"), np.asarray(doclen, np.uint32))
    np.save(MARCO / (PRE + "idf.npy"), idf)
    with open(MARCO / (PRE + "meta.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab, "N": N, "avgdl": total / N}, f)
    print(f"  cached {PRE}* ({N:,} docs, {V:,} terms, {P:,} postings) in {time.perf_counter()-t0:.0f}s", flush=True)


def main():
    nq = 3000
    build_raw()
    di = np.load(MARCO / (PRE + "di.npy")); tf = np.load(MARCO / (PRE + "tf.npy"))
    ptr = np.load(MARCO / (PRE + "ptr.npy")); doclen = np.load(MARCO / (PRE + "doclen.npy")).astype(np.float32)
    idfa = np.load(MARCO / (PRE + "idf.npy"))
    with open(MARCO / (PRE + "meta.pkl"), "rb") as f:
        m = pickle.load(f)
    vocab, N, avgdl = m["vocab"], m["N"], m["avgdl"]
    tid = {t: i for i, t in enumerate(vocab)}
    acc = np.zeros(N, dtype=np.float32)
    print(f"  loaded raw index: {N:,} docs, {len(vocab):,} terms", flush=True)

    def idf_of(w):
        i = tid.get(w); return float(idfa[i]) if i is not None else 0.0

    def bm25_top(qterms, k=100):
        touched = []; cand_src = []; best = None
        for w in qterms:
            i = tid.get(w)
            if i is None:
                continue
            wi = idfa[i]
            if wi < SCORE_FLOOR:
                continue
            s, e = int(ptr[i]), int(ptr[i + 1])
            dis = di[s:e]; tfs = tf[s:e].astype(np.float32); dl = doclen[dis]
            acc[dis] += wi * tfs * (K1 + 1.0) / (tfs + K1 * (1.0 - B + B * dl / avgdl))
            touched.append(dis)
            if wi >= CAND_FLOOR:
                cand_src.append(dis)
            if best is None or wi > best[0]:
                best = (wi, dis)
        if not touched:
            return []
        if not cand_src:
            cand_src = [best[1]]
        cand = np.unique(np.concatenate(cand_src))
        sc = acc[cand]
        top = cand[np.argpartition(-sc, k)[:k]] if len(cand) > k else cand
        order = top[np.argsort(-acc[top])]
        res = [int(d) for d in order]
        acc[np.concatenate(touched)] = 0.0
        return res

    qrels = defaultdict(set)
    with open(MARCO / "qrels.dev.tsv", encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(MARCO / "queries.dev.tsv", encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    mrr = r100 = 0.0; t0 = time.perf_counter()
    for n, q in enumerate(qids):
        rel = qrels[q]
        qs = [w for w in set(TOK.findall(queries[q].lower())) if idf_of(w) >= QGATE]
        if not qs:
            continue
        ranked = bm25_top(qs, 100)
        mrr += next((1.0 / i for i, d in enumerate(ranked[:10], 1) if str(d) in rel), 0.0)
        r100 += len(rel & set(str(d) for d in ranked[:100])) / len(rel)
        if (n + 1) % 500 == 0:
            print(f"    {n+1}/{nq} | raw BM25 {mrr/(n+1):.4f} | {time.perf_counter()-t0:.0f}s", flush=True)
    print(f"\nRAW (unstemmed) BM25 on full 8.8M -- {nq} dev queries")
    print(f"   MRR@10 {mrr/nq:.4f}   R@100 {r100/nq:.4f}")
    print(f"\n   COMPLETE full-collection ladder vs the pool:")
    print(f"     raw BM25   stem      + ladder")
    print(f"     full 8.8M: {mrr/nq:.4f} -> 0.1845 -> 0.1950")
    print(f"     pool 298k: 0.5419 -> 0.5777 -> 0.6030")


if __name__ == "__main__":
    main()
