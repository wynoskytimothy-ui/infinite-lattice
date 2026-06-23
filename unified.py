#!/usr/bin/env python3
"""UNIFIED engine: the MARCO slim substrate (numpy CSR + O(1) presence meet + slim-compressible) carrying
the BEIR-path accuracy (supervised corridors deployed as POOL EXPANSION, not just rerank). One engine,
run on BEIR corpora. Proves: BEIR accuracy (nDCG@10) at MARCO speed + small footprint, no neural.

  build_csr(corpus)        -> di/tf/ptr/idf numpy CSR (same shape as marco_full_eval.FullIndex)
  train_corridors()        -> {qt: [(dt, w=P(dt|qt)*idf(dt))]} from train qrels (== RelevanceBridges.learn)
  search(lexical | +corridor pool-expansion+fuse)   -> nDCG@10 / Recall@10 / MRR@10 on test
"""
import sys, time, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stem_safe import safe
from scripts.bench_supervised_bridges import load, ndcg10, recall10

WORD = re.compile(r"[a-z0-9]+")
K1, B = 1.2, 0.75          # BEIR-standard BM25
LAM, N_EXPAND, MINP, TOP = 0.25, 20, 2, 12
IDF_Q, IDF_D = 1.5, 1.5

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]


def build_csr(corpus):
    cids = list(corpus); id2row = {c: i for i, c in enumerate(cids)}; M = len(cids)
    dtok = [Counter(toks(corpus[c])) for c in cids]
    df = Counter()
    for ct in dtok:
        for w in ct:
            df[w] += 1
    vocab = list(df); tid = {w: i for i, w in enumerate(vocab)}
    idf = np.array([math.log(1 + (M - df[w] + 0.5) / (df[w] + 0.5)) for w in vocab], np.float32)
    post = defaultdict(list); postf = defaultdict(list)
    for r, ct in enumerate(dtok):
        for w, f in ct.items():
            post[tid[w]].append(r); postf[tid[w]].append(f)
    ptr = np.zeros(len(vocab) + 1, np.int64); di = []; tf = []
    for t in range(len(vocab)):
        ds = post[t]
        ptr[t + 1] = ptr[t] + len(ds); di.extend(ds); tf.extend(postf[t])
    di = np.array(di, np.int32); tf = np.array(tf, np.uint16)
    doclen = np.array([sum(ct.values()) for ct in dtok], np.float32)
    return dict(cids=cids, id2row=id2row, M=M, tid=tid, idf=idf, ptr=ptr, di=di, tf=tf,
                doclen=doclen, avgdl=float(doclen.mean()))


def train_corridors(eng, corpus, queries, train_q):
    tid, idf = eng["tid"], eng["idf"]
    cooc = defaultdict(Counter); npairs = Counter()
    for qid, rels in train_q.items():
        qts = [w for w in set(toks(queries.get(qid, ""))) if w in tid and idf[tid[w]] >= IDF_Q]
        if not qts:
            continue
        for cid, rel in rels.items():
            if rel <= 0 or cid not in corpus:
                continue
            dts = set(w for w in toks(corpus[cid]) if w in tid and idf[tid[w]] >= IDF_D)
            for qt in qts:
                npairs[qt] += 1
                cooc[qt].update(dts)
    corr = {}
    for qt, cnt in cooc.items():
        np_ = npairs[qt]
        sc = [(dt, (c / np_) * float(idf[tid[dt]])) for dt, c in cnt.items() if c >= MINP and dt != qt]
        sc.sort(key=lambda x: -x[1]); corr[qt] = sc[:TOP]
    return corr


def bm25(eng, qterms):
    M = eng["M"]; lex = np.zeros(M, np.float32)
    for w in set(qterms):
        t = eng["tid"].get(w)
        if t is None:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1]); dis = eng["di"][s:e]; tfs = eng["tf"][s:e].astype(np.float32)
        dl = eng["doclen"][dis]
        lex[dis] += eng["idf"][t] * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * dl / eng["avgdl"]))
    return lex


def search(eng, corr, query, use_corr, k=10):
    qt = toks(query); lex = bm25(eng, qt)
    if not use_corr:
        return [eng["cids"][r] for r in np.argsort(-lex)[:k]]
    M = eng["M"]; exp = np.zeros(M, np.float32)
    for w in set(qt):
        for (dt, wt) in corr.get(w, []):
            t = eng["tid"].get(dt)
            if t is None:
                continue
            s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1]); dis = eng["di"][s:e]; tfs = eng["tf"][s:e].astype(np.float32)
            exp[dis] += wt * tfs / (tfs + 1.0)
    cand = np.argsort(-lex)[:100]; expc = np.argsort(-exp)[:N_EXPAND]
    pool = np.unique(np.concatenate([cand, expc]))
    lmax = max(float(lex[cand].max()), 1e-9); emax = max(float(exp.max()), 1e-9)
    final = lex[pool] / lmax + LAM * exp[pool] / emax
    return [eng["cids"][r] for r in pool[np.argsort(-final)][:k]]


def run(name):
    corpus, queries, train_q, test_q = load(name)
    t0 = time.perf_counter(); eng = build_csr(corpus); bt = time.perf_counter() - t0
    fp = eng["di"].nbytes + eng["tf"].nbytes + eng["ptr"].nbytes + eng["idf"].nbytes
    corr = train_corridors(eng, corpus, queries, train_q) if train_q else {}
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*60}\n{name}: {eng['M']:,} docs | train {len(train_q):,} q | test {len(test_ids):,} q | "
          f"build {bt:.1f}s ({eng['M']/bt:,.0f} d/s) | CSR {fp/1e6:.1f} MB")

    def ev(use_corr):
        nd = rc = mr = 0.0; ts = []
        for q in test_ids:
            t0 = time.perf_counter(); r = search(eng, corr, queries[q], use_corr, 10); ts.append((time.perf_counter() - t0) * 1000)
            nd += ndcg10(r, test_q[q]); rc += recall10(r, test_q[q])
            for i, d in enumerate(r):
                if test_q[q].get(d, 0) > 0:
                    mr += 1.0 / (i + 1); break
        n = len(test_ids)
        return nd / n, rc / n, mr / n, float(np.median(ts))

    nd0, rc0, mr0, ms0 = ev(False)
    print(f"  lexical (BM25, word-only):  nDCG@10 {nd0:.4f}  Recall@10 {rc0:.4f}  MRR@10 {mr0:.4f}  {ms0:.2f} ms/q")
    if corr:
        nd1, rc1, mr1, ms1 = ev(True)
        print(f"  + corridor pool-expansion:  nDCG@10 {nd1:.4f}  Recall@10 {rc1:.4f}  MRR@10 {mr1:.4f}  {ms1:.2f} ms/q  ({nd1-nd0:+.4f})")


def main():
    for name in (sys.argv[1:] or ["scifact"]):
        run(name)


if __name__ == "__main__":
    main()
