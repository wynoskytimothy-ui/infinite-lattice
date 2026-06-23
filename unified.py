#!/usr/bin/env python3
"""UNIFIED engine: MARCO slim numpy-CSR substrate + BEIR-path accuracy (supervised corridors as POOL
EXPANSION) + MULTI-VIEW tokens (word + char-trigram + 4-char prefix gears, BEIR-style). One engine,
fast + small + accurate across BEIR/MARCO, no neural. Reports word-only vs multi-view side by side.

  build_csr(corpus, mv)    -> CSR over (view,token) terms; tf = gear-weighted bag value (w 1.0, tri .30, pre .20)
  train_corridors()        -> {qt: [(dt, P(dt|qt)*idf)]} on WORD view (== RelevanceBridges.learn)
  search(lex multi-view | + corridor pool-expansion+fuse)
"""
import sys, time, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stem_safe import safe
from scripts.bench_supervised_bridges import load, ndcg10, recall10

WORD = re.compile(r"[a-z0-9]+")
K1, B = 1.2, 0.75
LAM, N_EXPAND, MINP, TOP = 0.25, 20, 2, 12
IDF_Q, IDF_D = 1.5, 1.5
GEAR_TRI, GEAR_PRE, TRI_DF_FRAC = 0.30, 0.20, 0.5

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]


def doc_bag(text, mv):
    bag = Counter()
    for w in toks(text):
        bag[("w", w)] += 1.0
        if mv:
            s = "^" + w + "$"
            for i in range(len(s) - 2):
                bag[("3", s[i:i + 3])] += GEAR_TRI
            bag[("p", w[:4])] += GEAR_PRE
    return bag


def build_csr(corpus, mv):
    cids = list(corpus); id2row = {c: i for i, c in enumerate(cids)}; M = len(cids)
    bags = [doc_bag(corpus[c], mv) for c in cids]
    df = Counter()
    for b in bags:
        for k in b:
            df[k] += 1
    vocab = list(df); tid = {k: i for i, k in enumerate(vocab)}
    idf = np.array([math.log(1 + (M - df[k] + 0.5) / (df[k] + 0.5)) for k in vocab], np.float32)
    for i, k in enumerate(vocab):                       # tri_df_frac: drop common trigrams
        if k[0] == "3" and df[k] > TRI_DF_FRAC * M:
            idf[i] = 0.0
    post = defaultdict(list); postf = defaultdict(list)
    for r, b in enumerate(bags):
        for k, wt in b.items():
            t = tid[k]; post[t].append(r); postf[t].append(wt)
    ptr = np.zeros(len(vocab) + 1, np.int64); di = []; tf = []
    for t in range(len(vocab)):
        ds = post[t]; ptr[t + 1] = ptr[t] + len(ds); di.extend(ds); tf.extend(postf[t])
    di = np.array(di, np.int32); tf = np.array(tf, np.float32)
    doclen = np.array([sum(b.values()) for b in bags], np.float32)
    wtid = {k[1]: i for i, k in enumerate(vocab) if k[0] == "w"}
    return dict(cids=cids, id2row=id2row, M=M, tid=tid, wtid=wtid, idf=idf, ptr=ptr, di=di, tf=tf,
                doclen=doclen, avgdl=float(doclen.mean()))


def widf(eng, w):
    i = eng["wtid"].get(w)
    return float(eng["idf"][i]) if i is not None else 0.0


def train_corridors(eng, corpus, queries, train_q):
    cooc = defaultdict(Counter); npairs = Counter()
    for qid, rels in train_q.items():
        qts = [w for w in set(toks(queries.get(qid, ""))) if widf(eng, w) >= IDF_Q]
        if not qts:
            continue
        for cid, rel in rels.items():
            if rel <= 0 or cid not in corpus:
                continue
            dts = set(w for w in toks(corpus[cid]) if widf(eng, w) >= IDF_D)
            for qt in qts:
                npairs[qt] += 1; cooc[qt].update(dts)
    corr = {}
    for qt, cnt in cooc.items():
        np_ = npairs[qt]
        sc = [(dt, (c / np_) * widf(eng, dt)) for dt, c in cnt.items() if c >= MINP and dt != qt and widf(eng, dt) > 0]
        sc.sort(key=lambda x: -x[1]); corr[qt] = sc[:TOP]
    return corr


def bm25(eng, qbag):
    lex = np.zeros(eng["M"], np.float32)
    for k, qwt in qbag.items():
        t = eng["tid"].get(k)
        if t is None:
            continue
        wi = eng["idf"][t]
        if wi <= 0:
            continue
        s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1]); dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
        dl = eng["doclen"][dis]
        lex[dis] += qwt * wi * tfs * (K1 + 1) / (tfs + K1 * (1 - B + B * dl / eng["avgdl"]))
    return lex


def search(eng, corr, query, use_corr, mv, k=10):
    lex = bm25(eng, doc_bag(query, mv))
    if not use_corr:
        return [eng["cids"][r] for r in np.argsort(-lex)[:k]]
    exp = np.zeros(eng["M"], np.float32)
    for qt in set(toks(query)):
        for (dt, wt) in corr.get(qt, []):
            t = eng["tid"].get(("w", dt))
            if t is None:
                continue
            s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1]); dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
            exp[dis] += wt * tfs / (tfs + 1.0)
    cand = np.argsort(-lex)[:100]; expc = np.argsort(-exp)[:N_EXPAND]
    pool = np.unique(np.concatenate([cand, expc]))
    lmax = max(float(lex[cand].max()), 1e-9); emax = max(float(exp.max()), 1e-9)
    final = lex[pool] / lmax + LAM * exp[pool] / emax
    return [eng["cids"][r] for r in pool[np.argsort(-final)][:k]]


def run(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*64}\n{name}: {len(corpus):,} docs | train {len(train_q):,} q | test {len(test_ids):,} q")
    for mv in (False, True):
        t0 = time.perf_counter(); eng = build_csr(corpus, mv); bt = time.perf_counter() - t0
        fp = eng["di"].nbytes + eng["tf"].nbytes + eng["ptr"].nbytes + eng["idf"].nbytes
        corr = train_corridors(eng, corpus, queries, train_q) if train_q else {}

        def ev(uc):
            nd = mr = 0.0; ts = []
            for q in test_ids:
                t1 = time.perf_counter(); r = search(eng, corr, queries[q], uc, mv, 10); ts.append((time.perf_counter() - t1) * 1000)
                nd += ndcg10(r, test_q[q])
                for i, d in enumerate(r):
                    if test_q[q].get(d, 0) > 0:
                        mr += 1.0 / (i + 1); break
            n = len(test_ids); return nd / n, mr / n, float(np.median(ts))

        tag = "multi-view (w+tri+pre)" if mv else "word-only"
        nd0, mr0, ms0 = ev(False)
        line = f"  [{tag:<22}] lexical nDCG@10 {nd0:.4f} MRR {mr0:.4f}"
        if corr:
            nd1, mr1, _ = ev(True); line += f" | +corridor {nd1:.4f} ({nd1-nd0:+.4f})"
        print(f"{line} | build {bt:.1f}s {len(corpus)/bt:,.0f} d/s | CSR {fp/1e6:.1f} MB | {ms0:.2f} ms/q")


def main():
    for name in (sys.argv[1:] or ["scifact"]):
        run(name)


if __name__ == "__main__":
    main()
