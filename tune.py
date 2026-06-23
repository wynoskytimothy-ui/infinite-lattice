#!/usr/bin/env python3
"""Systematic glass-box tuning across BEIR corpora: find GENERAL rules (demote goblins / wrong-reason docs,
merge/filter, coordinate) that lift the cross-corpus AVERAGE nDCG -- the generalization gate (NOT scifact-
overfit). Build each corpus ONCE (multi-view + corridors), then sweep many cheap pool-rescoring rule configs.
Rules act only on the BM25-top-100 pool (no memory/speed blowup): containment, diversity (anti-goblin),
hub-demote (wrong-reason), rare-coordination, and the BM25 length knob.
"""
import sys, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stem_safe import safe
from scripts.bench_supervised_bridges import load, ndcg10

WORD = re.compile(r"[a-z0-9]+")
GEAR_TRI, GEAR_PRE, TRI_DF = 0.30, 0.20, 0.5
LAM, N_EXP, MINP, TOP, IDF_Q, IDF_D, RARE = 0.25, 20, 2, 12, 1.5, 1.5, 4.0
NAMES = ["scifact", "nfcorpus", "fiqa", "arguana"]

_st = {}
def st(w):
    r = _st.get(w)
    if r is None:
        r = safe(w); _st[w] = r
    return r
def toks(s):
    return [st(w) for w in WORD.findall(s.lower())]
def dbag(text):
    bag = Counter()
    for w in toks(text):
        bag[("w", w)] += 1.0; s = "^" + w + "$"
        for i in range(len(s) - 2):
            bag[("3", s[i:i + 3])] += GEAR_TRI
        bag[("p", w[:4])] += GEAR_PRE
    return bag


def build(corpus):
    cids = list(corpus); M = len(cids)
    bags = [dbag(corpus[c]) for c in cids]
    df = Counter()
    for b in bags:
        for k in b:
            df[k] += 1
    vocab = list(df); tid = {k: i for i, k in enumerate(vocab)}
    idf = np.array([math.log(1 + (M - df[k] + 0.5) / (df[k] + 0.5)) for k in vocab], np.float32)
    for i, k in enumerate(vocab):
        if k[0] == "3" and df[k] > TRI_DF * M:
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
    doc_w = [set(tid[("w", w)] for w in set(toks(corpus[c]))) for c in cids]
    ndist = np.array([len(s) for s in doc_w], np.float32)
    wlen = np.array([len(toks(corpus[c])) for c in cids], np.float32)
    wv = [k[1] for k in vocab if k[0] == "w"]
    awl = float(np.mean([len(w) for w in wv])) if wv else 5.0   # alignment signal: avg word length
    return dict(cids=cids, M=M, tid=tid, wtid=wtid, idf=idf, ptr=ptr, di=di, tf=tf,
                doclen=doclen, avgdl=float(doclen.mean()), doc_w=doc_w, ndist=ndist, wlen=wlen, awl=awl)


def widf(eng, w):
    i = eng["wtid"].get(w); return float(eng["idf"][i]) if i is not None else 0.0
def corridors(eng, corpus, queries, train_q):
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
        n = npairs[qt]
        sc = [(dt, (c / n) * widf(eng, dt)) for dt, c in cnt.items() if c >= MINP and dt != qt and widf(eng, dt) > 0]
        sc.sort(key=lambda x: -x[1]); corr[qt] = sc[:TOP]
    return corr


def bm25(eng, qbag, B):
    K1 = 1.2; lex = np.zeros(eng["M"], np.float32)
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


def search(eng, corr, query, cfg, k=10):
    if cfg.get("router2"):                        # multi-signal router: query length + corpus alignment
        if len(toks(query)) > 40:
            cfg = {"div": 0.5, "lenp": 0.3}                      # long argument query -> length+diversity
        elif eng["awl"] < cfg.get("awl_thresh", 6.3):
            cfg = {"wordonly": True, "div": 0.25}               # short + everyday/aligned -> fragment-cap
        else:
            cfg = {"div": 0.25}                                  # short + scientific -> keep gears, diversity
    elif cfg.get("router"):                       # wave-3 router: query length only
        cfg = {"div": 0.5, "lenp": 0.3} if len(toks(query)) > 40 else {"div": 0.25}
    qbag = dbag(query)
    if cfg.get("wordonly"):                       # fragment-cap: drop trigram/prefix gears at query time
        qbag = {kk: vv for kk, vv in qbag.items() if kk[0] == "w"}
    lex = bm25(eng, qbag, cfg.get("b", 0.75)); M = eng["M"]
    exp = np.zeros(M, np.float32)
    for qt in set(toks(query)):
        for (dt, wt) in corr.get(qt, []):
            t = eng["tid"].get(("w", dt))
            if t is None:
                continue
            s, e = int(eng["ptr"][t]), int(eng["ptr"][t + 1]); dis = eng["di"][s:e]; tfs = eng["tf"][s:e]
            exp[dis] += wt * tfs / (tfs + 1.0)
    cand = np.argsort(-lex)[:100]; expc = np.argsort(-exp)[:N_EXP]
    pool = np.unique(np.concatenate([cand, expc]))
    lmax = max(float(lex[cand].max()), 1e-9); emax = max(float(exp.max()), 1e-9)
    score = lex[pool] / lmax + LAM * exp[pool] / emax
    qw = set(t for t in (eng["tid"].get(("w", w)) for w in set(toks(query))) if t is not None)
    if cfg and any(cfg.get(x) for x in ("contain", "div", "hub", "rare", "lenp")):
        qidf = {t: float(eng["idf"][t]) for t in qw}
        for i, d in enumerate(pool):
            matched = qw & eng["doc_w"][d]
            if cfg.get("lenp"):                   # length de-pile-up: demote long docs
                score[i] *= max(eng["wlen"][d], 1.0) ** (-cfg["lenp"])
            if cfg.get("contain"):
                score[i] *= 1 + cfg["contain"] * len(matched) / max(len(qw), 1)
            if cfg.get("div"):
                score[i] *= (eng["ndist"][d] / max(eng["wlen"][d], 1.0)) ** cfg["div"]
            if cfg.get("hub") or cfg.get("rare"):
                rare = sum(1 for t in matched if qidf[t] >= RARE); hub = len(matched) - rare
                if cfg.get("hub") and hub > rare:
                    score[i] *= cfg["hub"]
                if cfg.get("rare") and rare > 0:
                    rs = sum(qidf[t] for t in matched if qidf[t] >= RARE)
                    score[i] += cfg["rare"] * rs * (rare ** cfg.get("rareg", 1.5)) / lmax
    return [eng["cids"][r] for r in pool[np.argsort(-score)][:k]]


def evalc(eng, corr, queries, test_q, ids, cfg):
    return sum(ndcg10(search(eng, corr, queries[q], cfg, 10), test_q[q]) for q in ids) / len(ids)


def main():
    engs = {}
    for nm in NAMES:
        corpus, queries, train_q, test_q = load(nm)
        eng = build(corpus); corr = corridors(eng, corpus, queries, train_q) if train_q else {}
        ids = [q for q in test_q if q in queries]
        engs[nm] = (eng, corr, queries, test_q, ids)
        print(f"  built {nm}: {eng['M']:,} docs, {len(ids)} test q", flush=True)
    configs = {                                   # WAVE 3: the per-query ROUTER vs single rules vs oracle
        "baseline": {},
        "div0.25 (best single)": {"div": 0.25},
        "len0.3": {"lenp": 0.3},
        "div0.5+len0.3": {"div": 0.5, "lenp": 0.3},
        "ROUTER (qlen>40)": {"router": True},
    }
    rows = []
    for cname, cfg in configs.items():
        per = {nm: evalc(*engs[nm], cfg) for nm in NAMES}
        rows.append((cname, sum(per.values()) / len(NAMES), per))
    base = next(a for n, a, p in rows if n == "baseline")
    base_per = next(p for n, a, p in rows if n == "baseline")
    oracle_per = {nm: max(p[nm] for _, _, p in rows) for nm in NAMES}   # router ceiling: each corpus its best rule
    oracle_avg = sum(oracle_per.values()) / len(NAMES)
    rows.sort(key=lambda x: -x[1])
    print(f"\n  RULE SWEEP WAVE 2 -- avg nDCG@10 across {NAMES} (baseline {base:.4f})\n")
    print(f"  {'config':<28}{'avg':>8}{'delta':>9}  " + "".join(f"{n[:8]:>9}" for n in NAMES))
    for cname, avg, per in rows:
        gen = avg > base and all(per[n] >= base_per[n] - 0.003 for n in NAMES)
        print(f"  {cname:<28}{avg:>8.4f}{avg-base:>+9.4f}  " + "".join(f"{per[n]:>9.4f}" for n in NAMES) +
              ("  <- generalizes" if gen and cname != "baseline" else ""))
    print(f"\n  {'ORACLE ROUTER (per-corpus best)':<28}{oracle_avg:>8.4f}{oracle_avg-base:>+9.4f}  " +
          "".join(f"{oracle_per[n]:>9.4f}" for n in NAMES) + "  = ceiling if we pick the right rule per corpus")


if __name__ == "__main__":
    main()
