#!/usr/bin/env python3
"""Head-to-head: current per-pair bridges vs the champion's gold-doc triangulation
(cross-query union + doc-rare + discriminative df), on held-out SciFact.
Ports trng/aethos_master gold_doc_triangulation into this pipeline to measure if
it beats the current +3.5pp bridge."""
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10

ds = sys.argv[1] if len(sys.argv) > 1 else "scifact"
corpus, queries, train_q, test_q = load(ds)
test_ids = [q for q in test_q if q in queries]
idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)

_idf = {}
def idf(w):
    v = _idf.get(w)
    if v is None:
        p = idx.token_prime.get(("w", w))
        v = idx._idf(p, N) if p else 0.0
        _idf[w] = v
    return v

RARE = 3.0
def rare(text):
    return {w for w in set(words(text)) if idf(w) >= RARE}

# current per-pair bridges (Berger-Lafferty)
br = RelevanceBridges(idx, N, min_pairs=2).learn(queries, train_q, corpus)

# champion gold-doc triangulation, ported
gold_to_qids = defaultdict(set)
for qid, rels in train_q.items():
    for d, sc in rels.items():
        if sc > 0:
            gold_to_qids[d].add(qid)
cands, dfc = {}, Counter()
for d, qids in gold_to_qids.items():
    if d not in corpus:
        continue
    aqr = set()
    for qid in qids:
        if qid in queries:
            aqr |= rare(queries[qid])
    c = aqr | rare(corpus[d])
    cands[d] = (c, aqr)
    for w in c:
        dfc[w] += 1
ngold = len(gold_to_qids)
thr = max(ngold * 0.15, 3)
word_docs = defaultdict(list)       # BRIDGE:w  -> docs
ng_docs = defaultdict(list)         # BRIDGE_NG:w1+w2 -> docs
for d, (c, aqr) in cands.items():
    disc = sorted([w for w in c if dfc[w] <= thr], key=lambda w: -idf(w))[:25]
    for w in disc:
        word_docs[w].append(d)
    qr = sorted(aqr, key=lambda w: -idf(w))[:8]
    for a, b in combinations(sorted(qr), 2):
        ng_docs[(a, b)].append(d)

def gold_exp(q):
    exp = defaultdict(float)
    rw = rare(q)
    for w in rw:
        for d in word_docs.get(w, ()):
            exp[d] += idf(w)
    for a, b in combinations(sorted(rw, key=lambda w: -idf(w))[:8], 2):
        for d in ng_docs.get((a, b), ()):
            exp[d] += idf(a) + idf(b)
    return exp

def fused(q, expfn, lam=0.25, n_expand=20, k=10):
    lex = idx._score(q)
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    exp = expfn(q)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=exp.get, reverse=True) if d not in cset][:n_expand]
    pool = cand + extra
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=final.get, reverse=True)[:k]

def br_exp(q):
    exp = defaultdict(float)
    for qt in set(words(q)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            for d, tf in idx.postings.get(p, {}).items():
                exp[d] += w * tf / (tf + 1.0)
    return exp

def both_exp(q):
    exp = br_exp(q)
    for d, v in gold_exp(q).items():
        exp[d] += v
    return exp

def lex_only(q, k=10):
    s = idx._score(q)
    return sorted(s, key=s.get, reverse=True)[:k]

def ev(fn):
    nd = rc = 0.0
    for qid in test_ids:
        r = fn(queries[qid])
        nd += ndcg10(r, test_q[qid])
        rc += recall10(r, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n

gold_reuse = sum(1 for d in gold_to_qids if len(gold_to_qids[d]) > 1)
print(f"\n{ds}: {len(corpus)} docs | test {len(test_ids)} q | {ngold} train-gold docs "
      f"({gold_reuse} reused across >1 query)")
b = ev(lex_only)
print(f"  baseline lexical:          nDCG {b[0]:.4f}  Recall {b[1]:.4f}")
for name, fn in [("current per-pair bridges", lambda q: bridge_search(idx, br, q)),
                 ("gold-doc triangulation", lambda q: fused(q, gold_exp)),
                 ("both combined", lambda q: fused(q, both_exp))]:
    nd, rc = ev(fn)
    print(f"  {name:<26} nDCG {nd:.4f}  Recall {rc:.4f}  "
          f"({nd-b[0]:+.4f} / {rc-b[1]:+.4f})")
