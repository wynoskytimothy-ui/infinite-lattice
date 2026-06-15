#!/usr/bin/env python3
"""Zero-shot knowledge injection on any BEIR corpus: define the rare gap terms
(external knowledge the corpus lacks), inject as full-weight query expansion,
RRF-fuse with lexical. No qrels.  python zeroshot_kb.py nfcorpus"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10

DS = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
GATE = {"scifact": 5.5, "nfcorpus": 4.5, "fiqa": 5.0}.get(DS, 5.0)
MODS = {"scifact": ["scifact_glossary", "scifact_glossary_full"],
        "nfcorpus": ["nfcorpus_glossary"]}.get(DS, [])
GLOSSARY = {}
for m in MODS:
    try:
        GLOSSARY.update(__import__(m).GLOSSARY)
    except Exception:
        pass

corpus, queries, train_q, test_q = load(DS)
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

def knowledge_terms(q):
    extra = []
    for w in set(words(q)):
        if w in GLOSSARY and idf(w) >= GATE:
            for dw in dict.fromkeys(words(GLOSSARY[w])):
                if dw != w and idf(dw) >= 2.0:
                    extra.append(dw)
    return extra[:14]

def lex(text, k=100):
    s = idx._score(text)
    return sorted(s, key=s.get, reverse=True)[:k]

def rrf(weighted, k=60):
    sc = defaultdict(float)
    for r, w in weighted:
        for i, d in enumerate(r):
            sc[d] += w / (k + i + 1)
    return sorted(sc, key=sc.get, reverse=True)

def baseline(q):
    return lex(q, 10)

def knowledge(q, w=0.5):
    kt = knowledge_terms(q)
    if not kt:
        return lex(q, 10)
    return rrf([(lex(q, 100), 1.0), (lex(" ".join(kt), 100), w)])[:10]

def ev(fn):
    nd = rc = 0.0
    for qid in test_ids:
        r = fn(queries[qid])
        nd += ndcg10(r, test_q[qid])
        rc += recall10(r, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n

covered = sum(1 for qid in test_ids if knowledge_terms(queries[qid]))
b = ev(baseline)
k = ev(knowledge)
print(f"ZERO-SHOT knowledge injection on {DS} (no qrels) -- glossary {len(GLOSSARY)} "
      f"terms, fires on {covered}/{len(test_ids)} queries\n")
print(f"  baseline lexical:          nDCG {b[0]:.4f}  Recall {b[1]:.4f}")
print(f"  + knowledge (definitions): nDCG {k[0]:.4f}  Recall {k[1]:.4f}  "
      f"({k[0]-b[0]:+.4f} nDCG, {k[1]-b[1]:+.4f} recall)")
