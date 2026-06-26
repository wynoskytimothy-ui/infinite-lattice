"""STAGE 3 push: how far does 'a prime + a rule' really compress, and what is
the fidelity cliff?

Tests three regeneration regimes against the STORED explicit matrix:
  A) qrels-evidence rule  (faithful regen, baseline from _stage3_regen_vs_store)
  B) qrels-evidence rule, evidence stored as COMPACT integer codes (how small
     can the irreducible supervision get?)
  C) UNSUPERVISED lattice-meet regen: NO qrels at all. Regenerate correlations
     from the pure corpus co-occurrence (lattice meet of doc-chains) -- the
     truest "store the corpus in primes + one rule, no labels" reading. Measure
     the fidelity cost of dropping supervision.
"""
from __future__ import annotations
import sys, time, math, pickle, zlib
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_supervised_bridges import load, ndcg10
from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search


def recall_at(ranked, rels, k):
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


NAME = "scifact"
corpus, queries, train_q, test_q = load(NAME)
idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)
test_ids = [q for q in test_q if q in queries]

def idf(w):
    p = idx.token_prime.get(("w", w))
    return idx._idf(p, N) if p else 0.0

br_stored = RelevanceBridges(idx, N).learn(queries, train_q, corpus)
explicit_matrix = br_stored.bridge
IDF_GATE, MIN_PAIRS, TOP = br_stored.idf_gate, br_stored.min_pairs, br_stored.top_per_term
prime_to_word = {p: tok[1] for tok, p in idx.token_prime.items() if tok[0] == "w"}

def doc_gated_words(d):
    return [prime_to_word[p] for p in idx.doc_words.get(d, ())
            if p in prime_to_word and idf(prime_to_word[p]) >= IDF_GATE]

class _Shim:
    def __init__(self, b): self.bridge=b; self.corridor_bridge={}

def evalsys(bridge, lam=0.25):
    shim=_Shim(bridge); R10=N10=R100=0.0; n=0
    for qid in test_ids:
        rels=test_q[qid]
        if not any(s>0 for s in rels.values()): continue
        r=bridge_search(idx,shim,queries[qid],lam=lam,k=100)
        R10+=recall_at(r,rels,10); R100+=recall_at(r,rels,100); N10+=ndcg10(r[:10],rels); n+=1
    return R10/n, R100/n, N10/n

def blob(o): return len(zlib.compress(pickle.dumps(o),9))

# ---- C) UNSUPERVISED: regenerate correlations from pure corpus co-occurrence ----
# the lattice meet: for each gated word qt, its corpus partners are the gated
# words that co-occur with it in the SAME docs, weighted by P(dt|qt)*idf(dt).
# NO qrels. This is "the corpus in primes + one co-occurrence rule".
t0=time.time()
gated_doc_words = {d: set(doc_gated_words(d)) for d in idx.alive}
# build co-occurrence by walking docs (the meet of doc-chains)
cooc=defaultdict(Counter); termdf=Counter()
for d, ws in gated_doc_words.items():
    for w in ws: termdf[w]+=1
    for w in ws:
        cooc[w].update(ws)
unsup={}
for qt, partners in cooc.items():
    npq=termdf[qt]
    scored=[(dt,(c/npq)*idf(dt)) for dt,c in partners.items() if dt!=qt and c>=MIN_PAIRS]
    scored.sort(key=lambda x:(-x[1],x[0]))
    if scored: unsup[qt]=scored[:TOP]
unsup_time=time.time()-t0

# restrict unsup to the SAME query-terms the stored matrix learned (fair: same coverage)
unsup_restricted={qt:unsup[qt] for qt in explicit_matrix if qt in unsup}

print(f"{NAME}: test {len(test_ids)} q")
print("\n--- FIDELITY ---")
for name, m, lam in [("lex-only (lam=0)", explicit_matrix, 0.0),
                     ("STORED matrix    ", explicit_matrix, 0.25),
                     ("UNSUP meet (all terms)", unsup, 0.25),
                     ("UNSUP meet (stored-term subset)", unsup_restricted, 0.25)]:
    r10,r100,nd=evalsys(m,lam)
    print(f"{name:34s}: R@10={r10:.4f} R@100={r100:.4f} nDCG@10={nd:.4f}")

print(f"\nunsup regen time {unsup_time*1000:.0f} ms, "
      f"{len(unsup)} terms, {sum(len(v) for v in unsup.values())} entries")

# ---- B) compact integer-coded supervision evidence ----
# map qt-words and gold doc-ids to dense ints; store as arrays -> max compressibility
word_id={}; doc_id={}
ev_compact=[]
for qid,rels in train_q.items():
    if qid not in queries: continue
    qts=[w for w in set(words(queries[qid])) if idf(w)>=IDF_GATE]
    golds=[c for c,sc in rels.items() if sc>0 and c in corpus]
    if not qts or not golds: continue
    qi=tuple(word_id.setdefault(w,len(word_id)) for w in qts)
    gi=tuple(doc_id.setdefault(c,len(doc_id)) for c in golds)
    ev_compact.append((qi,gi))
import numpy as np
# flatten to varint-friendly arrays
flat=[]
for qi,gi in ev_compact:
    flat.append(len(qi)); flat.extend(qi); flat.append(len(gi)); flat.extend(gi)
arr=np.array(flat,dtype=np.uint16)
compact_blob=zlib.compress(arr.tobytes(),9)+zlib.compress(("\n".join(word_id).encode()),9)
print("\n--- FOOTPRINT (zlib-9) ---")
print(f"STORED explicit matrix          : {blob(explicit_matrix):>8,} B")
rule_ev=[(tuple(sorted(w for w in set(words(queries[qid])) if idf(w)>=IDF_GATE)),
          tuple(sorted(c for c,sc in rels.items() if sc>0 and c in corpus)))
         for qid,rels in train_q.items() if qid in queries]
print(f"rule (raw qrels evidence)       : {blob((IDF_GATE,MIN_PAIRS,TOP,rule_ev)):>8,} B")
print(f"rule (int-coded compact evidence): {len(compact_blob):>8,} B "
      f"(word-table {len(word_id)} terms incl.)")
print(f"UNSUP rule: NO evidence at all  :        ~0 B (just 3 float params)")
