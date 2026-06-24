#!/usr/bin/env python3
"""STAGE 2 AUDIT: correlation -> intersection -> 3-way semantics. Two-sided, real numbers.

(a) Corridor sanity: do high-weight learned corridor pairs actually co-occur in gold docs?
(b) Intersection: does the 2-way meet of two query terms reach the right doc pool?
(c) The 3-way: do co-relevant docs (>=2 gold for same query) share triple/corridor
    addresses MORE than random doc pairs? Quantify vs a random baseline.
"""
import math, random, statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_semantic_lattice import SemanticLattice, triple_cell
from scripts.bench_supervised_bridges import load

random.seed(0)
corpus, queries, qtrain, qtest = load("scifact")
print(f"scifact: {len(corpus)} docs, {len(queries)} queries, "
      f"train qrels {len(qtrain)} q, test qrels {len(qtest)} q")

idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)

# idf + word-prime -> docset (the exact posting lists)
def wprime(w): return idx.token_prime.get(("w", w))
def docs_of(w):
    p = wprime(w)
    return set(idx.postings[p].keys()) if p else set()
_idf = {}
def idf(w):
    v = _idf.get(w)
    if v is None:
        p = wprime(w); v = idx._idf(p, N) if p else 0.0; _idf[w] = v
    return v

# doc -> set of word-strings (rebuild from doc_words primes is costly; just retokenize)
doc_terms = {d: set(words(t)) for d, t in corpus.items()}

# ---------------------------------------------------------------------------
# Build the semantic lattice (learns corridors + triple cells)
# ---------------------------------------------------------------------------
sl = SemanticLattice(idx)
sl._N = N
# mine corridors over rare co-occurrence (mirror aethos_branch_meet)
RARE = 3.0
cooc = defaultdict(Counter)
for d, t in corpus.items():
    ws = [w for w in doc_terms[d] if idf(w) >= RARE]
    for a in ws:
        for b in ws:
            if a != b:
                cooc[a][b] += 1
# corridor weight = count(a,b) * P(b|a) * idf(b); top-K partners
corridor = {}
for a, partners in cooc.items():
    tot = sum(partners.values())
    scored = []
    for b, n in partners.items():
        if n < 2: continue
        w = n * (n / tot) * idf(b)
        scored.append((b, w, n))
    scored.sort(key=lambda x: -x[1])
    corridor[a] = scored[:6]

# =========================================================================
# (a) CORRIDOR SANITY: high-weight corridor pairs must co-occur in gold docs
# =========================================================================
# Build the set of gold doc ids per query (test split)
gold = {q: [d for d in ds if d in corpus] for q, ds in qtest.items()}
gold = {q: ds for q, ds in gold.items() if ds}

# Sanity: for top corridor pairs, the observed co-occurrence count IS the mining
# input, so instead verify the *direction*: high-weight partner b of a appears in
# docs containing a far more often than a random rare term does (lift over base rate).
def base_rate(b):  # fraction of docs containing b
    return len(docs_of(b)) / N
lifts = []
rare_terms = [w for w in corridor if idf(w) >= RARE and len(docs_of(w)) >= 5]
sample_a = random.sample(rare_terms, min(300, len(rare_terms)))
for a in sample_a:
    da = docs_of(a)
    if len(da) < 3: continue
    for b, w, n in corridor[a][:3]:
        p_b_given_a = sum(1 for d in da if b in doc_terms[d]) / len(da)
        br = base_rate(b)
        if br > 0:
            lifts.append(p_b_given_a / br)
print("\n(a) CORRIDOR SANITY (top-3 partners of 300 rare terms)")
print(f"    P(partner|term) / base_rate(partner)  median lift = {statistics.median(lifts):.1f}x, "
      f"mean = {statistics.mean(lifts):.1f}x, n={len(lifts)}")
print(f"    (lift >> 1 => corridor partners really are the company the term keeps)")

# =========================================================================
# (b) INTERSECTION: 2-way meet of two query terms reaches the right pool
# =========================================================================
# For each test query, take its 2 rarest terms; the AND-pool = docs(t1) ∩ docs(t2).
# Does this intersection contain gold docs at a high rate vs each term alone?
def rarest_terms(q_text, k=2):
    ws = sorted(set(words(q_text)), key=lambda w: -idf(w))
    return [w for w in ws if wprime(w)][:k]

inter_hits = []   # fraction of gold reached by 2-way AND
single_hits = []  # fraction of gold reached by rarest single term
pool_sizes = []
for q, gds in gold.items():
    if q not in queries: continue
    ts = rarest_terms(queries[q], 2)
    if len(ts) < 2: continue
    d1, d2 = docs_of(ts[0]), docs_of(ts[1])
    inter = d1 & d2
    union_single = d1  # rarest single
    g = set(gds)
    if not g: continue
    inter_hits.append(len(inter & g) / len(g))
    single_hits.append(len(union_single & g) / len(g))
    pool_sizes.append((len(inter), len(union_single)))
print("\n(b) 2-WAY INTERSECTION (rarest 2 query terms, test queries with 2 usable terms)")
print(f"    n={len(inter_hits)} queries")
print(f"    gold-recall in AND-pool   = {statistics.mean(inter_hits):.3f}  "
      f"(median pool size {statistics.median([p[0] for p in pool_sizes]):.0f} docs)")
print(f"    gold-recall in rarest term = {statistics.mean(single_hits):.3f}  "
      f"(median pool size {statistics.median([p[1] for p in pool_sizes]):.0f} docs)")
print(f"    => AND narrows the pool {statistics.median([p[1] for p in pool_sizes])/max(1,statistics.median([p[0] for p in pool_sizes])):.1f}x; "
      f"recall {'holds' if statistics.mean(inter_hits)>=statistics.mean(single_hits)-0.05 else 'drops'}")

# =========================================================================
# (c) THE 3-WAY: do co-relevant docs share addresses more than random pairs?
# =========================================================================
# For each query with >=2 gold docs, every co-relevant pair (di, dj) is "found
# together". Signal we test: shared RARE-term Jaccard, and whether they share a
# TRIPLE-MEET cell. A triple cell for a doc = for each rare term-pair in the doc
# plus a shared corridor anchor, the triple_cell address. Two docs co-locate in a
# triple node iff they share such a cell. Compare co-relevant pairs vs random pairs.

def rare_set(d):
    return frozenset(w for w in doc_terms[d] if idf(w) >= RARE)

def jaccard(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)

# triple-cell fingerprint of a doc: the set of triple_cell addresses formed by its
# top rare terms (anchored). Cap to keep it bounded.
anchor = {}
def anch(w):
    a = anchor.get(w)
    if a is None:
        a = 2 * len(anchor) + 3; anchor[w] = a
    return a
def triple_fps(d, cap=12):
    rs = sorted(rare_set(d), key=lambda w: -idf(w))[:cap]
    fps = set()
    for i in range(len(rs)):
        for j in range(i+1, len(rs)):
            for k in range(j+1, len(rs)):
                fps.add(triple_cell(anch(rs[i]), anch(rs[j]), anch(rs[k])))
    return fps

# Co-relevant pairs
co_pairs = []
for q, gds in gold.items():
    gds = [d for d in gds if d in corpus]
    for i in range(len(gds)):
        for j in range(i+1, len(gds)):
            co_pairs.append((gds[i], gds[j]))
print(f"\n(c) THE 3-WAY: queries with >=2 gold = "
      f"{sum(1 for q,g in gold.items() if len([d for d in g if d in corpus])>=2)}, "
      f"co-relevant pairs = {len(co_pairs)}")

# Random baseline: random doc pairs (same count)
all_docs = list(corpus.keys())
rand_pairs = [(random.choice(all_docs), random.choice(all_docs)) for _ in range(max(2000, len(co_pairs)*4))]
rand_pairs = [(a,b) for a,b in rand_pairs if a != b]

co_jac = [jaccard(rare_set(a), rare_set(b)) for a,b in co_pairs]
rd_jac = [jaccard(rare_set(a), rare_set(b)) for a,b in rand_pairs]

# triple-cell sharing (subsample for cost)
sub_co = co_pairs if len(co_pairs) <= 400 else random.sample(co_pairs, 400)
sub_rd = rand_pairs if len(rand_pairs) <= 400 else random.sample(rand_pairs, 400)
fp_cache = {}
def fps(d):
    v = fp_cache.get(d)
    if v is None: v = triple_fps(d); fp_cache[d] = v
    return v
co_share = [len(fps(a) & fps(b)) for a,b in sub_co]
rd_share = [len(fps(a) & fps(b)) for a,b in sub_rd]
co_share_frac = sum(1 for x in co_share if x>0)/len(co_share)
rd_share_frac = sum(1 for x in rd_share if x>0)/len(rd_share)

print(f"    rare-term Jaccard:   co-relevant mean = {statistics.mean(co_jac):.4f}  "
      f"random mean = {statistics.mean(rd_jac):.4f}  "
      f"=> {statistics.mean(co_jac)/max(1e-9,statistics.mean(rd_jac)):.1f}x")
print(f"    share >=1 triple-cell: co-relevant = {co_share_frac:.3f}  random = {rd_share_frac:.3f}  "
      f"=> {co_share_frac/max(1e-9,rd_share_frac):.1f}x")
print(f"    mean shared triple-cells: co = {statistics.mean(co_share):.2f}  random = {statistics.mean(rd_share):.3f}")

# Corridor-bridge signal: do co-relevant docs share a CORRIDOR partner-anchor more?
def corridor_terms(d, cap=12):
    rs = sorted(rare_set(d), key=lambda w: -idf(w))[:cap]
    out = set(rs)
    for w in rs:
        for b,_,_ in corridor.get(w, [])[:3]:
            out.add(b)
    return out
co_cor = [jaccard(corridor_terms(a), corridor_terms(b)) for a,b in sub_co]
rd_cor = [jaccard(corridor_terms(a), corridor_terms(b)) for a,b in sub_rd]
print(f"    corridor-expanded Jaccard: co = {statistics.mean(co_cor):.4f}  random = {statistics.mean(rd_cor):.4f}  "
      f"=> {statistics.mean(co_cor)/max(1e-9,statistics.mean(rd_cor)):.1f}x")

print("\nSUMMARY: co-relevant docs share rare-term/triple structure "
      f"{statistics.mean(co_jac)/max(1e-9,statistics.mean(rd_jac)):.1f}x (Jaccard) / "
      f"{co_share_frac/max(1e-9,rd_share_frac):.1f}x (triple-cell) over random.")
