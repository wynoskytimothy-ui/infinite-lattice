"""STAGE 3 audit: "store a corpus in a prime + a rule, regenerate correlations
with NO extra footprint."

Two systems, IDENTICAL retrieval algorithm (lex candidates + bridge pool-expand,
fused). They differ ONLY in how the correlated-words expansion is obtained:

  STORED      : persist the EXPLICIT correlation matrix  bridge[qt] -> [(dt,w)..]
                (a precomputed doc-word correlation table). Query = dict lookup.

  REGENERATED : persist ONLY (i) the inverted index = doc-primes/composites
                (already needed for retrieval) and (ii) ONE global RULE object
                = the corridor formula's PARAMETERS + the per-relevant-pair
                (qt,gold-doc) evidence in its rawest form. At query time the
                rule REGENERATES bridge[qt] by applying the corridor formula to
                the lattice MEET of the stored doc-chains (qt's relevant gold
                docs intersected with the corpus postings) -- no stored matrix.

We measure two things on scifact, held-out test queries:
  (1) FOOTPRINT: bytes(doc-primes + rule)  vs  bytes(doc-primes + explicit matrix)
  (2) FIDELITY : recall@10 / nDCG@10 of REGENERATED vs STORED (and vs lex-only).
"""
from __future__ import annotations
import sys, time, math, pickle, zlib
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.bench_supervised_bridges import load, ndcg10, recall10
from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_search


def recall_at(ranked, rels, k):
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


# ------------------------------------------------------------------ build index
NAME = "scifact"
corpus, queries, train_q, test_q = load(NAME)
idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)
test_ids = [q for q in test_q if q in queries]
print(f"{NAME}: {len(corpus)} docs | train {len(train_q)} q | test {len(test_ids)} q")

# idf helper (corpus-level, from the live index)
def idf(w):
    p = idx.token_prime.get(("w", w))
    return idx._idf(p, N) if p else 0.0

# ------------------------------------------------------------------ STORED system
# learn the explicit correlation matrix once (the canonical RelevanceBridges)
br_stored = RelevanceBridges(idx, N).learn(queries, train_q, corpus)
explicit_matrix = br_stored.bridge          # qt -> [(dt, w), ...]

# ------------------------------------------------------------------ THE RULE (global)
IDF_GATE = br_stored.idf_gate
MIN_PAIRS = br_stored.min_pairs
TOP_PER_TERM = br_stored.top_per_term

# The rule's EVIDENCE in its rawest, most-compressible form: for each training
# query, (its gated query-terms, its gold-doc ids). This is NOT the matrix -- it
# is the raw supervision the rule consumes. The doc-word content is regenerated
# from the stored doc-chains (postings), NOT stored again.
rule_evidence = []   # list of (tuple_of_qt, tuple_of_gold_doc_ids)
for qid, rels in train_q.items():
    if qid not in queries:
        continue
    qts = tuple(sorted(w for w in set(words(queries[qid])) if idf(w) >= IDF_GATE))
    if not qts:
        continue
    golds = tuple(sorted(cid for cid, sc in rels.items() if sc > 0 and cid in corpus))
    if not golds:
        continue
    rule_evidence.append((qts, golds))

# doc -> gated doc-word SET regenerated from the stored doc-chains (lattice meet:
# a doc's word-primes ARE its stored chain; we recover the gated word set from the
# index's doc_words via the prime->token map, i.e. from the doc-primes themselves).
prime_to_word = {p: tok[1] for tok, p in idx.token_prime.items() if tok[0] == "w"}

def doc_gated_words(d):
    """Regenerate a gold doc's gated content words from its STORED prime-chain."""
    out = []
    for p in idx.doc_words.get(d, ()):
        w = prime_to_word.get(p)
        if w is not None and idf(w) >= IDF_GATE:
            out.append(w)
    return out

# ------------------------------------------------------------------ REGENERATE the matrix
def regenerate_bridge():
    """Apply the corridor formula to the lattice meet of stored doc-chains."""
    cooc = defaultdict(Counter)
    qt_pairs = Counter()
    # cache regenerated gold-doc word sets (regenerated from postings once)
    dcache = {}
    for qts, golds in rule_evidence:
        # regenerate each gold doc's gated word set from the doc-prime chain
        gold_word_sets = []
        for g in golds:
            ws = dcache.get(g)
            if ws is None:
                ws = frozenset(doc_gated_words(g))
                dcache[g] = ws
            gold_word_sets.append(ws)
        for qt in qts:
            for ws in gold_word_sets:
                qt_pairs[qt] += 1
                if ws:
                    cooc[qt].update(ws)
    bridge = {}
    for qt, partners in cooc.items():
        np_ = qt_pairs[qt]
        scored = [(dt, (c / np_) * idf(dt)) for dt, c in partners.items()
                  if dt != qt and c >= MIN_PAIRS]
        scored.sort(key=lambda x: (-x[1], x[0]))
        if scored:
            bridge[qt] = scored[:TOP_PER_TERM]
    return bridge

t0 = time.time()
regen_matrix = regenerate_bridge()
regen_time = time.time() - t0
print(f"regenerated matrix in {regen_time*1000:.0f} ms")

# ------------------------------------------------------------------ FIDELITY: are they identical?
def sig(m):
    # round weights to compare faithfully (formula is deterministic)
    return {qt: [(dt, round(w, 9)) for dt, w in v] for qt, v in m.items()}

ident = sig(explicit_matrix) == sig(regen_matrix)
print(f"regenerated == stored matrix (bit-faithful)? {ident}")
if not ident:
    sk = set(explicit_matrix) ^ set(regen_matrix)
    print("  term-set diff:", len(sk))

# ------------------------------------------------------------------ retrieval eval
class _BrShim:
    def __init__(self, bridge):
        self.bridge = bridge
        self.corridor_bridge = {}

def eval_system(bridge, lam=0.25):
    shim = _BrShim(bridge)
    R10 = N10 = R100 = R20 = 0.0
    n = 0
    for qid in test_ids:
        rels = test_q[qid]
        if not any(s > 0 for s in rels.values()):
            continue
        ranked = bridge_search(idx, shim, queries[qid], lam=lam, k=100)
        R10 += recall_at(ranked, rels, 10)
        R20 += recall_at(ranked, rels, 20)
        R100 += recall_at(ranked, rels, 100)
        N10 += ndcg10(ranked[:10], rels)
        n += 1
    return dict(R10=R10/n, R20=R20/n, R100=R100/n, nDCG10=N10/n, n=n)

def eval_lex(lam=0.0):
    R10 = N10 = R100 = 0.0; n = 0
    for qid in test_ids:
        rels = test_q[qid]
        if not any(s > 0 for s in rels.values()):
            continue
        ranked = sorted(idx._score(queries[qid]), key=idx._score(queries[qid]).get, reverse=True)[:100]
        # simpler: just use search via dict path
        sc = idx._score(queries[qid])
        ranked = sorted(sc, key=sc.get, reverse=True)[:100]
        R10 += recall_at(ranked, rels, 10)
        R100 += recall_at(ranked, rels, 100)
        N10 += ndcg10(ranked[:10], rels)
        n += 1
    return dict(R10=R10/n, R100=R100/n, nDCG10=N10/n, n=n)

print("\n--- FIDELITY (held-out test) ---")
lex = eval_lex()
print(f"lex-only      : R@10={lex['R10']:.4f}  R@100={lex['R100']:.4f}  nDCG@10={lex['nDCG10']:.4f}")
sto = eval_system(explicit_matrix)
print(f"STORED  matrix: R@10={sto['R10']:.4f}  R@100={sto['R100']:.4f}  nDCG@10={sto['nDCG10']:.4f}")
reg = eval_system(regen_matrix)
print(f"REGEN   matrix: R@10={reg['R10']:.4f}  R@100={reg['R100']:.4f}  nDCG@10={reg['nDCG10']:.4f}")

# ------------------------------------------------------------------ FOOTPRINT
def blob_size(obj):
    return len(zlib.compress(pickle.dumps(obj), 9))

# the inverted index (doc-primes) is COMMON to both -> report it but it cancels
inv_size = blob_size({p: list(pl.items()) for p, pl in idx.postings.items()
                      if any(tok[0] == "w" and pp == p for tok, pp in idx.token_prime.items())})
# simpler: word-gear postings only (the doc-chains)
word_primes = {p for tok, p in idx.token_prime.items() if tok[0] == "w"}
word_postings = {p: list(idx.postings[p].items()) for p in word_primes if p in idx.postings}
inv_size = blob_size(word_postings)

explicit_size = blob_size(explicit_matrix)
rule_size = blob_size((IDF_GATE, MIN_PAIRS, TOP_PER_TERM, rule_evidence))

print("\n--- FOOTPRINT (zlib-9 compressed bytes) ---")
print(f"inverted index (doc-primes, word gear), COMMON to both : {inv_size:>9,} B")
print(f"STORED  add-on: explicit correlation matrix            : {explicit_size:>9,} B")
print(f"REGEN   add-on: global rule (params + raw qrels evidence): {rule_size:>9,} B")
print(f"matrix saving by regenerating (add-on only)            : "
      f"{explicit_size - rule_size:>9,} B  ({100*(explicit_size-rule_size)/explicit_size:.1f}%)")
print(f"total STORED  : {inv_size + explicit_size:>9,} B")
print(f"total REGEN   : {inv_size + rule_size:>9,} B")
print(f"total saving  : {100*(explicit_size-rule_size)/(inv_size+explicit_size):.2f}% of full footprint")

# raw (uncompressed) too for honesty
print("\n--- FOOTPRINT (raw pickled bytes, no compression) ---")
print(f"explicit matrix : {len(pickle.dumps(explicit_matrix)):>9,} B")
print(f"rule evidence   : {len(pickle.dumps((IDF_GATE,MIN_PAIRS,TOP_PER_TERM,rule_evidence))):>9,} B")
print(f"  rule_evidence: {len(rule_evidence)} train pairs, "
      f"{sum(len(q)+len(g) for q,g in rule_evidence)} total ids")
print(f"  explicit matrix: {len(explicit_matrix)} terms, "
      f"{sum(len(v) for v in explicit_matrix.values())} (qt->dt) entries")
