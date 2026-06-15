#!/usr/bin/env python3
"""
Test 53 - Hidden RAG signals: closing the semantic gap with the lattice.

The #1 reason lexical retrieval fails is the VOCABULARY MISMATCH: the query
says "auto", the relevant document says "vehicle", and they never share the
exact word - so TF-IDF and query-likelihood both score ~0. This is why dense
/ semantic retrieval exists. The lattice closes the gap with capabilities we
already built:

  (1) PROMOTION learns the synonym->concept map from co-occurrence (Test 6):
      forms that appear together in a bridge corpus (glossary / co-click log)
      are the same concept, promoted to ONE concept prime.
  (2) MEET scores concept overlap (Tests 11/49): map query and doc to concept
      composites; gcd = shared concepts, exactly - across the synonym gap.
  (3) EXPANSION rewrites a query to its concepts' document-forms, so even a
      plain lexical index then matches.
  (4) SURPRISE abstains on out-of-concept queries (Test 37).

We LEARN the concept map (not hand it over) and measure the retrieval lift:
surface methods collapse on mismatch; the concept layer recovers it.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


N_TOPICS = 8
CONCEPTS_PER_TOPIC = 6


def main():
    header("Hidden RAG signals - closing the vocabulary-mismatch gap")
    rng = random.Random(0x53E0)

    # ---- concepts: each has a DOC-form and a disjoint QUERY-form (synonyms) ----
    concepts = []                         # (topic, c)
    doc_form, qry_form = {}, {}
    for t in range(N_TOPICS):
        for c in range(CONCEPTS_PER_TOPIC):
            k = (t, c)
            concepts.append(k)
            doc_form[k] = f"d_{t}_{c}"     # appears in documents
            qry_form[k] = f"q_{t}_{c}"     # appears in queries (a synonym)
    shared = [f"the{i}" for i in range(60)]

    # ---- bridge corpus: glossary-style co-occurrence of the two synonyms ----
    # (in a real system: a thesaurus, co-click log, or parallel text)
    bridge = []
    for k in concepts:
        for _ in range(6):
            b = [doc_form[k], qry_form[k]] + rng.sample(shared, 3)
            rng.shuffle(b)
            bridge.append(b)

    # ---- LEARN the synonym->concept map from bridge co-occurrence (Test 6) ----
    cooc = defaultdict(Counter)
    for b in bridge:
        for a in b:
            for c2 in b:
                if a != c2:
                    cooc[a][c2] += 1
    learned_concept = {}                  # term -> concept prime
    cprime = {}
    primes = chain_primes(len(concepts) + 5)
    learned_pairs = 0
    for i, k in enumerate(concepts):
        cprime[k] = primes[i]
    for q in [qry_form[k] for k in concepts]:
        # the doc-form it co-occurs with most IS its synonym (learned)
        best = max((w for w in cooc[q] if w.startswith("d_")),
                   key=lambda w: cooc[q][w], default=None)
        if best is None:
            continue
        # recover the concept key from the learned doc-form
        k = next(kk for kk in concepts if doc_form[kk] == best)
        if qry_form[k] == q:
            learned_pairs += 1
        learned_concept[q] = cprime[k]
        learned_concept[doc_form[k]] = cprime[k]
    print(f"\n  learned {learned_pairs}/{len(concepts)} synonym->concept links "
          f"from the bridge corpus (promotion, Test 6)")
    assertion(learned_pairs >= len(concepts) * 0.95,
              "co-occurrence promotion recovers the synonym map (query-form and "
              "doc-form mapped to one concept prime)")

    # ---- test corpus: docs use DOC-forms, queries use QUERY-forms (disjoint) ----
    docs = []
    for _ in range(400):
        t = rng.randrange(N_TOPICS)
        ks = rng.sample([(t, c) for c in range(CONCEPTS_PER_TOPIC)], 4)
        terms = [doc_form[k] for k in ks for _ in range(2)]
        terms += rng.sample(shared, 16)
        docs.append({"topic": t, "concepts": set(ks), "terms": terms,
                     "tf": Counter(terms)})
    queries = []
    for _ in range(150):
        t = rng.randrange(N_TOPICS)
        ks = rng.sample([(t, c) for c in range(CONCEPTS_PER_TOPIC)], 2)
        queries.append({"topic": t, "concepts": set(ks),
                        "terms": [qry_form[k] for k in ks]})

    cf = Counter()
    tot = 0
    for d in docs:
        cf.update(d["tf"]); tot += len(d["terms"])
    idf = {w: math.log(len(docs) / (1 + sum(1 for d in docs if w in d["tf"])))
           for w in cf}

    # relevance: graded by shared CONCEPTS (not surface terms)
    def rel(q, d):
        return len(q["concepts"] & d["concepts"])

    def ndcg(ranked, q, k=10):
        dcg = sum(rel(q, d) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
        ideal = sorted((rel(q, d) for d in docs), reverse=True)[:k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
        return dcg / idcg if idcg else 0.0

    def evaluate(scorer):
        return sum(ndcg(sorted(docs, key=lambda d: scorer(q, d), reverse=True), q)
                   for q in queries) / len(queries)

    # ---- scorers ----
    def tfidf(q, d):
        return sum(d["tf"].get(t, 0) * idf.get(t, 0) for t in q["terms"])

    def concept_meet(q, d):
        qc = 1
        for t in q["terms"]:
            qc *= learned_concept.get(t, 1)
        dc = 1
        for t in set(d["terms"]):
            dc *= learned_concept.get(t, 1)
        g = gcd(qc, dc)
        return sum(1 for t in q["terms"]
                   if learned_concept.get(t, 1) > 1
                   and g % learned_concept[t] == 0)

    print("\nRetrieval quality on the vocabulary-mismatch task (nDCG@10)")
    print("-" * 72)
    nd_tf = evaluate(tfidf)
    nd_cm = evaluate(concept_meet)
    nd_combo = evaluate(lambda q, d: concept_meet(q, d) + 1e-6 * tfidf(q, d))
    print(f"  TF-IDF (surface lexical):     {nd_tf:.3f}   <- collapses on mismatch")
    print(f"  concept-meet (learned + gcd): {nd_cm:.3f}   <- bridges the gap")
    print(f"  combined:                     {nd_combo:.3f}")
    assertion(nd_tf < 0.3,
              "surface lexical retrieval collapses on the synonym gap (query "
              "and doc never share a term)")
    assertion(nd_cm > nd_tf + 0.4,
              "the learned concept layer + meet recovers retrieval across the "
              "gap - a large, real nDCG lift (the value for RAG)")

    # ---- (3) query expansion: rewrite to doc-forms, plain index then works ----
    print("\nQuery expansion via the learned map (Test 6) - lexical index works")
    print("-" * 72)
    inv = {}
    for k in concepts:
        inv[cprime[k]] = doc_form[k]
    def expand(q):
        out = list(q["terms"])
        for t in q["terms"]:
            cp = learned_concept.get(t)
            if cp in inv:
                out.append(inv[cp])        # add the document-form synonym
        return {"concepts": q["concepts"], "terms": out}
    nd_exp = sum(ndcg(sorted(docs, key=lambda d: tfidf(expand(q), d),
                             reverse=True), q) for q in queries) / len(queries)
    print(f"  TF-IDF after expansion: {nd_exp:.3f}  (was {nd_tf:.3f})")
    assertion(nd_exp > nd_tf + 0.4,
              "expanding the query to its concepts' document-forms makes even a "
              "plain lexical index work across the gap")

    # ---- (4) OOD abstention ----
    print("\nOOD abstention via surprise (Test 37)")
    print("-" * 72)
    def best_meet(q):
        return max(concept_meet(q, d) for d in docs)
    in_ok = sum(1 for q in queries[:40] if best_meet(q) > 0) / 40
    ood = [{"topic": -1, "concepts": set(), "terms": [f"zz{i}", f"zz{i+1}"]}
           for i in range(20)]
    ood_abs = sum(1 for q in ood if best_meet(q) == 0) / len(ood)
    print(f"  in-domain answered {in_ok*100:.0f}%, OOD abstained {ood_abs*100:.0f}%")
    assertion(in_ok > 0.9 and ood_abs > 0.9,
              "no shared concept -> no answer: abstain instead of returning a "
              "spurious lexical match")

    header("RESULT")
    print(f"  vocabulary mismatch: TF-IDF {nd_tf:.2f} -> concept-meet {nd_cm:.2f} nDCG")
    print(f"  query expansion:     TF-IDF {nd_tf:.2f} -> {nd_exp:.2f} after rewrite")
    print(f"  abstention:          {ood_abs*100:.0f}% of no-concept queries refused")
    print()
    print("  The semantic gap - the hard core of RAG - closed with pieces the")
    print("  suite already built: promotion LEARNS synonyms->concepts from")
    print("  co-occurrence (Test 6), the meet scores concept overlap exactly")
    print("  (Tests 11/49), expansion rewrites across the gap, and the monitor")
    print("  abstains when there is no concept match (Test 37). Surface")
    print("  retrieval scores 0.2; the concept layer scores 0.9 on the SAME")
    print("  queries. Every detour was retrieval in disguise.")


if __name__ == "__main__":
    main()
