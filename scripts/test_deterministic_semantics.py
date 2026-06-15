#!/usr/bin/env python3
"""
Test 58 - Deterministic, verifiable semantics from co-occurrence counting.

Does adding a semantic layer break the lattice's determinism and verifiability?
NO - if it is COUNTING (co-occurrence -> PPMI), not a neural embedding.

  - word2vec / neural: random init + SGD -> non-deterministic, opaque weights,
    retrain to add words. BREAKS determinism + verifiability + append-only.
  - co-occurrence counting: same corpus -> identical counts (any order),
    every association explained by shared-context EVIDENCE, new docs just add
    counts. PRESERVES all three. (Levy & Goldberg 2014: word2vec implicitly
    factorizes a PPMI matrix - neural-quality semantics from counting.)

We verify, on a corpus with planted synonyms:
  (A) DETERMINISM   build in two different doc orders -> identical PPMI
  (B) VERIFIABILITY every learned association has shared-context evidence
  (C) APPEND-ONLY   adding docs only increments counts; old counts untouched
  (D) SEMANTIC      PPMI-context similarity recovers the planted synonyms
  (E) RETRIEVAL     PPMI bridges the synonym gap (a real, deterministic lift)
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


def assertion(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


class CoocSemantics:
    """Deterministic co-occurrence -> PPMI semantic layer (counting only)."""

    def __init__(self):
        self.cooc = defaultdict(Counter)     # word -> Counter(context word)
        self.wc = Counter()                  # word doc-frequency
        self.D = 0

    def add(self, doc_words):                # APPEND only
        s = set(doc_words)
        for w in s:
            self.wc[w] += 1
        for a, b in combinations(sorted(s), 2):
            self.cooc[a][b] += 1
            self.cooc[b][a] += 1
        self.D += 1

    def ppmi(self, a, b):
        c = self.cooc[a].get(b, 0)
        if c == 0 or self.wc[a] == 0 or self.wc[b] == 0:
            return 0.0
        p_ab = c / self.D
        p_a = self.wc[a] / self.D
        p_b = self.wc[b] / self.D
        return max(0.0, math.log2(p_ab / (p_a * p_b)))

    def context_vec(self, w):
        return {c: self.ppmi(w, c) for c in self.cooc[w]}

    def similarity(self, a, b):
        va, vb = self.context_vec(a), self.context_vec(b)
        shared = set(va) & set(vb)
        if not shared:
            return 0.0, []
        dot = sum(va[c] * vb[c] for c in shared)
        na = math.sqrt(sum(v * v for v in va.values()))
        nb = math.sqrt(sum(v * v for v in vb.values()))
        sim = dot / (na * nb) if na and nb else 0.0
        # evidence = shared context words contributing most
        ev = sorted(shared, key=lambda c: -(va[c] * vb[c]))[:5]
        return sim, ev


def build_corpus(rng):
    # concepts with two synonym surface forms that co-occur with shared anchors
    topics = {}
    for t in range(6):
        anchors = [f"anchor{t}_{i}" for i in range(6)]
        concepts = []
        for c in range(5):
            concepts.append((f"q{t}_{c}", f"d{t}_{c}"))   # (query form, doc form)
        topics[t] = (anchors, concepts)
    docs = []
    truth = {}                                # surface form -> concept id
    for t, (anchors, concepts) in topics.items():
        for c, (qf, df) in enumerate(concepts):
            truth[qf] = (t, c)
            truth[df] = (t, c)
    for _ in range(1200):
        t = rng.randrange(6)
        anchors, concepts = topics[t]
        doc = rng.sample(anchors, 4)
        # both forms of a few concepts appear with the topic anchors
        for (qf, df) in rng.sample(concepts, 3):
            doc.append(rng.choice([qf, df]))    # one form per occurrence
        docs.append(doc)
    return docs, truth, topics


def main():
    header("Deterministic, verifiable semantics from co-occurrence counting")
    rng = random.Random(0x58E0)
    docs, truth, topics = build_corpus(rng)

    # ---- (A) DETERMINISM: two different ingestion orders ----
    header("(A) DETERMINISM - counting is order-independent")
    s1 = CoocSemantics()
    for d in docs:
        s1.add(d)
    s2 = CoocSemantics()
    for d in reversed(docs):                  # opposite order
        s2.add(d)
    # compare PPMI on a sample of pairs
    sample_words = list(truth)[:30]
    diff = 0
    for a in sample_words:
        for b in sample_words:
            if abs(s1.ppmi(a, b) - s2.ppmi(a, b)) > 1e-12:
                diff += 1
    print(f"  PPMI values compared across both orders: {len(sample_words)**2}; "
          f"differences: {diff}")
    assertion(diff == 0,
              "identical PPMI regardless of ingestion order - deterministic "
              "(no random init, no SGD, just counts)")

    # ---- (B) VERIFIABILITY: every association has evidence ----
    header("(B) VERIFIABILITY - associations explained by shared-context evidence")
    qf, df = "q0_0", "d0_0"                    # a known synonym pair
    sim, ev = s1.similarity(qf, df)
    print(f"  '{qf}' ~ '{df}': similarity {sim:.3f}")
    print(f"  WHY (shared high-PPMI context words): {ev}")
    assertion(sim > 0 and len(ev) >= 2,
              "the association is explained by concrete shared-context words - "
              "you can audit WHY two terms are semantically linked (no opaque "
              "embedding)")

    # ---- (C) APPEND-ONLY: adding docs only increments counts ----
    header("(C) APPEND-ONLY - new docs add counts; old counts untouched")
    snap = {a: dict(s1.cooc[a]) for a in sample_words}
    extra, _, _ = build_corpus(random.Random(99))
    for d in extra[:300]:
        s1.add(d)
    monotone = all(s1.cooc[a].get(b, 0) >= cnt
                   for a, row in snap.items() for b, cnt in row.items())
    print(f"  added 300 docs; every prior co-occurrence count >= its old value: "
          f"{monotone}")
    assertion(monotone,
              "counts only grow as data arrives - append-only, never rewritten "
              "(consistent with Test 54: teach forward, never relearn)")

    # ---- (D) SEMANTIC: PPMI recovers planted synonyms ----
    header("(D) SEMANTIC - PPMI-context similarity recovers synonyms")
    sclean = CoocSemantics()
    for d in docs:
        sclean.add(d)
    # for each query-form, is its true doc-form synonym its top neighbour?
    forms = [w for w in truth if w.startswith("q")]
    hit = 0
    for qf in forms:
        t, c = truth[qf]
        df_true = f"d{t}_{c}"
        cands = [w for w in truth if w.startswith("d")]
        best = max(cands, key=lambda d: sclean.similarity(qf, d)[0])
        hit += (best == df_true)
    rate = hit / len(forms)
    print(f"  synonym recovery: query-form's nearest doc-form is its true "
          f"synonym {rate*100:.0f}% of the time")
    assertion(rate > 0.8,
              "PPMI-context similarity recovers the planted synonyms (semantic "
              "signal from pure counting - no neural net)")

    # ---- (E) RETRIEVAL: PPMI bridges the synonym gap ----
    header("(E) RETRIEVAL - PPMI expansion bridges the vocabulary gap")
    # query uses q-forms, docs use d-forms (disjoint surface). Lexical = 0.
    # build a tiny test: docs hold d-forms; queries hold q-forms.
    test_docs = []
    for t, (anchors, concepts) in topics.items():
        for _ in range(20):
            doc = rng.sample(anchors, 3) + [df for (_, df) in
                                            rng.sample(concepts, 2)]
            test_docs.append((t, set(doc)))
    test_q = [(truth[qf], qf) for qf in forms]

    def expand(qf):
        # PPMI nearest doc-form (deterministic) bridges to the doc vocabulary
        cands = [w for w in truth if w.startswith("d")]
        best = max(cands, key=lambda d: sclean.similarity(qf, d)[0])
        return {qf, best}

    lex_hit = sem_hit = 0
    for (t, c), qf in test_q[:60]:
        relevant = [i for i, (tt, dd) in enumerate(test_docs) if tt == t
                    and f"d{t}_{c}" in dd]
        if not relevant:
            continue
        lex = [i for i, (_, dd) in enumerate(test_docs) if qf in dd]      # surface
        exp = expand(qf)
        sem = [i for i, (_, dd) in enumerate(test_docs) if exp & dd]
        lex_hit += any(i in relevant for i in lex)
        sem_hit += any(i in relevant for i in sem)
    print(f"  lexical (surface) retrieval found relevant: {lex_hit}/60")
    print(f"  PPMI-bridged retrieval found relevant:      {sem_hit}/60")
    assertion(sem_hit > lex_hit + 10,
              "deterministic PPMI bridges the synonym gap lexical match cannot - "
              "semantic lift WITHOUT breaking determinism or verifiability")

    header("RESULT - semantics added, determinism & verifiability intact")
    print("  (A) deterministic: identical PPMI in any ingestion order")
    print("  (B) verifiable: every association has shared-context evidence")
    print("  (C) append-only: counts only grow; old values untouched")
    print("  (D) semantic: recovers planted synonyms from counting alone")
    print("  (E) retrieval: bridges the vocabulary gap lexical match misses")
    print()
    print("  The answer to 'does this break determinism/verifiability': NO.")
    print("  Semantics done as CO-OCCURRENCE COUNTING (PPMI) is deterministic,")
    print("  auditable, and append-only - the lattice's properties intact.")
    print("  Only the NEURAL route (random init + SGD) would break them, and")
    print("  we do not need it: Levy-Goldberg showed word2vec just factorizes")
    print("  this same PPMI matrix. Counting gets the semantics for free,")
    print("  reproducibly, with a receipt for every link.")


if __name__ == "__main__":
    main()
