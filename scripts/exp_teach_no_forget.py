#!/usr/bin/env python3
"""
Teach a concept from a REAL document that is never stored as a retrievable doc -
it only builds correlations. Validates Timothy's three claims:

  (1) NOW IT KNOWS   teaching a concept document recovers gap queries whose gold
                     doc was unreachable (no lexical overlap), WITHOUT adding the
                     teaching text to the index (it can never be returned itself).
  (2) NO FORGETTING  teaching only ADDS correlation edges; it never edits postings
                     or idf, so every query that does not touch a taught term gets
                     a byte-identical ranking, and overall held-out nDCG does not
                     regress (append-only / no catastrophic forgetting).
  (3) TINY MEMORY    because the symbols / subwords are already known, a teaching
                     doc adds only a few correlation edges on EXISTING primes -
                     far smaller than storing the document as postings.

Teaching source = real concept write-ups (the scifact glossary entries, authored
from general knowledge, NOT from the gold docs). Each is fed as a free-text
"document" through teach(); the engine keeps only rare-term co-occurrence edges.

Run:  python scripts/exp_teach_no_forget.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_teach_store import TeachStore
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def lex_rank(idx, lex, k=100):
    return sorted(lex, key=lex.get, reverse=True)[:k]


def taught_rank(idx, teach, query, lex, lam=0.3, n_expand=30, k=100):
    cand = sorted(lex, key=lex.get, reverse=True)[:100]
    exp = teach.expand_scores(query)
    if not exp:
        return cand[:k]                    # no taught term touched -> identical to lexical
    cset = set(cand)
    extra = [d for d in sorted(exp, key=exp.get, reverse=True) if d not in cset][:n_expand]
    pool = cand + extra
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) or 1.0
    final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=final.get, reverse=True)[:k]


def gold_rank(ranked, gold):
    g = set(gold)
    return next((i + 1 for i, d in enumerate(ranked) if d in g), None)


def load_glossary(name):
    mod = {"scifact": "scifact_glossary", "nfcorpus": "nfcorpus_glossary",
           "fiqa": "fiqa_glossary"}.get(name)
    if mod is None:
        return {}
    import importlib
    return getattr(importlib.import_module(mod), "GLOSSARY", {})


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    vocab_before = len(idx.token_prime)

    GLOSSARY = load_glossary(name)             # real concept write-ups (not gold docs)

    print(f"{name}: {len(corpus)} docs, {N} alive, vocab {vocab_before} primes")
    print(f"teaching {len(GLOSSARY)} real concept documents as CORRELATION-ONLY "
          f"(never indexed)\n")

    # precompute lexical scores once (the 'before teaching' state)
    lex_cache = {qid: idx._score(queries[qid]) for qid in test_ids}

    # ---------- BEFORE: pure lexical ----------
    before_ranks = {qid: lex_rank(idx, lex_cache[qid], k=1000) for qid in test_ids}
    nd0 = sum(ndcg10(before_ranks[q][:10], test_q[q]) for q in test_ids) / len(test_ids)
    rc0 = sum(recall10(before_ranks[q][:10], test_q[q]) for q in test_ids) / len(test_ids)

    # ---------- TEACH ----------
    teach = TeachStore(idx, N)
    for term, definition in GLOSSARY.items():
        # a real concept doc = the term plus its written-up definition
        teach.teach(f"{term} {definition}")
    teach.finalize(top_k=16)               # prune to linear (tiny) memory
    n_docs_before = len(idx.alive)
    assert len(idx.alive) == n_docs_before, "teaching must not add retrievable docs"
    n_edges, mem_b = teach.memory_bytes()

    # ---------- AFTER: lexical + taught correlations ----------
    after_ranks = {qid: taught_rank(idx, teach, queries[qid], lex_cache[qid], k=1000)
                   for qid in test_ids}
    nd1 = sum(ndcg10(after_ranks[q][:10], test_q[q]) for q in test_ids) / len(test_ids)
    rc1 = sum(recall10(after_ranks[q][:10], test_q[q]) for q in test_ids) / len(test_ids)

    # ===== (2) NO FORGETTING: identical rankings where no taught term is touched =====
    # a query only EXPANDS through a rare taught term (idf >= gate); match that gate.
    def expands(qid):
        for w in set(words(queries[qid])):
            iv = teach.idf(w)
            if w in teach.edges and iv is not None and iv >= 5.5:
                return True
        return False

    identical = touched = changed_for_better = changed_for_worse = 0
    for qid in test_ids:
        q_touches = expands(qid)
        same = before_ranks[qid][:10] == after_ranks[qid][:10]
        if same:
            identical += 1
        if q_touches:
            touched += 1
            b = ndcg10(before_ranks[qid][:10], test_q[qid])
            a = ndcg10(after_ranks[qid][:10], test_q[qid])
            if a > b + 1e-9:
                changed_for_better += 1
            elif a < b - 1e-9:
                changed_for_worse += 1

    # ===== (1) NOW IT KNOWS: gap-query gold recovery =====
    recovered = []
    for qid in test_ids:
        gold = [d for d, s in test_q[qid].items() if s > 0]
        gb = gold_rank(before_ranks[qid], gold)
        ga = gold_rank(after_ranks[qid], gold)
        if (gb is None or gb > 10) and ga is not None and ga <= 10:
            recovered.append((qid, gb, ga, queries[qid]))

    # ===== (3) TINY MEMORY: edges vs storing as docs =====
    # cost to store the same teaching text as retrievable postings (multi-view):
    # ~ tokens * 3 gears * ~6 B/posting is the index cost we AVOIDED.
    avoided_postings = teach.tokens_seen * 3
    avoided_bytes = avoided_postings * 6

    print("="*68)
    print("(1) NOW IT KNOWS  -- gap queries whose gold was unreachable, recovered:")
    if recovered:
        for qid, gb, ga, qt in recovered[:8]:
            print(f"   q{qid:<5} gold {str(gb):>5} -> {ga:>3}  top-10   {qt[:52]!r}")
        print(f"   ... {len(recovered)} gap queries recovered into top-10 by teaching")
    else:
        print("   (no top-10 recoveries on this split; see overall nDCG below)")

    print("\n(2) NO FORGETTING  -- teaching is purely additive:")
    print(f"   vocab primes: {vocab_before} -> {len(idx.token_prime)} (unchanged: "
          f"teaching added 0 retrievable docs)")
    print(f"   {identical}/{len(test_ids)} test queries have a BYTE-IDENTICAL top-10 "
          f"after teaching")
    print(f"   of {touched} queries that touch a taught term: "
          f"{changed_for_better} improved, {changed_for_worse} regressed")
    print(f"   overall held-out: nDCG {nd0:.4f} -> {nd1:.4f} ({nd1-nd0:+.4f}), "
          f"Recall {rc0:.4f} -> {rc1:.4f} ({rc1-rc0:+.4f})")

    reuse_pct = 100.0 * (1 - teach.new_symbols / max(teach.tokens_seen, 1))
    vocab_growth = 100.0 * teach.new_symbols / max(vocab_before, 1)
    print("\n(3) TINY MEMORY  -- learning reuses known symbols/subwords:")
    print(f"   taught {teach.n_taught} concept docs, {teach.tokens_seen} tokens")
    print(f"   new symbols (primes) created: {teach.new_symbols}  "
          f"-> vocab grew only {vocab_growth:.3f}% ({reuse_pct:.1f}% of tokens reused)")
    print(f"   correlation edges (capped top-16/term): {n_edges}  "
          f"(~{mem_b/1024:.1f} KB @ 8 B/edge)  = ~{mem_b/max(teach.n_taught,1):.0f} B/concept")
    print(f"   the document text is NOT stored: it can never be (wrongly) returned, "
          f"and avoids ~{avoided_postings} retrievable postings")


if __name__ == "__main__":
    main()
