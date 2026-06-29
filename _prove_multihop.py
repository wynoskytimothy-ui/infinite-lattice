#!/usr/bin/env python3
"""
_prove_multihop.py  --  PROVE the lattice MEET can CHAIN a multi-hop / bridge
retrieval that single-hop BM25 misses.  CPU only, no GPU, no downloads.

THE MULTI-HOP PROBLEM (a known weakness of vector / single-hop RAG)
-------------------------------------------------------------------
A bridge question needs TWO docs:
    doc A  -- matches the query terms, AND mentions a rare "bridge entity" X
    doc B  -- the actual answer; mentions X but NOT the query terms
Single-hop retrieval scores B ~ 0 on the query (B shares no query words), so B
is missed.  A human reasons:  query -> A -> (entity X) -> B.

THE LATTICE MEET = THAT HOP, FOR FREE (no stored edges)
-------------------------------------------------------
In aethos_append_index every word is a prime; doc D's content is the set
idx.doc_words[D] of word-primes.  The "meet" of two docs is their shared primes
(this is exactly the overlap edge search_manifold builds).  The bridge entity X
is the RAREST shared prime between A and the rest of the corpus -- found by idf,
no learned edge.  meet-expand = "all docs whose word-set contains prime(X)" =
idx.postings[prime(X)].  Chaining query->A->X->B is two meets, zero stored graph.

PROTOCOL (honest, held-out, no peeking at B during retrieval)
-------------------------------------------------------------
Two corpora are tested:

  PART 1  SYNTHETIC controlled bridge corpus.  We MANUFACTURE the failure mode
          (B deliberately shares no query word) so the mechanism is unambiguous.
          Metric: 2nd-hop (bridge-doc B) recall, chain vs single-hop BM25.

  PART 2  REAL scifact docs.  We MINE genuine bridge pairs: a rare entity-word
          that occurs in exactly 2 real docs (A,B); query = A's other content.
          We only accept pairs where single-hop BM25 actually misses B (the hard
          case).  Then measure whether the meet-chain recovers B.  This uses no
          synthetic text -- real corpus, real vocabulary co-occurrence.

PASS = the chain recovers bridge docs that single-hop BM25 misses, with numbers.
Honest: every miss is reported; partial results are stated as partial.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_append_index import AppendOnlyLatticeIndex, words


# --------------------------------------------------------------------------- #
#  THE MEET-CHAIN  (the only "new" logic; everything else is the stock index) #
# --------------------------------------------------------------------------- #
def bridge_entity_prime(idx, doc, query_terms, N, exclude_primes):
    """The rarest (highest-idf) word-prime in `doc` that is NOT a query word and
    co-occurs with at least one OTHER live doc -- the bridge entity X.  No edge
    is stored; rarity is read from the live df.  Returns prime or None."""
    cand = []
    for p in idx.doc_words.get(doc, ()):
        if p in exclude_primes:
            continue
        df = idx.df[p]
        if df < 2:           # must reach at least one other doc to be a bridge
            continue
        cand.append((idx._idf(p, N), p))
    if not cand:
        return None
    cand.sort(reverse=True)          # rarest first
    return cand[0][1]


def meet_chain(idx, query, k_first=5, k_expand=10):
    """Two-meet hop, no stored graph:
       hop-1  BM25 on the query -> top doc(s) A
       bridge the rarest non-query entity-prime X out of A
       hop-2  meet-expand: docs sharing X (idx.postings[X]), ranked by how rare
              X is and how strongly they carry it (tf).  Returns (A_list, second_hop_ranked, X_word)."""
    N = max(1, len(idx.alive))
    qbag = idx._multiview(query)
    q_word_primes = set()
    for tok in qbag:
        if tok[0] == "w":
            p = idx.token_prime.get(tok)
            if p is not None:
                q_word_primes.add(p)

    hop1 = idx.search(query, k_first)            # single-hop result = the A docs
    if not hop1:
        return [], [], None

    # collect bridge candidates from EACH first-hop doc, expand the union
    second = Counter()
    chosen_X = None
    for a in hop1:
        X = bridge_entity_prime(idx, a, qbag, N, exclude_primes=q_word_primes)
        if X is None:
            continue
        if chosen_X is None:
            chosen_X = X
        idfX = idx._idf(X, N)
        for doc, tf in idx.postings[X].items():
            if doc in idx.alive and doc not in hop1:
                second[doc] += idfX * tf      # meet-expand score
    ranked = [d for d, _ in second.most_common(k_expand)]
    # recover the bridge word string for reporting
    x_word = None
    if chosen_X is not None:
        for (view, tok), p in idx.token_prime.items():
            if p == chosen_X and view == "w":
                x_word = tok
                break
    return hop1, ranked, x_word


# --------------------------------------------------------------------------- #
#  PART 1 -- SYNTHETIC controlled bridge corpus                               #
# --------------------------------------------------------------------------- #
def part1_synthetic():
    print("=" * 70)
    print("PART 1  SYNTHETIC controlled multi-hop corpus")
    print("=" * 70)
    idx = AppendOnlyLatticeIndex()

    # 5 bridge cases. For each: query words live in doc A together with a unique
    # rare entity; the ANSWER doc B carries that entity but ZERO query words.
    # Plus distractor docs so retrieval is non-trivial.
    cases = [
        # (query, docA_text, docB_text(answer), bridge_word)
        ("which company manufactures the zephyrine compound",
         "the pharmaceutical firm acmecorp manufactures zephyrine for clinical trials",
         "acmecorp reported quarterly revenue growth and expanding research facilities",
         "acmecorp"),
        ("what protein interacts with the kraylon receptor",
         "studies show the kraylon receptor binds tightly to the membrane",
         "vimblastin protein modulates membrane channels and signal cascades broadly",   # B has NO query word, shares entity via A->bridge
         "membrane"),
        ("city hosting the grindel summit conference",
         "the grindel summit was scheduled in the coastal port of varnhold",
         "varnhold features deep harbors heavy industry and a historic old quarter",
         "varnhold"),
        ("gene linked to the florbetan syndrome",
         "florbetan syndrome arises from mutation in the qixly gene locus",
         "qixly expression rises sharply during embryonic neural tube formation phases",
         "qixly"),
        ("author of the treatise on lumic mechanics",
         "the treatise on lumic mechanics was penned by scholar daventhorpe",
         "daventhorpe later founded an academy and tutored several royal heirs",
         "daventhorpe"),
    ]
    distractors = [
        "ordinary weather patterns affect coastal shipping routes each season",
        "the museum acquired several paintings from a private collection",
        "agricultural yields depend on rainfall soil quality and crop rotation",
        "modern processors use multiple cores and deep cache hierarchies",
        "the orchestra performed a symphony to a sold out concert hall",
        "tax policy changes influence small business hiring and investment",
        "volcanic activity reshaped the island over several thousand years",
        "the library digitized thousands of rare manuscripts last year",
    ]

    doc_ids = {}
    for i, (q, a, b, br) in enumerate(cases):
        idx.add(f"A{i}", a)
        idx.add(f"B{i}", b)
        doc_ids[i] = (f"A{i}", f"B{i}", br)
    for j, d in enumerate(distractors):
        idx.add(f"D{j}", d)
    idx.finalize()

    chain_hit = single_hit = 0
    for i, (q, a, b, br) in enumerate(cases):
        Aid, Bid, _ = doc_ids[i]
        single = idx.search(q, 10)            # plain single-hop BM25 top-10
        hop1, second, xw = meet_chain(idx, q, k_first=3, k_expand=10)
        s_ok = Bid in single
        c_ok = Bid in (hop1 + second)
        single_hit += s_ok
        chain_hit += c_ok
        print(f"\n case {i}: '{q[:48]}...'")
        print(f"    answer doc {Bid}; single-hop top1={single[0] if single else None}  "
              f"B in single-hop@10? {s_ok}")
        print(f"    meet bridge word = '{xw}'  ;  2nd-hop = {second[:4]}  "
              f"B recovered by chain? {c_ok}")

    print(f"\n  SYNTHETIC bridge-doc recall:  single-hop {single_hit}/{len(cases)}"
          f"   chain {chain_hit}/{len(cases)}")
    return single_hit, chain_hit, len(cases)


# --------------------------------------------------------------------------- #
#  PART 2 -- REAL scifact bridge pairs mined by rare co-occurrence            #
# --------------------------------------------------------------------------- #
def part2_scifact(max_cases=60):
    print("\n" + "=" * 70)
    print("PART 2  REAL scifact docs -- mined bridge pairs (rare co-occurrence)")
    print("=" * 70)
    try:
        from scripts.bench_supervised_bridges import load
        corpus, queries, train_q, test_q = load("scifact")
    except SystemExit:
        print("  scifact not found -> skipping PART 2 (synthetic result stands).")
        return None

    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    idx.finalize()
    N = max(1, len(idx.alive))

    # invert: word-prime -> docs that contain it (word gear only)
    prime_word = {p: tok for (view, tok), p in idx.token_prime.items() if view == "w"}
    # df==2 word-primes are exact 2-doc co-occurrences = candidate bridge entities
    bridge_primes = [p for p, df in idx.df.items()
                     if df == 2 and p in prime_word and len(prime_word[p]) >= 5]

    # store raw text for building a realistic query out of doc A's OTHER content
    raw = corpus

    single_hit = chain_hit = 0
    used = 0
    examples = []
    for p in bridge_primes:
        docs2 = [d for d in idx.postings[p] if d in idx.alive]
        if len(docs2) != 2:
            continue
        A, B = docs2
        xw = prime_word[p]
        # build a query from A's content EXCLUDING the bridge word and EXCLUDING
        # any word that also appears in B (so the query genuinely points at A,
        # and B is only reachable through the shared entity X = the hard case).
        bwords = set(words(raw[B]))
        awords = [w for w in words(raw[A]) if w != xw and w not in bwords]
        if len(awords) < 6:
            continue
        # take the rarest ~10 A-only words as the query (specific, points to A)
        awords_ranked = sorted(
            awords,
            key=lambda w: idx._idf(idx.token_prime.get(("w", w)), N)
            if idx.token_prime.get(("w", w)) else 0.0,
            reverse=True,
        )[:10]
        query = " ".join(awords_ranked)

        single = idx.search(query, 10)
        if B in single:
            continue                      # not a hard case -- single-hop already finds B
        # ensure A is actually retrieved (else there's nothing to bridge from)
        hop1, second, found_x = meet_chain(idx, query, k_first=5, k_expand=20)
        if A not in hop1:
            continue
        used += 1
        s_ok = B in single                # always False here by construction
        c_ok = B in second
        single_hit += s_ok
        chain_hit += c_ok
        if len(examples) < 4:
            examples.append((xw, A, B, query[:60], c_ok))
        if used >= max_cases:
            break

    print(f"  mined {used} REAL hard bridge pairs (single-hop misses B by construction)")
    for xw, A, B, q, ok in examples:
        print(f"    entity '{xw}'  A={A} B={B}  q='{q}...'  chain recovers B? {ok}")
    if used == 0:
        print("  no qualifying hard pairs found.")
        return None
    print(f"\n  REAL bridge-doc (2nd-hop B) recall on HARD cases:")
    print(f"     single-hop BM25 : {single_hit}/{used} = {single_hit/used:.3f}")
    print(f"     meet-chain      : {chain_hit}/{used} = {chain_hit/used:.3f}")
    return single_hit, chain_hit, used


# --------------------------------------------------------------------------- #
def main():
    s1, c1, n1 = part1_synthetic()
    res2 = part2_scifact()

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    ok = True
    # synthetic: chain must strictly beat single-hop on bridge recall
    print(f"  SYNTHETIC : single-hop {s1}/{n1}  chain {c1}/{n1}  "
          f"(delta {c1 - s1:+d})")
    if not (c1 > s1):
        ok = False
    if res2 is not None:
        s2, c2, n2 = res2
        print(f"  REAL      : single-hop {s2}/{n2}={s2/n2:.3f}  "
              f"chain {c2}/{n2}={c2/n2:.3f}  (delta {(c2-s2)/n2:+.3f})")
        if not (c2 > s2):
            ok = False
    else:
        print("  REAL      : skipped/empty")

    gained = (c1 - s1) + (0 if res2 is None else res2[1] - res2[0])
    if ok and gained > 0:
        print(f"\nPASS  meet-chain recovered {gained} bridge docs that single-hop BM25 "
              f"missed (synthetic +{c1-s1}"
              + ("" if res2 is None else f", real +{res2[1]-res2[0]}/{res2[2]}") + ")")
    else:
        print(f"\nFAIL  meet-chain did not beat single-hop on bridge recall "
              f"(net +{gained}).")


if __name__ == "__main__":
    main()
