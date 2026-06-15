#!/usr/bin/env python3
"""
Test 49 - Analogical reasoning: A:B :: C:? via the meet algebra (exact).

Word embeddings do analogies approximately (king - man + woman ~ queen, and
they get it right maybe 70% of the time). The lattice does the SAME analogy
EXACTLY and interpretably, because a concept is a set of feature-primes and
the "relation" between two concepts is a precise factor difference (a meet).

  CONCEPT     a composite of feature primes (royal, male, adult, human, ...)
  RELATION    A:B = the features removed (A\\B) and added (B\\A) - read off
              by factoring A/gcd and B/gcd (the meet)
  SOLVE       A:B :: C:D  =>  D = (C minus removed) plus added
  ODD-ONE-OUT the concept NOT sharing the common factor (gcd) of a group
  SAME-RELATION  two pairs are analogous iff their (removed, added) match

Verified: exact analogy completion on a feature ontology, odd-one-out via
gcd, and relation-equivalence grouping - cognition as prime arithmetic.
"""

from __future__ import annotations

import sys
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


# feature -> prime
FEATURES = ["royal", "common", "male", "female", "adult", "child",
            "human", "animal", "canine", "feline", "big", "small"]
P = {f: chain_primes(40)[i] for i, f in enumerate(FEATURES)}


def concept(*feats):
    c = 1
    for f in feats:
        c *= P[f]
    return c


# a small ontology
C = {
    "king":     concept("royal", "male", "adult", "human"),
    "queen":    concept("royal", "female", "adult", "human"),
    "man":      concept("common", "male", "adult", "human"),
    "woman":    concept("common", "female", "adult", "human"),
    "prince":   concept("royal", "male", "child", "human"),
    "princess": concept("royal", "female", "child", "human"),
    "boy":      concept("common", "male", "child", "human"),
    "girl":     concept("common", "female", "child", "human"),
    "dog":      concept("animal", "canine", "adult", "big"),
    "puppy":    concept("animal", "canine", "child", "small"),
    "cat":      concept("animal", "feline", "adult", "small"),
    "kitten":   concept("animal", "feline", "child", "small"),
}
NAME = {v: k for k, v in C.items()}


def features_of(comp):
    return {f for f, p in P.items() if comp % p == 0}


def relation(a, b):
    """The A:B relation as (removed features, added features) - a meet read."""
    fa, fb = features_of(a), features_of(b)
    return (frozenset(fa - fb), frozenset(fb - fa))


def apply_relation(c, rel):
    removed, added = rel
    fc = features_of(c)
    fd = (fc - removed) | added
    comp = 1
    for f in fd:
        comp *= P[f]
    return comp


def main():
    header("Analogical reasoning - A:B :: C:? via the meet algebra")

    # ------------------------------------------------------------------
    print("\nExact analogy completion (A:B :: C:D)")
    print("-" * 72)
    analogies = [
        ("king", "man", "queen", "woman"),     # royal -> common
        ("king", "queen", "man", "woman"),     # male -> female
        ("king", "prince", "queen", "princess"),  # adult -> child
        ("dog", "puppy", "cat", "kitten"),     # adult/big -> child/small
        ("boy", "girl", "prince", "princess"), # male -> female
        ("man", "king", "woman", "queen"),     # common -> royal (reverse)
    ]
    correct = 0
    for a, b, c, expected in analogies:
        rel = relation(C[a], C[b])
        d_comp = apply_relation(C[c], rel)
        d_name = NAME.get(d_comp, "?")
        ok = d_name == expected
        correct += ok
        rstr = (("-" + ",".join(sorted(rel[0]))) if rel[0] else "") + \
               (("+" + ",".join(sorted(rel[1]))) if rel[1] else "")
        print(f"  {a}:{b} :: {c}:{d_name:<9} [{rstr}]  "
              f"{'OK' if ok else 'expected ' + expected}")
        assertion(ok, f"{a}:{b}::{c}:{expected} solved exactly")
    print(f"  {correct}/{len(analogies)} analogies solved exactly")
    assertion(correct == len(analogies),
              "every analogy completed EXACTLY - the relation is a precise "
              "factor difference, not an approximate vector offset")

    # ------------------------------------------------------------------
    print("\nOdd-one-out via the common factor (gcd)")
    print("-" * 72)
    groups = [
        (["king", "queen", "prince", "dog"], "dog"),     # human vs animal
        (["dog", "puppy", "cat", "king"], "king"),       # animal vs human
        (["king", "prince", "boy", "queen"], "queen"),   # male vs female
    ]
    for members, outlier in groups:
        comps = [C[m] for m in members]
        # the common factor of the MAJORITY: gcd of all is too weak, so find
        # the feature shared by all but one (the outlier breaks the common set)
        g_all = comps[0]
        for x in comps[1:]:
            g_all = gcd(g_all, x)
        # outlier = the member whose removal maximizes the shared gcd
        best, best_g = None, 0
        for i, m in enumerate(members):
            rest = [comps[j] for j in range(len(comps)) if j != i]
            gg = rest[0]
            for x in rest[1:]:
                gg = gcd(gg, x)
            if gg > best_g:
                best_g, best = gg, m
        shared = features_of(best_g)
        print(f"  {members} -> outlier '{best}' "
              f"(rest share {sorted(shared)})")
        assertion(best == outlier,
                  f"gcd identifies '{outlier}' as the odd one out")

    # ------------------------------------------------------------------
    print("\nSame-relation detection (analogy grouping)")
    print("-" * 72)
    pairs = [("king", "man"), ("queen", "woman"), ("prince", "boy"),
             ("king", "queen"), ("man", "woman")]
    rels = {}
    for a, b in pairs:
        r = relation(C[a], C[b])
        rels.setdefault(r, []).append(f"{a}:{b}")
    print(f"  distinct relations found: {len(rels)}")
    for r, members in rels.items():
        rstr = (("-" + ",".join(sorted(r[0]))) if r[0] else "") + \
               (("+" + ",".join(sorted(r[1]))) if r[1] else "")
        print(f"    [{rstr}] : {members}")
    # king:man, queen:woman, prince:boy all share royal->common
    royal_common = [m for r, ms in rels.items() if r == (frozenset({"royal"}),
                    frozenset({"common"})) for m in ms]
    assertion(set(royal_common) == {"king:man", "queen:woman", "prince:boy"},
              "pairs with the same relation group together (analogy classes "
              "are equal meets) - king:man :: queen:woman :: prince:boy")

    header("RESULT")
    print(f"  {correct}/{len(analogies)} analogies solved EXACTLY (word2vec is")
    print(f"  approximate and ~70% on such tasks); odd-one-out via gcd; same-")
    print(f"  relation pairs grouped by equal meets.")
    print()
    print("  Analogical reasoning - the engine of human cognition and the")
    print("  party trick of word embeddings - is exact prime arithmetic here:")
    print("  the relation between two concepts is a factor difference (a meet),")
    print("  applying it is multiply/divide, and the answer is exact and")
    print("  interpretable. The same meet that scheduled tasks and found")
    print("  sunflowers now reasons by analogy.")


if __name__ == "__main__":
    main()
