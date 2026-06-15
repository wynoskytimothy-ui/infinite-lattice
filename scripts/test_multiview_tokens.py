#!/usr/bin/env python3
"""
Test 56 - Multi-view tokenization across the gears (robust tokens & sub-words).

"Look at all the different ways we can make tokens and distribute them into
the rotating gears." A token need not be ONE thing. The same text can be cut
many ways at once - whole word, character n-grams, prefix, sub-word - and each
view rides its own gear (its own prime namespace). A term is then indexed
under ALL views, so it is findable through ANY of them.

This is the cure for the failure that breaks lexical RAG in the real world:
typos and variants. Whole-word matching collapses on a single typo;
character n-grams are robust; a prefix view catches truncations. Distribute
all of them across the gears and the union is robust where each alone is
brittle.

  views (gears):  word | char-trigram | char-bigram | prefix
  index:          term -> set of (view, token) prime addresses
  match:          shared (view, token) addresses across query and term (a meet)

Verified: under increasing typos, whole-word recall collapses, char-grams
stay high, and the multi-view union beats every single view.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# ---- the tokenization views (each a 'gear') ----

def v_word(s):
    return {("w", s)}


def v_tri(s):
    p = f"^{s}$"
    return {("3", p[i:i + 3]) for i in range(len(p) - 2)}


def v_bi(s):
    p = f"^{s}$"
    return {("2", p[i:i + 2]) for i in range(len(p) - 1)}


def v_prefix(s):
    return {("p", s[:4])}


VIEWS = {"word": [v_word], "char3": [v_tri], "prefix": [v_prefix],
         "multi": [v_word, v_tri, v_bi, v_prefix]}


def tokens(s, viewfns):
    out = set()
    for fn in viewfns:
        out |= fn(s)
    return out


def make_words(rng, n):
    cons, vowels = "bcdfghklmnprst", "aeiou"
    words = set()
    while len(words) < n:
        L = rng.randint(4, 8)
        w = "".join((rng.choice(cons) if i % 2 == 0 else rng.choice(vowels))
                    for i in range(L))
        words.add(w)
    return list(words)


def typo(s, k, rng):
    s = list(s)
    for _ in range(k):
        if not s:
            break
        op = rng.choice(["sub", "del", "ins", "swap"])
        i = rng.randrange(len(s))
        if op == "sub":
            s[i] = rng.choice("abcdefghiklmnoprstu")
        elif op == "del":
            s.pop(i)
        elif op == "ins":
            s.insert(i, rng.choice("abcdefghiklmnoprstu"))
        elif op == "swap" and i + 1 < len(s):
            s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def main():
    header("Multi-view tokenization across the gears - typo-robust retrieval")
    rng = random.Random(0x56E0)
    corpus = make_words(rng, 800)

    # pre-index the corpus under each view set
    index = {name: {w: tokens(w, fns) for w in corpus}
             for name, fns in VIEWS.items()}

    def recall_at(name, n_typos, k=1, trials=400):
        fns = VIEWS[name]
        idx = index[name]
        hit = 0
        for _ in range(trials):
            target = rng.choice(corpus)
            q = typo(target, n_typos, rng)
            qt = tokens(q, fns)
            # score every corpus term by shared (view, token) addresses (a meet)
            scored = sorted(corpus, key=lambda w: len(qt & idx[w]), reverse=True)
            if target in scored[:k]:
                hit += 1
        return hit / trials

    print("\nRecall@1 of the correct term under typos")
    print("-" * 72)
    print(f"  {'view (gear)':<12} | {'0 typos':>8} | {'1 typo':>7} | "
          f"{'2 typos':>8}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*7} | {'-'*8}")
    res = {}
    for name in VIEWS:
        r0 = recall_at(name, 0)
        r1 = recall_at(name, 1)
        r2 = recall_at(name, 2)
        res[name] = (r0, r1, r2)
        print(f"  {name:<12} | {r0*100:>7.0f}% | {r1*100:>6.0f}% | {r2*100:>7.0f}%")

    # whole-word collapses under typos; char-grams stay robust
    assertion(res["word"][1] < 0.15,
              "whole-word matching COLLAPSES on a single typo (exact token gone)")
    assertion(res["char3"][1] > 0.7,
              "char-trigram view stays robust to a typo (most n-grams survive)")
    # the multi-view union is best at every typo level
    assertion(all(res["multi"][i] >= max(res[v][i] for v in VIEWS) - 1e-9
                  for i in range(3)),
              "the multi-view union (all gears) matches or beats every single "
              "view at every typo level - distributing tokens across gears wins")
    assertion(res["multi"][2] > res["word"][2] + 0.4,
              "at 2 typos the multi-view gears crush whole-word matching - "
              "the robustness comes from indexing under many tokenizations")

    # ---- show the sub-word bridge: a variant shares sub-word gears ----
    print("\nSub-word bridge - morphological variants share gears")
    print("-" * 72)
    base = corpus[0]
    variant = base + "es"                  # a suffixed variant (not in corpus)
    shared = tokens(base, VIEWS["multi"]) & tokens(variant, VIEWS["multi"])
    print(f"  '{base}' vs variant '{variant}': {len(shared)} shared "
          f"(view,token) gears -> still matchable")
    assertion(len(shared) >= 3,
              "a morphological variant shares many sub-word gears with its base "
              "(prefix + most n-grams) - sub-words bridge inflections for free")

    header("RESULT")
    print(f"  recall@1 at 2 typos: word {res['word'][2]*100:.0f}%, "
          f"char3 {res['char3'][2]*100:.0f}%, multi {res['multi'][2]*100:.0f}%")
    print()
    print("  A token is not one thing. Cutting the same text many ways - word,")
    print("  trigram, bigram, prefix - and giving each its own gear (prime")
    print("  namespace) makes a term findable through ANY view. Whole-word")
    print("  retrieval dies on a typo; the multi-view union shrugs it off, and")
    print("  morphological variants share sub-word gears automatically.")
    print()
    print("  This is the tokenization design space the gears were built for:")
    print("  don't pick one tokenizer - run them all in parallel chambers and")
    print("  index under the union. For RAG it means robustness to typos,")
    print("  inflections, and spelling variants that lexical and even dense")
    print("  retrieval routinely miss - a direct, practical upgrade.")


if __name__ == "__main__":
    main()
