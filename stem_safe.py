#!/usr/bin/env python3
"""Conservative stemming: GRAMMAR ONLY (plurals + -ing/-ed), never derivational
(tion/ment/ization/ity...). Preserves word specificity; semantic links (diabetes<->diabetic)
are the CORRIDOR's job, not stemming's. Re-audit (confirm over-spreads gone) + re-measure
rung 2 honestly vs the crude stemmer.
"""
import re, gc
from collections import defaultdict, Counter
from marco_lab import load_pool, Index, evaluate

WORD = re.compile(r"[a-z0-9]+")
CRUDE_SUF = ("ization", "ational", "tion", "ment", "ness", "ing", "ed", "es", "ly", "s")


def crude(w):
    for suf in CRUDE_SUF:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def safe(w):
    """Inflectional only. Derivational suffixes are LEFT INTACT (organization stays organization)."""
    if len(w) <= 4:
        return w
    if w.endswith("ies") and len(w) > 5:
        w = w[:-3] + "y"                                   # studies -> study
    elif w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]                                         # addresses -> address, dishes -> dish
    elif w.endswith("s") and not w.endswith(("ss", "us", "is", "ous", "ics")):
        w = w[:-1]                                         # drugs -> drug  (but glass/virus/analysis kept)
    if w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2]:
            w = w[:-1]                                     # running -> runn -> run
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2]:
            w = w[:-1]
    return w


def tok(fn):
    return lambda s: [fn(w) for w in WORD.findall(s.lower())]


if __name__ == "__main__":
    qids, queries, qrels, texts = load_pool()

    # re-audit the SAFE stemmer on the meaning-changing words
    s2w = defaultdict(set)
    for t in texts.values():
        for w in WORD.findall(t.lower()):
            s2w[safe(w)].add(w)
    print("RE-AUDIT safe stemmer (should NOT over-strip, SHOULD merge plurals/tenses):")
    for w in ["organization", "organ", "united", "unit", "pressure", "press",
              "running", "run", "address", "addresses", "diabetes", "diabetic", "secret", "secrete"]:
        st = safe(w)
        print(f"   {w:>13} -> '{st}'   merged: {sorted(s2w[st])[:6]}")

    print("\nRE-MEASURE rung 2 honestly (MRR@10; floor words = 0.5419):")
    ref = None
    for label, fn in [("words", lambda w: w), ("crude stem", crude), ("safe stem", safe)]:
        idx = Index(tok(fn)).build(texts)
        res = evaluate(idx, qids, queries, qrels, ref_rr=ref, label=label)
        if ref is None:
            ref = res["rrs"]
        del idx
        gc.collect()
    print("\n  safe >= crude with the over-spreads GONE = a representation we can trust to build on.")
