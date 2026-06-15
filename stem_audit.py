#!/usr/bin/env python3
"""SLOW DOWN -- audit what stemming ACTUALLY does to words before climbing further.
The +5.1% average can hide damage: words over-collapsing (losing specificity) or failing
to merge (missing correlations). Inspect the real merges, no retrieval averages.
"""
import re
from collections import defaultdict, Counter
from marco_lab import load_pool

WORD = re.compile(r"[a-z0-9]+")
SUF = ("ization", "ational", "tion", "ment", "ness", "ing", "ed", "es", "ly", "s")


def _stem(w):
    for suf in SUF:
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


qids, queries, qrels, texts = load_pool()
stem2words = defaultdict(Counter)
wf = Counter()
for t in texts.values():
    for w in WORD.findall(t.lower()):
        wf[w] += 1
        stem2words[_stem(w)][w] += 1

nw, ns = len(wf), len(stem2words)
print(f"vocabulary: {nw} words -> {ns} stems  (collapsed {1 - ns/nw:.0%})\n")

print("WORST OVER-SPREADS -- one stem swallowing many DISTINCT words (rare = bigger risk):")
spread = sorted(((len(ws), st, ws) for st, ws in stem2words.items() if len(ws) >= 3), reverse=True)
for n, st, ws in spread[:18]:
    print(f"   {st:>12}  <- {[w for w, _ in ws.most_common(7)]}")

print("\nPROBE specific words -- does it merge RIGHT, or over-strip into a different word?")
probe = ["diabetes", "diabetic", "diabetics", "organization", "organ", "organic",
         "united", "unit", "pressure", "press", "analysis", "analyses",
         "heart", "hearts", "running", "run", "address", "addresses"]
for w in probe:
    st = _stem(w)
    co = [x for x, _ in stem2words[st].most_common(6)]
    flag = ""
    if st in wf and st != w and len(w) - len(st) >= 4:
        flag = "  <-- OVER-STRIP onto a real different word"
    print(f"   {w:>13} -> '{st}'   merged: {co}{flag}")

bad_merges = sum(1 for st, ws in stem2words.items()
                 if st in wf and len(ws) > 1 and any(len(w) - len(st) >= 4 for w in ws))
lost_merges = sum(1 for w in ["diabetes", "studies", "countries", "analyses", "matrices"]
                  if _stem(w) != _stem(w.rstrip("es") + "ic") and True)
print(f"\nover-strips (long word collapsed onto a different short real word): ~{bad_merges} stems")
print("=> verify: keep ONLY safe inflectional merges (plurals, tenses); drop derivational"
      " (ization/ment/ation) that change meaning. then re-measure rung 2 honestly.")
