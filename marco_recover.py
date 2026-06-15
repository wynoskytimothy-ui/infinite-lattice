#!/usr/bin/env python3
"""Recovery cascade. Rarest word R1 is in the gold ~89%. The 2nd rare word R2 is often NOT
(the missing ~30%). For that R1-yes / R2-no population, how many golds are recovered by:
  (a) a SUBWORD of R2  (shares a 5-char prefix -- diabetes<->diabetic), or
  (b) a MEDIUM query word present (the rare x medium compound path).
"""
import re
from collections import Counter
from marco_lab import load_pool, Index
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
RARE, MED_LO, MED_HI = 4.0, 1.5, 4.0


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)

n2 = 0; r1_in = 0; r2_in = 0
pop = 0; rec_sub = 0; rec_med = 0; rec_either = 0
for q in qids:
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3]
    rare = sorted([w for w in qs if idx.idf.get(w, 0) >= RARE], key=lambda w: -idx.idf[w])
    if len(rare) < 2:
        continue
    n2 += 1
    R1, R2 = rare[0], rare[1]
    gold = set()
    for pid in qrels[q]:
        gold |= set(stoks(texts.get(pid, "")))
    r1 = R1 in gold; r2 = R2 in gold
    r1_in += r1; r2_in += r2
    if r1 and not r2:                                    # the missing-2nd-rare population
        pop += 1
        sub = len(R2) >= 5 and any(len(w) >= 5 and w != R2 and w[:5] == R2[:5] for w in gold)
        medium = [w for w in qs if MED_LO <= idx.idf.get(w, 0) < MED_HI]
        med = any(M in gold for M in medium)
        rec_sub += sub
        rec_med += med
        rec_either += (sub or med)

print(f"queries with >=2 rare words: {n2}\n")
print(f"  rarest R1 in gold: {r1_in} ({r1_in/n2:.0%})")
print(f"  2nd rare R2 in gold: {r2_in} ({r2_in/n2:.0%})")
print(f"\n  R1-in / R2-missing population: {pop} ({pop/n2:.0%} of 2-rare queries)")
print(f"     recovered by SUBWORD of R2 (5-char prefix): {rec_sub} ({rec_sub/max(1,pop):.0%})")
print(f"     recovered by a MEDIUM word present:        {rec_med} ({rec_med/max(1,pop):.0%})")
print(f"     recovered by EITHER:                       {rec_either} ({rec_either/max(1,pop):.0%})")
print(f"\n  so of the 2nd-rare-word misses, the cascade recovers {rec_either/max(1,pop):.0%}"
      f" via subword/medium -- the rest need the company/encyclopedia.")
