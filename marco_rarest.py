#!/usr/bin/env python3
"""How many docs does the RAREST word in each query touch (its postings size), and is the
gold doc among them? If the rarest word touches few docs AND the gold contains it, that one
word nearly identifies the answer -- the entity anchor.
"""
import re
from collections import Counter
from marco_lab import load_pool, Index
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
idx = Index(stoks).build(texts)

touched = []
gold_has = 0
buckets = Counter()
rare_idf = []
no_term = 0
gold_touch_when_present = []
for q in qids:
    R = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 2.0 and w in idx.post]
    if not R:
        no_term += 1
        continue
    rarest = max(R, key=lambda w: idx.idf[w])
    posts = len(idx.post[rarest])
    touched.append(posts)
    rare_idf.append(idx.idf[rarest])
    gold_stems = set()
    for pid in qrels[q]:
        gold_stems |= set(stoks(texts.get(pid, "")))
    present = rarest in gold_stems
    gold_has += present
    if present:
        gold_touch_when_present.append(posts)
    b = ("1-10" if posts <= 10 else "11-50" if posts <= 50 else "51-200" if posts <= 200
         else "201-1000" if posts <= 1000 else "1000+")
    buckets[b] += 1


def med(xs):
    s = sorted(xs); return s[len(s) // 2] if s else 0


n = len(touched)
print(f"rarest query word: how many docs it touches (300k pool, {n} queries with a rare term)\n")
print(f"  docs touched by the rarest word:   median {med(touched)}   mean {sum(touched)//n}")
print(f"  rarest word avg idf: {sum(rare_idf)/n:.1f}  (higher = rarer)")
print(f"  gold doc CONTAINS the rarest word: {gold_has} ({gold_has/n:.0%})")
print(f"  when it does, the rarest word touches median {med(gold_touch_when_present)} docs")
print(f"     -> the gold is 1 of ~{med(gold_touch_when_present)} docs from the rarest word ALONE\n")
print("  distribution (how many docs the rarest query word touches):")
for b in ["1-10", "11-50", "51-200", "201-1000", "1000+"]:
    c = buckets[b]
    print(f"     {b:>9}: {c:>5}  ({c/n:.0%})")
print("\n  rarest word touches few docs + gold contains it = one word nearly picks the answer.")
