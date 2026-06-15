#!/usr/bin/env python3
"""Query<->gold-doc lexical overlap distribution: how many queries share >=1,2,3,4,6 content
words with their gold doc. Low overlap = needs semantic bridging (company/encyclopedia);
high overlap = raw word-match (BM25) territory. Counts over CONTENT stems (idf>=2, no stopwords)
and RARE stems (idf>=4, the discriminative triggers).
"""
import re, math
from collections import Counter
from marco_lab import load_pool
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")


def stoks(s):
    return set(safe(w) for w in WORD.findall(s.lower()))


qids, queries, qrels, texts = load_pool()
N = len(texts)
df = Counter()
for t in texts.values():
    for w in stoks(t):
        df[w] += 1
idf = {w: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}

THRESHOLDS = [1, 2, 3, 4, 6]


def overlap_hist(gate):
    content = {t: (idf.get(t, 0) >= gate) for t in idf}
    counts = Counter()
    zero = 0
    total_words = 0
    for q in qids:
        qw = {w for w in stoks(queries[q]) if idf.get(w, 0) >= gate}
        total_words += len(qw)
        # best gold doc by overlap
        best = 0
        for pid in qrels[q]:
            gw = {w for w in stoks(texts.get(pid, "")) if idf.get(w, 0) >= gate}
            best = max(best, len(qw & gw))
        if best == 0:
            zero += 1
        for th in THRESHOLDS:
            if best >= th:
                counts[th] += 1
    return counts, zero, total_words / max(1, len(qids))


nq = len(qids)
print(f"query<->gold lexical overlap, {nq} queries\n")
for label, gate in [("CONTENT words (idf>=2)", 2.0), ("RARE words (idf>=4)", 4.0)]:
    counts, zero, avgw = overlap_hist(gate)
    print(f"  {label}  (avg {avgw:.1f} such words per query):")
    print(f"     0 shared (pure semantic gap): {zero:>5}  ({zero/nq:.0%})")
    for th in THRESHOLDS:
        c = counts[th]
        print(f"     >= {th} word{'s' if th>1 else ' '} to gold:        {c:>5}  ({c/nq:.0%})")
    print()
print("  high overlap = raw word-match solvable; low/zero = needs company + encyclopedia.")
