#!/usr/bin/env python3
"""Verify the CORRELATIONS are working right -- build the corridor on the clean (conservative-
stem) base and AUDIT it at the word level before it scores anything. Does `insulin` find
`glucose/diabetes/pancreas`? Does `diabetic` link to `diabetes` now (via correlation, the link
stemming correctly refused to force)? Sensible company, or junk? Inspect, don't average.
"""
import re, math, time
from collections import defaultdict, Counter
from marco_lab import load_pool
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
IDF_GATE, TOP, MIN_CO = 4.0, 10, 3


def stoks(text):
    return [safe(w) for w in WORD.findall(text.lower())]


qids, queries, qrels, texts = load_pool()

t0 = time.perf_counter()
df = Counter()
N = len(texts)
for t in texts.values():
    for w in set(stoks(t)):
        df[w] += 1
idf = {w: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}

cooc = defaultdict(Counter)
freq = Counter()
for t in texts.values():
    rare = [w for w in set(stoks(t)) if idf.get(w, 0) >= IDF_GATE]
    for a in rare:
        freq[a] += 1
    for a in rare:
        ca = cooc[a]
        for b in rare:
            if a != b:
                ca[b] += 1
corridor = {}
for a, cnt in cooc.items():
    fa = freq[a]
    scored = [(b, (c / fa) * idf.get(b, 0.0)) for b, c in cnt.items() if c >= MIN_CO]
    scored.sort(key=lambda x: -x[1])
    if scored:
        corridor[a] = scored[:TOP]
print(f"corridor on clean stem base: {len(corridor)} anchor terms ({time.perf_counter()-t0:.0f}s)\n")

print("WORD-LEVEL CORRELATION AUDIT -- right company, or junk?")
probe = ["insulin", "diabetes", "diabetic", "glucose", "cancer", "tumor", "chemotherapy",
         "vaccine", "asthma", "metformin", "cholesterol", "pregnancy", "antibiotic",
         "seizure", "migraine", "arthritis", "thyroid", "vitamin"]
for w in probe:
    st = safe(w)
    co = corridor.get(st)
    print(f"   {w:>13} -> " + (", ".join(b for b, _ in co[:8]) if co else "(no corridor)"))

print("\nSANITY -- are known-related partners actually in each other's corridor?")
pairs = [("insulin", "glucose"), ("diabetes", "insulin"), ("vaccine", "immune"),
         ("cancer", "tumor"), ("asthma", "lung"), ("cholesterol", "heart")]
for a, b in pairs:
    ca = {x for x, _ in corridor.get(safe(a), [])}
    cb = {x for x, _ in corridor.get(safe(b), [])}
    hit = "LINKED" if (safe(b) in ca or safe(a) in cb) else "(not linked)"
    print(f"   {a} <-> {b}: {hit}")
print("\n  sensible company + known pairs linked = correlations work -> THEN measure retrieval.")
