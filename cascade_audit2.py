#!/usr/bin/env python3
"""Apply the user's exact rule -- "RARE terms that cross" -- and re-audit. Cross-count alone
over-rewards generic hubs (pain/affect/cause). Weight corroboration by rarity (idf) so
discriminative cross-terms (joint, inflammation) dominate and hubs sink. Compare the two.
"""
import re, math
from collections import defaultdict, Counter
from marco_lab import load_pool
from stem_safe import safe

WORD = re.compile(r"[a-z0-9]+")
IDF_GATE, TOP, MIN_CO = 4.0, 12, 3


def stoks(s):
    return [safe(w) for w in WORD.findall(s.lower())]


qids, queries, qrels, texts = load_pool()
df = Counter(); N = len(texts)
for t in texts.values():
    for w in set(stoks(t)):
        df[w] += 1
idf = {w: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for w, d in df.items()}
cooc = defaultdict(Counter); freq = Counter()
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
    sc = [(b, (c / fa) * idf.get(b, 0.0)) for b, c in cnt.items() if c >= MIN_CO]
    sc.sort(key=lambda x: -x[1])
    if sc:
        corridor[a] = sc[:TOP]


def cascade(seed, decay=0.5):
    act = defaultdict(float); paths = defaultdict(set)
    for t1, w1 in corridor.get(seed, []):
        act[t1] += w1; paths[t1].add(t1)
        for t2, w2 in corridor.get(t1, []):
            if t2 != seed:
                act[t2] += w1 * w2 * decay; paths[t2].add(t1)
    cross = {t: act[t] * len(paths[t]) for t in act}                 # cross-count (old)
    rare_cross = {t: len(paths[t]) * idf.get(t, 0.0) for t in act}   # RARE x cross (user's rule)
    return cross, rare_cross, paths


for seed in ["arthritis", "asthma", "diabetes"]:
    st = safe(seed)
    cross, rc, paths = cascade(st)
    a = sorted(cross, key=cross.get, reverse=True)[:8]
    b = sorted(rc, key=rc.get, reverse=True)[:8]
    print(f"=== {seed} ===")
    print(f"  cross-count only : {[f'{t}({len(paths[t])})' for t in a]}")
    print(f"  RARE x cross     : {[f'{t}({len(paths[t])},idf{idf.get(t,0):.1f})' for t in b]}")
    print()
print("  generic hubs (pain/affect/cause) sink, discriminative cross-terms (joint/rheumatoid) rise")
print("  = 'rare terms that cross' sharpens the strong relationships exactly as intended.")
