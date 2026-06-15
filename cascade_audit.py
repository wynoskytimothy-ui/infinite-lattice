#!/usr/bin/env python3
"""Verify the CASCADE WALK before it scores anything. Spreading activation through the
corridor: seed -> hop1 -> hop2, with decay, and CROSS-REINFORCEMENT (a term reached by
MULTIPLE paths is corroborated = a strong relationship; single-path = drift, decays).
Audit: does `arthritis` cascade to a TIGHT reinforced set (joint/inflammation/cartilage lit
by several paths) while unrelated terms stay weak? Compare single-hop vs cascade+cross.
"""
import re, math, time
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
print(f"corridor ready: {len(corridor)} anchors\n")


def cascade(seed, decay=0.5, topk=12):
    """hop1 + hop2 with decay; cross = # distinct hop-1 paths reaching a term (corroboration)."""
    act = defaultdict(float)
    paths = defaultdict(set)
    for t1, w1 in corridor.get(seed, []):
        act[t1] += w1
        paths[t1].add(t1)
        for t2, w2 in corridor.get(t1, []):
            if t2 == seed:
                continue
            act[t2] += w1 * w2 * decay
            paths[t2].add(t1)
    score = {t: act[t] * len(paths[t]) for t in act}        # cross-reinforcement
    top = sorted(score, key=score.get, reverse=True)[:topk]
    return top, paths


for seed in ["arthritis", "diabetes", "asthma", "insulin"]:
    st = safe(seed)
    hop1 = [b for b, _ in corridor.get(st, [])][:8]
    top, paths = cascade(st)
    print(f"=== {seed} ===")
    print(f"  single-hop : {hop1}")
    print(f"  cascade+cross (term×paths):")
    for t in top[:10]:
        npaths = len(paths[t])
        mark = " <<< CROSS" if npaths >= 2 else ""
        print(f"      {t:<16} paths={npaths}{mark}")
    print()
print("  tight reinforced set + cross-terms (joint/inflammation) on top + drift decayed = cascade works.")
