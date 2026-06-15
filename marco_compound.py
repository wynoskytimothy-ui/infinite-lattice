#!/usr/bin/env python3
"""For queries with <2 rare words (no native 2-way crossover), does compounding the RAREST
word with a MEDIUM-frequency word (rarest x medium = a P*Q composite) manufacture the 2-way
discrimination? Measure: among the ~76 docs the rarest word touches, how few also contain the
best medium word (the compound set), and is the gold doc in it.
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
pid2di = {pid: di for di, pid in enumerate(idx.docids)}


def med(xs):
    s = sorted(xs); return s[len(s) // 2] if s else 0


rare_alone = []; compound = []; gold_in_rare = 0; gold_in_comp = 0; npop = 0
for q in qids:
    qs = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= 0.3 and w in idx.post]
    rare = [w for w in qs if idx.idf.get(w, 0) >= RARE]
    medium = [w for w in qs if MED_LO <= idx.idf.get(w, 0) < MED_HI]
    if len(rare) != 1 or not medium:
        continue                                          # focus: exactly 1 rare word + a medium word
    npop += 1
    R = rare[0]
    cand = [di for di, _ in idx.post[R]]                  # ~76 docs touched by the rarest word
    cstem = {di: set(stoks(texts[idx.docids[di]])) for di in cand}
    rare_alone.append(len(cand))
    gdi = {pid2di[pid] for pid in qrels[q] if pid in pid2di}
    gir = bool(gdi & set(cand)); gold_in_rare += gir
    best = len(cand); bestM = None
    for M in medium:
        c = sum(1 for di in cand if M in cstem[di])
        if 0 < c < best:
            best = c; bestM = M
    compound.append(best)
    if bestM is not None and any((di in gdi) and (bestM in cstem[di]) for di in cand):
        gold_in_comp += 1

n = npop
print(f"queries with exactly 1 rare word + a medium word: {n} ({n/len(qids):.0%} of pool)\n")
print(f"  rarest word alone:        median {med(rare_alone):>5} docs   gold inside {gold_in_rare} ({gold_in_rare/n:.0%})")
print(f"  rarest x medium COMPOUND: median {med(compound):>5} docs   gold inside {gold_in_comp} ({gold_in_comp/n:.0%})")
print(f"\n  the compound (P*Q of rarest x medium) manufactures the 2-way: {med(rare_alone)} -> {med(compound)} docs,")
print(f"  keeping the gold {gold_in_comp/max(1,gold_in_rare)*100:.0f}% of the time it was reachable by the rare word.")
