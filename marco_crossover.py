#!/usr/bin/env python3
"""How discriminating are the crossovers? For each query's terms, how many docs share >=2
of them (2-way cross) and >=3 (3-way cross). Small 3-way set = the meet narrows to a handful
(strong discrimination); large = the meet alone can't pick the doc. Plus: is the gold doc
inside the 2-way / 3-way set?
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
pid2di = {pid: di for di, pid in enumerate(idx.docids)}


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else 0


def crossover(gate):
    n1 = []; n2 = []; n3 = []; g1 = g2 = g3 = 0
    for q in qids:
        R = [w for w in set(stoks(queries[q])) if idx.idf.get(w, 0) >= gate and w in idx.post]
        cnt = Counter()
        for w in R:
            for di, _ in idx.post[w]:
                cnt[di] += 1
        sz = Counter(cnt.values())
        n1.append(len(cnt))
        n2.append(sum(v for k, v in sz.items() if k >= 2))
        n3.append(sum(v for k, v in sz.items() if k >= 3))
        gdi = [pid2di[pid] for pid in qrels[q] if pid in pid2di]
        gc = max((cnt.get(d, 0) for d in gdi), default=0)
        g1 += gc >= 1; g2 += gc >= 2; g3 += gc >= 3
    return n1, n2, n3, g1, g2, g3


nq = len(qids)
print(f"crossover discrimination over the 300k pool, {nq} queries\n")
for label, gate in [("CONTENT terms (idf>=2)", 2.0), ("RARE terms (idf>=4)", 4.0)]:
    n1, n2, n3, g1, g2, g3 = crossover(gate)
    print(f"  {label}:")
    print(f"     docs sharing >=1 term   median {med(n1):>6}   mean {sum(n1)//nq:>6}")
    print(f"     docs sharing >=2 (2-way) median {med(n2):>6}   mean {sum(n2)//nq:>6}")
    print(f"     docs sharing >=3 (3-way) median {med(n3):>6}   mean {sum(n3)//nq:>6}")
    print(f"     gold doc in 1-way {g1}({g1/nq:.0%})  2-way {g2}({g2/nq:.0%})  3-way {g3}({g3/nq:.0%})\n")
print("  small 3-way set + gold inside it = the meet narrows to ~the answer (strong discrimination).")
print("  (full 8.8M would have ~30x more docs per set, but the gold-in-set % is corpus-independent.)")
