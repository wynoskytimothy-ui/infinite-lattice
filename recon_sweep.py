"""Sweep IBLT overhead factor + hash count to find where erasure-decode
(missing-member peel) recovers the EXACT symmetric difference, with many trials."""
import random
from depth_measure_sketch import ReconSketch

def trial(set_size, d, overhead, k, seed):
    rnd = random.Random(seed)
    universe = max(200000, set_size * 5)
    base = set(rnd.sample(range(universe), set_size))
    pool = list(set(range(universe)) - base)
    rnd.shuffle(pool)
    onlyA = set(pool[:d // 2]); onlyB = set(pool[d // 2:d])
    A = base | onlyA; B = base | onlyB
    true_sd = (A - B) | (B - A)
    b = max(16, int(d * overhead))
    sa, sb = ReconSketch(b, k), ReconSketch(b, k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    recA, recB, ok = sa.subtract(sb).decode()
    rec = recA | recB
    return ok and rec == true_sd

print(f"{'k':>2} {'overhead':>9} {'|symdiff|':>9} {'success_rate(50 trials)':>24}")
for k in (3, 4):
    for overhead in (1.5, 2.0, 2.5, 3.0):
        for d in (50, 200, 1000):
            ok = sum(trial(40000, d, overhead, k, s) for s in range(50))
            print(f"{k:>2} {overhead:>9} {d:>9} {ok/50*100:>22.0f}%")
    print()
