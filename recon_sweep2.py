import random
from depth_measure_sketch import ReconSketch

def trial(set_size, d, overhead, k, seed):
    rnd = random.Random(seed)
    universe = set_size * 5
    base = set(rnd.sample(range(universe), set_size))
    pool = list(set(range(universe)) - base); rnd.shuffle(pool)
    onlyA = set(pool[:d//2]); onlyB = set(pool[d//2:d])
    A = base | onlyA; B = base | onlyB
    true_sd = (A-B)|(B-A)
    b = max(16, int(d*overhead))
    sa, sb = ReconSketch(b,k), ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    recA, recB, ok = sa.subtract(sb).decode()
    return ok and (recA|recB)==true_sd

print(f"{'k':>2}{'overhead':>9}{'d':>6}{'success(30 trials)':>20}")
for k in (3,4):
  for overhead in (1.5, 2.0, 3.0):
    for d in (50, 500):
      ok = sum(trial(5000, d, overhead, k, s) for s in range(30))
      print(f"{k:>2}{overhead:>9}{d:>6}{ok}/30")
  print()
