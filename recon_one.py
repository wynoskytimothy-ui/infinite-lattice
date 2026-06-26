import random, time
from depth_measure_sketch import ReconSketch
rnd = random.Random(7)
set_size, d = 5000, 200
universe = set_size*5
base = set(rnd.sample(range(universe), set_size))
pool = [x for x in range(universe) if x not in base]; rnd.shuffle(pool)
onlyA=set(pool[:d//2]); onlyB=set(pool[d//2:d])
A=base|onlyA; B=base|onlyB
true_sd=(A-B)|(B-A)
for overhead in (1.5, 2.0, 3.0):
    for k in (3,4):
        b=int(d*overhead)
        sa,sb=ReconSketch(b,k),ReconSketch(b,k)
        for x in A: sa.add(x)
        for x in B: sb.add(x)
        t=time.time()
        recA,recB,ok=sa.subtract(sb).decode()
        rec=recA|recB
        print(f"k={k} overhead={overhead} b={b} recovered={len(rec)}/{len(true_sd)} clean={ok} EXACT={rec==true_sd} ({time.time()-t:.2f}s)")
