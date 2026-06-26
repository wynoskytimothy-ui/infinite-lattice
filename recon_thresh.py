import random
from depth_measure_sketch import ReconSketch
def trial(d, overhead, k, seed, set_size=5000):
    rnd=random.Random(seed); universe=set_size*5
    base=set(rnd.sample(range(universe),set_size))
    pool=[x for x in range(universe) if x not in base]; rnd.shuffle(pool)
    A=base|set(pool[:d//2]); B=base|set(pool[d//2:d])
    sd=(A-B)|(B-A); b=int(d*overhead)
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd
print("k=3, d=200, 40 trials each:")
for ov in (1.5,2.0,2.5,3.0,3.5):
    s=sum(trial(200,ov,3,seed) for seed in range(40))
    print(f"  overhead={ov}: {s}/40 = {s/40*100:.0f}%")
print("k=4, d=200, 40 trials each:")
for ov in (1.5,2.0,2.5,3.0):
    s=sum(trial(200,ov,4,seed) for seed in range(40))
    print(f"  overhead={ov}: {s}/40 = {s/40*100:.0f}%")
