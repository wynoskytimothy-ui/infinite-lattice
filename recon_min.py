import random, time
from depth_measure_sketch import ReconSketch
t0=time.time()
def trial(d, overhead, k, seed):
    rnd=random.Random(seed)
    base=set(range(500))
    extra=rnd.sample(range(2000, 100000), d)
    A=base|set(extra[:d//2]); B=base|set(extra[d//2:])
    sd=(A-B)|(B-A); b=int(d*overhead)
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd
for ov in (1.5,2.0,2.5,3.0):
    s=sum(trial(200,ov,3,seed) for seed in range(10))
    print(f"k=3 overhead={ov}: {s}/10  ({time.time()-t0:.1f}s elapsed)", flush=True)
