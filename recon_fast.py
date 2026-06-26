import random
from depth_measure_sketch import ReconSketch
def trial(d, overhead, k, seed):
    rnd=random.Random(seed)
    # small base so universe construction is cheap; d differences dominate the sketch anyway
    base=set(range(1000))
    extra=rnd.sample(range(2000, 100000), d)
    A=base|set(extra[:d//2]); B=base|set(extra[d//2:])
    sd=(A-B)|(B-A); b=int(d*overhead)
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd
for k in (3,4):
    print(f"k={k}, d=200, 30 trials:")
    for ov in (1.5,2.0,2.5,3.0):
        s=sum(trial(200,ov,k,seed) for seed in range(30))
        print(f"  overhead={ov}: {s}/30")
