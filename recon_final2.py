import random, time
from depth_measure_sketch import ReconSketch
def trial(d, overhead, k, seed):
    rnd=random.Random(seed)
    extra=rnd.sample(range(2000, 500000), d)
    A=set(extra[:d//2]) | set(range(100)); B=set(extra[d//2:]) | set(range(100))
    sd=(A-B)|(B-A); b=max(8,int(d*overhead))
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd
print("Decode success (30 trials) | d=symmetric-diff size, k=4 hashes")
print(f"{'d':>6} {'ov=1.3':>7} {'ov=1.5':>7} {'ov=2.0':>7}")
t0=time.time()
for d in (50,200):
    row=[]
    for ov in (1.3,1.5,2.0):
        s=sum(trial(d,ov,4,seed) for seed in range(30))
        row.append(f"{int(s/30*100):>5d}%")
    print(f"{d:>6} "+" ".join(row), flush=True)
print(f"({time.time()-t0:.1f}s)")
