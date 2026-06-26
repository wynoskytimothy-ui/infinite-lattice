import random, time
from depth_measure_sketch import ReconSketch
rnd=random.Random(0)
d=50; extra=rnd.sample(range(2000,500000),d)
A=set(extra[:25])|set(range(100)); B=set(extra[25:])|set(range(100))
sd=(A-B)|(B-A)
for ov in (1.3,1.5,2.0):
    b=int(d*ov)
    sa,sb=ReconSketch(b,4),ReconSketch(b,4)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    t=time.time()
    a,bb,ok=sa.subtract(sb).decode()
    print(f"ov={ov} b={b}: ok={ok} exact={(a|bb)==sd} time={time.time()-t:.3f}s", flush=True)
