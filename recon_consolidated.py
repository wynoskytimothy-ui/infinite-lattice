import random
from depth_measure_sketch import ReconSketch
def trial(d, overhead, k, seed):
    rnd=random.Random(seed)
    extra=rnd.sample(range(2000,500000), d)
    A=set(extra[:d//2])|set(range(50)); B=set(extra[d//2:])|set(range(50))
    sd=(A-B)|(B-A); b=max(8,int(d*overhead))
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd, b
out=[]
out.append("RECONCILIATION DECODE SUCCESS (20 trials each), k=4 hashes")
out.append(f"{'|symdiff|':>10} {'ov=1.5':>7} {'ov=2.0':>7}  bytes@ov2.0")
for d in (50,200,500):
    cells=[]; ok15=ok20=0; bsz=0
    for seed in range(20):
        r15,_=trial(d,1.5,4,seed); ok15+=r15
        r20,b=trial(d,2.0,4,seed); ok20+=r20; bsz=b*24
    out.append(f"{d:>10} {ok15*5:>5d}% {ok20*5:>5d}%  {bsz//1024 or bsz}{'KB' if bsz>=1024 else 'B'}")
print("\n".join(out))
