import random
from depth_measure_sketch import ReconSketch, cells_of
def trial(d, overhead, k, seed):
    rnd=random.Random(seed)
    base=set(range(500))
    extra=rnd.sample(range(2000, 500000), d)
    A=base|set(extra[:d//2]); B=base|set(extra[d//2:])
    sd=(A-B)|(B-A); b=max(8,int(d*overhead))
    sa,sb=ReconSketch(b,k),ReconSketch(b,k)
    for x in A: sa.add(x)
    for x in B: sb.add(x)
    a,bb,ok=sa.subtract(sb).decode()
    return ok and (a|bb)==sd
print("Decode success rate (50 trials), k=4 hashes:")
print(f"{'d':>6} " + " ".join(f"ov={o}" for o in (1.3,1.5,2.0,2.5)))
for d in (50,200,1000):
    row=[]
    for ov in (1.3,1.5,2.0,2.5):
        s=sum(trial(d,ov,4,seed) for seed in range(50))
        row.append(f"{s*2:>4d}%")
    print(f"{d:>6} "+" ".join(row))
print("\nk=3 hashes:")
print(f"{'d':>6} " + " ".join(f"ov={o}" for o in (1.3,1.5,2.0,2.5)))
for d in (50,200,1000):
    row=[]
    for ov in (1.3,1.5,2.0,2.5):
        s=sum(trial(d,ov,3,seed) for seed in range(50))
        row.append(f"{s*2:>4d}%")
    print(f"{d:>6} "+" ".join(row))
