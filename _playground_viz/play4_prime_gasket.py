"""Play 4: tie it to PRIMES + count check.

(1) Count check: #Sierpinski points in [0,2^k)^2 (min<=max) should follow 3^k growth.
(2) Prime overlay: among prime pairs (a,p), which sit ON the gasket ((a&p)==0)?
    Plot prime meet-nodes, color = on-gasket vs off, to see the fractal carve
    the prime field.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (1) count: full square (a,p in [0,2^k)) carrying (a&p)==0 is exactly 3^k (Stern/Lucas)
for k in range(1, 9):
    G = 1 << k
    A, Pm = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
    c = int(((A & Pm) == 0).sum())
    print(f"k={k}: 2^k={G:4d}  #(a&p==0)={c:6d}  3^k={3**k:6d}  match={c==3**k}")

# (2) prime overlay
def primes_up_to(N):
    s=np.ones(N+1,bool); s[:2]=False
    for i in range(2,int(N**0.5)+1):
        if s[i]: s[i*i::i]=False
    return np.flatnonzero(s)
P = primes_up_to(512)
xs_on, ys_on, xs_off, ys_off = [], [], [], []
for i,a in enumerate(P):
    for p in P[i:]:
        X=int(a+p); Y=int(min(a,p))
        if (int(a) & int(p))==0:
            xs_on.append(X); ys_on.append(Y)
        else:
            xs_off.append(X); ys_off.append(Y)
print(f"prime pairs on-gasket {len(xs_on)} / total {len(xs_on)+len(xs_off)}")
fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(xs_off, ys_off, s=2, c="lightgray", label="off-gasket (a&p)!=0")
ax.scatter(xs_on, ys_on, s=6, c="crimson", label="on-gasket (a&p)==0")
ax.set_xlabel("X=a+p"); ax.set_ylabel("Y=min"); ax.legend()
ax.set_title("Prime meet field: which prime pairs land on the Sierpinski gasket")
fig.savefig("_playground_viz/prime_gasket.png", dpi=120)
print("saved prime_gasket.png")
# which on-gasket prime pairs? bitwise-disjoint primes are interesting
ex = [(int(min(a,p)), int(max(a,p))) for i,a in enumerate(P) for p in P[i:] if (int(a)&int(p))==0][:12]
print("sample bitwise-disjoint prime pairs:", ex)
