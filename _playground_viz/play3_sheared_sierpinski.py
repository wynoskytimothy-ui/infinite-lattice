"""Play 3: the meet IS a shear of Sierpinski; plot it in native (X,Y) coords.

Claim: meet(a,p) = (X=a+p, Y=min). Sierpinski lives on (min,max) via (min&max)==0.
Since min=Y and max=X-Y, the meet plane is the linear shear (min,max)->(min+max, min)
of the Sierpinski gasket. Plot the Sierpinski set DIRECTLY in meet coords and see
the gasket sheared into the triangular wedge we saw in play1.

Also: nail the exact rule of the (X^Y)&1 'nested-L staircase' from play2.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 256
mn, mx = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
mask = (mn <= mx) & ((mn & mx) == 0)          # Sierpinski on (min,max), upper tri
Xs = (mn + mx)[mask]                           # meet X = sum
Ys = mn[mask]                                  # meet Y = min

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Xs, Ys, s=1, c="black")
ax.set_xlabel("X = min+max (meet sum)"); ax.set_ylabel("Y = min (meet)")
ax.set_title("Sierpinski gasket plotted in MEET coordinates (sheared)")
fig.savefig("_playground_viz/sheared_sierpinski.png", dpi=120)
print("saved sheared_sierpinski.png; #points", mask.sum())

# Exact rule of (X^Y)&1 staircase, X=a+p, Y=min(a,p).
# parity(a+p) XOR parity(min). Let's decode: it's parity(a)^parity(p)^parity(min(a,p))
#  = parity(max(a,p))  since a+p+min = max + 2*min  -> parity(a+p) ^ parity(min)=parity(max).
A, Pm = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
lhs = ((A + Pm) ^ np.minimum(A, Pm)) & 1
rhs = np.maximum(A, Pm) & 1
print("(X^Y)&1 == parity(max(a,p)) everywhere?", np.array_equal(lhs, rhs))
