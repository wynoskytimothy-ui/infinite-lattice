"""Play 2: mod-2 fields (Sierpinski hunt) + octant field over a grid.

Three sub-plays:
 (A) Meet-coord parity:  for grid (a,p), color = (X & Y) parity-ish.
     The classic Sierpinski test is (a AND p)==0 -> binomial C(a+p,a) odd.
     Here we test the MEET coords: does any boolean combo of (X=a+p, Y=min)
     reproduce Sierpinski / Pascal-mod-2?
 (B) Pascal-mod-2 ground truth via Lucas: C(i+j, i) mod 2 == ((i & j)==0).
     Compare to meet parity to see if the lattice secretly encodes it.
 (C) Octant field octant(a*p) over a grid of primes — does XOR-homomorphism
     tile the grid into a recognizable 8-color pattern?
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "C:/Users/wynos/trng")
from prime_hotel.premonition import prime_octant  # noqa

G = 256

# (A)+(B): Sierpinski hunt on a full integer grid a,p in [0,G)
A, Pm = np.meshgrid(np.arange(G), np.arange(G), indexing="ij")
sierp = ((A & Pm) == 0).astype(np.uint8)          # ground-truth Pascal mod 2
X = A + Pm
Ymin = np.minimum(A, Pm)
meet_parity = (X ^ Ymin) & 1                       # parity of (sum XOR min)
xy_and = ((X & Ymin) == 0).astype(np.uint8)        # Sierpinski-shaped on meet coords?

print("Sierpinski (a&p)==0 density:", sierp.mean().round(4))
print("meet (X^Y)&1 matches Sierpinski?", np.array_equal(meet_parity, sierp))
print("meet (X&Y)==0 matches Sierpinski?", np.array_equal(xy_and, sierp))
# X = a+p, Y=min. Is (a&p)==0 expressible from (X,Y)? a = X-Y or Y; recover both:
amin = Ymin; amax = X - Ymin
recov = ((amin & amax) == 0).astype(np.uint8)
print("recovered (min&max)==0 == Sierpinski?", np.array_equal(recov, sierp))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(sierp, cmap="binary", origin="lower"); axs[0].set_title("Pascal mod2 (a&p)==0")
axs[1].imshow(meet_parity, cmap="binary", origin="lower"); axs[1].set_title("meet (X^Y)&1")
axs[2].imshow(recov, cmap="binary", origin="lower"); axs[2].set_title("(min & max)==0 from meet")
fig.savefig("_playground_viz/sierpinski.png", dpi=110)
print("saved sierpinski.png")

# (C) Octant field over prime grid
def primes_up_to(N):
    s = np.ones(N+1, bool); s[:2]=False
    for i in range(2,int(N**0.5)+1):
        if s[i]: s[i*i::i]=False
    return np.flatnonzero(s)
P = primes_up_to(700)        # ~125 primes
P = P[P > 7]                 # homomorphism holds for primes > 7
oc = np.array([prime_octant(int(p)) for p in P])
# grid of octant(p)^octant(q) (== octant(p*q) by homomorphism)
M = oc[:, None] ^ oc[None, :]
fig2, ax2 = plt.subplots(figsize=(8, 8))
im = ax2.imshow(M, cmap="tab10", origin="lower", interpolation="nearest")
ax2.set_title(f"octant(p)^octant(q) field, {len(P)} primes >7")
ax2.set_xlabel("prime index q"); ax2.set_ylabel("prime index p")
fig2.colorbar(im, ax=ax2, ticks=range(8))
fig2.savefig("_playground_viz/octant_field.png", dpi=110)
print("saved octant_field.png")
# octant value distribution
vals, cnts = np.unique(oc, return_counts=True)
print("octant distribution over primes>7:", dict(zip(vals.tolist(), cnts.tolist())))
print("diagonal of M (p^p) all zero?", np.all(np.diag(M) == 0))
