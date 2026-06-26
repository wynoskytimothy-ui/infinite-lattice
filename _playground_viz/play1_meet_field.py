"""Play 1: the prime x prime MEET field.

meet(a,p) = (X, Y) = (a+p, min(a,p)).  Plot every prime pair's meet node.
No agenda — just look at what shape the (sum, min) field makes.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def primes_up_to(N):
    sieve = np.ones(N + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return np.flatnonzero(sieve)

N = 300
P = primes_up_to(N)
print(f"#primes <= {N}: {len(P)}")

# all ordered pairs a<=p
xs, ys = [], []
for i, a in enumerate(P):
    for p in P[i:]:
        xs.append(a + p)      # X = sum
        ys.append(min(a, p))  # Y = min
xs = np.array(xs); ys = np.array(ys)
print(f"#meet nodes: {len(xs)}")
print(f"X range [{xs.min()},{xs.max()}], Y range [{ys.min()},{ys.max()}]")
print(f"#distinct (X,Y): {len(set(zip(xs.tolist(), ys.tolist())))}")

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(xs, ys, s=2, c=xs - 2*ys, cmap="twilight", alpha=0.7)
ax.set_xlabel("X = a+p (sum)"); ax.set_ylabel("Y = min(a,p)")
ax.set_title(f"Prime x Prime meet field, primes<= {N}")
fig.savefig("_playground_viz/meet_field.png", dpi=110)
print("saved meet_field.png")

# structural probe: for fixed Y=min=a (a prime), X = a + p ranges over a+each larger prime.
# So each horizontal row Y=a is the set {a + p : p prime >= a} = a-shifted prime line.
# Check: are the GAPS within a row exactly the prime gaps?
for a in [3, 5, 7, 11]:
    row = sorted(x for x, y in zip(xs.tolist(), ys.tolist()) if y == a)
    gaps = np.diff(row)
    pgaps = np.diff([p for p in P if p >= a])
    print(f"Y={a}: row-gaps == prime-gaps? {np.array_equal(gaps, pgaps)}  (first gaps {gaps[:6].tolist()})")
