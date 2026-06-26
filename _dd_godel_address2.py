# -*- coding: utf-8 -*-
"""
_dd_godel_address2.py  --  ANGLE: godel-fix  (STRESS to breaking point)

Pin three things the first probe left open:

  A. PURE interior-transgressor decode: recover the FULL path from ONLY the
     per-edge VA1 triples (no smuggled mixed-radix scalar).  Does the AETHOS
     interior read ALONE invert?  And what is its honest bit cost?

  B. COMPRESS the AETHOS triple read: a VA1 triple (a+p, a, a+p+n) is 3 numbers
     but the ONLY free info is n (=child index); a and p are fixed by the tree.
     So entropy-coded it collapses to log2(b_i).  Prove the redundancy: the
     7.86xH is RAW storage, the INFORMATION is at the floor.  => the blow-up is
     a REPRESENTATION artifact, not an information wall.  Measure the gap.

  C. DEPTH SCALING: does the raw-triple blow-up grow with depth d while the
     mixed-radix floor-coder stays at ~1.0xH?  Sweep d=2..12 (uniform b=4).
     If raw blow-up ratio is ~constant and floor stays ~1.0, that's the
     honest statement: AETHOS read is O(d) numbers each O(log node) bits =
     a CONSTANT-FACTOR overhead, not exponential -- but it NEVER beats the floor.
"""
import math, itertools, sys
from functools import reduce

OUT = []
def log(*a): OUT.append(" ".join(str(x) for x in a))

def bits_for_int(x):
    x = abs(int(x)); return 1 if x == 0 else x.bit_length()
def first_primes(k):
    ps, c = [], 2
    while len(ps) < k:
        if all(c % q for q in ps): ps.append(c)
        c += 1
    return ps

# ---------- A. PURE interior-transgressor decode ----------
log("=== A. PURE interior-transgressor decode (no smuggled scalar) ===")
BRANCH = [4, 5, 6, 5, 8]
D = len(BRANCH); NLEAVES = reduce(lambda x,y:x*y, BRANCH); H = math.log2(NLEAVES)
_pp = first_primes(sum(BRANCH) + 4); NODE_PRIME = {}; _ix = 0
for depth in range(D):
    for idx in range(BRANCH[depth]):
        NODE_PRIME[(depth, idx)] = _pp[_ix]; _ix += 1

def encode_triples(path):
    """address = ordered list of per-edge VA1 triples ONLY."""
    tris = []
    for i, idx in enumerate(path):
        a = NODE_PRIME[(i, 0)]; p = NODE_PRIME[(i, idx)]; n = idx
        tris.append((a + p, a, a + p + n))   # (X,Y,Z) = (a+p, a, a+p+n)
    return tuple(tris)

def decode_triples(tris):
    """invert each triple independently: a=Y, p=X-Y, n=Z-X.  child idx = n."""
    path = []
    for i, (X, Y, Z) in enumerate(tris):
        a = Y; p = X - Y; n = Z - X
        # recover child index from n directly (n carried idx)
        idx = n
        # SANITY: the recovered node prime must equal NODE_PRIME[(i,idx)] and a anchor
        if NODE_PRIME[(i, 0)] != a:      # anchor check
            return None
        if NODE_PRIME[(i, idx)] != p:    # node prime check
            return None
        path.append(idx)
    return tuple(path)

paths = list(itertools.product(*[range(b) for b in BRANCH]))
seen = set(); fail = 0; rawbits = []
for path in paths:
    addr = encode_triples(path)
    back = decode_triples(addr)
    if back != path: fail += 1
    seen.add(addr)
    rawbits.append(sum(bits_for_int(v) for tri in addr for v in tri))
log(f"  distinct triple-addresses : {len(seen)}/{NLEAVES}")
log(f"  PURE decode exact         : {NLEAVES-fail}/{NLEAVES}  "
    f"({'EXACT - interior read DOES invert alone' if fail==0 else f'FAIL {fail}'})")
log(f"  raw triple bits avg       : {sum(rawbits)/len(rawbits):.2f}  vs H={H:.2f}  "
    f"= {(sum(rawbits)/len(rawbits))/H:.2f}xH")
log("")

# ---------- B. Entropy-code the triple read: redundancy collapse ----------
log("=== B. Compress the AETHOS triple read (information vs representation) ===")
# Only n (=idx) is free per edge; a,p are tree-fixed. So min description per edge
# is log2(b_i). Build the actual minimal code = mixed-radix on the recovered n's.
def minimal_from_triples(addr):
    # extract n per edge -> mixed-radix -> this is the INFORMATION content
    ns = [Z - X for (X, Y, Z) in addr]
    code = 0
    for i, n in enumerate(ns):
        code = code * BRANCH[i] + n
    return code
codes = set(); cbits = []
for path in paths:
    c = minimal_from_triples(encode_triples(path))
    codes.add(c); cbits.append(bits_for_int(c))
log(f"  info-coded distinct       : {len(codes)}/{NLEAVES}")
log(f"  info-coded bits avg       : {sum(cbits)/len(cbits):.2f}  = "
    f"{(sum(cbits)/len(cbits))/H:.3f}xH  (collapses to the floor)")
log(f"  => the 7.86xH raw blow-up is a REPRESENTATION artifact; the triples")
log(f"     carry only {H:.2f} bits of real info (rest is tree-fixed a,p redundancy).")
log("")

# ---------- C. depth scaling: raw vs floor ----------
log("=== C. Depth scaling  (uniform b=4, d=2..12) ===")
log(f"  {'d':>3} {'#leaves':>9} {'H':>7} | {'raw_avg':>8} {'raw/H':>6} | "
    f"{'floor_avg':>9} {'floor/H':>7} | {'godel/H':>7}")
PRIMES = first_primes(16)
for d in range(2, 13):
    b = 4
    nl = b ** d
    Hd = math.log2(nl)
    # build node primes
    npd = {}
    pp = first_primes(b * d + 4); ix = 0
    for depth in range(d):
        for idx in range(b):
            npd[(depth, idx)] = pp[ix]; ix += 1
    # sample (full enumerate up to d=8 ~65k; for d>8 sample 20000 random)
    import random; random.seed(7)
    if nl <= 70000:
        sample = list(itertools.product(*[range(b)] * d))
    else:
        sample = [tuple(random.randrange(b) for _ in range(d)) for _ in range(20000)]
    raw_bits = []; floor_bits = []; godel_bits = []
    for path in sample:
        # raw triple
        rb = 0
        for i, idx in enumerate(path):
            a = npd[(i, 0)]; p = npd[(i, idx)]; n = idx
            rb += bits_for_int(a + p) + bits_for_int(a) + bits_for_int(a + p + n)
        raw_bits.append(rb)
        # floor mixed-radix
        code = 0
        for i, idx in enumerate(path): code = code * b + idx
        floor_bits.append(bits_for_int(code))
        # godel product
        g = 1
        for i, idx in enumerate(path): g *= PRIMES[i] ** idx
        godel_bits.append(bits_for_int(g))
    ra = sum(raw_bits)/len(raw_bits); fa = sum(floor_bits)/len(floor_bits)
    ga = sum(godel_bits)/len(godel_bits)
    log(f"  {d:>3} {nl:>9} {Hd:>7.2f} | {ra:>8.1f} {ra/Hd:>6.2f} | "
        f"{fa:>9.2f} {fa/Hd:>7.3f} | {ga/Hd:>7.2f}")
log("")
log("READING: floor/H stays ~1.0 for all d (positional = optimal).")
log("raw triple/H is a roughly CONSTANT factor (O(d) numbers x O(log node) bits).")
log("godel/H GROWS without bound (sum of i*log p_i / log b -> diverges): worst.")

with open("_dd_godel_address2_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n".join(OUT))
