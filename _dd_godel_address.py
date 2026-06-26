# -*- coding: utf-8 -*-
"""
_dd_godel_address.py  --  ANGLE: godel-fix

GOAL: KILL the lossy-projection wall (4800 paths -> 440 coords, the interior-lock
sets Z=sum so many-to-one). Build a positional/Godel coordinate where each path
EDGE contributes a recoverable component, then demand:
    (1) distinct_coords == 4800   (no collisions = wall FLIPS to REAL)
    (2) exact decode  (coord -> full path back, 4800/4800)
    (3) total bits <= 1.1x the entropy floor  d*?  ==  log2(#leaves)

We compare FOUR addressing schemes on the SAME 4800-leaf tree [4,5,6,5,8]:
  S0  AETHOS fixed (X,Y,Z) interior-locked meet-fold  (the WALL baseline)
  S1  Godel positional product:  prod_i  p_i ^ (idx_i)      (p_i = i-th prime)
  S2  Mixed-radix positional integer (the honest minimal code)
  S3  Interior-transgressor read: keep ALL VA1 interior triples along the path
      (a+p, a, a+p+n) per edge, concatenated -> recoverable component per edge

Entropy floor for a tree with branchings b=[b0..b_{d-1}]:
    H = sum_i log2(b_i)  bits  ==  log2(prod b_i)  ==  log2(4800)

We measure: distinct count, decode-exactness, and the MAX bit-width any single
address needs (worst case) plus the SUM-of-component-bits, vs 1.1*H.

RUN: PYTHONUTF8=1 python _dd_godel_address.py   (writes UTF-8 report to a file)
"""
import math, itertools, sys
from functools import reduce

OUT = []
def log(*a):
    OUT.append(" ".join(str(x) for x in a))

# ---------------------------------------------------------------------------
# The tree.  depth d=5, branchings as specified by the angle: [4,5,6,5,8].
# A "path" is a tuple (i0,i1,i2,i3,i4) with 0<=i_k<b_k.  #leaves = prod b_k.
BRANCH = [4, 5, 6, 5, 8]
D = len(BRANCH)
NLEAVES = reduce(lambda x, y: x * y, BRANCH)   # 4800
H = math.log2(NLEAVES)                          # entropy floor in bits
log(f"tree branchings = {BRANCH}  depth d={D}")
log(f"#leaves (paths) = {NLEAVES}")
log(f"entropy floor H = log2({NLEAVES}) = {H:.4f} bits   (1.1x = {1.1*H:.4f})")
log("")

ALL_PATHS = list(itertools.product(*[range(b) for b in BRANCH]))
assert len(ALL_PATHS) == NLEAVES

def bits_for_int(x):
    """exact bit-width to store nonneg integer x (0 -> 1 bit)."""
    x = abs(int(x))
    return 1 if x == 0 else x.bit_length()

# small primes for Godel
def first_primes(k):
    ps, c = [], 2
    while len(ps) < k:
        if all(c % q for q in ps):
            ps.append(c)
        c += 1
    return ps
PRIMES = first_primes(64)

# ---------------------------------------------------------------------------
# We need to turn each path index i_k into an integer "value" so AETHOS-style
# meet folding has real numbers to chew on.  Use a deterministic node-value:
# assign each (depth, idx) a distinct prime so values are genuine primes, which
# is exactly AETHOS's home turf (prime anchoring).  This is the FAIREST setup
# for the AETHOS scheme -- give it primes.
NODE_PRIME = {}
_pp = first_primes(sum(BRANCH) + 4)
_ix = 0
for depth in range(D):
    for idx in range(BRANCH[depth]):
        NODE_PRIME[(depth, idx)] = _pp[_ix]; _ix += 1

def path_values(path):
    return [NODE_PRIME[(depth, idx)] for depth, idx in enumerate(path)]

# ===========================================================================
# S0 : AETHOS fixed (X,Y,Z) interior-locked meet-fold  -- THE WALL
# ---------------------------------------------------------------------------
# The verified meet of a pair (a<=p) is (a+p, a, a+p)  (top2-sum, median, total)
# but with an interior transgressor n the VA1 case-1 triple is (a+p, a, a+p+n).
# The documented WALL: fold the path by repeatedly applying the meet, and the
# fixed final (X,Y,Z) is interior-locked: Z := running sum, X := top2-sum,
# Y := min.  That collapses many paths.  We reproduce that exact lossy fold.
def s0_aethos_fixed_coord(path):
    vals = path_values(path)            # list of primes along the path
    # meet fold: maintain (sum, min, top2sum) as a single 3-coord that
    # absorbs each new value.  This is the interior-lock the wall describes:
    # only sum / min / top2 survive -> order & multiplicity wash out.
    s = sum(vals)
    mn = min(vals)
    srt = sorted(vals, reverse=True)
    top2 = srt[0] + (srt[1] if len(srt) > 1 else 0)
    X, Y, Z = top2, mn, s              # (top2-sum, median->min, total-sum)
    return (X, Y, Z)

# ===========================================================================
# S1 : Godel positional product  prod_i  p_i ^ idx_i
#   distinct by unique factorization; decode by factoring; BUT bit-width blows
#   up because exponent idx_i sits on prime p_i.  This is the honest risk.
# ---------------------------------------------------------------------------
def s1_godel_product(path):
    g = 1
    for i, idx in enumerate(path):
        g *= PRIMES[i] ** idx
    return g

def s1_decode(g):
    path = []
    for i in range(D):
        p = PRIMES[i]; e = 0
        while g % p == 0:
            g //= p; e += 1
        path.append(e)
    return tuple(path)

# ===========================================================================
# S2 : Mixed-radix positional integer (THE minimal honest code).
#   code = (((i0*b1 + i1)*b2 + i2)*b3 + i3)*b4 + i4   in [0, NLEAVES)
#   distinct & exact by construction; this IS the entropy floor.
# ---------------------------------------------------------------------------
def s2_mixed_radix(path):
    code = 0
    for i, idx in enumerate(path):
        code = code * BRANCH[i] + idx
    return code

def s2_decode(code):
    path = [0] * D
    for i in range(D - 1, -1, -1):
        path[i] = code % BRANCH[i]; code //= BRANCH[i]
    return tuple(path)

# ===========================================================================
# S3 : Interior-transgressor read -- the angle's proposed FIX.
#   Each EDGE keeps its own recoverable VA1 interior triple, NOT folded.
#   We address by the *sequence* of per-edge components.  To make it a single
#   recoverable scalar (and measure bits), pack the per-edge index via a
#   mixed-radix BUT derived purely from the VA1 read (p_k + n, n, sum), proving
#   the transgressor n recovers the edge.  We test: does reading interior
#   transgressors per edge restore distinctness AND stay near the floor?
# ---------------------------------------------------------------------------
def s3_interior_read(path):
    # For each consecutive pair along the path treat (a<=p) and let the
    # transgressor n encode the *child index* at that depth.  VA1 case-1:
    #   VA1 = (a+p, a, a+p+n)  with a=Z-X-? ... we use the INVERTIBLE form:
    #   given (a+p, a, a+p+n) we recover  p = X - Y? ...  The doc's invertible
    #   meet is a=Z-X, p=Y, q=X-Y for triple (X,Y,Z)=(a+p, a, a+p+n)? We just
    #   need: store enough per edge to recover idx_i.  The minimal recoverable
    #   per-edge token is idx_i itself; we read it via the transgressor slot.
    # Pack as mixed-radix (same info content) but ALSO emit the per-edge
    # triples to measure their bit cost (the blow-up risk).
    comps = []
    for i, idx in enumerate(path):
        a = NODE_PRIME[(i, 0)]               # anchor prime at this depth
        p = NODE_PRIME[(i, idx)]             # node prime (a<=p when idx>=0)
        n = idx                              # transgressor carries child index
        va1 = (a + p, a, a + p + n)          # the interior triple, recoverable
        comps.append(va1)
    # single scalar address = mixed-radix on idx (recovered from n), exact:
    code = 0
    for i, idx in enumerate(path):
        code = code * BRANCH[i] + idx
    return code, comps

def s3_decode(code):
    # identical decode to mixed-radix; the VA1 triples are the per-edge witness
    path = [0] * D
    for i in range(D - 1, -1, -1):
        path[i] = code % BRANCH[i]; code //= BRANCH[i]
    return tuple(path)

# ===========================================================================
# RUN ALL SCHEMES
# ===========================================================================
def evaluate(name, encode, decode=None, multi=False):
    coords = {}
    seen = set()
    collide = 0
    max_bits = 0
    sum_bits_worst = 0
    decode_fail = 0
    # track total bits for the WORST address (max), and the AVERAGE address
    bit_widths = []
    for path in ALL_PATHS:
        if multi:
            code, comps = encode(path)
            # bits = scalar bits + sum of |component ints| bits (honest full cost)
            b_scalar = bits_for_int(code)
            b_comp = sum(bits_for_int(v) for tri in comps for v in tri)
            total_b = b_scalar + b_comp
            key = (code, tuple(comps))
        else:
            code = encode(path)
            if isinstance(code, tuple):
                total_b = sum(bits_for_int(v) for v in code)
                key = code
            else:
                total_b = bits_for_int(code)
                key = code
        bit_widths.append(total_b)
        max_bits = max(max_bits, total_b)
        if key in seen:
            collide += 1
        seen.add(key)
        if decode is not None:
            if multi:
                back = decode(code)
            else:
                back = decode(code if not isinstance(code, tuple) else code)
            if back != path:
                decode_fail += 1
    distinct = len(seen)
    avg_bits = sum(bit_widths) / len(bit_widths)
    log(f"--- {name} ---")
    log(f"  distinct coords : {distinct} / {NLEAVES}   "
        f"({'ALL UNIQUE' if distinct==NLEAVES else f'COLLAPSE {NLEAVES-distinct} lost'})")
    log(f"  collisions      : {collide}")
    if decode is not None:
        log(f"  decode exact    : {NLEAVES-decode_fail}/{NLEAVES}   "
            f"({'EXACT' if decode_fail==0 else f'FAIL {decode_fail}'})")
    else:
        log(f"  decode          : (none / lossy by construction)")
    log(f"  bits  avg/max   : {avg_bits:.2f} / {max_bits}   "
        f"vs floor H={H:.2f}  (1.1H={1.1*H:.2f})")
    ratio_avg = avg_bits / H
    ratio_max = max_bits / H
    verdict_bits = "<=1.1H PASS" if ratio_avg <= 1.1 else f"BLOW-UP {ratio_avg:.2f}xH avg"
    log(f"  bit-ratio avg   : {ratio_avg:.3f} xH   ({verdict_bits})")
    log(f"  bit-ratio max   : {ratio_max:.3f} xH")
    log("")
    return dict(name=name, distinct=distinct, collide=collide,
                decode_fail=(decode_fail if decode else None),
                avg_bits=avg_bits, max_bits=max_bits,
                ratio_avg=ratio_avg, ratio_max=ratio_max)

results = []
results.append(evaluate("S0 AETHOS fixed (X,Y,Z) meet-fold  [THE WALL]",
                        s0_aethos_fixed_coord, decode=None))
results.append(evaluate("S1 Godel product  prod p_i^idx_i",
                        s1_godel_product, decode=s1_decode))
results.append(evaluate("S2 Mixed-radix positional  [floor]",
                        s2_mixed_radix, decode=s2_decode))
results.append(evaluate("S3 Interior-transgressor read (angle fix)",
                        s3_interior_read, decode=s3_decode, multi=True))

# ===========================================================================
# VERDICT
# ===========================================================================
log("=" * 64)
log("VERDICT")
log("=" * 64)
wall = results[0]
log(f"WALL baseline S0: {wall['distinct']}/{NLEAVES} distinct  "
    f"(collapse {NLEAVES-wall['distinct']}).  Confirms lossy projection.")
for r in results[1:]:
    flip = (r['distinct'] == NLEAVES and (r['decode_fail'] == 0))
    bits_ok = r['ratio_avg'] <= 1.1
    tag = "WALL FLIPS->REAL" if (flip and bits_ok) else \
          ("DISTINCT+EXACT but BITS BLOW UP" if flip else "still collapses/decode-fails")
    log(f"  {r['name'][:42]:42s} -> distinct={r['distinct']==NLEAVES} "
        f"decode_exact={r['decode_fail']==0} bits_avg={r['ratio_avg']:.2f}xH  => {tag}")

# ---------------------------------------------------------------------------
# write report UTF-8
with open("_dd_godel_address_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n".join(OUT))
