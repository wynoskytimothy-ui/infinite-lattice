"""
Deep-dive probe: zeno-pi construct.

Questions:
 1. What IS Timothy's additive pi mathematically? (the dimensionless layer sum)
 2. What IS its convergence rate? Compare to the standard method it most resembles
    (inscribed-polygon / Archimedes area, doubling sides).
 3. Are the accelerators (Richardson / double-Richardson / tail) novel, or the
    textbook Richardson/Romberg/Aitken applied to a geometric error sequence?
 4. Is "frame descent" (prime-primorial subdivision in zeno_text) a convergence
    method at all, or just a positional number system?
 5. Head-to-head: digits-correct vs work, Timothy(+accel) vs Archimedes(+accel)
    vs Machin vs Gauss-Legendre. Where does it actually land?

All numbers measured here. mpmath used for high-precision ground truth + to push
past float64 so we can see true asymptotic rate (float64 floors near 1e-16).
"""
import math
import time
import sys

sys.path.insert(0, r"C:/Users/wynos/aethos_master/src")
from aethos_master.pi import richardson as R   # build_pi, accelerators
from aethos_master.pi import timothy as T       # additive layer sum

from mpmath import mp, mpf, sin, cos, tan, sqrt, pi as MPPI, atan

def hr(t=""):
    print("\n" + "=" * 78)
    if t: print(t)
    print("=" * 78)

# ----------------------------------------------------------------------------
# 0. Ground truth
# ----------------------------------------------------------------------------
mp.dps = 80
PI = +MPPI
PI_STR = mp.nstr(PI, 70)

def digits_correct(approx_mpf):
    """Number of correct significant decimal digits vs true pi."""
    err = abs(mpf(approx_mpf) - PI)
    if err == 0:
        return mp.dps
    return float(-mp.log10(err / PI))

# ----------------------------------------------------------------------------
# 1. Identify Timothy's formula in closed form and confirm it IS Archimedes area
# ----------------------------------------------------------------------------
hr("1. WHAT IS TIMOTHY'S ADDITIVE PI? (closed form)")
# From richardson.show_formula docstring:
#   pi = 2 + SUM_{n>=0} 2^(n+2) * sin(pi/2^(n+2)) * (1 - cos(pi/2^(n+2)))
# Each partial sum to step n is EXACTLY the area of an inscribed regular
# 2^(n+3)-gon... let's just verify the partial sums equal inscribed-polygon area.
def inscribed_area(sides):
    # area of regular N-gon inscribed in unit circle = (1/2) N sin(2pi/N)
    return 0.5 * sides * math.sin(2 * math.pi / sides)

print("Timothy layer-sum partial vs inscribed regular-polygon area:")
print(f"{'step n':>6} {'sides':>10} {'Timothy cumsum':>20} {'inscribed area':>20} {'match?':>8}")
hist = R.build_pi_fast(12)   # (it, n_added, running_area)
for it, _added, total in hist[:9]:
    sides = 4 * (2 ** it)          # 4,8,16,... inscribed 2^(it+2)-gon
    ia = inscribed_area(sides)
    match = abs(total - ia) < 1e-12
    print(f"{it:6d} {sides:10d} {total:20.15f} {ia:20.15f} {str(match):>8}")
print("\n=> Timothy's 'running_area' at step n == area of inscribed regular 4*2^n-gon.")
print("   It is the AREA form of Archimedes' inscribed-polygon method (side-doubling).")
print("   (The classic Archimedes uses perimeter; this uses area. Same sequence family.)")

# Also confirm the dimensionless 'timothy_layers' sum is the identical sequence
print("\nTimothy additive (timothy.py) vs richardson build_pi (richardson.py):")
for L in (5, 10, 15, 20):
    a = T.pi_timothy_additive(L)
    b = R.build_pi_fast(L)[-1][2]
    print(f"  L={L:2d}: timothy={a:.15f}  build_pi={b:.15f}  diff={abs(a-b):.2e}")

# ----------------------------------------------------------------------------
# 2. Convergence RATE of the base method (high precision, past float64 floor)
# ----------------------------------------------------------------------------
hr("2. BASE CONVERGENCE RATE (mpmath, past float64 floor)")
mp.dps = 60
def timothy_area_mp(steps):
    """Inscribed-polygon AREA via the Timothy additive recurrence, high precision."""
    total = mpf(2)        # square area (4 right triangles, legs 1)
    k = 4                 # sides
    # area added at step n going from k-gon to 2k-gon
    for n in range(steps):
        ang = MPPI / k          # = pi / sides
        added = k * sin(ang) * (1 - cos(ang))
        total += added
        k *= 2
    return total

print(f"{'step':>5} {'sides':>10} {'err':>16} {'err ratio':>12} {'digits':>10}")
prev = None
errs = []
for n in range(0, 26):
    val = timothy_area_mp(n)
    err = abs(val - PI)
    ratio = (prev / err) if prev else float('nan')
    errs.append(err)
    dc = digits_correct(val)
    if n % 2 == 0 or n < 6:
        print(f"{n:5d} {4*2**n:10d} {float(err):16.3e} {float(ratio):12.4f} {dc:10.2f}")
    prev = err
print("\n=> error ratio -> 4.0  =>  error ~ C/4^n  (each side-doubling quarters the error).")
print("   This is the KNOWN quadratic-in-sides convergence of Archimedes area; the")
print("   'right-triangle additive' framing does not change the rate.")

# ----------------------------------------------------------------------------
# 3. The accelerators: are they novel or textbook Richardson/Romberg/Aitken?
# ----------------------------------------------------------------------------
hr("3. ACCELERATORS: novel or textbook Richardson/Romberg?")
# Build a high-precision sequence S_n = inscribed area, error = C/4^n.
# Richardson for error ~ 4^-n: S* ~ (4 S_n - S_{n-1})/3. Then next order /15, /63...
# That is EXACTLY Romberg / repeated Richardson on a 4^-k error expansion.
mp.dps = 60
N = 16
S = [timothy_area_mp(n) for n in range(N)]

# Repeated Richardson (Romberg-style triangle) with factors 4^k
def romberg_triangle(seq):
    """Classic repeated Richardson assuming error expansion in powers of 1/4."""
    T = [list(seq)]
    k = 1
    while len(T[-1]) > 1:
        prev = T[-1]
        f = 4 ** k
        new = [(f * prev[i+1] - prev[i]) / (f - 1) for i in range(len(prev) - 1)]
        T.append(new)
        k += 1
    return T

tri = romberg_triangle(S)
print("Repeated Richardson (Romberg) on Timothy area sequence; best estimate per level:")
print(f"{'level':>6} {'best digits':>14}")
for lvl, row in enumerate(tri):
    best = row[-1]
    print(f"{lvl:6d} {digits_correct(best):14.2f}")
print("\nNow the module's OWN richardson_accelerate / double_richardson (float64):")
# NOTE: R.build_pi materializes 4*2^n EXPLICIT edges per step (a geometric memory
# blowup -> ~67M tuples at it=24, OOM/hang). Since section 1 PROVED running_area ==
# inscribed-polygon area, we feed the module's *accelerator* functions an
# equivalent float64 history built in O(n) (no edge materialization). This is a
# faithful drop-in: the accelerators only read history[i][2] (the running area).
def cheap_history(steps):
    hist = []
    total = 2.0; k = 4
    hist.append((0, 4, total))
    for it in range(1, steps + 1):
        ang = math.pi / k
        total += k * math.sin(ang) * (1 - math.cos(ang))
        k *= 2
        hist.append((it, k, total))
    return hist
hist50 = cheap_history(26)
# sanity: cheap_history matches the real build_pi for small n
_real = R.build_pi_fast(8)
print(f"  cheap_history vs real build_pi @it=8: "
      f"{abs(hist50[8][2]-_real[8][2]):.2e} (==0 confirms drop-in)")
r1 = R.richardson_accelerate(hist50)
r2 = R.double_richardson(hist50)
tail = R.tail_correction(hist50)
print(f"  base   final err  = {abs(hist50[-1][2]-math.pi):.3e}")
print(f"  tail   final err  = {tail[-1][4]:.3e}")
print(f"  Rich-1 final err  = {r1[-1][4]:.3e}")
print(f"  Rich-2 final err  = {r2[-1][3]:.3e}")
# Confirm module's richardson_accelerate == (4 S_n - S_{n-1})/3, i.e. textbook
chk = (4.0 * hist50[5][2] - hist50[4][2]) / 3.0
print(f"  module Rich-1 at it=5 = {r1[4][2]:.15f}")
print(f"  textbook (4Sn-Sn-1)/3 = {chk:.15f}  match={abs(r1[4][2]-chk)<1e-15}")
print("\n=> The module's 'Richardson'/'double Richardson'/'tail' are the LITERAL")
print("   textbook Richardson extrapolation (factor 4) and geometric-tail sum.")
print("   The docstring even says so. Novel framing, standard algorithm.")

# tail correction == 1 step of the geometric series remainder; equivalent to a
# weaker Richardson. Confirm:
print("\n  tail vs Richardson-1 (both kill leading 4^-n term):")
print(f"    tail final digits   = {digits_correct(mpf(tail[-1][2])):.2f}")
print(f"    Rich-1 final digits = {digits_correct(mpf(r1[-1][2])):.2f}")

# ----------------------------------------------------------------------------
# 4. HEAD-TO-HEAD: digits-correct vs WORK
#    Timothy(base) / Timothy+Romberg / Archimedes-perimeter / Machin / Gauss-Legendre
# ----------------------------------------------------------------------------
hr("4. HEAD-TO-HEAD: digits correct vs iterations (high precision)")
mp.dps = 70
PI = +MPPI
def dc(x):
    e = abs(mpf(x) - PI)
    return 0.0 if e == 0 else float(-mp.log10(e / PI))

# (a) Timothy base (inscribed area)
def timothy_base(n):
    total = mpf(2); k = 4
    for _ in range(n):
        ang = MPPI / k
        total += k * sin(ang) * (1 - cos(ang)); k *= 2
    return total

# (b) Timothy + full Romberg (use first n+1 terms)
def timothy_romberg(n):
    seq = [timothy_base(i) for i in range(n + 1)]
    tri = romberg_triangle(seq)
    return tri[-1][0]

# (c) Archimedes perimeter (the *standard* Archimedes), side-doubling, area-equiv rate
def archimedes_perimeter(n):
    # inscribed perimeter half = N sin(pi/N); pi ~ N sin(pi/N)
    k = 4
    for _ in range(n): k *= 2
    return k * sin(MPPI / k)   # uses true angle; rate identical C/N^2 = C/4^n

# (d) Machin's formula pi/4 = 4 atan(1/5) - atan(1/239), via arctan series, n terms each
def machin(nterms):
    def at(x, nt):
        x = mpf(x); s = mpf(0); xp = x; sign = 1
        for k in range(nt):
            s += sign * xp / (2*k + 1)
            xp *= x * x; sign = -sign
        return s
    return 4 * (4 * at(mpf(1)/5, nterms) - at(mpf(1)/239, nterms))

# (e) Gauss-Legendre AGM (quadratic-doubling-digits)
def gauss_legendre(n):
    a = mpf(1); b = 1/sqrt(mpf(2)); t = mpf(1)/4; p = mpf(1)
    for _ in range(n):
        an = (a + b) / 2
        b = sqrt(a * b)
        t = t - p * (a - an) ** 2
        a = an; p = 2 * p
    return (a + b) ** 2 / (4 * t)

print(f"{'n':>4} | {'Timothy base':>13} {'Timothy+Romb':>13} {'Archimedes':>11} {'Machin(n t)':>12} {'GaussLeg':>10}")
print(f"{'':>4} | {'digits':>13} {'digits':>13} {'digits':>11} {'digits':>12} {'digits':>10}")
print("-" * 78)
for n in [1, 2, 4, 6, 8, 10, 13, 16, 20]:
    db = dc(timothy_base(n))
    dr = dc(timothy_romberg(n))
    da = dc(archimedes_perimeter(n))
    dmach = dc(machin(n))
    dgl = dc(gauss_legendre(n)) if n <= 12 else dc(gauss_legendre(12))
    print(f"{n:4d} | {db:13.2f} {dr:13.2f} {da:11.2f} {dmach:12.2f} {dgl:10.2f}")

print("\nReading: 'digits' = correct significant decimal digits of pi.")
print(" - Timothy base & Archimedes: LINEAR (~0.6 digit / iter, the 4^-n rate).")
print(" - Timothy+Romberg: each Richardson level adds an order; competitive at low n,")
print("   but it's just Romberg on Archimedes (a textbook combination).")
print(" - Machin: linear but steeper (~1.4 digit/term), the classic hand-computation king.")
print(" - Gauss-Legendre: QUADRATIC (digits DOUBLE each iter) -> uncatchable.")

# ----------------------------------------------------------------------------
# 5. Is 'frame descent' (prime/primorial subdivision, zeno_text) a pi method?
# ----------------------------------------------------------------------------
hr("5. ZENO 'FRAME DESCENT': is the prime-primorial subdivision a pi accelerator?")
# zeno_text frame descent: width_n = 1/primorial(n); position = sum i_k/primorial_k.
# It is a POSITIONAL NUMBER SYSTEM (mixed-radix / factorial-number-system cousin),
# NOT a pi series. The only 'pi' link in the file is via the SAME inscribed-polygon
# build (Timothy). Let's show frame-descent is mixed-radix and reconstructs any x in [0,1].
def primorial_descent_encode(x, primes):
    digits = []
    frac = x
    cum = 1
    for p in primes:
        cum *= p
        # child index at this level: floor(frac * p), then descend into that child
        i = int(frac * p)
        if i == p: i = p - 1
        digits.append((p, i))
        frac = frac * p - i
    return digits
def primorial_descent_decode(digits):
    x = mpf(0); cum = mpf(1)
    for p, i in digits:
        cum *= p
        x += mpf(i) / cum
    return x
primes = [2,3,5,7,11,13,17,19,23,29]
for target in [mpf(1)/3, mpf(1)/7, mpf('0.123456789')]:
    d = primorial_descent_encode(float(target), primes)
    back = primorial_descent_decode(d)
    print(f"  x={mp.nstr(target,10):>12}  reconstructed={mp.nstr(back,10):>12}  |err|={float(abs(back-target)):.2e}")
print("\n=> Frame descent is an EXACT mixed-radix (primorial-base) positional system.")
print("   It addresses points in [0,1]; it is NOT itself a pi-series accelerator.")
print("   Its only connection to pi is reusing the SAME inscribed-polygon build (Timothy).")

# ----------------------------------------------------------------------------
# 6. Best honest framing: does Timothy beat the method it resembles?
# ----------------------------------------------------------------------------
hr("6. VERDICT METRICS")
# work-normalized: digits per polygon side (both Timothy area & Archimedes use 4*2^n sides)
mp.dps = 60
n = 20
tb = dc(timothy_base(n)); ap = dc(archimedes_perimeter(n))
print(f"Timothy base   @ n={n} (sides={4*2**n:,}): {tb:.2f} digits")
print(f"Archimedes per.@ n={n} (sides={4*2**n:,}): {ap:.2f} digits")
print(f"   -> identical asymptotic rate; Timothy is the AREA variant of Archimedes.")
tr = dc(timothy_romberg(n))
print(f"Timothy+Romberg@ n={n}: {tr:.2f} digits  (Romberg-accelerated Archimedes-area)")
print(f"Gauss-Legendre @ n=8 : {dc(gauss_legendre(8)):.2f} digits  (8 iters, quadratic)")
print("\nDONE.")
