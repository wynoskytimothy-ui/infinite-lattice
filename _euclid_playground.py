"""
PLAYGROUND — Euclid / GCD / continued-fraction lens on the AETHOS meet.

Ground-truth meet:  meet(a,p) = (a+p, min(a,p), a+p)   X=Z=sum, Y=min
Inverse (unmeet):   from (X,Y):  small = Y,  large = X - Y.

KID QUESTION: the inverse meet hands back (min, max-min) = (smaller, larger-smaller).
That is EXACTLY one subtraction step of the subtractive Euclidean algorithm.
So if I keep meeting/unmeeting, do I fall into GCD? continued fractions? Stern-Brocot?
"""
import math
from fractions import Fraction

# ---- the two primitive lattice ops, taken verbatim from the verified formula ----
def meet(a, p):
    """forward meet -> (X, Y, Z)"""
    return (a + p, min(a, p), a + p)

def unmeet(X, Y):
    """invert a meet node (X,Y) back to its two operands {small, large}"""
    small = Y
    large = X - Y
    return small, large

# sanity: meet then unmeet is identity (round trip)
def roundtrip_check():
    bad = 0
    for a in range(1, 60):
        for p in range(1, 60):
            X, Y, Z = meet(a, p)
            s, l = unmeet(X, Y)
            if {s, l} != {a, p}:
                bad += 1
    return bad


# =====================================================================
# EXPERIMENT 1 — does iterating the meet's own subtraction compute GCD?
# The unmeet of (a, b) hands back (min, max-min). Replace the larger by
# (larger - smaller) using ONLY lattice ops, loop until one side is 0.
# =====================================================================
def lattice_gcd(a, b, trace=False):
    """
    Subtractive GCD driven purely by meet/unmeet.
    State is a pair {x, y}. Each step:
      node = meet(x, y)           -> (X, Y, Z) with Y=min, X=sum
      small, _ = unmeet(X, Y)     -> small = Y = min
      big_minus_small = X - 2*Y   ... NO. We want max - min.
    Actually max = max(x,y); max-min = (x+y) - 2*min = X - 2*Y.
    But the *lattice* already exposes Y=min and X=sum. So max-min = X - 2*Y,
    one add/sub away. We use ONLY values the meet node exposes (X, Y).
    """
    steps = []
    while min(a, b) != 0:
        X, Y, Z = meet(a, b)        # Y = min, X = a+b
        m = Y                       # the smaller, straight off the node
        diff = X - 2 * Y            # the larger minus the smaller (max - min)
        if trace:
            steps.append((a, b, m, diff))
        a, b = m, diff              # subtractive step
    g = max(a, b)
    return (g, steps) if trace else g


# =====================================================================
# EXPERIMENT 2 — continued fraction of a/b via REPEATED meets.
# Subtractive Euclid produces unary quotients; batch the equal-subtractions
# to get the CF partial quotients [a0; a1, a2, ...]. Count how many times the
# meet-subtraction fires before the min flips -> that count IS the CF digit.
# =====================================================================
def lattice_cf(num, den, maxlen=40):
    """
    Continued fraction of num/den, computed by counting meet-subtractions.
    Classic subtractive trick: repeatedly subtract the smaller from the larger;
    the run-length of subtractions of the same value = next CF partial quotient.
    Every subtraction here is the lattice unmeet step (max-min = X - 2Y).
    """
    a, b = num, den
    cf = []
    count = 0
    # we track subtractions of (smaller from larger); when larger drops below
    # smaller, the run ends and we emit the run-length.
    while b != 0 and len(cf) < maxlen:
        # how many times does b go into a, done by REPEATED lattice subtraction?
        q = 0
        while a >= b and b != 0:
            X, Y, Z = meet(a, b)     # Y=min=b (since a>=b), X=a+b
            # subtract smaller (Y) from larger: larger-min = X - 2Y = a - b
            a = X - 2 * Y            # == a - b
            q += 1
            count += 1
        cf.append(q)
        a, b = b, a                  # swap roles (now reduce b by the remainder a)
    return cf, count


def true_cf(num, den, maxlen=40):
    a, b = num, den
    cf = []
    while b != 0 and len(cf) < maxlen:
        cf.append(a // b)
        a, b = b, a % b
    return cf


# =====================================================================
# EXPERIMENT 3 — CF of irrationals (sqrt2, phi) by feeding rational
# convergent approximations through the lattice subtractor.
# =====================================================================
def cf_of_real(x, terms=12):
    """true CF of a real via the floor/reciprocal algorithm (reference)"""
    cf = []
    for _ in range(terms):
        a = math.floor(x)
        cf.append(a)
        frac = x - a
        if frac < 1e-12:
            break
        x = 1.0 / frac
    return cf


# =====================================================================
# EXPERIMENT 4 — does the unmeet tree draw the Stern-Brocot / Farey mediant?
# Stern-Brocot: a node (a/b, c/d) has child mediant (a+c)/(b+d).
# The meet's X = a+p is literally a MEDIANT NUMERATOR+DENOMINATOR sum when we
# feed it (num, den) pairs. Test: does iterating meet on fraction-pairs walk
# the Stern-Brocot tree toward a target rational, and does the L/R path equal
# the CF digits?
# =====================================================================
def stern_brocot_path(target: Fraction, maxsteps=64):
    """Walk Stern-Brocot tree to target; return L/R path + mediants visited."""
    lo = Fraction(0, 1)   # left boundary 0/1
    hi = Fraction(1, 0) if False else None
    # use the classic 0/1 .. 1/0 boundaries with explicit num/den
    ln, ld = 0, 1
    hn, hd = 1, 0
    path = []
    mediants = []
    for _ in range(maxsteps):
        # MEDIANT via the meet's additive law: numerator = ln+hn, denom = ld+hd
        mn = ln + hn       # == meet(ln,hn).X
        md = ld + hd       # == meet(ld,hd).X
        med = Fraction(mn, md)
        mediants.append(med)
        if med == target:
            path.append('=')
            break
        elif target < med:
            path.append('L')
            hn, hd = mn, md
        else:
            path.append('R')
            ln, ld = mn, md
    return path, mediants


def runlength(path):
    """compress an L/R path into run-lengths -> these should be the CF digits"""
    if not path:
        return []
    runs = []
    cur = path[0]
    c = 0
    for ch in path:
        if ch == '=':
            break
        if ch == cur:
            c += 1
        else:
            runs.append(c)
            cur = ch
            c = 1
    runs.append(c)
    return runs


if __name__ == "__main__":
    print("=== roundtrip meet/unmeet identity (1..59 x 1..59) ===")
    print("mismatches:", roundtrip_check())

    print("\n=== EXP1: lattice_gcd vs math.gcd ===")
    pairs = [(48, 18), (1071, 462), (17, 5), (100, 75), (270, 192), (13, 13), (1, 99)]
    ok = True
    for a, b in pairs:
        lg = lattice_gcd(a, b)
        mg = math.gcd(a, b)
        flag = "OK" if lg == mg else "FAIL"
        if lg != mg:
            ok = False
        print(f"  gcd({a:5d},{b:5d}) lattice={lg:4d}  math={mg:4d}  {flag}")
    print("  ALL MATCH:", ok)

    print("\n=== EXP2: lattice_cf vs true CF (run-length of meet-subtractions) ===")
    cf_ok = True
    for a, b in [(415, 93), (1071, 462), (355, 113), (45, 16), (649, 200)]:
        lcf, nsub = lattice_cf(a, b)
        tcf = true_cf(a, b)
        match = (lcf == tcf)
        if not match:
            cf_ok = False
        print(f"  {a}/{b}: lattice_cf={lcf}  ({nsub} subtractions)")
        print(f"           true_cf   ={tcf}   match={match}")
    print("  ALL CF MATCH:", cf_ok)

    print("\n=== EXP3: irrationals via Stern-Brocot mediant walk (meet's additive law) ===")
    # 355/113 ~ pi ; F(n+1)/F(n) -> phi ; p/q -> sqrt2
    print("  -- pi via 355/113 --")
    p_pi, _ = stern_brocot_path(Fraction(355, 113))
    print("    SB run-lengths:", runlength(p_pi))
    print("    true CF 355/113:", true_cf(355, 113))

    print("  -- phi via Fibonacci 6765/4181 --")
    p_phi, _ = stern_brocot_path(Fraction(6765, 4181))
    print("    SB run-lengths:", runlength(p_phi))
    print("    true CF       :", true_cf(6765, 4181))
    print("    cf_of_real(phi):", cf_of_real((1 + 5 ** 0.5) / 2, 14))

    print("  -- sqrt2 via 19601/13860 --")
    p_s2, _ = stern_brocot_path(Fraction(19601, 13860))
    print("    SB run-lengths:", runlength(p_s2))
    print("    true CF        :", true_cf(19601, 13860))
    print("    cf_of_real(sqrt2):", cf_of_real(2 ** 0.5, 14))

    print("\n=== EXP4: is the meet's X literally the Stern-Brocot mediant? ===")
    # meet((ln,hn)).X = ln+hn ; meet((ld,hd)).X = ld+hd ; mediant = that pair
    ln, ld, hn, hd = 1, 2, 1, 3   # between 1/2 and 1/3
    mx = meet(ln, hn)[0]
    my = meet(ld, hd)[0]
    print(f"  mediant of 1/2,1/3 via meet.X = {mx}/{my}  (expect 2/5)")
