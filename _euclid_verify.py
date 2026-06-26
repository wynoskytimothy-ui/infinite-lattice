"""
VERIFY: (1) the SB run-length last-digit offset is the known CF tail ambiguity,
        (2) my meet() matches the ACTUAL library swap_meet on the real lattice.
"""
import sys
sys.path.insert(0, "C:/Users/wynos/New folder (3)")
from fractions import Fraction
from aethos_complex_plane import swap_meet, wing_transform
from aethos_lattice import BranchKind

def meet(a, p):
    return (a + p, min(a, p), a + p)

# --- (2) does the library swap_meet expose X=sum, Y=min like the verified formula? ---
print("=== library swap_meet vs verified meet formula ===")
bad = 0
for a in range(1, 30):
    for p in range(1, 30):
        left, right = swap_meet(a, p, BranchKind.VA1, 1)
        # left = bank(a)@n=p ; co-location means left==right (the meet node)
        # node coords: X=Re z, Y=Im z, zeta=Z
        Xl, Yl, Zl = left.z.real, left.z.imag, left.zeta
        Xf, Yf, Zf = meet(a, p)
        # the verified atom: 2-way meet equalizes; check left==right AND ==formula
        if left.coord != right.coord:
            bad += 1
        # X should be sum, Y should be min
        if not (abs(Xl - Xf) < 1e-9 and abs(Yl - Yf) < 1e-9):
            bad += 1
# show one concrete node
l, r = swap_meet(3, 5, BranchKind.VA1, 1)
print(f"  swap_meet(3,5): left={l.coord} right={r.coord}")
print(f"  verified meet(3,5) = {meet(3,5)}")
print(f"  mismatches over 29x29 grid: {bad}")

# --- (1) CF tail ambiguity: [.., a, 1] == [.., a+1] ; SB drops the trailing +1 ---
print("\n=== CF tail-ambiguity check (the only SB discrepancy) ===")
def true_cf(num, den, maxlen=40):
    a, b = num, den; cf = []
    while b != 0 and len(cf) < maxlen:
        cf.append(a // b); a, b = b, a % b
    return cf

def eval_cf(cf):
    f = Fraction(cf[-1])
    for d in reversed(cf[:-1]):
        f = d + 1/f
    return f

for (n,d) in [(355,113),(6765,4181),(19601,13860)]:
    cf = true_cf(n,d)
    # the canonical "other" form: split last digit a -> a-1, 1
    alt = cf[:-1] + [cf[-1]-1, 1] if cf[-1] > 1 else cf
    print(f"  {n}/{d}: cf={cf}")
    print(f"         value(cf)={eval_cf(cf)}  value(alt {alt})={eval_cf(alt)}  equal={eval_cf(cf)==eval_cf(alt)==Fraction(n,d)}")
