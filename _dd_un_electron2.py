"""
_dd_un_electron2.py -- strongest-form charity + the Pauli/shell counting claim.

(1) Most charitable Bell reading: the two electrons are ONE shared pump, partner
    "snaps opposite". Implement as a genuine singlet LHV: shared lambda, B uses the
    SAME deterministic rule so that at equal angles A=-B perfectly (anti-correlated)
    or A=B (correlated). Still LOCAL. Confirm it still caps at 2.

(2) Probabilistic local model tuned to best-match QM marginals (the absolute best a
    local theory can do). Report its CHSH.

(3) Pauli / shell capacity: model claims "each pump holds 2, shell n holds n^2 pumps
    => 2,8,18 electrons". Check the arithmetic vs real 2n^2 shell filling AND vs the
    real reason (quantum numbers). Is the *number* right? Is the *mechanism* right?
"""
import numpy as np
rng = np.random.default_rng(7)

def chsh(Eaa):
    a, ap, b, bp = 0.0, 90.0, 45.0, 135.0
    return Eaa(a-b) - Eaa(a-bp) + Eaa(ap-b) + Eaa(ap-bp)

# (1) genuine shared-pump singlet LHV (anti-correlated partner)
def E_singlet_lhv(d, n=400000):
    lam = rng.uniform(0,2*np.pi,n)
    a=0.0; b=np.deg2rad(d)
    A = np.where(np.cos(lam-a)>=0, 1,-1)
    B = -np.where(np.cos(lam-b)>=0, 1,-1)   # partner snaps OPPOSITE (shared pump)
    return np.mean(A*B)
S1 = chsh(E_singlet_lhv)
print(f"(1) shared-pump singlet LHV CHSH |S| = {abs(S1):.4f}   (still <=2, Bell holds)")

# (2) optimal local probabilistic model: best fit gives at most the triangle => S<=2
# We brute force a family: A,B = sign(cos(lam-theta)) with shared lam, plus a fraction
# p of 'noise' shots; sweep p, take best CHSH.
def E_noisy(d, p, n=300000):
    lam = rng.uniform(0,2*np.pi,n)
    A = np.where(np.cos(lam)>=0,1,-1)
    B = -np.where(np.cos(lam-np.deg2rad(d))>=0,1,-1)
    mask = rng.uniform(size=n) < p
    B = np.where(mask, rng.choice([-1,1],n), B)   # local noise
    return np.mean(A*B)
best=0
for p in np.linspace(0,0.5,11):
    s=abs(chsh(lambda d,p=p: E_noisy(d,p)))
    best=max(best,s)
print(f"(2) best local probabilistic model CHSH |S| = {best:.4f}   (cannot exceed 2)")
print(f"    QM/Tsirelson = {2*np.sqrt(2):.4f}; loophole-free experiments measured ~2.4-2.42 (Hensen 2015).")

# (3) Pauli / shell counting
print()
print("(3) PAULI / SHELL CAPACITY claim:")
print("    model: 'each pump holds 2; shell n holds n^2 pumps => 2 n^2 electrons'")
real = [2*n*n for n in (1,2,3,4)]
model_text = [2,8,18]   # the text lists shells 1,2,3 as 2,8,18
print(f"    real 2n^2 for n=1..4 : {real}")
print(f"    model text lists     : {model_text} (+ would predict 32 for n=4)")
print(f"    -> the NUMBERS 2,8,18,32 are CORRECT (= 2n^2).")
print("    BUT the mechanism is wrong: real degeneracy = sum over l of 2(2l+1) = 2n^2,")
print("    driven by 4 quantum numbers (n,l,m_l,m_s) + antisymmetry of fermion wavefn.")
print("    'pump holds 2, n^2 pumps' is a numerology re-labeling: it ASSUMES 2/pump and")
print("    ASSUMES n^2 pumps to hit 2n^2. It does not derive WHY 2 or WHY n^2.")
print("    The spin-statistics theorem (the actual reason fermions exclude) is absent.")

# bonus: the model's '2 per orbital = opposite phases of one pump' is just relabeling
# the two spin states. It gives no prediction distinguishing it from standard QM.
print()
print("SUMMARY: every quantitative hook (1836, g=2.002, 2-8-18, cos) is QUOTED/ASSUMED.")
print("The one DERIVED prediction (Bell via local pump) is WRONG: triangle, S=2, not QM.")
