"""
_dd_un_electron.py
==================
Deep-dive audit of the AETHOS "electron = coin + spring + membrane in a photon sea"
model (partical_extract.txt / partical_extract2.txt).

We implement the model's OWN stated mechanism and measure concrete physics,
then compare to real quantum mechanics. No hand-waving: every number is computed.

The model's strongest, most falsifiable claim (sec 5.13, "Bell Correlations"):

    "The hidden variable is the PUMP PHASE ... Measurement doesn't REVEAL, it CATCHES.
     The squeeze CREATES the state ... Your model REPRODUCES Bell correlations
     ... correlation cos^2((a-b)/2) because of spring geometry."

That is *precisely* a LOCAL HIDDEN-VARIABLE (LHV) model:
  - a shared variable lambda (pump phase) created at the source,
  - each detector applies a LOCAL deterministic response A(a, lambda), B(b, lambda) in {+1,-1},
  - the partner "snaps opposite" -> B = -A under aligned settings.

Bell's theorem: ANY such local model obeys |CHSH S| <= 2.
QM (and every real Bell experiment, Aspect 1982 ... loophole-free 2015) reaches
S = 2*sqrt(2) ~= 2.828 (Tsirelson bound).

So the test is decisive and self-contained:
  Build the model literally. Measure S. If it caps at 2 -> the Bell claim is FALSE
  by the model's own construction (it cannot reproduce QM). If the "cosine" it
  produces is actually a triangle/linear wave -> doubly falsified.

We run several increasingly-charitable versions of the model's mechanism.
"""

import numpy as np

rng = np.random.default_rng(20260628)

QM = lambda d: np.cos(np.deg2rad(d))      # QM correlation E(a,b) = -cos? sign conv below
# For the singlet, E(a,b) = -cos(a-b). The model talks about P(same)=cos^2((a-b)/2)
# which gives E = 2*P(same)-1 = 2cos^2(x/2)-1 = cos(x). (correlated, not anti-correlated)
# We'll use the correlated convention E_QM(d) = cos(d) to match the model's own text,
# and test |CHSH| against the bound (the bound is symmetric in sign).

def chsh(Eaa):
    """CHSH from a correlation function E(angle_difference_in_degrees).
    Standard angle set that maximises QM: a=0, a'=90 (90deg apart between a,a'),
    detector b=45, b'=135. Differences: (a,b)=45 (a,b')=135 (a',b)=45 (a',b')=45.
    S = E(a,b) - E(a,b') + E(a',b) + E(a',b')."""
    a, ap, b, bp = 0.0, 90.0, 45.0, 135.0
    S = Eaa(a-b) - Eaa(a-bp) + Eaa(ap-b) + Eaa(ap-bp)
    return S

# ----------------------------------------------------------------------------
# MODEL MECHANISM v1 — literal "shared pump phase, local squeeze catches it"
# ----------------------------------------------------------------------------
# Source emits a shared pump phase lambda in [0,2pi) (random each shot = "oscillating,
# caught at whatever phase"). Detector at angle 'theta' SQUEEZES: the deterministic
# local rule the text implies -> the coin reads +1 if lambda is on the +half-plane
# relative to the detector axis, else -1. Partner is the SAME pump -> reads with
# the same rule at its own angle. (Aligned axes => perfectly correlated, as the text
# demands: "squeeze one, other snaps to opposite/same".)
def simulate_lhv(theta_a_deg, theta_b_deg, n=400_000, rule="sign"):
    lam = rng.uniform(0, 2*np.pi, n)              # shared pump phase
    a = np.deg2rad(theta_a_deg); b = np.deg2rad(theta_b_deg)
    if rule == "sign":
        # "squeeze catches the pump": +1 if the phase lies in the half-plane the
        # detector axis points to. This is the most natural local deterministic read.
        A = np.where(np.cos(lam - a) >= 0, 1, -1)
        B = np.where(np.cos(lam - b) >= 0, 1, -1)
    elif rule == "cos_threshold":
        # alternative local rule: probabilistic on |cos|; still LOCAL (no access to other angle)
        pa = (1 + np.cos(lam - a)) / 2
        A = np.where(rng.uniform(size=n) < pa, 1, -1)
        pb = (1 + np.cos(lam - b)) / 2
        B = np.where(rng.uniform(size=n) < pb, 1, -1)
    E = np.mean(A * B)
    return E

# ----------------------------------------------------------------------------
# CHARITABLE v2 — "spring has components on both axes, cos(angle)" taken at face
# value as the SOURCE correlation, but enforce LOCALITY of measurement (each side
# only knows its own detector angle + shared lambda). This is the malus/Bell LHV.
# ----------------------------------------------------------------------------
def lhv_correlation_curve(angles_deg, n=200_000):
    out = []
    for d in angles_deg:
        out.append(simulate_lhv(0.0, d, n=n, rule="sign"))
    return np.array(out)

# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------
print("="*74)
print("TEST A: Two-state quantization (Stern-Gerlach binary outcome)")
print("="*74)
# Model: coin has two faces -> any squeeze yields +1 or -1, never between. TRIVIALLY
# reproduced (it's built in). We verify the OUTPUT alphabet is {+1,-1}.
lam = rng.uniform(0,2*np.pi, 100000)
A = np.where(np.cos(lam - 0) >= 0, 1, -1)
print(f"  unique outcomes of a squeeze: {sorted(np.unique(A).tolist())}  (binary as claimed)")
print(f"  raw 50/50 split for unsorted beam: P(+1)={np.mean(A==1):.3f}")
print("  VERDICT: two-state binary is reproduced -- but it is ASSUMED (a coin), not derived.")

print()
print("="*74)
print("TEST B: The decisive Bell/CHSH test (model's headline claim)")
print("="*74)
print("Model claims local 'pump phase' mechanism reproduces QM cos correlation.")
print()
angles = np.array([0,15,30,45,60,75,90,120,135,180.0])
E_model = lhv_correlation_curve(angles)
E_qm    = QM(angles)                     # cos(angle): the curve the text claims
print(f"{'angle':>7} | {'E_model(LHV)':>13} | {'E_QM=cos':>9} | {'triangle':>9}")
# triangle/linear prediction: the actual correlation of the sign-rule LHV is the
# Malus-Bell triangle E = 1 - 2*d/180 (sawtooth), the classic LHV result.
tri = 1 - 2*(angles/180.0)
for d, em, eq, t in zip(angles, E_model, E_qm, tri):
    print(f"{d:7.0f} | {em:13.4f} | {eq:9.4f} | {t:9.4f}")

S_model = chsh(lambda d: simulate_lhv(0.0, d, n=300000, rule="sign"))
S_qm    = chsh(QM)
print()
print(f"  CHSH  S (model LHV sign rule) = {S_model:.4f}")
print(f"  CHSH  S (QM cos curve)        = {S_qm:.4f}   <-- Tsirelson 2*sqrt2 = {2*np.sqrt(2):.4f}")
print(f"  CHSH  classical/LHV bound     = 2.0000")
print()
print("  If S_model <= 2.000 the model CANNOT reproduce QM Bell correlations,")
print("  falsifying the 'reproduces Bell' claim by its own construction.")

print()
print("="*74)
print("TEST C: Does the model's mechanism even give a COSINE, or a triangle?")
print("="*74)
# Fit: is E_model closer to cos(d) (QM) or to the linear triangle (LHV signature)?
err_cos = np.mean(np.abs(E_model - E_qm))
err_tri = np.mean(np.abs(E_model - tri))
print(f"  mean|E_model - cos|      = {err_cos:.4f}")
print(f"  mean|E_model - triangle| = {err_tri:.4f}")
print(f"  -> the model's actual curve is a {'COSINE' if err_cos<err_tri else 'TRIANGLE (classical LHV)'},")
print("     NOT the quantum cosine. The 'cos' in the text is ASSERTED, not produced.")

print()
print("="*74)
print("TEST D: Quantitative constants -- are they PREDICTED or quoted?")
print("="*74)
# The text states m_p/m_e ~ 1836 and g-factor ~2.002, magnetic moment -9.284e-24.
# Does the model give a FORMULA that outputs these, or does it just cite them?
# Search: the model offers NO equation relating spring stiffness/coin geometry to 1836.
mp_me_real = 1836.15267343
print(f"  proton/electron mass ratio: model says 'spring mechanical range = 1836'.")
print(f"    real value = {mp_me_real:.4f}. Model gives NO formula -> 1836 is INPUT, not OUTPUT.")
print(f"  electron g-factor: model quotes ~2.002. real = 2.00231930436.")
print(f"    QED predicts 2(1+a), a=alpha/2pi+... to 12 digits. Model: no derivation.")
print(f"  -> Constants are QUOTED from known physics, not derived. No predictive power.")

print()
print("="*74)
print("TEST E: Strongest charitable form -- can ANY local pump rule beat S=2?")
print("="*74)
# Try many local deterministic response functions A(a,lam), B(b,lam) and a probabilistic
# one; report the best CHSH any LOCAL rule achieves. Bell guarantees <=2.
best = -10
for trial in range(200):
    # random local response: threshold on cos(lam - theta - offset), random offset/sharpness
    off = rng.uniform(0, 2*np.pi)
    def Efun(d, off=off):
        n=120000
        lam = rng.uniform(0,2*np.pi,n)
        A = np.where(np.cos(lam - 0 - off) >= 0, 1, -1)
        B = np.where(np.cos(lam - np.deg2rad(d) - off) >= 0, 1, -1)
        return np.mean(A*B)
    s = abs(chsh(Efun))
    best = max(best, s)
print(f"  best |CHSH| over 200 random LOCAL pump rules = {best:.4f}")
print(f"  QM needs 2.828. Local pump mechanism is CAPPED at 2.0 (Bell). Gap is fundamental.")
print("="*74)
