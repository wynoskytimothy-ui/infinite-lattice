#!/usr/bin/env python3
"""HE-3 / HE-4 DISCRIMINATOR — the concrete, runnable experiment from Packets & Strings ch9.3 (O8).
The ONE genuinely-novel falsifiable thread: collisional decoherence of a matter-wave in a He buffer gas.

THE CLAIM:
  * Standard collisional decoherence (Joos-Zeh / Hornberger-Sipe): the decoherence rate Lambda = n * sigma *
    v_bar.  With ³He and ⁴He at MATCHED density n and (near-)identical scattering cross-section sigma (same
    element), the only difference is the mean thermal speed v ~ 1/sqrt(m):
        Lambda(³He)/Lambda(⁴He) = v3/v4 = sqrt(m4/m3) = 1.152   (PURE KINEMATICS — the SM prediction)
  * Timothy's model (aethos_physics.lambda_*): an extra ISOTOPE-DEPENDENT COUPLING f_coin (motivated by the
    real physical difference: ³He is a spin-1/2 FERMION, ⁴He a spin-0 BOSON), pulling the ratio AWAY from
    pure kinematics:  Lambda(³He)/Lambda(⁴He) ~ 1.075   (BELOW the SM 1.152, in the 5-10% band)

THE TEST: measure both rates to ~2-3% in a Talbot-Lau / KDTLI interferometer; the ratio decides.
HONEST: the exact 1.075 is calibrated (f_coin tuned to a 5-10% band); the ROBUST falsifiable content is
'ratio significantly BELOW the kinematic 1.152'. A measured 1.152 +- 0.02 falsifies the model; 1.075 +- 0.02
falsifies pure-kinematic decoherence and supports an isotope coupling."""
import math
import numpy as np
import aethos_physics as P

kB = 1.380649e-23
m3, m4 = P.M_HE3, P.M_HE4


def vbar(m, T): return math.sqrt(8 * kB * T / (math.pi * m))


def main():
    print("=" * 80)
    print("HE-3 / HE-4 DISCRIMINATOR — concrete experiment spec (collisional decoherence)")
    print("=" * 80)
    T = 300.0
    v3, v4 = vbar(m3, T), vbar(m4, T)
    sm_ratio = math.sqrt(m4 / m3)                      # = v3/v4, the kinematic SM prediction
    model_ratio = P.lambda_he3_he4_ratio_calibrated()  # Timothy's model

    print(f"\n  CONSTANTS (T={T:.0f} K):")
    print(f"    m(³He)={m3*1e27:.4f}e-27 kg   m(⁴He)={m4*1e27:.4f}e-27 kg   m4/m3={m4/m3:.4f}")
    print(f"    v̄(³He)={v3:.1f} m/s   v̄(⁴He)={v4:.1f} m/s   v3/v4={v3/v4:.4f}")
    print(f"    nuclear spin: ³He = 1/2 (FERMION)   ⁴He = 0 (BOSON)   <- the physical basis for f_coin")

    print(f"\n  THE TWO PREDICTIONS for Lambda(³He)/Lambda(⁴He):")
    print(f"    Standard collisional decoherence (kinematic): {sm_ratio:.4f}   (= +{(sm_ratio-1)*100:.1f}%)")
    print(f"    Timothy's model (isotope coupling f_coin):    {model_ratio:.4f}   (= +{(model_ratio-1)*100:.1f}%)")
    print(f"    SEPARATION to detect: {abs(sm_ratio-model_ratio):.4f} ({abs(sm_ratio-model_ratio)/sm_ratio*100:.1f}% of the ratio)")

    # ---- the interferometer + measurement protocol ----
    print(f"\n  APPARATUS — Kapitza-Dirac-Talbot-Lau interferometer (Arndt-class; existing technology):")
    print(f"    * test particle: a heavy fullerene/functionalized molecule (e.g. C60 / TPP), mass ~720-1000 u")
    print(f"    * 3 gratings, period d=266 nm, separation L=0.105 m; Talbot length L_T=d^2/lambda_dB")
    print(f"    * buffer gas cell of length l_cell ~ 0.10 m, ³He or ⁴He at controlled pressure P_gas, T=300 K")
    print(f"    * observable: fringe visibility V(P_gas) = V_0 * exp(-gamma_iso * P_gas)  (decoherence decay)")

    # decoherence coefficient gamma_iso ∝ n_density * sigma * v_rel ; n=P/(kT); v_rel ≈ v_He (light gas)
    # gamma per unit pressure (relative, sigma matched): gamma_iso ∝ v_iso / (kT)   [* sigma * l_cell / v_part]
    print(f"\n  PREDICTED VISIBILITY CURVES V(P)/V_0 (matched sigma, n; ratio is what matters):")
    Pgrid = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8])    # Pa
    # set an absolute scale so ⁴He loses half visibility at 0.4 Pa (typical), then scale ³He by each model
    g4 = -math.log(0.5) / 0.4
    g3_sm = g4 * sm_ratio
    g3_model = g4 * model_ratio
    print(f"    {'P(Pa)':>7}{'V(⁴He)':>10}{'V(³He) SM':>12}{'V(³He) model':>14}")
    for p in Pgrid:
        print(f"    {p:>7.2f}{math.exp(-g4*p):>10.3f}{math.exp(-g3_sm*p):>12.3f}{math.exp(-g3_model*p):>14.3f}")

    # ---- required precision / statistics ----
    print(f"\n  REQUIRED PRECISION & STATISTICS:")
    sep_frac = abs(sm_ratio - model_ratio) / sm_ratio
    sigma_needed = sep_frac / 5.0                       # 5-sigma separation
    print(f"    to separate {sm_ratio:.3f} vs {model_ratio:.3f} at 5σ: measure each Lambda to ~{sigma_needed*100:.1f}% (1σ)")
    print(f"    => fit gamma from ~6-8 pressure points x ~{int((0.02/sigma_needed)**2*200)} molecules/point (visibility SNR)")
    print(f"    achievable: Arndt-group KDTLI routinely measures collisional decoherence to 2-3% per isotope.")

    # ---- the decision rule ----
    print(f"\n  DECISION RULE (pre-registered):")
    print(f"    measured ratio R = Lambda(³He)/Lambda(⁴He), with 1σ error ~σ_R:")
    print(f"    * R = {sm_ratio:.3f} ± 0.02  -> PURE-KINEMATIC decoherence; Timothy's isotope coupling FALSIFIED.")
    print(f"    * R = {model_ratio:.3f} ± 0.02 -> isotope coupling beyond kinematics CONFIRMED; falsifies standard")
    print(f"      collisional-decoherence's mass-only prediction. A genuine new effect (spin/statistics-dependent).")
    print(f"    * R between -> partial; tightens f_coin.")
    print(f"\n  HONEST CAVEAT: the model's exact 1.075 is calibrated (f_coin tuned to 5-10%). The FALSIFIABLE,")
    print(f"  robust claim is: R is significantly BELOW the kinematic 1.152 because ³He(fermion) and ⁴He(boson)")
    print(f"  couple to the decohering substrate differently. That is a clean, runnable, pre-registerable test.")


if __name__ == "__main__":
    main()
