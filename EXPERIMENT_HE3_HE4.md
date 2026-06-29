# Experiment spec — the ³He/⁴He decoherence discriminator

*The one genuinely-novel, falsifiable thread in* Packets and Strings *(ch9.3, O8), worked into a concrete,
runnable, pre-registerable test. Computation: `_math_he_experiment.py`. Honest status: a real discriminator
between standard collisional decoherence and an isotope-dependent coupling; the model's exact number is
calibrated, so the robust claim is the **direction + magnitude band**, not a first-principles point value.*

## 1. The question
When a matter wave decoheres by collisions with a helium buffer gas, does the decoherence rate ratio
between ³He and ⁴He follow **pure kinematics** (mass/velocity only), or is there an **extra
isotope-dependent coupling** — as the lattice model predicts, motivated by the real physical difference that
**³He is a spin-½ fermion and ⁴He is a spin-0 boson**?

## 2. The two predictions (exact)
Standard collisional decoherence (Joos–Zeh / Hornberger–Sipe): Λ = n·σ·v̄. At matched gas density n and the
(near-identical) scattering cross-section σ of the same element, only the mean speed v̄ ∝ 1/√m differs:

| model | Λ(³He)/Λ(⁴He) | basis |
|---|---|---|
| **Standard (kinematic)** | **1.152** (+15.2%) | v̄₃/v̄₄ = √(m₄/m₃), m₄/m₃ = 1.327 |
| **Lattice model (this work)** | **1.075** (+7.5%) | extra isotope coupling f_coin (fermion vs boson) |

Separation to detect: **0.077** (≈6.7% of the ratio). v̄(³He)=1870 m/s, v̄(⁴He)=1623 m/s at 300 K.

## 3. Apparatus — Kapitza-Dirac-Talbot-Lau interferometer (Arndt-class, existing technology)
- **Test particle:** a heavy fullerene / functionalized molecule (C₆₀ or TPP-class), m ≈ 720–1000 u.
- **Gratings:** 3 gratings, period d = 266 nm, separation L = 0.105 m (Talbot length L_T = d²/λ_dB).
- **Buffer-gas cell:** length ℓ ≈ 0.10 m, filled with ³He **or** ⁴He at controlled pressure P_gas, T = 300 K
  (cryogenic T optional — lowers v̄, raises contrast, same ratio).
- **Observable:** fringe visibility decay `V(P_gas) = V₀·exp(−γ_iso·P_gas)`; fit γ_iso per isotope.

## 4. Protocol (the controls are everything)
1. Establish V₀ (no gas) for the chosen molecule.
2. For **each** isotope, sweep P_gas over ~6–8 points (0.05–0.8 Pa), measure V(P), fit γ_iso.
3. **Match exactly:** same molecule, same T, same number density n = P/(k_BT) (set by pressure), same cell
   geometry, same gratings. The *only* difference between runs is the isotope. (³He/⁴He are chemically
   identical — this is the cleanest possible matched control.)
4. Report **R = γ(³He)/γ(⁴He)** with its 1σ error. Cross-section σ and apparatus factors cancel in the ratio.

### Predicted visibility curves V(P)/V₀ (the two hypotheses are visibly distinct)
| P (Pa) | V(⁴He) | V(³He) standard | V(³He) model |
|---|---|---|---|
| 0.10 | 0.841 | 0.819 | 0.830 |
| 0.40 | 0.500 | 0.450 | 0.475 |
| 0.80 | 0.250 | 0.203 | 0.225 |

## 5. Precision & statistics
To separate 1.152 vs 1.075 at **5σ**, measure each γ to ~**1.3%** (1σ). The Arndt group routinely measures
collisional decoherence to **2–3%** per gas — so a first ~3σ result is immediately feasible, and 5σ needs
only more integration time (≈ a few hundred molecules per pressure point at the visibility SNR).

## 6. Decision rule (pre-register before running)
- **R = 1.152 ± 0.02** → pure-kinematic decoherence; the lattice isotope coupling is **falsified.**
- **R = 1.075 ± 0.02** → isotope coupling **beyond kinematics confirmed**; this *also* falsifies the standard
  mass-only collisional-decoherence prediction — a genuine new (spin/statistics-dependent) effect.
- **R in between** → partial; tightens f_coin, motivates a second isotope (e.g. ²⁰Ne/²²Ne) replication.

## 7. Honest caveat
The model's exact 1.075 is **calibrated** (f_coin tuned to a 5–10% band, not derived from first principles).
So the *robust* falsifiable content is: **R is significantly below the kinematic 1.152**, because the fermionic
³He and bosonic ⁴He couple to the decohering substrate differently. A measured R = 1.152 kills it cleanly; a
measured R ≈ 1.07–1.10 is a real anomaly worth chasing. Either outcome is publishable — that is what makes it
a genuine experiment rather than a story.
