# AETHOS Parameter Calibration Sheet

Single reference for **empirical anchors (E)**, **phenomenological fits (MODEL)**, and **how to close PARTIAL items** using measured data.

**Related files:** `symbol_registry.md`, `open_items_rollup.md`, `section_XX_derivations.md`.

---

## 1. How to use this sheet

| Column | Meaning |
|--------|---------|
| **ID** | Short calibration key (for code/logs) |
| **Symbol** | Primary symbol(s) |
| **Closure** | Equation or rule from derivations |
| **Anchor** | Experiment / CODATA target |
| **Calibrated value** | Current best fit in this pass |
| **Sec** | Primary derivation section |
| **Status** | `E` fixed, `FIT` one-parameter match, `OPEN` not yet numerically closed |

**Work order:** fix all `E` rows first → one-parameter `FIT` rows → multi-parameter regressions → update section open tables.

---

## 2. Fixed empirical anchors (do not fit)

| ID | Quantity | Value | Sec |
|----|----------|-------|-----|
| E-c | Speed of light | `2.99792458×10⁸ m/s` | 1,12 |
| E-h | Planck constant | `6.62607015×10⁻³⁴ J·s` | 1,2 |
| E-hbar | Reduced Planck | `1.054571817×10⁻³⁴ J·s` | all |
| E-e | Elementary charge | `1.602176634×10⁻¹⁹ C` | 2–4 |
| E-me | Electron mass | `9.1093837015×10⁻³¹ kg` | 2,3 |
| E-mp | Proton mass | `1.67262192369×10⁻²⁷ kg` | 3,4 |
| E-mn | Neutron mass | `1.67492749804×10⁻²⁷ kg` | 4 |
| E-muB | Bohr magneton | `9.2740100783×10⁻²⁴ J/T` | 2,4,10 |
| E-muN | Nuclear magneton | `5.0507837461×10⁻²⁷ J/T` | 4 |
| E-mun | Neutron magnetic moment | `−1.91304273 μ_N` | 4 |
| E-geff | Effective leakage factor | `g_eff ≈ 1.76×10³` from `(2m_e/m_n)` law | 4 |
| E-Rpe | Proton/electron mass ratio | `1836.15267343` | 3 |
| E-taun | Free neutron lifetime | `879.4 s` (order `879 s` in text) | 4 |
| E-Qbeta | Neutron β Q-value | `0.78233 MeV` | 4 |
| E-Dmnp | Mass excess n−p | `1.293332 MeV/c²` | 4 |
| E-EH | Hydrogen ionization | `13.605693 eV` | 9 |
| E-EH2 | H₂ bond energy | `4.52 eV` | 9 |
| E-rH2 | H₂ bond length | `0.74 Å` | 9 |
| E-eps0 | Vacuum permittivity | `8.8541878128×10⁻¹² F/m` | 3,9 |
| E-G | Newton constant | `6.67430×10⁻¹¹ m³/(kg·s²)` | 10,11 |
| E-H0 | Hubble rate (approx.) | `67.4 km/s/Mpc` | 10,11 |
| E-rhoL | Dark-energy density | `~6×10⁻²⁷ kg/m³` | 1,11 |

---

## 3. Tier-1 / foundation parameters

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-ks | `k_s` | `T = k_s u` | STM/spring scale (material) | *OPEN* | 2 | OPEN |
| F-alpha | `α` | `u = α Re(ψ)` | Born normalization | *OPEN* | 2 | OPEN |
| F-gE | `g_E` | `H_coin` coupling; `Γ_n∝g_E²E_obs²` | Measurement strength | *FIT via SG* | 2,5 | **PARTIAL** |
| F-LambdaN | `Λ_n` | `Λ_n=2∫Γ_n dt≈2(g_E E_obs)²τ_m` | SG strong-pin (`Λ_n≥5`) | `g_E` from `calibrate_g_e_for_lambda(5,…)` | 5 | **PARTIAL** |
| F-Delta | `Δ(κ)`, `Ω(κ)` | coin bias/bounce | pump splitting | *OPEN* | 2 | OPEN |
| F-Lbounce | `L` | `ω_b = π v_int/L` | Compton/cavity scale | `~2.4×10⁻¹² m` (order) | 2 | FIT |
| F-Piobs | `Π_pin` | `|Δ_eff|/(|Δ_eff|+Ω)` | strong-pin limit | `→1` (neutron) | 4,5 | MODEL |

**Born chain (O1-1 / O2-2):** once `k_s, α` set, verify `P(x) ∝ |ψ|²` on coin domain; sea level uses `k_S` (F-kS).

**P1-v spectrum (O1-5):** calibrate `B_full` and `φ_B` from source spectra; helpers in `aethos_physics.py` (`spectral_fill`, `report_vapor_spectrum`).

---

## 4. Fusion / proton / neutron block

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-Kf | `K_f` | `R_pe = 1/(1−αK_f)` | `R_pe = 1836.15` | `K_f ≈ 0.999456` at `α=1` | 3 | FIT |
| F-alphaK | `α` (compression) | same | independent geometry | `1` (normalized) | 3 | MODEL |
| F-gobs | `g_obs`, `Ē_obs` | `dP/dt = α Γ_obs`, `t_escape=(P_c−P_0)/(αΓ_obs)` | `τ_n = 879.4 s` | **Tier A:** `Γ_obs=1/τ_n`; **Tier B (cavity):** `Γ_obs≈5.76×10⁻³ s⁻¹` | 4 | **FIT** |
| F-geff | `g_eff` | `μ_n = −g_eff (2m_e/m_n) μ_N` | `μ_n = −1.913 μ_N` | **`g_eff ≈ 1.76×10³`** | 4 | FIT |
| F-CN | `C_0,N_0,N−Z_0,R_0,b_net` | `C_N(N,Z)`, `B_share` | stability ridge | *OPEN* | 4,9 | OPEN |
| F-eta | `η` | `R_share = η N C_N` | nuclear vs free τ | *OPEN* | 4 | OPEN |

**τ_n calibration procedure** (implemented in `aethos_physics.py`)

1. Set free-neutron `R_share ≈ 0`, `Π_pin ≈ 1`.
2. Choose gap energy `P_c−P_0` (`Q_β` or `Δm_np c²`) and pump scale `ω_in0` (scale-locked or Compton cavity).
3. `Γ_obs = (P_c−P_0)/(α τ_n)` with `α = ℏ ω_in0 Π_pin`.
4. Run `python aethos_physics.py` for numeric table.

**Tier A (default consistency):** `P_gap=Q_β`, `ω_in0=Q_β/ℏ` → `Γ_obs = 1/τ_n ≈ 1.137×10⁻³ s⁻¹`.

---

## 5. Measurement / entanglement / slit

**`g_E` / `Λ_n` procedure** (implemented in `aethos_physics.py`)

1. Choose reference Stern–Gerlach: `|dB/dz|`, `L_mag`, beam speed `v` → `τ_m=L_mag/v`.
2. `E_obs = μ_B |dB/dz| L_mag`.
3. Pick target `Λ_n` (e.g. `5` for projective limit).
4. `g_E = calibrate_g_e_for_lambda(Λ_target, …)`.
5. Run `python aethos_physics.py` or `report_measurement_calibration()` for table.

Default reference: `|dB/dz|=10³ T/m`, `L=0.04 m`, `v=10⁵ m/s` → `τ_m=4×10⁻⁷ s`.

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-Gform | `Γ_0` | `Γ_form = Γ_0 C` | fringe recovery | *OPEN* | 6,8 | OPEN |
| F-Gbreak | `Γ_break` | decoherence ODE | `V(P)` slopes | *OPEN* | 6,8 | OPEN |
| F-LambdaHe | `Λ` (He) | `V = V_0 e^{−ΛP}` | Arndt–Zeilinger He | **`81 bar⁻¹ s⁻¹`** (text) | 8 | E |
| F-LambdaAr | `Λ` (Ar) | same | same experiment | **`7.1 bar⁻¹ s⁻¹`** (text) | 8 | E |
| F-etawake | `η_wake` | `A_0 ∝ η_wake √(ℏω_b)` | fringe contrast | *OPEN* | 8 | OPEN |
| F-sigmae | `σ_{e,i}` | `Γ_partner = Σ n_i σ_i v̄_i f_i` | gas scan | *OPEN* | 8 | OPEN |
| F-delta | `δ` | `τ_re^mol/τ_re^e` scaling | e vs C₇₀ pressure | `~10²–10⁴` (order) | 8 | FIT |

**3He / 4He test:** fit `Λ_{3He}`, `Λ_{4He}` under matched `(P,T)`; model predicts `Λ_{3He} ≠ Λ_{4He}` via `f_{coin,i}`.

---

## 6. Tunneling / chemistry / atom

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-kappa | `κ̄`, `E_bar` | `T ~ e^{−2κ̄L}` | STM / alpha decay | *per barrier* | 7,9 | OPEN |
| F-Cb | `C_b` | `E_bond = U_C − C_b ℏω_b |η|² Π_pin` | H₂ `4.52 eV` | *solve given `η,ω_b`* | 9 | FIT |
| F-etaAB | `η_AB` | overlap integral | ab initio | *OPEN* | 9 | OPEN |
| F-omegaB | `ω_b` | pump frequency | shell spectrum | *from `E_n`* | 2,9 | OPEN |

**H₂ `C_b` procedure:** set `r_0 = 0.74 Å`, compute `U_C(r_0)`, choose `η_AB` from overlap model, then

`C_b = |E_bond,H₂ − U_C(r_0)| / (ℏω_b |η_AB|² Π_pin)`.

---

## 7. Cosmic / dark sector

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-fpart | `f_part` | `N_eff = f_part n_n V_c` | Earth `B` surface | *OPEN* | 10 | OPEN |
| F-xiNS | `ξ_NS` | `B_NS` scaling | magnetar `B` | *OPEN* | 10 | OPEN |
| F-tauflip | `τ_flip` | geodynamo bridge | `~4.5×10⁵ yr` | *OPEN* | 10 | OPEN |
| F-PiS | `Π_s(z)` | `w = −1 + Π_s/(ρ_DE c²)` | CPL `w_0,w_a` | *OPEN* | 10,11 | OPEN |
| F-Pivac | `Π_vac` | `ρ_{S,eff} = Π_vac u_S/c²` | `ρ_Λ` | `Π_vac ≪ 1` required | 1 | OPEN |
| F-Lcell | `L_cell` | `ω_max ~ c/L_cell` | UV cutoff | *OPEN* | 1 | OPEN |
| F-Rspring | `R_spring`, `E_spring` | `σ_{γDM} = σ_geom K_sup` | direct-detection null | upper bound | 11 | OPEN |

**Cosmology fit order:** (1) `ρ_DM, ρ_b` from rotation/lensing; (2) `w(z)` from distance data; (3) constrain `Q(z)` so `Q(z=0) ≪ 3H_0 ρ_DE`.

---

## 8. Time / Zeno (Section 12 ↔ simulation)

| ID | Symbol | Closure | Anchor | Calibrated value | Sec | Status |
|----|--------|---------|--------|------------------|-----|--------|
| F-Gammaobs | `Γ_obs` | prime-split rate | Zeno width descent | reuse `1/τ_n` scale or fit from observation model | 12 | PARTIAL |
| F-GAMMA_OBS | `GAMMA_OBS` | code alias for neutron | `aethos_physics` | `1.137e-3` s⁻¹ (Tier A) | 4 | FIT |
| F-Pmax | `P_max` | `P(p) ∝ log p` | detector radix | *OPEN* | 12 | OPEN |
| F-chi | `χ` (charge) | `q = q_0 χ` | `q_e=−e`, `q_p=+e` | `χ_e=−1`, `χ_p=+1` | 2 | MODEL |

**Link to `aethos_active.py`:** prime chains and `ActiveRole` implement **address topology**, not yet physical `Γ_obs`. Map simulation step `Δt` → `∫Γ_obs dt = 1` per observation event when coupling codec to Section 12.3.1.

---

## 9. Consistency constraints (must hold simultaneously)

| Constraint | Equation | Sections |
|------------|----------|----------|
| Charge neutrality | `q_n = 0` | 4 |
| Mass ratio | `R_pe ≈ 1836.15` ↔ `K_f` | 3 |
| Neutron moment | `μ_n = −1.913 μ_N` ↔ `g_eff` | 4 |
| Lifetime | `t_escape ≈ 879 s` ↔ `Γ_obs,P_c` | 4 |
| SR clock | `f/f_0 = 1/γ`, `v_space²+v_time²=c²` | 2,12 |
| GR static | `dτ/dt = √A`, Schwarzschild `A=1−2GM/(rc²)` | 10,12 |
| No-signaling | marginals independent | 6,8 |
| DM EM null | `S_{res,DM}=0`, small `σ_{γDM}` | 11 |
| β kinematics | `P_n = P_p + P_e + P_ν`, `Q_β` | 4 |

---

## 10. Priority calibration queue

1. **F-gobs** — lock `τ_n` (single most diagnostic Section 4 parameter).
2. **F-geff** — `≈1.76×10³` from `(2m_e/m_n)` law; cross-check against `μ_cell` in Section 10.
3. **F-Cb + F-etaAB** — H₂ bond with one overlap model.
4. **F-LambdaHe/Ar + F-sigmae** — environment table for Section 8 tests.
5. **F-CN, F-eta** — nuclear stability ridge (needs `(Z,N)` dataset).
6. **F-Pivac, F-Lcell** — cosmological constant (extreme fine-tuning; document `Π_vac` only).
7. **F-PiS, F-tauflip** — astrophysical time scales.

---

## 11. Simulation hooks (`aethos_active.py` / codec stack)

| Calibration ID | Suggested code symbol | Notes |
|----------------|----------------------|--------|
| F-Kf | `K_FUSION` | compression threshold |
| F-gobs | `GAMMA_OBS` | neutron pressure build |
| F-Gform | `gamma_form` | entanglement recovery |
| F-etawake | `eta_wake` | slit wake amplitude |
| F-Gammaobs | `gamma_obs` | Zeno prime-step rate |
| F-Pmax | `P_MAX` | largest split prime |

Keep **physics calibration** in this sheet; keep **address/combinatorics** in lattice/origin modules.

---

## 12. Revision log

| Date | Change |
|------|--------|
| 2026-06-02 | Initial sheet after Tier 4 partial closure; 0 active OPEN in rollup |
