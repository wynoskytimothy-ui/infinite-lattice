# AETHOS — Physics Questions, Formulas, and Answer Map

**Purpose:** One pass over all 12 derivation files: which **open physics questions** get an explicit formula, which are **re-explanations** of known physics, and which remain **honestly open**.

**Companion files:** `section_XX_derivations.md`, `symbol_registry.md`, `calibration_sheet.md`, `conflict_log.md`, `open_items_rollup.md`.

**Legend**

| Status | Meaning |
|--------|---------|
| **PROVEN** | Math in-file, assumptions explicit |
| **ANCHORED** | Matches standard physics when parameters chosen |
| **MODEL** | Follows AETHOS postulates; needs independent test |
| **PARTIAL** | Formula exists; calibration or uniqueness incomplete |
| **GAP** | Physics wants an answer; AETHOS does not yet deliver |

---

## Part A — Master formula index (all sections)

### Section 1 — Photon sea

| # | Formula | Role |
|---|---------|------|
| 1 | `E = hf`, `p = h/λ`, `E² = (pc)² + (m²c⁴)²` | Energy–momentum bridge |
| 2 | `ds² = c²dt² − dx²`, `dτ = dt√(1−v²/c²)` | Spacetime / proper time |
| 3 | `dτ = 0` (photon) | Null clock |
| 4 | `iℏ∂ψ/∂t = Ĥψ`, `ρ = |ψ|²`, `∂ρ/∂t + ∇·J = 0` | Wave dynamics + conservation |
| 5 | `P_S ∝ T_S² ∝ |ψ_S|²`, `T_S = k_S Re(ψ_S)` | Sea-level Born (O1-1) |
| 6 | `ρ_{S,eff} = Π_vac u_S(ω_max)/c²` | Vacuum / Λ interface (O1-3) |
| 7 | Retarded Green’s function; **no FTL signaling** (T3) | Causality vs correlation |

### Section 2 — Electron (coin)

| # | Formula | Role |
|---|---------|------|
| 1 | `Ĥ_coin = (ℏ/2)Δσ_z + (ℏ/2)Ωσ_x + ℏg_E E_obs σ_z` | Trapped-photon pump |
| 2 | `f_b = v_int/(2L)`, `f = f_0/γ` | Internal clock + SR |
| 3 | `T = k_s u`, `P ∝ T² ∝ |ψ|²` | Born from spring (O2-2) |
| 4 | `q = q_0 χ`, `χ = sign[(v_pump×n_spring)·n_coin]` | Charge sign (O2-3) |
| 5 | `Δx Δp ≥ ℏ/2` | Uncertainty |

### Section 3 — Proton (fusion)

| # | Formula | Role |
|---|---------|------|
| 1 | `K < K_f` elastic; `K ≥ K_f` fused | Regime switch |
| 2 | `R_pe = 1/(1−αK_f) = L_0/L_p` | Mass ratio geometry (O3-1) |
| 3 | `K_f ≈ 1 − 1/R_pe` | Calibrated to 1836 |
| 4 | `f_b^(p) ≈ 0`, `S_p(ω_b) = 0` | No proton pump clock (O3-4) |
| 5 | `∇·(ε_eff ∇Φ) = −ρ_q`, `U = Q_p Q_e/(4πε_eff r)` | Coulomb from drain (O3-3) |
| 6 | `Ĥ_e → Ĥ_p` at `K_f` (`Ω → 0`) | Fusion Hamiltonian (O3-2) |

### Section 4 — Neutron

| # | Formula | Role |
|---|---------|------|
| 1 | `n = p + e⁻ + γ_obs`, `q_n = 0` | Composite |
| 2 | `ḋφ_in = ω_in0 − κ I_obs(t)` | Outer photon pins inner |
| 3 | `dP/dt = αΓ_obs − βR_share`, `t_escape ∼ τ_n` | Pressure → decay |
| 4 | `n → p + e⁻ + ν̄` ↔ escaped `γ_obs` | β decay (O4-3) |
| 5 | `μ_n = −g_eff (2m_e/m_n) μ_N`, `g_eff ≈ 1.76×10³` | Magnetic moment (O4-4) |
| 6 | `C_N(N,Z)`, `NC_N > (αΓ_obs)/(βη)` | Nuclear stability |
| 7 | `B_share = −b_net N C_N` | Strong-force bridge (O4-2) |

### Section 5 — Measurement

| # | Formula | Role |
|---|---------|------|
| 1 | Kraus `M_{s,n}` from `Ĥ_coin` + `Λ_n = 2∫Γ_n dt` | Compression channel (O5-1) |
| 2 | `P(up|θ) = cos²(θ/2)` | Stern–Gerlach |
| 3 | `E(α,β) = −cos(α−β)` | Bell kernel |
| 4 | Sequential noncommuting measurements | Disturbance |

### Section 6 — Entanglement

| # | Formula | Role |
|---|---------|------|
| 1 | `ḊC = Γ_form(1−C) − Γ_break C` | Coherence ODE |
| 2 | `σ_obs = A_eff η_geom S_res(ω)` | Observation cross-section |
| 3 | `Γ_form = k_lock |O_AB| J_AB` | Phase-lock formation |
| 4 | No-signaling marginals | Causality |

### Section 7 — Tunneling

| # | Formula | Role |
|---|---------|------|
| 1 | `T ~ e^(−2κL)`, `κ = √(2m(V−E))/ℏ` | WKB anchor |
| 2 | `κ` from `Ĥ_x` (coin + spring + barrier) | Microscopic (O7-1) |
| 3 | `χ̇ = Γ_rec(1−χ) − Γ_sh χ`, `T_eff = T_WKB χ_ss` | Recoherence (O7-2) |
| 4 | `ΔV ∝ η_share Γ_sh` | Entanglement cost (O7-3) |

### Section 8 — Double slit

| # | Formula | Role |
|---|---------|------|
| 1 | `Ψ = A_L + A_R`, `I = |Ψ|²`, `V = (I_max−I_min)/(I_max+I_min)` | Interference |
| 2 | `A_s = ∫ G S_s d³r'`, `φ_R = φ_L + π` | Sea wakes (O8-2) |
| 3 | `Γ_partner = Σ n_i σ_i v̄_i f_i` | Partner sourcing (O8-1) |
| 4 | `V(P) = V_0 e^(−ΛP)` | Decoherence vs pressure |
| 5 | `τ_re^mol/τ_re^e ~ (M_mol/m_e)^(1/2)(L_mol/L_e) N_dof^(−δ)` | e vs C₇₀ (O8-3) |

### Section 9 — Atom

| # | Formula | Role |
|---|---------|------|
| 1 | `E_n ≈ −13.6 Z_eff²/n² eV`, Rydberg lines | Spectra |
| 2 | `N_max(n) = 2n²` | Shells |
| 3 | `E_bond = U_C − C_b ℏω_b |η_AB|² Π_pin` | Chemistry (O9-2) |
| 4 | `ψ_nlm = R_nl A_lm`, winding → `l, m_l` | Orbitals (O9-3) |
| 5 | `λ_α ∝ e^(−2G)`, fusion `P ~ e^(−2κL)` | Decay / fusion |

### Section 10 — Cosmic scales

| # | Formula | Role |
|---|---------|------|
| 1 | `M_core = Σ μ_i`, `B ∝ |M_core|/R³` | Magnetization |
| 2 | `τ_flip` from `ΔU(P)` geodynamo bridge | Field reversal (O10-2) |
| 3 | `v_flow = √(2GM/r)`, `dτ/dt = √(1−2GM/(rc²))` | Gravity–time |
| 4 | `P_sea,DE`, `w(z) = −1 + Π_s/(ρ_DE c²)` | Dark energy (O10-3) |
| 5 | Friedmann + structure-growth constraints | Cosmology fit layer |

### Section 11 — Dark sector

| # | Formula | Role |
|---|---------|------|
| 1 | `∇²Φ = 4πG(ρ_b + ρ_DM)` | Gravity |
| 2 | `σ_γDM = σ_geom K_sup`, `S_res,DM = 0` | EM silence (O11-1) |
| 3 | `Q = ρ_NM (Γ_sep/(m_N c²)) Ē_γ` | Sector transfer (O11-2) |
| 4 | Classes A–D no-detection theorem | Detectability (O11-3) |
| 5 | `f_clock^DM,coh = 0`, thermal tail `S_clock ≪ 1` | Clockless DM (→ Sec 12) |

### Section 12 — Zeno + time

| # | Formula | Role |
|---|---------|------|
| 1 | `w_n = w_0/∏p_k > 0` finite; `w_n → 0` only asymptotically | No terminal instant |
| 2 | `v_space² + v_time² = c²`, `v_time = c√A` | Motion budget |
| 3 | `Γ_obs` prime-split: `w_{k+1} = w_k/p_k` | Zeno microprocess (O12-1) |
| 4 | SR + static GR + weak field + FLRW maps | Unified dilation |

---

## Part B — What physics wants answered (and what AETHOS offers)

### B1. Interpretational / foundational

| Physics question | Standard situation | AETHOS answer (formula / story) | Status | Honest limit |
|------------------|-------------------|----------------------------------|--------|--------------|
| Why Born rule `P∝|ψ|²`? | Postulate / Gleason | `P ∝ T²`, `T = k_s Re(ψ)` (coin + sea) | **PARTIAL** | Uniqueness not proven; Gleason route open |
| What is measurement? | Interpretations disagree | Compression pins inner photon; Kraus `M_n` from `Ĥ_coin` | **PARTIAL** | Full unitary+environment derivation incomplete |
| Wave–particle duality? | Formalism unifies | Disturbance in sea; coin cavity | **MODEL** | Not a new prediction |
| EPR / Bell? | QM wins | Same `E = −cos(α−β)`; sea correlations, no signaling | **ANCHORED** | Kernel not derived from coin geometry alone (O5-3) |
| “Connected ocean” vs causality? | Relativity + QM | Correlation without FTL (T3, retarded G) | **PARTIAL** | Wording discipline required (`conflict_log` #1) |
| Path before measurement? | No classical path | Wake / partner picture; channel decoherence | **GAP/MODEL** | Path realism not proven (`conflict_log` #3) |
| Origin of probability? | Deep open | Tension-squared deposition | **PARTIAL** | Needs env + nonlinear spring proof |

### B2. Mass, charge, and particle identity

| Physics question | Standard situation | AETHOS answer | Status | Honest limit |
|------------------|-------------------|---------------|--------|--------------|
| Why `m_p/m_e ≈ 1836`? | Input parameter | `R_pe = 1/(1−αK_f)`, `K_f ≈ 0.999456` | **PARTIAL** | `K_f` from geometry not derived (needs O2-1 material) |
| Why charge quantization? | Gauge theory | `q = q_0 χ` topological | **PARTIAL** | Why fusion flips `χ` not proven |
| What is electron “spin”? | Dirac / SU(2) | Inner-photon two-sided coin | **MODEL** | Matches SG stats when mapped |
| Proton internal structure? | QCD | Fused drain; no electron-like clock | **MODEL** | Not a QCD derivation |
| Neutron lifetime `τ_n`? | Measured | `t_escape=(P_c−P_0)/(αΓ_obs)`; **FIT:** `Γ_obs=1/τ_n` (Tier A) or `≈5.76×10⁻³ s⁻¹` (cavity) | **FIT** | `aethos_physics.py`; not SM weak-rate derivation |
| Negative `μ_n`? | QCD + lattice | Trapped-electron leakage: `μ_n = −g_eff(2m_e/m_n)μ_N` | **PARTIAL** | `g_eff` fitted; micro leakage open |
| What is neutrino? | Standard Model extension | Escaped outer observation photon; `P_ν² ≈ 0` | **PARTIAL** | Not full weak interaction theory |
| Lepton number? | Conserved in SM | `N_obs` bookkeeping | **MODEL** | Convention, not derivation |

### B3. Forces (as reframed)

| Physics question | Standard situation | AETHOS answer | Status | Honest limit |
|------------------|-------------------|---------------|--------|--------------|
| What is EM? | QED | Drain/pump potentials → Coulomb at leading order | **PARTIAL** | `ε_eff = ε_0` leading; not QED |
| What is strong force? | QCD gluons | Trapped-electron entanglement network across neutrons | **MODEL** | `B_share`, `C_N` phenomenological |
| What is weak force? | W/Z bosons | β channel via outer-photon escape | **GAP** | No full `G_F` derivation from micro |
| Why nuclear stability ridge? | SEMF + shell | `B_total = B_SEMF + B_share(N,Z)` | **PARTIAL** | Parameters `C_0, b_net, …` open |
| Coulomb from first principles? | QED | `U = Q_p Q_e/(4πε_eff r)` from drain | **PARTIAL** | `χ_sea` micro term open |

### B4. Quantum phenomena

| Physics question | Standard situation | AETHOS answer | Status | Honest limit |
|------------------|-------------------|---------------|--------|--------------|
| Double-slit without which-path? | Standard QM | Sea wake interference `I = |A_L+A_R|²` | **ANCHORED** math | Ontology extra |
| Environment decoherence? | Decoherence theory | `V(P)=V_0 e^(−ΛP)`, `Γ_partner(Σ n_i…)` | **PARTIAL** | Species constants open |
| **3He vs ⁴He** discriminator? | Nearly identical in SM | Different `f_coin`, `Λ_3He ≠ Λ_4He` | **MODEL — testable** | **Key novel prediction** |
| e vs large molecule recoherence? | Known trend | `τ_re^mol/τ_re^e` scaling | **PARTIAL** | Order-of-magnitude |
| Tunneling through barrier? | WKB | Same + `κ` from `Ĥ_x`, `T_eff = T_WKB χ_ss` | **PARTIAL** | Barrier params open |
| Quantum Zeno? | Standard | Repeated pin → suppressed transition | **ANCHORED** qualitatively | |
| “No instant” in time? | Philosophy / limits | Finite `w_n > 0`; prime-split descent | **PROVEN** (math) + **PARTIAL** (physics link) |

### B5. Atoms and chemistry

| Physics question | Standard situation | AETHOS answer | Status | Honest limit |
|------------------|-------------------|---------------|--------|--------------|
| Shell structure `2n²`? | QM + Pauli | Same count; geometry interpretation | **PROVEN** count | Interpretation only |
| Orbital shapes? | Spherical harmonics | Membrane winding → `A_lm` | **PARTIAL** | Full spectrum without hydrogenic import open |
| Bond energies? | Quantum chemistry | `E_bond` from `η_AB`, `ω_b`, `U_C` | **PARTIAL** | `C_b`, `η_AB` calibration |
| Periodic table trends? | Aufbau + QED | Valence / `Z_eff` layer | **ANCHORED** | No new table |

### B6. Astrophysics and cosmology

| Physics question | Standard situation | AETHOS answer | Status | Honest limit |
|------------------|-------------------|---------------|--------|--------------|
| What is dark matter? | Unknown particle | Spring without inner photon | **MODEL** | Not a particle catalog |
| Why DM no EM? | Assumed | `S_res,DM=0`, `σ_γDM ∝ (ℏω/E_spring)⁴` | **PARTIAL** | Universal detection theorem incomplete |
| What is dark energy? | Λ or field | Freed inner photons pressurize sea; `w(z)` | **PARTIAL** | `Π_s(z)`, `Q` calibration open |
| Cosmological constant problem? | 10¹²⁰ mismatch | `Π_vac ≪ 1` locked-mode fraction | **GAP** | No derived `Π_vac` value |
| Flat rotation curves? | DM halo | `ρ_DM` in Poisson | **ANCHORED** if `ρ_DM` chosen | Ontology ≠ automatic fit |
| Geodynamo reversal time? | MHD models | `τ_flip` from `ΔU(P)` bridge | **PARTIAL** | `~4.5×10⁵ yr` target not fit |
| Gravity + time dilation unity? | GR | `v_space²+v_time²=c²`, `v_time=c√A` | **PARTIAL** | Not full Einstein equations |
| Black hole interior? | Open in QG | Budget → `dτ/dt→0` at `r_s` | **ANCHORED** limit | No firewall resolution |

---

## Part C — Where AETHOS is strongest (novel or discriminating)

These are the places the theory **most clearly tries to answer what physics still debates**, with a formula attached:

1. **Neutron as pinned composite** — explains stability in nucleus vs `τ_n` free via `R_share(N)`; β as outer-photon escape.
2. **Strong force as entanglement sharing** — `B_share(N,Z)` replaces gluons with network pressure relief (needs data fit).
3. **Dark matter = spring-only** — explains EM null + gravity; predicts tiny `σ_γDM`, no spectral clocks.
4. **Dark energy = freed pump photons** — `w(z) = −1 + Π_s/(ρ_DE c²)` with sector transfer `Q`.
5. **Time as motion budget** — unifies SR `γ` and static GR redshift without extra time postulate.
6. **No realized instant** — theorem on finite prime descent vs Zeno language.
7. **Environment-specific interferometry** — `Γ_partner` composition; **3He/⁴He** inequality.
8. **Mass ratio from fusion compression** — `R_pe ↔ K_f` (if `K_f` ever derived from geometry, this becomes major).

---

## Part D — Where physics still has no answer (AETHOS also GAP)

| Open problem in physics | AETHOS status |
|-------------------------|---------------|
| Cosmological constant value | `Π_vac` introduced; **no number** |
| Dark matter particle identity / detection | Ontology yes; **micro parameters open** |
| Baryon asymmetry | **Not addressed** |
| Three generations | **Not addressed** |
| Quantum gravity (full) | Budget language only; **no Einstein derivation** |
| Proton radius / form factors from QCD | **Not QCD** |
| Hierarchy problem (Higgs scale) | **Not addressed** |
| Black hole information | **Not addressed** |
| Measurement problem (unique solution) | **PARTIAL** Born + channels |
| Why these constants? | Mostly **E** inputs + fits |

---

## Part E — Derivation agenda: questions → next formulas to prove

Priority order to turn **MODEL** into **discriminating physics**:

| Priority | Question | Target derivation | Depends on |
|----------|----------|-------------------|------------|
| 1 | Why `τ_n = 879 s`? | **Done (FIT):** Sec 4.5.5 + `aethos_physics.py`; next: weak-rate comparison / `Φ_obs` env | Sec 4, 6 |
| 2 | Is ³He ≠ ⁴He in `Λ`? | Fit `σ_{e,i}`, `f_{coin,i}` | Sec 8 experiment |
| 3 | Does `K_f` follow from coin geometry? | `K_f` from `(L_0, k_s, Δ)` | O2-1 material |
| 4 | Unique Born rule? | Nonlinear spring + env → only `|ψ|²` | O1-1, O2-2 |
| 5 | `Π_vac` or `L_cell` from sea modes? | Numeric `ρ_Λ` | O1-3 |
| 6 | Halos + CMB with one DM model? | Joint `ρ_DM`, `σ_γDM` | Sec 11 sim |
| 7 | `B_share` vs `(N,Z)` table | Fit `C_0, b_net` | Sec 4,9 data |
| 8 | Bell kernel from coin only? | `E(α,β)` from geometry | O5-3 |
| 9 | `w(z)` vs SN/BAO | Fit `Π_s(z)`, `Q` | Sec 10–11 |
| 10 | Proton clock null experiment | `S_p(ω)` bound | O3-4 |

---

## Part F — How to use this with the rest of the repo

1. **Per-section detail** → `section_XX_derivations.md` (proofs, tests, tags).
2. **Symbols** → `symbol_registry.md`.
3. **Numbers** → `calibration_sheet.md` (fit queue).
4. **Contradictions** → `conflict_log.md` (wording + theorem gaps).
5. **Simulation** → map `F-*` IDs to code; physics constants ≠ lattice primes (`calibration_sheet` §11).

---

## Part G — One-sentence verdict

**AETHOS does not yet replace the Standard Model or GR;** it offers a **single mechanical narrative** (sea + trapped photon + compression + networks) that **re-anchors much known physics** and **points at specific new tests** (neutron pressure, DM EM nulls, ³He/⁴He, sector transfer, motion-budget time). The **honest frontier** is turning **PARTIAL** closures into **calibrated, falsified** predictions—not adding more postulates.

---

*Revision: 2026-06-02 — initial master map after Tier 4 + calibration sheet.*
