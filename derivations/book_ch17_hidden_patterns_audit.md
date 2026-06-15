# Chapter 17.2 — Hidden Patterns Audit (repo cross-check)

**Date:** 2026-06-05  
**Sources:** `physics_questions_map.md`, `cross_reference_matrix.md`, `conflict_log.md`, `section_*_derivations.md`, `aethos_physics.py`, `test_aethos_physics.py`.

---

## Five unification patterns (same mechanism, many questions)

### Pattern 1 — **Compression spine**

One story: observation = compression of inner photon / frame width.

| Link | Equation | Sections |
|------|----------|----------|
| Measurement | `Λ_n = 2∫Γ_n dt`, Kraus `M_{s,n}` | 5, `aethos_physics.lambda_n_*` |
| Born deposition | `P ∝ T² ∝ \|ψ\|²` | 1.5.5a, 2.8 |
| Zeno width | `w_{k+1} = w_k/p_k`, `dw/dt = −λ_desc w` | 12.3.1 |
| Pin probability | `p = 1 − e^{−Λ_n}` | 5, code `measurement_pin_probability` |

**Answers together:** measurement problem, quantum Zeno, zeptosecond discreteness (at `T_bounce`), fresh-electron statistics.

**Still open:** `Γ_n(κ, geometry)` micro closure (O5-1 tail); Gleason uniqueness.

---

### Pattern 2 — **Γ_form / Γ_break competition**

| Link | Equation | Status |
|------|----------|--------|
| Entanglement ODE | `Ċ = Γ_form(1−C) − Γ_break C` | **PROVEN** closed form (Sec 6.3.2) |
| Steady state | `C_* = Γ_form/(Γ_form+Γ_break)` | **PROVEN** + `coherence_steady_state()` |
| Lifetime | `τ_E ~ 1/(Γ_form+Γ_break)` | **DERIVED** |
| Geometry | `Γ_form = k_lock \|O_AB\| J_AB φ_AB`, `k_lock ~ f_b` | **PARTIAL** Stage A in code |

**Answers together:** collapse vs unitary (effective), Schrödinger cat (as `Γ_break` dominance), entanglement lifetime.

**Book under-tags:** 17.2.2 listed MODEL — ODE math is **PROVEN**; interpretation remains MODEL.

---

### Pattern 3 — **Partner-photon / environment spine**

| Link | Equation | Sections |
|------|----------|----------|
| Partner rate | `Γ_partner = Σ n_i σ_i v̄_i f_i` | 8.7.1, `gamma_partner_rate()` |
| Visibility | `V(P) = V_0 e^{−ΛP}` | 8 |
| Double slit | `I = \|A_L+A_R\|²`, `φ_R = φ_L + π` | 8, `demo_slit_fringe_intensity()` |
| Eraser | restore when `Γ_form > Γ_partner` | 8 narrative |

**Answers together:** which-path, quantum eraser, environment decoherence, double-slit ontology.

**Still open:** `σ_{e,i}` from geometry (O8-1); path realism wording (`conflict_log` #3).

---

### Pattern 4 — **φ_AB fill bridge (Bell + DM + entanglement)**

| Link | Equation | Status |
|------|----------|--------|
| Bell contract | `E(a,b) = −φ_AB cos(a−b)` | **ANCHORED** at `φ_AB=1` (C7 Stage B) |
| Half-scale partial | `φ_AB = 0.5` → half QM correlation | **PARTIAL** (`bell_correlation_joint_ripple_linear`) |
| Rejected sketch | `sgn(cos θ)` coins | **REJECT** (test falsifier) |
| Formation | `Γ_form ∝ φ_AB` | Sec 6.13, P11-3 |

**Answers together:** Bell kernel, EPR correlations, entanglement path on DM string.

**Book fix:** 17.1.5 / 17.2.4 should cite **contract + φ_AB**, not only "PARTIAL kernel."

---

### Pattern 5 — **Address / descent spine (π, Zeno, mass)**

| Link | Equation | Status |
|------|----------|--------|
| π recurrence | `B_{k+1}=C_k/2`, `π = lim 2^{k+1}C_k` | **PROVEN** (`pi/`) |
| No instant | `w_n > 0` finite steps | **PROVEN** (Sec 12) |
| Mixed-radix position | `x_n = Σ i_k/∏p_j` | **PROVEN** |
| Mass gap | `R_pe^model = (π²/8) × M_lat` | **GEOMETRY** path (C2, C6) |

**Answers together:** Zeno, time quantization, "no bottom" vs operational `λ_C`, mass ratio program.

**Why (2026-06-05, `derivations/book_ch17_why_calibration_patterns.md`, `scripts/pattern_why_discriminators.py`):**

- `R_pe = (π²/8) × M_lat` — reciprocal of proton length shrink `L_p=(8/π²)L_0`
- `M_lat = (32/52)×Σμ` with **40 origins** (=1+3+9+27) fixed at depth 3 → linear in node count
- `n=80` = **16×5** balanced role ledger (`i%5` assignment) — not arbitrary count tuning alone
- He: `Λ_3/Λ_4 = (f_3/f_4)(m_4/m_3)`; mass gives 33%, structure factor **0.81** cancels to ~7.5%

**Calibration (2026-06-05, `scripts/calibrate_discriminators.py`):**

| Discriminator | Best profile | Result |
|---------------|--------------|--------|
| `R_pe^pred` | `SequenceKind.PRIMES`, `count=80`, `depth=3` | **1847.1** (0.6% vs CODATA) |
| `Λ_{3He}/Λ_{4He}` | `f_{coin,3}=0.405`, `f_{coin,4}=0.5` | **1.075** (7.5% band) |

Default bootstrap `count=100` still overshoots (~2444); use `r_pe_model_reference_bootstrap()` for E-check. He defaults `0.75/0.15` were placeholders — use `lambda_he3_he4_ratio_calibrated()`.

---

## Re-tag table: book Ch 17.2 vs repo truth

| §17.2 | Book tag | Repo tag | Evidence |
|-------|----------|----------|----------|
| 17.2.1 Measurement | PARTIAL | **PARTIAL+** | Kraus derived Sec 5.3; SG cal in code |
| 17.2.2 Collapse | MODEL | **PROVEN ODE + MODEL interpret** | Sec 6.3.2 |
| 17.2.3 Wave–particle | MODEL | MODEL | — |
| 17.2.4 EPR | PARTIAL | **ANCHORED kernel + PARTIAL ontology** | Sec 6.7, T3 |
| 17.2.5 Path | GAP/MODEL | **OPEN** | `conflict_log` #3 |
| 17.2.6 Which-path | A+M | A+M | `interference_intensity` |
| 17.2.7 Eraser | PARTIAL | PARTIAL | Γ_partner law |
| 17.2.8 Decoherence | PARTIAL | PARTIAL | V(P) + Γ_break |
| 17.2.9 Zeno | A+P | **PROVEN + PARTIAL link** | 12.G.1, 12.3.1 |
| 17.2.10 Uncertainty | A; mech OPEN | **A + MODEL (Sec 9.3.2)** | kinetic pressure in wells |
| 17.2.11 Tunneling | PARTIAL | **PARTIAL+code** | `t_eff_soft`, `chi_steady` |
| 17.2.14 R_pe | PARTIAL | **PARTIAL+ forward M_lat** | `r_pe_model_with_lattice()` |
| 17.2.17 Complex ψ | GAP | **PARTIAL** | `wake_amplitude_complex` — phase in sea wakes |
| 17.2.19 ³He/⁴He | TEST | **TEST + formula** | `lambda_he3_he4_ratio()` — defaults need fit |

---

## Genuinely still OPEN (not hidden in repo)

1. Path realism / definite trajectories (`conflict_log` #3)
2. Born rule uniqueness (Gleason / full nonlinear env)
3. Full unitary environment → projection
4. Pauli–Dirac statistics from coin exclusivity
5. `σ_{e,i}`, `f_{coin,i}` first-principles for He discriminator
6. `M_lat` species selection → match 1836 without tuning
7. Weak force / `G_F` from micro
8. Baryon asymmetry, three generations, hierarchy, BH information
9. Einstein field equations from lattice
10. Absolute DM non-detection theorem (`conflict_log` #4)

---

## Recommended book updates (applied in Ch 17.2 v1.2)

- Add §17.2.0 five-pattern overview box
- Re-tag collapse ODE, Zeno, Bell/φ_AB, complex wakes, M_lat path
- Add §17.2.22–24: unified compression law, φ_AB Bell closure, M_lat consequence check
- Appendix G: sync re-tags

---

*Audit complete — use with `physics_questions_map.md` Part C (strongest discriminators).*
