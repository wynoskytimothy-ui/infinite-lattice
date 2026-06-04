# Section 5 — Geometry Audit (Step 5 Gate)

**Mandates:** C1 (`L_0`), C5 (joint DM ripple; **REJECT** `sign(cos θ)` sketch).  
**Imports:** Step 2 — `H_{coin}`, `\Delta(κ)`, `\Omega(κ)`, `\Pi_{pin}`, `\omega_b`, `L_0`.

**Goal:** `\Lambda_n` and effective `M_n` from compression geometry; Bell kernel path without silent QM import.

---

## 5.G.1 `H_{coin}` measurement limit (GEOMETRY)

\[
\hat H_{coin}=\frac{\hbar}{2}\Delta(\kappa)\sigma_z+\frac{\hbar}{2}\Omega(\kappa)\sigma_x+\hbar g_E E_{obs}\sigma_z
\]

**Axis dominance (strong pin):**

\[
|g_E E_{obs}|\gg \Omega(\kappa)
\]

Code: `strong_measurement_ratio(b_grad, g_e, kappa)` = `\omega_{eff}/\Omega` with  
`\omega_{eff}=\mu_B|dB/dz|L_0/\hbar` (**C1** micro coupling).

Tag: **GEOMETRY** (Step 2 imports + field gradient).

---

## 5.G.2 `\Lambda_n` and Kraus `M_n` (GEOMETRY core)

Dephasing rate (Sec 5.3):

\[
\Gamma_n \propto g_E^2 E_{obs}^2,
\qquad
\Lambda_n = 2\int \Gamma_n\,dt \approx 2\Gamma_n\tau_m
\]

**Micro coupling (C1):**

\[
E_{obs}^{micro}=\mu_B|dB/dz|\,L_0,
\qquad
\tau_m=\frac{L_{mag}}{v_{beam}}
\]

**Macro apparatus path:** `E_{obs}=\mu_B|dB/dz|L_{mag}` — used in SG calibration (**FIT** `g_E` to target `\Lambda_n`).

Pin strength:

\[
p=\frac{1-e^{-\Lambda_n}}{2}
\]

Off-diagonal suppression `\sim e^{-\Lambda_n}`; `\Lambda_n\ge 5` ⇒ projective limit.

Code: `lambda_n_from_coin_gradient`, `kraus_decoherence_factor`, `calibrate_measurement_sg`.

Tag: **GEOMETRY** (structure); **FIT** (`g_E` for reference SG row).

---

## 5.G.3 Projection law (PROVEN / ANCHORED)

\[
P(\uparrow|\theta)=\cos^2(\theta/2)
\]

Unbiased prep `\rho=I/2` ⇒ `P=1/2`. Sequential noncommuting axes: **ANCHORED**.

Not re-derived from coin here — standard spin projection consistent with `M_n` limit.

---

## 5.G.4 Bell / O5-3 (C5)

| Model | `E(0,\pi/4)` | Status |
|-------|--------------|--------|
| QM singlet | `-\sqrt{2}/2` | **ANCHORED** (comparison) |
| `sign(cos)` half-plane | `\approx -0.5` | **REJECT** (C5, numeric falsifier) |
| Shared-phase linear projection | `-\cos(a-b)/2` | **PARTIAL** (half-scale; not full kernel) |

**C5 target (OPEN):** joint mechanical ripple on DM string — compression at A and B reads **correlated projections** on shared `\phi_{AB}` mode (P11-3, Stage B C7), not independent sign maps.

Code: `bell_correlation_joint_ripple_linear` — documents partial; full `-cos(a-b)` needs Stage B fill.

---

## 5.G.5 Observer criterion (MODEL)

Observation ⟺ nontrivial channel `\mathcal M\neq I` or `\mathcal D\neq 0`. Requires `H_{int}\neq 0` over `\Delta t>0` (Sec 5.10).

---

## 5.G.6 Geometry Gate checklist (Step 5)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `\Lambda_n` chain from `L_0`, `H_{coin}` | ✅ |
| 2 | Strong-pin ratio from `\Omega`, `\omega_{eff}` | ✅ |
| 3 | Kraus / pin `p` from `\Lambda_n` | ✅ |
| 4 | `cos²` law documented | ✅ |
| 5 | O5-3 sign sketch **REJECT** | ✅ |
| 6 | C5 joint-ripple partial + forward contract | ✅ **PARTIAL** |
| 7 | Code + test | ✅ |

**Gate verdict:** **PASS (geometry core + honest O5-3 boundary)** — full Bell kernel **OPEN** until `\phi_{AB}` joint mode (C7 Stage B / Sec 6).

---

## 5.G.7 Exports

- `\Lambda_n`, `p`, `e^{-\Lambda_n}` → Sec 6 `\Gamma_{break}`, Sec 7 decoherence  
- `strong_measurement_ratio` → pin vs hop crossover  
- O5-3 **REJECT** + partial ripple → Sec 6.12, Sec 11 P11-3  

---

*Next: Step 6 entanglement (`\Gamma_{form}`, `\phi_{AB}` Stage A) or deepen O5-3 with joint-mode projection.*
