# Section 8 — Geometry Audit (Step 8 Gate)

**Mandates:** C1 wake scale; C6/C7 `\phi_{AB}` partner link; Sec 6 `\Gamma_{form}`, `\Gamma_{break}`.  
**Imports:** Step 2 — `\omega_b`, `\Omega`, `L_0`; Step 5 — path marking; Step 6 — `C_*`, `\phi_{AB}`; Step 7 — `\mu` / decoherence.

**Goal:** `A_L`, `A_R` wake kernels from coin pump; `\Lambda_{^3He}\neq\Lambda_{^4He}` discriminator structure.

---

## 8.G.1 Two-path interference (ANCHORED)

\[
\Psi=A_L+A_R,\qquad
I=|A_L|^2+|A_R|^2+2\mu\,\mathrm{Re}(A_LA_R^*)
\]

\[
V=\frac{I_{max}-I_{min}}{I_{max}+I_{min}}\approx \mu
\]

(balanced amplitudes). Tag: **ANCHORED**.

Code: `interference_intensity`, `fringe_visibility`, `coherence_mu_steady`.

---

## 8.G.2 Wake amplitude scale (GEOMETRY)

From Sec 8.2.2 with Step 2 exports:

\[
\mathcal A_0=\eta_{wake}\sqrt{\hbar\omega_b}\,\frac{|\Omega(\kappa)|}{\Omega_0},
\qquad
\Omega_0=\frac{\omega_b}{2}
\]

Geometric kernel at slit `s`:

\[
K_s(\mathbf r)=
\frac{\exp\!\big(-r_{xy}^2/(2\sigma_{wake}^2)\big)}{\sqrt{r_{xy}^2+z^2}}
\exp\!\left(-\frac{r}{\ell_{wake}}\right)
\]

**Default:** `\sigma_{wake}\sim L_0` (C1).

Opposite-phase lock (Sec 6): `\phi_R=\phi_L+\pi` ⇒ `A_R\approx -A_L` (balanced).

**Defaults:** `\sigma_{wake}^{micro}\sim L_0` (C1); apparatus demos use `\sigma_{wake}\sim w_{slit}/4` via `sigma_wake_default()`. Opposite-phase lock gives `I\propto|K_L-K_R|^2` on symmetric screen — small lateral offset breaks symmetry for fringes.

Code: `a0_wake_scale`, `wake_kernel_xy`, `wake_amplitude_complex`, `demo_slit_fringe_intensity`.

---

## 8.G.3 Which-path / visibility loss (ANCHORED + Sec 5)

Path marking strength `p`:

\[
V \to V_0(1-p),\qquad p\to 1 \Rightarrow I_{int}\to 0
\]

Code: `path_mark_visibility`.

Tag: **ANCHORED** (decoherence); links Step 5 `\Lambda_n`.

---

## 8.G.4 Pressure visibility (ANCHORED form)

\[
V(P)=V_0 e^{-\Lambda P}
\]

`\mu` dynamics from Sec 6:

\[
\dot\mu=\Gamma_{form}(1-\mu)-\Gamma_{break}\mu
\]

Code: `visibility_vs_pressure`, `coherence_mu_steady`.

Tag: **ANCHORED** envelope; `\Lambda` species extraction below.

---

## 8.G.5 `\Gamma_{partner}` and ³He vs ⁴He (MODEL + discriminator)

\[
\Gamma_{partner}=\sum_i n_i\,\sigma_i\,\bar v_i\,f_{coin,i}
\]

Ideal gas: `n=P/(k_B T)`, `\bar v=\sqrt{8k_B T/(\pi m)}`.

**Discriminator (Sec 8.7.3):**

\[
\frac{\Lambda_{^3He}}{\Lambda_{^4He}}
\approx
\frac{\sigma_{e,^3He}\,f_{coin,^3He}}{\sigma_{e,^4He}\,f_{coin,^4He}}
\neq 1
\]

Model: `^3He` unpaired → larger `f_{coin}`; `^4He` closed-shell → smaller.

Code: `gamma_partner_rate`, `lambda_decoherence_proxy`, `lambda_he3_he4_ratio`.

Tag: **MODEL** (composition law); `\sigma_{e,i}`, `\eta_{spin}` **OPEN** (O8-1).

---

## 8.G.6 Geometry Gate checklist (Step 8)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `A_0` from `\omega_b`, `\Omega` | ✅ |
| 2 | Wake kernel + opposite-phase demo | ✅ |
| 3 | `V` vs `\mu`, path marking | ✅ |
| 4 | `V(P)` envelope | ✅ |
| 5 | `\Lambda_{^3He}\neq\Lambda_{^4He}` structure | ✅ |
| 6 | `\sigma_{e,i}` from geometry | ⚠️ **OPEN** (O8-1) |
| 7 | Code + test | ✅ |

**Gate verdict:** **PASS (geometry core + honest O8-1 boundary)**.

---

## 8.G.7 Exports

- Wake scale → Sec 10 cosmic / beam narratives  
- `\Lambda` gas test → experimental discriminator  
- `\mu`, `\Gamma_{partner}` → Sec 6, Sec 9  

---

*Next: Step 9 atom/bonds OR Step 11 `\phi_{AB}` / O8-1 `\sigma` calibration.*
