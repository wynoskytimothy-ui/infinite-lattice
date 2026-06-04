# Section 6 — Geometry Audit (Step 6 Gate)

**Mandates:** C7 Stage A (`ℓ_c`); Stage B forward (`φ_AB`, P11-3); C5 Bell via joint fill.  
**Imports:** Step 2 — `L_0`, `λ_C`, `f_b`, `σ_obs` area; Step 5 — `\Lambda_n`, `\Gamma_n`, `M_n`.

**Goal:** `\Gamma_{form}`, `\Gamma_{break}`, `C(t)` from coin geometry; `\phi_{AB}` contract for full Bell kernel.

---

## 6.G.1 Coherence ODE (PROVEN)

\[
\frac{dC}{dt}=\Gamma_{form}(1-C)-\Gamma_{break}C,
\qquad
C_*=\frac{\Gamma_{form}}{\Gamma_{form}+\Gamma_{break}}
\]

\[
C(t)=C_*+(C_0-C_*)e^{-(\Gamma_{form}+\Gamma_{break})t},
\qquad
\tau_E\sim\frac{1}{\Gamma_{form}+\Gamma_{break}}
\]

Tag: **PROVEN** (linear ODE). Code: `coherence_steady_state`, `coherence_at_time`, `entanglement_lifetime`.

---

## 6.G.2 Break rate (GEOMETRY + env)

\[
\sigma_{obs}=A_{eff}\,\eta_{geom}\,S_{res},
\qquad
A_{eff}=\pi L_0^2
\]

\[
\Gamma_{break}\approx \Phi_{env}\,\sigma_{obs}+\Gamma_{other}
\]

**C1 lock:** `R_{coin}=L_0`. `\Phi_{env}` = environmental flux (**OPEN** micro; not derived from coin alone).

Tag: **GEOMETRY** (`\sigma_{obs}`); **MODEL** (`\Phi_{env}`).

Code: `sigma_obs_geometry`, `gamma_break_rate`.

---

## 6.G.3 Formation rate — Stage A (C7)

**Stage A mandate:** effective `\ell_c` from geometry, not silent FIT.

\[
\ell_c^{geom}=\lambda_C=2L_0
\]

\[
\mathcal J_{AB}=e^{-d/\ell_c}\,S_{freq}\,S_{axis}
\]

\[
\Gamma_{form}=k_{lock}\,|\mathcal O_{AB}|\,\mathcal J_{AB}\,\phi_{AB}
\]

**Geometry default:**

\[
k_{lock}^{geom}=f_b=\omega_b/(2\pi)
\]

Tag: **GEOMETRY** (`\ell_c`, `k_{lock}`, `J_{AB}` structure); `|\mathcal O_{AB}|`, `S_{freq}`, `S_{axis}` **MODEL** until pump overlap derived.

Code: `ell_c_from_geometry`, `k_lock_from_geometry`, `gamma_form_rate`.

---

## 6.G.4 Stage B — `\phi_{AB}` fill (forward contract)

Fill on DM path (P11-3, Sec 6.13 / Sec 11):

\[
\frac{d\phi_{AB}}{dt}=\Gamma_{fill}(1-\phi_{AB})-\Gamma_{snap}\phi_{AB}-\eta_{obs}\Gamma_{break}
\]

- `\phi_{AB}=0`: no connective joint mode → `\Gamma_{form}\to 0`, Bell off.
- `\phi_{AB}\to 1`: filled ripple → full lock + Bell kernel.

**Bell scaling contract (Stage B):**

\[
E(a,b)=-\phi_{AB}\cos(a-b)
\]

- `\phi_{AB}=1` ⇒ **QM kernel** (ANCHORED check).
- `\phi_{AB}=\tfrac12` ⇒ Step 5 partial (`bell_correlation_joint_ripple_linear`).

Tag: **GEOMETRY contract** for `E`; **OPEN** — derive `\phi_{AB}` steady state from `\Gamma_{fill}`, `\Gamma_{snap}`, `\rho_{DM}` (O11-5).

Code: `phi_ab_derivative`, `bell_correlation_phi_fill`.

---

## 6.G.5 O5-3 / C5 status

| Model | Status |
|-------|--------|
| `sign(cos θ)` half-plane | **REJECT** |
| `E=-\tfrac12\cos(a-b)` (`\phi=0.5`) | **PARTIAL** (Stage B half-fill) |
| `E=-\phi_{AB}\cos(a-b)` | **CONTRACT** (full at `\phi=1`) |
| Derive `\phi_{AB}` from mesh | **OPEN** (P11-3, Step 11) |

No-signaling marginals: **ANCHORED** (unchanged by `\phi_{AB}` scaling of correlation).

---

## 6.G.6 Sustained entanglement criterion

\[
\Gamma_{form}>\Gamma_{break}\quad\Leftrightarrow\quad C_*>0.5
\]

Engineering prediction: lower `\Phi_{env}` or higher `\phi_{AB}` / shorter `d` → higher `C_*`.

---

## 6.G.7 Geometry Gate checklist (Step 6)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `C(t)` ODE + closed form | ✅ **PROVEN** |
| 2 | `\sigma_{obs}` from `L_0` | ✅ |
| 3 | `\ell_c^{geom}=\lambda_C` (Stage A) | ✅ C7 |
| 4 | `\Gamma_{form}`, `\Gamma_{break}` structure | ✅ |
| 5 | `\phi_{AB}` Bell contract | ✅ **CONTRACT** |
| 6 | `\phi_{AB}` dynamics from mesh | ⚠️ **OPEN** (Stage B) |
| 7 | Code + test | ✅ |

**Gate verdict:** **PASS (Stage A core + Stage B contract + honest OPEN)**.

---

## 6.G.8 Exports

- `\ell_c^{geom}`, `\Gamma_{form}`, `\Gamma_{break}`, `C_*` → Sec 8 `\tau_{re}`, Sec 9 sharing  
- `bell_correlation_phi_fill` → closes O5-3 **when** `\phi_{AB}=1` justified  
- `\phi_{AB}` ODE → Sec 11 P11-3  

---

*Next: Step 7 tunneling (`\kappa`, P7-2) or Step 11 `\Gamma_{fill}` calibration for `\phi_{AB}`.*
