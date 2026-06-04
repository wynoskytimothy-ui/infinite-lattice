# Section 11 — Geometry Audit (Step 11 Gate)

**Imports:** Step 2 — `k_s`, `L_0`, `\omega_b`; Step 6 — `\Gamma_{form}`, `\ell_c`, `\phi_{AB}`; Step 7 — DM path `\eta_{DM}`; Step 10 — `w(z)`, Friedmann continuity.

**Goal:** Spring-only DM suppression, P11-3 mesh fill, sector transfer `Q`, and cosmology bridges with honest MODEL boundaries.

---

## 11.G.1 Postulates (MODEL — not silent)

| ID | Claim | Tag |
|----|-------|-----|
| P11-1 | DM unit = spring only (no inner photon) | **MODEL** |
| P11-2 | DE = freed inner photon sea pressure | **MODEL** |
| P11-3 | Connective DM mesh; `\phi_{AB}` fill | **MODEL** (structure in code) |

---

## 11.G.2 EM suppression (GEOMETRY structure — O11-1)

`S_{res,DM}=0` ⇒ Rayleigh law:

\[
\sigma_{\gamma DM}=\sigma_{geom}\mathcal K_{sup},\qquad
\mathcal K_{sup}=\left(\frac{\hbar\omega}{E_{spring}}\right)^4
\]

\[
E_{spring}=\hbar\sqrt{k_s/m_{spring}},\qquad
\sigma_{geom}=\pi R_{spring}^2
\]

**Geometry inputs:** `k_s`, `L_0` from C1; `R_{spring}\sim L_0`.

Code: `e_spring_j`, `sigma_geom_spring`, `k_sup_rayleigh`, `sigma_gamma_dm`.

**E-check:** at `\hbar\omega\sim 1` eV, `\sigma_{\gamma DM}\ll\sigma_{max}^{exp}`.

Tag: **GEOMETRY** (structure from coin spring); `(m_{spring}, R_{spring})` primary calibration **OPEN**.

---

## 11.G.3 P11-3 fill dynamics (PARTIAL — O11-5)

\[
\frac{d\phi}{dt}=\Gamma_{fill}(1-\phi)-\Gamma_{snap}\phi
\Rightarrow
\phi_{ss}=\frac{\Gamma_{fill}}{\Gamma_{fill}+\Gamma_{snap}}
\]

Sec 6 modifier (already wired):

\[
\Gamma_{form}\leftarrow k_{lock}|O_{AB}|J_{AB}\,\phi_{AB}\,e^{-d/\ell_c}
\]

Mesh reach model:

\[
\ell_c(\rho_{DM})\sim n_{DM}^{-1/3},\qquad n_{DM}=\rho_{DM}/m_{DM}
\]

Code: `phi_ab_steady_state`, `phi_ab_at_time`, `ell_c_from_rho_dm`, `gamma_form_rate(..., phi_ab=...)`.

Tag: **PARTIAL** — `\Gamma_{fill}`, `\Gamma_{snap}` ab-initio **OPEN**.

---

## 11.G.4 Sector transfer Q (PARTIAL — O11-2)

\[
Q=\rho_{NM}\,\frac{\Gamma_{sep}}{m_N c^2}\,\bar{\mathcal E}_\gamma,
\qquad
\Gamma_{sep}=\max(0,\Gamma_{unpin}-\Gamma_{pin})
\]

\[
\Gamma_{unpin}=\nu_0\exp(-\Delta_{sep}/k_B T_{env}),\qquad
\Delta_{sep}\sim\hbar\omega_b
\]

Split: `f_e=\bar E_\gamma/\mathcal E_N`, `f_m=\bar E_{spring}/\mathcal E_N`.

Code: `gamma_unpin_arrhenius`, `gamma_sep_rate`, `q_transfer_rate`, `sector_split_fractions`.

Tag: **PARTIAL** — `\nu_0`, `\Gamma_{pin}` micro closure **OPEN**.

---

## 11.G.5 No-detection channel classes (PARTIAL — O11-3)

| Class | Result | Code |
|-------|--------|------|
| A (pin/readout) | `\sigma_{det}=0` exact | `sigma_det_class_a` |
| B/C (EM/recoil) | `\sigma\le\sigma_{\gamma DM}` | `sigma_det_class_bc_upper` |
| D (gravity) | `\nabla^2\Phi=4\pi G(\rho_b+\rho_{DM})` | `circular_velocity_squared` |

**Not claimed:** universal impossibility over all Hamiltonians.

Tag: **PARTIAL-PROVEN** under stated assumptions.

---

## 11.G.6 Cosmology bridge (PARTIAL — O11-4)

From Step 10:

\[
p_{DE}=-\rho_{DE}c^2+\Pi_s,\qquad
w=-1+\Pi_s/(\rho_{DE}c^2)
\]

Continuity: `w=-1` ⇒ `\dot\rho_{DE}=0`.

\[
\dot\rho+3H(\rho+p/c^2)=0
\]

Acceleration: `w<-1/3`.

Code: `p_de_effective`, `w_eos_from_sea_pressure`, `rho_dot_frw`, `rho_dot_de_const`, `calibrate_pi0_for_w0`, `u_gamma_free_density`, `p_sea_dark_energy`.

Tag: **PARTIAL** — joint halo+lensing+SN fit **OPEN**.

---

## 11.G.7 Halo kinematics (ANCHORED)

\[
v_c^2(r)=\frac{G M(<r)}{r}
\]

Code: `circular_velocity_squared`, `enclosed_mass_flat_halo` (illustrative flat curve).

Tag: **ANCHORED** observable; DM ontology must fit joint constraints (T11-1).

---

## 11.G.8 Gate checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Robust core tagged | ✅ |
| 2 | OPEN O11-1–5 listed | ✅ |
| 3 | conflict #4 addressed conditionally | ✅ Class A–D |
| 4 | Tests | ✅ |
| 5 | symbol_registry | defer batch |
| 6 | Import/export | ✅ |

**Exports:** Step 12 — `f_{clock}^{DM}\approx 0` as model postulate; lattice `\mathcal{M}_{lat}` may set `m_{DM}` scale.

**Imports:** Step 2 spring; Step 6 `\phi_{AB}`; Step 10 `w(z)`.

---

## 11.G.9 Verdict

**CORE GATE PASS** — suppression law, fill mesh, transfer bookkeeping, and cosmology hooks implemented; joint dataset fit and `\Gamma_{fill}` calibration remain **OPEN**.

*Next: Step 12 time / Zeno / lattice `\mathcal{M}_{lat}`.*
