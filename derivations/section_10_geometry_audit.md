# Section 10 — Geometry Audit (Step 10 Gate)

**Imports:** Step 2 — `\mu_B`, `\Pi_{pin}`, `\omega_b`; Step 4 — `P_{core}`, `\Gamma_{obs}`, `\alpha_{core}`; Step 9 — fusion `Q` (stellar source); Step 12 — motion budget `\sqrt{1-v^2/c^2}`.

**Goal:** Micro-to-macro dipole chain, reversal barrier, GR anchors, and `w(z)` bridge with honest participation fractions.

---

## 10.G.1 Cell magnetization (GEOMETRY structure — O10-1)

\[
\mu_{cell}=g_{eff}\,\mu_B\,\langle\sigma_z\rangle\,\Pi_{pin}
\]

Code: `mu_cell_neutron_network` — uses Step 2 `\Pi_{pin}(\kappa)` and Sec 4 `g_{eff}` anchor.

Tag: **GEOMETRY** (structure); `g_{eff}` primary still **FIT** from `\mu_n`.

---

## 10.G.2 Dipole closure (GEOMETRY + E check)

\[
N_{eff}=f_{part}\,n_n V_c,
\qquad
B(R)=\frac{\mu_0}{2\pi}\,\frac{N_{eff}\mu_{cell}|m|}{R^3}
\]

Neutron star (same law):

\[
B_{NS}\approx\frac{\mu_0}{2\pi}\,
\frac{f_{NS}(M_{NS}/m_n)\,|\mu_n|\xi_{NS}|m|}{R_{NS}^3}
\]

Code: `n_eff_participating`, `b_dipole_surface_t`, `b_neutron_star_dipole_t`, `calibrate_f_part_dipole`.

**Earth E-check:** invert `f_{part}` from `B_\oplus\sim 31\,\mu\text{T}` — not a definition of `f_{part}`.

Tag: **GEOMETRY** chain; **`f_{part}`, `f_{NS}`, `\xi_{NS}` OPEN** (material / lattice — Step 12 link).

---

## 10.G.3 Reversal dynamics (PARTIAL — O10-2)

Double well:

\[
U(m)=\frac{U_0}{4}(m^2-1)^2
\]

Pressure lowers barrier:

\[
\Delta U(P)=\Delta U_0-\zeta_P(P_{core}-P_{eq})
\]

Activated timescale:

\[
\tau_{flip}\sim\tau_0\exp\!\left(\frac{\Delta U(P_{core})}{D_{core}}\right),
\qquad
\tau_0\sim\frac{2\pi\hbar}{\alpha_{core}\Gamma_{obs,core}}
\]

Code: `double_well_u`, `flip_barrier_j`, `tau0_core_attempt`, `d_core_noise`, `tau_flip_seconds`.

Paleomagnetic anchor `\langle\tau_{flip}\rangle\sim 4.5\times10^5` yr — **E check** for `( \zeta_P, D_{core}, \Phi_{obs,core})`.

Tag: **PARTIAL** — bridge from Sec 4 pressure variables; Earth-core calibration **OPEN**.

---

## 10.G.4 GR anchors (ANCHORED)

| Item | Formula | Tag |
|------|---------|-----|
| Schwarzschild | `r_s=2GM/c^2` | **ANCHORED** |
| Escape speed | `v_{esc}=\sqrt{2GM/r}` | **ANCHORED** |
| Time dilation | `\sqrt{1-2GM/(rc^2)}` | **ANCHORED** |
| Hydrostatic | `dP/dr=-GM\rho/r^2` | **ANCHORED** |

Code: `schwarzschild_radius`, `escape_speed`, `gravitational_time_dilation_factor`, `hydrostatic_dp_dr`.

Section 12 back-link: `v_{flow}=\sqrt{2GM/r}`, clock `\propto\sqrt{1-v_{flow}^2/c^2}`.

---

## 10.G.5 Dark-energy bridge (PARTIAL — O10-3)

\[
p_{DE}=-\rho_{DE}c^2+\Pi_s(z),
\qquad
w(z)=-1+\frac{\Pi_s(z)}{\rho_{DE}c^2}
\]

CPL map:

\[
\Pi_s(z)=\Pi_0(1+z)^n
\Rightarrow
w_0\approx -1+\Pi_0/(\rho_{DE,0}c^2),
\quad
w_a\approx n\Pi_0/(\rho_{DE,0}c^2)
\]

Code: `w_eos_from_sea_pressure`, `w_z_cpl`, `cpl_from_sea_pressure`.

Acceleration requires `w<-1/3` ⇔ `\Pi_s/(\rho_{DE}c^2)<2/3`.

Tag: **PARTIAL** — structure links Sec 11; `(Q,f_e,\Pi_0,n)` dataset fit **OPEN** (O11-4).

---

## 10.G.6 Gate checklist

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Robust core tagged | ✅ |
| 2 | OPEN items listed | ✅ O10-1–3 |
| 3 | No conflict_log breach | ✅ |
| 4 | Tests in `test_aethos_physics.py` | ✅ |
| 5 | symbol_registry | defer batch |
| 6 | Import/export paragraph | ✅ below |

**Exports:** Sec 11 — `\Pi_s`, `\rho_{DE}`; Sec 12 — `r_s`, dilation, motion budget.

**Imports:** Sec 2/4 micro moments and pressure; Sec 9 fusion energy for stellar `\epsilon`.

---

## 10.G.7 Verdict

**CORE GATE PASS** — dipole chain, reversal structure, GR anchors, and `w(z)` bridge implemented with participation fractions as **E checks**, not silent FIT prose.

*Next: Step 11 dark sector OR Step 12 lattice `\mathcal{M}_{lat}` + time/Zeno.*
