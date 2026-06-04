# Section 9 — Geometry Audit (Step 9 Gate)

**Imports:** Step 2 — `\omega_b`, `L_0`, `\Pi_{pin}`; Step 3 — drain `U_C`; Step 4 — `C_N`, `B_{share}`; Step 7 — Gamow `\kappa`.

**Goal:** Bond energy `E_{bond}`, coupling `C_b`, orbital map `A_{lm}` from coin geometry with honest calibration boundaries.

---

## 9.G.1 Atomic anchors (PROVEN / ANCHORED)

| Item | Formula | Tag |
|------|---------|-----|
| Shell capacity | `N_{max}(n)=2n^2` | **PROVEN** |
| Subshell | `2(2l+1)` | **PROVEN** |
| Hydrogenic levels | `E_n=-13.6\,Z_{eff}^2/n^2` eV | **ANCHORED** |
| Bohr radius | `a_0=\hbar/(m_e c\alpha)` | **ANCHORED** |

Code: `shell_capacity`, `hydrogen_energy_ev`, `bohr_radius_geometry`.

**Geometry compare:** `a_0/L_0 \sim O(10^2)` — coin is micro anchor; atomic scale is **E** check, not derived from `L_0` alone.

---

## 9.G.2 Bond energy closure (GEOMETRY structure — O9-2)

\[
\Delta E_{share}=-C_b\,\hbar\omega_b\,|\eta_{AB}|^2\,\Pi_{pin}
\]

\[
E_{bond}(r)=U_C(r)+\Delta E_{share},
\qquad
U_C=\frac{Q_AQ_B}{4\pi\varepsilon_0 r}
\]

**Geometry inputs:**
- `\hbar\omega_b` from C1 (`hbar_omega_b_ev`)
- `\Pi_{pin}(\kappa)` from Step 2
- `|\eta_{AB}|` — overlap proxy `exp(-(r/2L_0)^2)` until ab-initio `\psi` (**MODEL**)

**H₂ calibration (E check, not definition):**

\[
C_b=\frac{U_C(r_0)+B_{H_2}}{\hbar\omega_b|\eta_{AB}|^2\Pi_{pin}}
\]

with `E_{bond}(r_0)=-B_{H_2}` (negative well depth). `|\eta_{AB}|` uses molecular scale `\max(a_0,r/2)` — not `L_0` alone.

Tag: **GEOMETRY** (structure); **`C_b` primary** needs `\eta_{AB}` geometry (**PARTIAL**).

---

## 9.G.3 Radial / angular orbitals (O9-3 partial)

**Radial:** `k_{nl}\sim(\pi/L_{bounce})(n-(l+1)/2)`, `L_{bounce}=4L_0`.

**Angular (coin membrane):** `A_{lm}(\theta,\phi)\propto Y_l^m` — minimal `l=0,1` map in code.

**Radial envelope:** `R_{nl}\sim r^l e^{-r/a_0}` (hydrogenic leading term — **MODEL bridge**).

Tag: **PARTIAL** — winding map started; full spectrum without hydrogenic import **OPEN**.

Code: `k_nl_geometry`, `a_lm_coin`, `radial_envelope_nl`.

---

## 9.G.4 Nuclear sharing bridge (MODEL — O9-1)

\[
C_N(N,Z)=\mathcal C_0 f_N(1-e^{-N/N_0})\exp\!\left[-\left(\frac{N-Z}{N-Z_0}\right)^2\right]e^{-R_0/R_A}
\]

\[
B_{share}=-b_{net}\,N\,C_N
\]

Imports Sec 4 closure; microscopic node graph **OPEN**.

Code: `c_n_sharing`, `b_share_ev`.

---

## 9.G.5 Decay / fusion links (imports)

- **Alpha / fusion:** Step 7 `kappa_wkb`, `t_wkb` — **ANCHORED** channel forms  
- **Beta:** Sec 4 pressure escape — **MODEL** narrative  

No new Step 9 geometry beyond cross-links.

---

## 9.G.6 Geometry Gate checklist (Step 9)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `\hbar\omega_b` in bond formula | ✅ |
| 2 | `\Pi_{pin}` in bond formula | ✅ |
| 3 | `C_b` from H₂ **E check** | ✅ |
| 4 | `|\eta_{AB}|` from geometry | ⚠️ **MODEL** proxy |
| 5 | `k_{nl}`, `A_{lm}` partial | ✅ **PARTIAL** |
| 6 | `C_N` structure | ✅ MODEL |
| 7 | Code + test | ✅ |

**Gate verdict:** **PASS (geometry core + honest O9-2/O9-3 boundaries)**.

---

## 9.G.7 Exports

- `C_b`, `E_{bond}` → chemistry / periodic table narrative  
- `k_{nl}` → spectral fingerprint bridge  
- `B_{share}` → Sec 10 stellar / Sec 4 neutron network  

---

*Next: Step 10 cosmic scales OR Step 12 `\mathcal{M}_{lat}` lattice.*
