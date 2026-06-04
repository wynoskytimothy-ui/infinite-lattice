# Section 7 — Geometry Audit (Step 7 Gate)

**Mandates:** P7-2 soft/hard; C7 `\phi_{path}` transit (Stage B forward).  
**Imports:** Step 2 — `\Delta(κ)`, `\Omega(κ)`, `k_s`, `U_{max}`, `\Pi_{pin}`; Step 4 — `P_c`; Step 5 — `E_{bar}`; Step 6 — `\phi_{AB}` fill language.

**Goal:** `\bar\kappa` from `H_x`; P7-2 regime map; DM transit modifiers with honest `\eta_{DM}` boundary.

---

## 7.G.1 WKB anchor (ANCHORED)

\[
T\sim e^{-2\bar\kappa L},
\qquad
\bar\kappa=\frac{1}{\hbar}\sqrt{2m_{eff}[U_{bar}-E]_+}
\]

Tag: **ANCHORED** (standard WKB); `\bar\kappa` **source** upgraded below.

---

## 7.G.2 `U_{bar}` from `H_x` (GEOMETRY)

\[
U_{bar}(x)=\frac{\hbar}{2}\sqrt{\Delta_{eff}^2+\Omega(\kappa)^2}+\cdots
\]

**Leading coin term (Step 2 export):**

\[
U_{bar}^{geom}(\kappa)=\frac{\hbar}{2}\sqrt{\Delta(\kappa)^2+\Omega(\kappa)^2}
\]

\[
\kappa_{WKB}=\frac{1}{\hbar}\sqrt{2m_{eff}\,\big[U_{bar}^{geom}(\kappa)-E\big]_+}
\]

Code: `u_bar_from_h_coin`, `kappa_wkb_from_h_x`.

Tag: **GEOMETRY** (core); `\Delta_{eff}` with `g_E E_{bar}` **PARTIAL** (field map open).

---

## 7.G.3 Shredding and `m_{eff}` (GEOMETRY structure)

\[
\xi_{shred}=\frac{|E_{bar}|}{|E_{bar}|+E_{ref}},
\qquad
E_{ref}^{geom}=\hbar\omega_b
\]

\[
m_{eff}=m_e\,(1+\lambda_m\xi_{shred})
\]

- `\lambda_m` small (electron) vs large (fused hadron) — **MODEL** mass row.

Code: `xi_shred_from_field`, `e_ref_from_geometry`, `m_eff_from_shred`.

---

## 7.G.4 P7-2 soft vs hard (GEOMETRY + Sec 4)

| Regime | `\Pi_{pin}` | `P` vs `P_c` | Outcome |
|--------|-------------|--------------|---------|
| **Soft** | `\ll 1` | `P<P_c` | `T_{eff}=T_{WKB}\,\chi_{ss}`; recapture |
| **Hard** | `\to 1` | `P\to P_c` | escape / collapse (Sec 4); no ordinary tunnel |

\[
\Pi_{pin}(\kappa)=\frac{|\Delta(\kappa)|}{|\Delta(\kappa)|+\Omega(\kappa)}
\]

Code: `classify_compression_from_coin`, `classify_compression`.

Tag: **GEOMETRY** (`\Pi_{pin}`); `P_c` from Sec 4.

---

## 7.G.5 Recoherence (PARTIAL)

\[
\chi_{ss}=\frac{\Gamma_{rec}}{\Gamma_{rec}+\Gamma_{sh}},
\qquad
\Gamma_{rec}=\frac{k_s}{m_{eff}}\,\frac{1}{2\pi}\,\Pi_{pin}^{-1}
\]

\[
T_{eff}=e^{-2\bar\kappa L}\,\chi_{ss}
\]

Code: `gamma_rec_from_geometry`, `chi_steady`, `t_eff_soft`, `transmission_soft_pipeline`.

Tag: **GEOMETRY** (`\Gamma_{rec}` structure); `\Gamma_{sh}` env **OPEN**.

---

## 7.G.6 DM path modifier (MODEL — O7-4)

Sec 7.3.4 / P11-3:

\[
\xi_{shred}\to\xi_{shred}\,(1-\eta_{DM}\phi_{path}),
\qquad
\bar\kappa\to\bar\kappa\,(1+\eta_\kappa(1-\phi_{path}))
\]

- `\phi_{path}`: fill on transit segment (same family as `\phi_{AB}`, Sec 6).
- `\eta_{DM}`, `\eta_\kappa`: **OPEN** until P11-3 calibration.

Code: `xi_shred_with_dm`, `kappa_bar_with_dm_path`.

---

## 7.G.7 Thickness law (PROVEN)

\[
\ln T = -2\bar\kappa L + \mathrm{const}
\]

Tag: **PROVEN** from WKB form.

---

## 7.G.8 Geometry Gate checklist (Step 7)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `\bar\kappa` from `H_x` / coin terms | ✅ |
| 2 | `E_{ref}^{geom}=\hbar\omega_b` | ✅ |
| 3 | P7-2 via `\Pi_{pin}(κ)` | ✅ |
| 4 | `T_{eff}` soft pipeline | ✅ |
| 5 | DM `\phi_{path}` modifiers | ✅ **MODEL** |
| 6 | `\eta_{DM}` micro derivation | ⚠️ **OPEN** (O7-4) |
| 7 | Code + test | ✅ |

**Gate verdict:** **PASS (geometry core + honest O7-4 boundary)**.

---

## 7.G.9 Exports

- `\kappa_{WKB}`, `U_{bar}`, `T_{eff}` → Sec 8 visibility, Sec 9 `\alpha` decay, Sec 10 fusion  
- P7-2 **hard** link → Sec 4 `P_c`  
- `\phi_{path}` → Sec 11 P11-3  

---

*Next: Step 8 double-slit wakes OR Step 11 `\eta_{DM}` calibration.*
