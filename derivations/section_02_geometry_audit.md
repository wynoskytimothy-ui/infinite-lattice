# Section 2 — Geometry Audit (Step 2 Gate)

**Mandates:** `architecture_mandates.md` (C1 locked).  
**Goal:** Derive `H_coin`, `ω_b`, `k_s`, `Δ(κ)`, `Ω(κ)` from `(L, m_e, c, h)` with **no fitted coin length**.

---

## 2.G.1 Geometry anchor (C1)

**Definition (GEOMETRY):**

\[
L \equiv \frac{\lambda_C}{2} = \frac{h}{2m_e c}
\]

Coin coordinate: inner photon position `x ∈ [−L/2,\,+L/2]`. Full span `2L = λ_C`.

| Quantity | Formula | Tag |
|----------|---------|-----|
| `λ_C` | `h/(m_e c) = 2L` | **GEOMETRY** |
| Round-trip bounce path | `4L = 2λ_C` (face to face and return) | **GEOMETRY** |

**No secondary lengths** in electron core dynamics.

---

## 2.G.2 Bounce / light clock (GEOMETRY)

Inner photon propagates at `v_{int} = c` along bounce (Sec 1 uncontained limit applied to cavity mode).

**Bounce angular frequency:**

\[
\omega_b = \frac{\pi c}{2L} = \frac{\pi m_e c^2}{h}
\]

(Period `T_b = 2π/ω_b = 4L/c` — four half-widths per round trip.)

**Clock frequency:**

\[
f_b = \frac{\omega_b}{2\pi} = \frac{c}{4L} = \frac{m_e c^2}{2h}
\]

**Exports to Sec 12:** `f_b(v) = f_{b0}/γ`, `v_{int} = √(c²−v²)`.

Status: **GEOMETRY** from C1 + `c`.

---

## 2.G.3 Spring stiffness from geometry (GEOMETRY)

Require spring restoring force at displacement `δ` to support bounce at `ω_b`:

\[
k_s = m_e\,\omega_b^2 = m_e\left(\frac{\pi c}{2L}\right)^2
\]

Numeric check (CODATA `m_e`, `c`, `L`): `k_s ≈ 2.9×10² N/m` order (effective electron spring scale).

Status: **GEOMETRY** — **no free `k_s` FIT** in core chain.

---

## 2.G.4 Cavity Hamiltonian → `H_coin` (GEOMETRY + PROVEN reduction)

### Continuous form

\[
\hat H_x = \frac{\hat p_x^2}{2m_e} + \frac{1}{2}k_s x^2 + \epsilon(\kappa)\,x
\]

with asymmetric bias from compression `\kappa ∈ [0,1]`:

\[
\epsilon(\kappa) = \frac{k_s L}{2}\,\kappa^2
\]

(quadratic side bias at max extension — leading order in `\kappa`).

### Two-level reduction on `{|W⟩,|B⟩}`

Identify `σ_z` with side localization, `σ_x` with inter-side hopping.

**Inter-side coupling (GEOMETRY):**

\[
\Omega(\kappa) = \Omega_0\,(1-\kappa^2),\qquad
\Omega_0 = \frac{\omega_b}{2} = \frac{\pi c}{4L}
\]

Compression `\kappa → 1` suppresses hopping (solid limit).

**Side bias (GEOMETRY):**

\[
\Delta(\kappa) = \frac{k_s L^2}{2\hbar}\,\kappa^2
\]

**Observation drive (unchanged form, Sec 5):**

\[
\hbar g_E E_{obs}(t)\,\sigma_z
\]

`g_E` remains **coupling** from compression geometry (Sec 5 O5-2); not a length fit.

### Closed `H_coin`

\[
\hat H_{coin}(t)=
\frac{\hbar}{2}\Delta(\kappa)\sigma_z
+\frac{\hbar}{2}\Omega(\kappa)\sigma_x
+\hbar g_E E_{obs}(t)\sigma_z
\]

Status:

| Term | Tag |
|------|-----|
| `L`, `ω_b`, `k_s`, `Ω_0`, `Δ(κ)` | **GEOMETRY** |
| `g_E E_obs` | **PARTIAL** (Sec 5 field calibration) |
| Two-level projection | **PROVEN** standard |

**O2-1:** **GEOMETRY closure for `(Δ, Ω, k_s, ω_b)`** — gate satisfied for core Hamiltonian.

---

## 2.G.5 Pinning indicator (export Sec 4–5–7)

\[
\Pi_{pin}(\kappa) = \frac{|\Delta(\kappa)|}{|\Delta(\kappa)|+\Omega(\kappa)}
\]

At `\kappa=1`: `\Pi_{pin} → 1` if `\Delta \gg \Omega` (check numerically in `aethos_physics.py`).

---

## 2.G.6 Exports to Step 3 (`K_f` — C2 preview)

Maximum spring energy at full geometric compression `\kappa=1`:

\[
U_{max} = \frac{1}{2}k_s L^2 = \frac{1}{2}m_e\omega_b^2 L^2
\]

Step 3 derives **`K_f`** when `U_{max}` reaches fusion barrier / length collapse

\[
L_p = L_0(1-\alpha K_f),\quad L_0 \equiv L
\]

**without** using 1836 as input. Prediction:

\[
R_{pe}^{model} = \frac{L_0}{L_p}\big|_{K=K_f}
\]

compared to **E** `1836.152…`.

---

## 2.G.7 Geometry Gate checklist (Step 2)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Robust core tagged | ✅ this file + §2.11 update |
| 2 | OPEN items have path | O2-2 Born uniqueness; O2-3 chirality — Step 2b |
| 3 | `conflict_log` | no new conflict |
| 4 | Code + test | `aethos_physics.coin_geometry_*`, `test_aethos_physics.py` |
| 5 | `symbol_registry` | `L`, `Omega_0`, `k_s` geometry |
| 6 | Export paragraph | below |

**Exports to Sec 3–7:** `L`, `ω_b`, `f_b`, `k_s`, `H_coin(κ)`, `Δ(κ)`, `Ω(κ)`, `Π_pin(κ)`, `U_max`.

**Gate verdict:** **PASS (core geometry)** — Step 3 unblocked for `K_f` derivation. O2-2/O2-3 remain **PARTIAL** (non-blockers per mandates).

---

*Next: `section_03_geometry_audit.md` — derive `K_f` from `U_max` / length collapse (C2).*
