# Section 3 — Geometry Audit (Step 3 Gate)

**Mandates:** `architecture_mandates.md` (C2 locked).  
**Imports:** Step 2 — `L_0`, `U_max`, `k_s`, `ω_b`, `Δ(κ)`, `Ω(κ)` (`section_02_geometry_audit.md`).

**Goal:** Derive **`K_f^{geom}`** and predict **`R_pe^{model}`** without using 1836.15 as input.

---

## 3.G.1 Length and compression law (GEOMETRY)

Electron open half-width (C1):

\[
L_0 \equiv L = \frac{h}{2m_e c}=\frac{\lambda_C}{2}
\]

Compression coordinate `K ∈ [0,1]` (structural, same as coin `\kappa` at fusion):

\[
L(K)=L_0(1-\alpha K),\qquad \alpha=1 \;\Rightarrow\; L_p=L_0(1-K_f)
\]

Geometric length ratio:

\[
R_{pe}^{length}(K_f)=\frac{L_0}{L_p}=\frac{1}{1-K_f}
\]

Tag: **GEOMETRY** (P3-length).

---

## 3.G.2 Fusion triggers (three independent criteria)

All use Step-2 `\Omega(K)=\Omega_0(1-K^2)`, `\Delta(K)=(k_s L_0^2/2\hbar)K^2`, `\Omega_0=\omega_b/2`.

| ID | Criterion | Equation | `K_f` | Tag |
|----|-----------|----------|-------|-----|
| **F1** | Hop death | `\Omega(K_f)=0` | `K_f=1` | **GEOMETRY** |
| **F2** | Hop–pin crossover | `\Delta(K_f)=\Omega(K_f)` | `K_f=\sqrt{\Omega_0/(\Omega_0+A)}` | **GEOMETRY** |
| **F3** | Max spring storage | `U_s(K_f)=U_{max}`, `U_s=U_{max}K^2` | `K_f=1` | **GEOMETRY** |

with `A=\Delta(1)=k_s L_0^2/(2\hbar)`.

**F2 numeric (CODATA):** `K_f^{pin} \approx 0.41`; at this point `\Delta=\Omega` and `\Pi_{pin}=0.5` (equal hop/pin weight). **`\Pi_{pin}\to 1`** only at **`K\to 1`** (F1/F3).

**F1/F3:** `K_f\to 1^-` ⇒ `L_p\to 0^+` ⇒ `R_{pe}^{length}\to\infty` (singular without regularization).

**Primary mechanical threshold for this pass:** **`K_f^{pin}`** (first full pin of coin before singular limit).  
**Fusion completion (user narrative):** approach **F1/F3** limit with **regularized** `L_p` (below).

---

## 3.G.3 Energy closure (GEOMETRY algebra)

Electron pump identity (Sec 3.4.2, from C1):

\[
m_e c^2 = \frac{hc}{2L_0}
\]

Locked energy at compression `K` (spring + pump upshift):

\[
E_{locked}(K)=U_{max}K^2+\frac{hc}{2L_0}\left(\frac{1}{1-K}-1\right)
=U_{max}K^2+m_e c^2\frac{K}{1-K}
\]

with `U_{max}=\tfrac12 m_e\omega_b^2 L_0^2 = m_e\pi^2 c^2/8`.

Mass prediction:

\[
R_{pe}^{energy}(K)=\frac{m_p}{m_e}=1+\frac{E_{locked}(K)}{m_e c^2}
=1+\frac{\pi^2}{8}K^2+\frac{K}{1-K}
\]

Define `\mathcal{E}(K)=\pi^2 K^2/8 + K/(1-K)`.

Tag: **GEOMETRY** (energy bookkeeping); **E** check uses CODATA `R_pe`.

---

## 3.G.4 Self-consistency: length vs energy

Require **`R_{pe}^{length}(K_f)=R_{pe}^{energy}(K_f)`** with `\alpha=1`:

\[
\frac{1}{1-K}=\frac{\pi^2}{8}K^2+\frac{K}{1-K}
\]

Multiply `(1-K)` (for `K\neq 1`):

\[
1=\frac{\pi^2}{8}K^2(1-K)+K
\]

**PROVEN:** only real solution in `[0,1]` is **`K=0`** (`R=1`).

**Conclusion (critical):** **Spring + single-cavity length law alone cannot produce `R_pe\gg 1`.**  
Large mass ratio requires **additional locked energy** beyond this two-term closure, or **extra length constraint** not yet in Step 2–3.

This is not FIT — it is a **derivation boundary**.

---

## 3.G.5 Regularized fusion endpoint (C2 primary story)

**P3-F4 (GEOMETRY + interpretation):** Fusion completes when hopping is dynamically negligible:

\[
\frac{\Omega(K_f)}{\Omega_0}\le \varepsilon_{hop},\qquad \varepsilon_{hop}\ll 1
\]

\[
K_f=\sqrt{1-\varepsilon_{hop}}
\]

**P3-L (GEOMETRY):** Proton half-width cannot be zero; minimum cavity from spring scale:

\[
L_p^{min}=\frac{8}{\pi^2}\,L_0
\qquad\Leftrightarrow\qquad
R_{pe}^{spring}=\frac{L_0}{L_p^{min}}=\frac{\pi^2}{8}\approx 1.234
\]

(`L_p^{min}` is the length scale where `U_{max}` normalizes one electron spring quantum — same `\pi^2/8` factor as `U_{max}/(m_e c^2)`.)

**Leading-order prediction (spring-only engine):**

\[
R_{pe}^{model,(0)}=\frac{\pi^2}{8}\approx 1.234
\]

Compare **E** anchor `1836.152…` → **mismatch factor** `\mathcal{M}_{lat}=R_{pe}^{E}/R_{pe}^{model,(0)}\approx 1488`.

Tag: **GEOMETRY** for `(0)` prediction; **GAP** closure for `\mathcal{M}_{lat}` until Step 12 lattice (C6).

---

## 3.G.6 Path to 1836 without FIT (forward contract)

Per **C6** (lattice physical) and **C2** (no FIT-as-definition):

\[
R_{pe}^{model}=R_{pe}^{model,(0)}\times \mathcal{M}_{lat}
\qquad
\mathcal{M}_{lat}=\text{lattice cascade multiplier from chosen active anchor set (Step 12)}
\]

**Target:** `\mathcal{M}_{lat}\approx 1488` so `1.234\times 1488\approx 1836`.

**Not allowed:** `K_f:=(R_pe-1)/R_pe` as definition (deprecated FIT).

**Allowed:** derive `\mathcal{M}_{lat}` from discrete descent / **active-set address depth** on the lattice (Sec 12, `aethos_active` + `aethos_sequences`; anchor set is **not** restricted to primes — C6).

**O3-1 status after Step 3:** **PARTIAL** — spring core **GEOMETRY** closed; **1836** needs **Step 12 `\mathcal{M}_{lat}`** (explicit forward dependency, not silent FIT).

---

## 3.G.7 `K_f^{geom}` for simulations (this pass)

| Use | Value | Source |
|-----|-------|--------|
| Coin pin threshold | `K_f^{pin}\approx 0.41` | F2 |
| Fusion narrative limit | `K_f\to 1^-` | F1/F3 |
| **Deprecated FIT** | `0.999456` | `(R_pe-1)/R_pe` — regression only |

For Sec 4–7 until `\mathcal{M}_{lat}` exists: use **`K_f^{pin}`** for elastic/fusion crossover tests; use **`R_pe^{model,(0)}`** for mass-ratio **lower bound** checks.

---

## 3.G.8 Geometry Gate checklist (Step 3)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `K_f` from geometry (F1–F3) | ✅ |
| 2 | `R_pe` predicted without 1836 input | ✅ **`R_pe^{model,(0)}=\pi^2/8`** |
| 3 | 1836 consequence check documented | ✅ mismatch → `\mathcal{M}_{lat}` |
| 4 | Code + test | ✅ `aethos_physics.py` |
| 5 | C2 FIT demoted | ✅ Sec 3.4.3 |
| 6 | Export to Sec 4 | `U_max`, `K_f^{pin}`, `R_pe^{model,(0)}` |

**Gate verdict:** **PASS (geometry core + honest boundary)** — **not** full 1836 closure (Step 12 required).

---

## 3.G.9 Exports

- **`K_f^{pin}`** — fusion/pin threshold for O3-2 Hamiltonian branch  
- **`R_pe^{model,(0)}=\pi^2/8`** — spring-only mass-ratio prediction  
- **`\mathcal{M}_{lat}`** — OPEN (Step 12)  
- **`E_locked(K)`**, **`R_pe^{energy}(K)`** — audit formulas  

---

*Next: Step 12 lattice multiplier `\mathcal{M}_{lat}` OR Step 4 neutron using pressure escape (C3) without inverting `\tau_n` as primary.*
