# Section 4 — Geometry Audit (Step 4 Gate)

**Mandates:** `architecture_mandates.md` (C3 locked).  
**Imports:** Step 2 — `L_0`, `ω_b`, `k_s`, `Π_{pin}(κ)`, `U_{max}`; Step 3 — proton fused drain base state.

**Goal:** **Pressure escape** is the primary `\tau_n` story; SM weak rate is **comparison only**.

---

## 4.G.1 Composite (model definition)

\[
n := p + e^- + \gamma_{obs}
\]

| Piece | Source | Tag |
|-------|--------|-----|
| `p` | Sec 3 fused drain | **MODEL** |
| `e^-` | Sec 2 coin + inner pump | **GEOMETRY** |
| `\gamma_{obs}` | outer trapped observation photon | **MODEL** |

Charge closure `q_n=0`: **PROVEN** arithmetic.

---

## 4.G.2 Pressure variable (GEOMETRY imports)

From Sec 4.5.3 with Sec 2 `\Delta(κ)`, `\Omega(κ)`:

\[
\Pi_{pin}(\kappa)=\frac{|\Delta_{eff}|}{|\Delta_{eff}|+\Omega},
\qquad
\Delta_{eff}=\Delta(\kappa)+2g_{obs}\bar E_{obs}
\]

\[
P(t)=\frac{1}{2}k_{el}(K-K_0)^2+\frac{\hbar}{2}|\Delta_{eff}|\Pi_{pin}-P_0
\]

Free-neutron build rate (Sec 4.5.4):

\[
\frac{dP}{dt}=\alpha\Gamma_{obs}-\beta R_{share},
\qquad
\alpha=\hbar\omega_{in0}\Pi_{pin}
\]

**C1 lock:** `\omega_{in0}^{geom}=\omega_b=\pi c/(2L_0)` with `L_0=h/(2m_e c)`.

**Free limit:** `R_{share}\approx 0`, `\Pi_{pin}\to 1` at `\kappa\to 1`.

Tag: **GEOMETRY** for `\alpha` when `\omega_b`, `\Pi_{pin}` from coin; **PARTIAL** for `g_{obs}\bar E_{obs}` micro closure.

---

## 4.G.3 Escape threshold (PRIMARY — C3)

**P4-Escape (GEOMETRY + interpretation):** Decay fires when trapped pressure crosses rupture threshold:

\[
P(t_{escape})=P_c
\]

Linear rise (free neutron):

\[
t_{escape}=\frac{P_c-P_0}{\alpha\Gamma_{obs}}
\]

**Decay channel (model):** `\gamma_{obs}` escape → `\bar\nu` identification; electron release; `p` remains.

**NOT primary:** Fermi `\beta`-decay rate as definition of `\tau_n`.  
**Allowed:** compare `t_{escape}` to electroweak benchmark `\tau_{weak}^{SM}\approx 886\ \text{s}` as **E** consistency check.

---

## 4.G.4 `P_c - P_0` candidates (geometry hierarchy)

| ID | Candidate | Scale (MeV) | Tag | Role |
|----|-----------|-------------|-----|------|
| **G1** | `U_{max}` | `\approx 0.63` | **GEOMETRY** | Spring rupture at full coin compression |
| **G2** | `Q_\beta` | `\approx 0.782` | **E + MODEL** | `\beta`-endpoint / outer-trap packet scale |
| **G3** | `\Delta m_{np}c^2` | `\approx 1.293` | **E** | Composite mass-excess bookkeeping |

Leading **micro falsifiable** choice for cavity mode: **G3** gap + **C1** `\omega_b` → invert `\Gamma_{obs}` for **E check** only.

Code: `p_c_gap_joules(kind)`, `predict_neutron_escape()`, `calibrate_neutron_pressure(gap="cavity")`.

---

## 4.G.5 `\Gamma_{obs}` boundary (honest OPEN)

Coin bounce rate `f_b\sim 10^{20}\ \text{Hz}` is **not** equal to `\Gamma_{obs}\sim 10^{-3}\ \text{s}^{-1}`.

**Conclusion:** outer-trap observation / leak rate `\Gamma_{obs}=\sigma_{obs}\Phi_{obs}` requires **environment + outer-layer microprocess** (Sec 6, Step 12 Zeno descent) — **not yet derived** from `L_0` alone.

**C3 compliance:**

| Direction | Status |
|-----------|--------|
| **Forward:** given `\Gamma_{obs}`, predict `t_{escape}` | **GEOMETRY** |
| **Inverse:** fit `\Gamma_{obs}` from `\tau_n` | **FIT / E check** — not primary definition |
| **Compare** to `\tau_{weak}^{SM}` | **ANCHORED** — agreement `\sim 1\%` is coincidence check, not mechanism |

---

## 4.G.6 Numeric closure (cavity + C1)

With `L_0` (C1), `\omega_b=\pi c/(2L_0)`, `\Pi_{pin}=1`, `P_c-P_0=\Delta m_{np}c^2`:

\[
\alpha=\hbar\omega_b\approx 0.255\ \text{MeV},
\qquad
\Gamma_{obs}=\frac{\Delta m_{np}c^2}{\alpha\,\tau_n}\approx 5.76\times 10^{-3}\ \text{s}^{-1}
\]

\[
t_{escape}=\frac{P_c-P_0}{\alpha\Gamma_{obs}}=\tau_n^{E}\approx 879.4\ \text{s}
\]

**Weak benchmark:** `\tau_{weak}^{SM}\approx 886\ \text{s}` — same order, **not** derived here.

---

## 4.G.7 Hard flatten link (P7-2 export)

Sec 7 **hard** regime: `\Pi_{pin}\to 1`, `P\to P_c` → escape/capture, **no** soft recapture.  
Neutron free decay is the reference **hard flatten** instance for `\gamma_{obs}`.

---

## 4.G.8 Geometry Gate checklist (Step 4)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | `\alpha` from C1 `\omega_b`, `\Pi_{pin}` | ✅ |
| 2 | `t_{escape}` law primary (not weak Fermi) | ✅ C3 |
| 3 | `P_c` candidates from geometry / E scales | ✅ G1–G3 |
| 4 | `\Gamma_{obs}` forward derivation | ⚠️ **OPEN** (env / Zeno) |
| 5 | Code + test | ✅ `aethos_physics.py` |
| 6 | Weak rate comparison documented | ✅ `TAU_WEAK_SM_EST` |
| 7 | Export to Sec 7 P7-2, Sec 9 `R_{share}` | ✅ forward contract |

**Gate verdict:** **PASS (geometry core + honest boundary)** — `\Gamma_{obs}` micro still OPEN (O4-1).

---

## 4.G.9 Exports

- `\alpha^{geom}`, `\Pi_{pin}(\kappa)`, `P_c` candidates  
- `t_{escape}(P_c,\alpha,\Gamma_{obs})` — **primary prediction**  
- `\Gamma_{obs}^{cal}` — inverse **E check** only  
- Hard-flatten reference for Sec 7  

---

*Next: Step 5 measurement (`\Lambda_n`, `M_n` from `H_{coin}`) or derive `\Gamma_{obs}` from Sec 6 / Step 12 active-set Zeno.*
