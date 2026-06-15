# Book Chapters 3–5 — Particles: Formal Derivations

Maps *Packets and Strings* Part II to `section_02/03/04_*.md`, geometry audits, and mandates **C1–C3**.

---

## Chapter 3 — Electron

### 3.1 Geometry (C1 — non-negotiable)

| Quantity | Formula | Value (e) |
|----------|---------|-----------|
| Cell width | `λ_C = h/(m_e c)` | `2.426×10⁻¹² m` |
| Coin half-width | `L = λ_C/2` | `1.213×10⁻¹² m` |
| Reduced Compton | `ƛ_C = λ_C/(2π)` | `3.862×10⁻¹³ m` |
| Cell crossing | `t_cell = λ_C/c` | `8.09×10⁻²¹ s` |
| Pump period | `T_bounce = 4L/c = 2 t_cell` | `1.619×10⁻²⁰ s` |
| Bounce frequency | `f_b = 1/T_bounce = m_e c²/(2h)` | `6.178×10¹⁹ Hz` |

**Forbidden:** second coin length; `f_0 = m_e c²/h` without factor 2; cell width = `ƛ_C`.

Code: `aethos_physics.coin_half_width`, `f_bounce`, `bounce_period`.

### 3.2 Four components

**P2-1:** coin + spring + membrane + trapped photon.

| Component | Role | Tag |
|-----------|------|-----|
| Coin | white/black sides, opposite coil spins | **P2-1** |
| Spring | polarizer; `T ∝ \|ψ\|²` → Born | **MODEL** + Sec 5 |
| Membrane | opposite visible color; phase ladder | **P2-1** |
| Trapped photon | pump engine | **P2-2** |

### 3.3 Mass and energy

\[
E = m_e c^2 = h f_b
\qquad
m_e = \frac{h f_b}{c^2}
\]

**Status:** **ANCHORED** with `f_b` from C1.

### 3.4 Spin-½

**MODEL:** net spin ½ from opposite coil rotation on two faces; one full pump cycle visits both sides; 720° return for spinor phase (Dirac **ANCHORED**).

Photon position: white ↔ up, black ↔ down, transit ↔ superposition (Sec 2 text).

### 3.4a Four mechanical states (book §3.1b–d)

| Label | Side | Tension | Pump / measurement role |
|-------|------|---------|-------------------------|
| WH | White | Hard | Photon pinned white; impact face |
| WS | White | Soft | Spiraled open; pre-compression / opposite inflate |
| BH | Black | Hard | Photon pinned black; post-flip measurement |
| BS | Black | Soft | Opposite face soft during WH pump leg |

**Free pump:** WH ↔ BS (hard at bounce face, soft opposite). **Transit:** spring soft → superposition (**MODEL**).

**Polarizer (unilateral):** WS → flat coin → axes flip → BH on release (**MODEL** deterministic path; Malus at angle **ANCHORED**, mechanism **PARTIAL** O1-1).

**Bilateral max:** ball state χ → 1 → fusion threshold (Ch 4, Sec 7 hard compression).

**Status:** quadrant map now primary **MODEL**; Dirac four-component = amplitude over WH/WS/BH/BS + transit (**MODEL** interpretation).

### 3.5 Time dilation

\[
f_b(v) = \frac{f_{b0}}{\gamma} = f_{b0}\sqrt{1-v^2/c^2}
\qquad
v_{int}=\sqrt{c^2-v^2}
\]

**Status:** **PROVEN** light-clock + SR; links Sec 12 motion budget.

### 3.6 Charge (O2-3)

\[
q = q_0 \chi
\]

Open/expansive pump → electron negative; positron = mirrored branch.

**Status:** **PARTIAL** — not derived from first principles in book.

### 3.7 Dirac equation

**MODEL interpretation:** four-component ψ as coin-state amplitudes; Dirac matrices as inter-side transitions.

**Status:** **ANCHORED** equation, **MODEL** mechanism.

### 3.8 Tests

| ID | Prediction | Tag |
|----|------------|-----|
| T3-1 | Zeptosecond discreteness at `t_cell`, `T_bounce` | **T** |
| T3-2 | `f_b` sidebands in compression spectroscopy | **MODEL** |

---

## Chapter 4 — Proton

### 4.1 Fusion (P3-1)

\[
K < K_f:\ \text{elastic electron};\qquad K \ge K_f:\ \text{fused proton}
\]

**C2:** `K_f` from geometry / `U_max` — **not** defined by fitting `R_pe`.

### 4.2 Post-fusion clock

\[
f_{clock}^{(p)} = 0
\]

Trapped mode locked; no free bounce pump (O3-4).

### 4.3 Mass ratio (C2 + C6)

Spring-only geometry: `R_pe^{(0)} = π²/8 ≈ 1.23`.

Full ratio needs lattice multiplier `\mathcal{M}_{lat}`:

\[
R_{pe} = R_{pe}^{(0)} \times \mathcal{M}_{lat}
\quad\text{vs}\quad
R_{pe}^E = 1836.152\ldots
\]

**Status:** **E-check**, not FIT-as-definition.

**Forbidden book text:** “compression enforced at 1836×” as primary closure.

### 4.4 Quark regions

Three zones of fused coin (poles + equator): **MODEL** map to `u,u,d` charges summing to +1.

**Status:** charge sum **PROVEN**; fractional quark charges **MODEL**.

### 4.5 Stability

No pressure-escape path; no trapped electron to release.

---

## Chapter 5 — Neutron

### 5.1 Composite (P4-1)

\[
n = p + e^- + \gamma_{obs}
\]

Outer `\gamma_{obs}` = constant observation / compression layer (Sec 4).

### 5.2 Mass excess (**E**)

\[
\Delta m_{np} c^2 \approx 1.293\ \text{MeV}
\qquad
m_e c^2 \approx 0.511\ \text{MeV}
\qquad
Q_\beta \approx 0.782\ \text{MeV}
\]

### 5.3 Lifetime — C3 primary

**Primary:** pressure escape when `P(t) \to P_c` from outer observation interrupting inner pump.

\[
t_{escape} \approx \frac{P_c - P_0}{\alpha \Gamma_{obs}} \sim \tau_n
\]

**Comparison layer:** SM `\Gamma_F` / Fermi estimate ≈ 879 s — **E** cross-check, not equal partner.

### 5.4 Decay channel

\[
n \to p + e^- + \bar\nu_e
\qquad
\gamma_{obs} \xrightarrow{\text{escape}} \nu
\]

**Status:** decay law **E**; photon→ν map **MODEL** (O4-3).

### 5.5 Magnetic moment (O4-4)

\[
\mu_n = -g_{eff}\,\frac{2m_e}{m_n}\,\mu_N
\qquad
g_{eff} \approx 1.76\times 10^3\ \text{(fit)}
\]

**Status:** **PARTIAL** — sign from trapped-electron leakage **MODEL**.

### 5.6 Bound stability

Network pressure sharing suppresses single-neutron escape in nuclei — **MODEL** (Sec 4.7).

---

## Cross-chapter review

- [x] C1 lengths/frequencies synced
- [x] C2 1836 as consequence check
- [x] C3 neutron escape priority
- [x] Membrane + coil spin from Sec 2
- [x] Fermi `\tau_n` demoted to comparison

**Book Ch 3–5 derivations: COMPLETE (v1.0).**
