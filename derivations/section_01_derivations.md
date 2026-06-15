# Section 1 — The Photon Sea: Formal Derivations

Maps every block in `section_01_photon_sea.md` to definitions, equations, proofs, and tests.

---

## 1.1 Objects and notation

| Symbol | Definition | Tag |
|--------|------------|-----|
| Sea **S** | Substrate; space ≡ S | **P1** |
| Photon packet | Minimal excitation of S, no sub-structure | **P2** |
| `h` | Quantum of action scale (Planck) | **E** |
| `c` | Phase speed of massless excitations in S | **E** |
| `f, λ, ω, k` | Frequency, wavelength, `ω=2πf`, `k=2π/λ` | **I** |
| `E, p` | Energy, momentum of packet | **E** |
| `ψ(x,t)` | Complex disturbance amplitude of S | **P3** |
| `ρ = \|ψ\|²` | Detection / occupancy density | **D** |
| `a_λ` | Mode amplitude for sea label `λ` (vapor quantum) | **P1-v** / **D** |
| `B_full` | Spectral band limit for a “full packet” | **MODEL** |
| `τ` | Proper time along worldline | **E** |

---

## 1.1b Lattice cell and E = mc² (book Ch 1 §1.3)

| Quantity | Formula | Tag |
|----------|---------|-----|
| Cell width | `λ_C = h/(mc)` | **C1** / **E** |
| Crossing time | `t_cell = λ_C/c = h/(mc²)` | **DERIVED** |
| Cell measure | `A_cell = λ_C · (c t_cell) = λ_C²` | **MODEL** (spacetime area reading) |
| Rest energy | `E = h/t_cell = mc²` | **ANCHORED** |
| Pump frequency | `f_b = 1/(2 t_cell) = mc²/(2h)` | **C1** |
| Per half-period | `h f_b = mc²/2` | **DERIVED**; full `mc²` per cell crossing quantum `h/t_cell` |

**Seed link:** `A₀² = B₀² = C₀²/2` — equal leg split on 45° seed only. **Not** `B² = C²/2` at every bisection step; step rule is `B_{k+1} = C_k/2`.

**Unification claim:** squared `c` in `E = mc²` and squared chord/hypotenuse in Pythagorean lattice steps share area bookkeeping — **MODEL** narrative; do not identify lattice chord `C_k` with speed `c` without context.

**Dispersion:** `E² = (mc²)² + (pc)²` — **ANCHORED** SR; lattice triangle overlay **MODEL**.

**Open:** universal “all v² in physics = geometric area” — **O1-6**.

---

## 1.2 Intuition: “Universe is a single connected ocean of photons”

**P1 (Sea as medium).** Physical space is identified with a single field medium S. Excitations in S are photons; there is no separate “empty container.”

**P1a (Full connectivity).** The state of S on a Cauchy slice is described by a global field configuration; correlations between separated regions are allowed without auxiliary signaling channels.

**Proof obligation:** P1 is not derivable from prior math — it is the foundational ontology.

**Consistency note (required):** “Connected” does **not** require superluminal signaling. Let `O_A`, `O_B` be observables at spacelike separation. Any viable completion must satisfy

\[
[O_A, O_B] = 0 \quad \text{(microcausality)}
\]

So: one ocean for **state**, not “send messages faster than c.”

**T1:** Bell tests — correlations can violate classical bounds; signaling tests must still show no FTL transfer.

---

## 1.3 Intuition: Properties (no structure, vapor, moves at c, scale h)

### 1.3.1 No internal structure → no internal clock

**P2.** A free photon packet has no internal bound state (no coin/spring).

**D1.** Internal proper time requires a timelike worldline with rest frame. For a massless packet:

\[
ds^2 = c^2 dt^2 - d\mathbf{x}^2 = 0 \quad \Rightarrow \quad d\tau = 0
\]

**Proof (PROVEN):** Along a lightlike curve, `dτ² = dt² - dℓ²/c² = 0` with `dℓ/dt = c`, hence `dτ = 0`.

Equivalent form (**E**):

\[
d\tau = dt\sqrt{1 - v^2/c^2} \xrightarrow[v=c]{} 0
\]

**Status:** Your Section 1 “proof” is **ANCHORED** in special relativity; **PROVEN** given **E**.

### 1.3.2 Moves at c because uncontained

**D2.** Massless plane-wave modes in vacuum propagate at phase velocity `c`.

**E anchor:** Maxwell/ QFT vacuum dispersion `ω = ck` for photons.

**P2 + E ⇒** uncontained photon travels at c in S.

### 1.3.3 Minimum packet scale h

**E (Planck–Einstein).** For a mode of frequency `f`:

\[
E = hf \quad \text{[Einstein 1905, photoelectric / quantum]}
\]

**I (reduced form).** `ħ = h/(2π)` is used in phase `e^{i(kx - ωt)}`.

**Clarification:** “Minimum packet size h” is better stated as **quantization of action per cycle** (`E=hf`), not a geometric diameter.

### 1.3.4 Postulate P1-v: light as vapor of h-quanta (spectrum / full packet)

**Intuition (book language):** the sea is **pure vapor** — a gas of minimal **h-molecules**, each molecule being one **mode quantum** labeled by `\lambda` (frequency, direction, polarization). **Full-spectrum light** is the **full structure** of populated modes in a band; a **single color** is one dominant fundamental packet; **different lights** are made by exciting, filtering, or compressing which modes survive.

**P1-v (Vapor modes).** Sea disturbances decompose into mode quanta:

\[
\Psi_{\mathrm{sea}}(\mathbf x,t)=\sum_{\lambda} a_\lambda(t)\,\epsilon_\lambda(\mathbf x,t),
\qquad
\epsilon_\lambda \propto e^{i(\mathbf k_\lambda\cdot\mathbf x-\omega_\lambda t)}
\]

Each `\lambda` is one **vapor molecule type**; one quantum carries

\[
E_\lambda = h f_\lambda = \hbar \omega_\lambda
\]

**P2 compatibility:** “No internal structure” means **no coin/spring/trapped engine** on a free sea quantum — not “no spectral label.” Free quanta still have `d\tau=0` (Sec 1.3.1).

#### Fundamental vs full packet

| Name | Definition | Tag |
|------|------------|-----|
| **Fundamental packet** | Single-mode excitation with one dominant `\lambda_0` | **D** + **E** (`E=hf`) |
| **Partial spectrum** | Finite set `\{a_\lambda\}` with several `\|a_\lambda\|^2>0` | **D** |
| **Full packet (band max)** | All modes in band `B_full=[f_{min},f_{max}]` populated up to source/coupling limit | **MODEL** |

Define band occupancy (spectral fill):

\[
\phi_B = \frac{\int_{B_{\mathrm{full}}} |a(f,t)|^2\,df}{\int_{B_{\mathrm{full}}} |a_{\max}(f)|^2\,df}
\]

with `\phi_B=1` = **full structure** on that band (white / broadband limit for the source).

#### How different lights are made (mechanism)

1. **Source** populates `\{a_\lambda\}` (thermal, line, LED, laser cavity).
2. **Medium** filters or scatters modes (absorption lines, `\sigma_{\gamma DM}`, etc.).
3. **Compression** (Sec 5) couples selected modes to coin/spring (`g_E E_{obs}(t)`).
4. **Detection** reads `\mathcal I \propto |a_\lambda|^2` / `\|ψ\|^2` (Sec 1.5.5a, Sec 2.8).

Examples:

- **Laser / line:** `\|a_{\lambda_0}\|^2 \approx 1` — one fundamental packet.
- **White / thermal band:** many `\lambda` — partial or near-full `\phi_B` on visible band.
- **Transform-limited pulse:** full spectral **support** on `B_full` with `\Delta\omega\,\Delta t \gtrsim 1` (time–bandwidth limit) — maximal **structured** burst on a finite band, not infinite frequencies.

#### Cross-links

| Process | Section | Vapor language |
|---------|---------|----------------|
| Inner photon **shed** in barrier | 7 **P7-2** soft | engine melts into sea vapor (`\chi\downarrow`) |
| **Recapture** by passing photon | 7.3.3 | one mode re-trapped into coin cavity |
| Born / tension detection | 2.8, 1.5.5a | `\|a_\lambda\|^2` ↔ spring tension² |
| Phase ladder vapor → solid | 2.10 | sea **vapor** → electron **liquid/gas** → proton **solid** |

Status: **P1-v MODEL** (interpretation on **ANCHORED** Fourier/`E=hf`); band calibration of `B_full`, `\phi_B` **OPEN** (O1-5).

**T1-v:** spectrometer resolves line vs broadband; filter changes `\phi_B`; laser cavity narrows to single `\lambda_0`.

---

## 1.4 Intuition: Key math `E=hf`, `c=fλ`, `p=h/λ`

These are **E** relations for photons in vacuum.

### 1.4.1 Closure: `E = pc`

**Given:**

\[
E = hf,\quad c = f\lambda,\quad p = \frac{h}{\lambda}
\]

**Derive (PROVEN — algebra):**

\[
p = \frac{h}{\lambda} = \frac{hf}{c} = \frac{E}{c}
\quad\Rightarrow\quad
E = pc
\]

**Status:** **D**, **PROVEN**.

### 1.4.2 Dispersion relation

From `c = fλ` and `ω = 2πf`, `k = 2π/λ`:

\[
\omega = ck
\]

**Status:** **D**, **ANCHORED** (massless vacuum).

### 1.4.3 Relativistic energy–momentum (bridge to Sec 2–3)

**E (general):**

\[
E^2 = (pc)^2 + (m_0 c^2)^2
\]

**Photon limit** `m_0 = 0`: recovers `E = pc` (**D**).

---

## 1.5 Intuition: Ocean fully connected → nonlocality, entanglement, ψ, collapse

### 1.5.1 Wave function = shape of disturbance in S

**P3.** For a system coupled to S, assign amplitude `ψ` such that `ρ = |ψ|²` gives event rates.

**D3 (Born rule — empirical anchor).** Measurement probability density:

\[
P(x) = |\psi(x)|^2
\]

**Status:** **E** (Born rule); **P3** identifies ψ with sea disturbance **MODEL**.

### 1.5.2 Evolution of disturbance

**E (Schrödinger, non-relativistic):**

\[
i\hbar \frac{\partial \psi}{\partial t} = \hat{H}\psi
\]

**E (Klein–Gordon / Dirac for relativistic extensions).**

**P3 + E ⇒** “shape in sea” evolves by standard wave dynamics until interaction.

### 1.5.3 Continuity (probability conservation)

Define current (1D example):

\[
J = \frac{\hbar}{2mi}\left(\psi^* \frac{\partial \psi}{\partial x} - \psi \frac{\partial \psi^*}{\partial x}\right)
\]

**Theorem (PROVEN):** If `ψ` solves Schrödinger equation,

\[
\frac{\partial \rho}{\partial t} + \frac{\partial J}{\partial x} = 0
\]

**Status:** **D**, standard QM.

### 1.5.4 Entanglement = correlated disturbances

Two subsystems A, B: joint state `ψ_{AB}`.

**P1a ⇒** single global `ψ_{AB}` is admissible.

**E:** Bell violations when `ψ_{AB}` not factorizable.

**Your phrase “two ripples in same water”:** **MODEL** language for **E** entangled states.

**T2:** CHSH `S ≤ 2` classical; quantum `S_max = 2√2`.

### 1.5.5 Collapse = impact / boundary interaction

**P4 (Interaction selection).** An interaction localizes coupling between S and a detector degree of freedom; effective update:

\[
\psi \rightarrow \frac{\hat{P}_k \psi}{\|\hat{P}_k \psi\|}
\]

where `P̂_k` projects onto outcome k.

**Status:** **MODEL** as effective rule; full derivation from unitary + large environment = **OPEN** (decoherence program, Sec 5).

**Your “ripple hits shore”:** **MODEL** consistent with **P4** if shore = detector boundary.

### 1.5.5a Born rule from sea tension mechanics (Tier-4 progress on O1-1)

Extend Section-2 constitutive chain to sea disturbance field `\psi_S` on `S`.

Define sea tension:

\[
T_S(\mathbf x,t)=k_S\,\mathrm{Re}\{\psi_S(\mathbf x,t)\}
\]

Detection/deposition intensity:

\[
\mathcal I_S(\mathbf x,t)\propto T_S(\mathbf x,t)^2
\propto |\psi_S(\mathbf x,t)|^2
\]

Normalized outcome density:

\[
P_S(\mathbf x,t)=\frac{|\psi_S(\mathbf x,t)|^2}{\int |\psi_S(\mathbf x',t)|^2\,d^3x'}
\]

Embedded coin limit (`\psi_S\to\psi_{coin}` on cavity support) recovers Section-2.8 Born closure.

Global uniqueness (why only `|\psi|^2`, not higher powers) remains open without nonlinear spring + environment ergodicity proof.

Status: **MODEL, PARTIAL-PROVEN** (sea-level `P\propto T^2` map; Gleason-level uniqueness still open).

### 1.5.6 “Instant propagation” — precision required

**Claim in text:** disturbance propagates “instantly through ocean.”

**Correction for rigor:**

- **Global correlation** of `ψ` can exist without communication.
- **Local update** of observables still respects light cone for **signaling**.

**D4.** Retarded Green’s function for massless field: influence on `ψ(x,t)` from source at `(x',t')` supported on light cone `|x-x'| = c(t-t')`.

**T3:** No observable superluminal signal; only correlation at spacelike separation.

---

## 1.6 Intuition: Photon time — emission = absorption

**D1 (reprise):** `dτ = 0` on photon worldline.

**Interpretation (MODEL):** Emission and absorption are the endpoints of one lightlike segment; no internal phase accumulation between them.

**Formal statement:** Phase along worldline:

\[
\phi = \int k_\mu dx^\mu = 0 \quad \text{(null, no rest phase clock)}
\]

**Status:** **ANCHORED** + **MODEL** interpretation.

### 1.6.2 Sea vacuum energy density (Tier-4 progress on O1-3)

Define sea zero-point energy density with mode cutoff:

\[
u_S=\int_0^{\omega_{\max}} \frac{\hbar\omega}{2}\,g(\omega)\,d\omega
\]

and vacuum mass-energy density:

\[
\rho_S=\frac{u_S}{c^2}
\]

Raw UV estimate diverges; impose sea locking fraction `\Pi_{vac}\in(0,1]` from non-propagating/locked modes (Sections 2–3):

\[
\rho_{S,eff}=\Pi_{vac}\,\rho_S
\]

Cosmological interface:

\[
\rho_\Lambda \equiv \rho_{S,eff}
\quad\Rightarrow\quad
\Pi_{vac}=\frac{\rho_\Lambda c^2}{u_S(\omega_{\max})}
\]

If `\omega_{\max}\sim c/L_{cell}` with microscopic cell size `L_{cell}`, then

\[
u_S(\omega_{\max})\sim \frac{\hbar c}{2L_{cell}^4}
\]

giving calibration equation for `(L_{cell},\Pi_{vac})` against observed `\rho_\Lambda`.

Status: **MODEL, PARTIAL-PROVEN** (cutoff vacuum bookkeeping + `\Pi_{vac}` interface; unique `(L_{cell},\Pi_{vac})` without external fit still open).

---

## 1.6.1 Interpretation note — “Riding a beam”

Your intuition can be formalized as:

- free photon paths are null (`dτ=0`),
- so along that null chain there is no internal elapsed clock-time,
- interactions are with globally correlated field structure in the same sea.

Compact statement:

> Along a null worldline, endpoint separation is external-coordinate time, not photon proper time; thus correlation structure can be treated as “already connected” from the photon-clock viewpoint.

Critical constraint:

- This does **not** imply controllable faster-than-light messaging.
- It implies nonlocal correlation access, while signaling remains causal.

Cross-link: see Section 6 no-signaling constraints.

---

## 1.7 Intuition: “All revealed from creation; observation selects endpoint”

**P5 (Unitary previsibility).** Closed system: `ψ(t)` from `ψ(0)` via `U(t)`.

**D5.** Born probabilities for complete set of outcomes are fixed by `ψ` at measurement — no extra hidden draw from “future.”

**P4 + P5 split:**

| Statement | Tag | Status |
|-----------|-----|--------|
| ψ contains all probabilities before measurement | **D** from unitarity + Born | **ANCHORED** |
| Path is single classical trajectory before measurement | — | **FALSE** in standard QM |
| “Observation selects endpoint” | **P4** effective | **MODEL** |

**Honest limit:** Your “path exists in sea from emission” is **not** classical path realism unless **hidden** variables added (Sec 6 discusses).

**T4:** Delayed-choice interferometry — consistent with no classical path before which-path info exists.

---

## 1.8 Section 1 — Complete equation sheet

| # | Equation | Tag | Proof status |
|---|----------|-----|--------------|
| 1 | `E = hf` | E | empirical |
| 2 | `p = h/λ` | E | empirical |
| 3 | `c = fλ` | E/I | definition + vacuum |
| 4 | `E = pc` | D | algebra from 1–3 |
| 5 | `ω = ck` | D | from 3 |
| 6 | `dτ = dt√(1-v²/c²)` | E | SR |
| 7 | `dτ = 0` for photon | D | from 6, v=c |
| 8 | `ds² = c²dt² - dx²` | E | Minkowski |
| 9 | `ρ = \|ψ\|²` | E | Born |
| 10 | `∂ρ/∂t + ∇·J = 0` | D | from Schrödinger |
| 11 | `E² = (pc)² + (mc²)²` | E | SR/QFT bridge |
| 12 | `\Psi_{\mathrm{sea}}=\sum_\lambda a_\lambda \epsilon_\lambda` | P1-v / D | mode decomposition |
| 13 | `E_\lambda = h f_\lambda` | E | per vapor quantum |
| 14 | `\phi_B` spectral fill on band `B_full` | MODEL | full-packet measure |
| 15 | `λ_C = h/(m_e c)`, `L = λ_C/2` | GEOMETRY / C1 | Sec 2 anchor |
| 16 | `t_{\mathrm{cell}} = λ_C/c = h/(m_e c²)` | D | light crosses one cell |
| 17 | `c = λ_C / t_{\mathrm{cell}}` | D | **PROVEN** (algebra) |
| 18 | `T_{\mathrm{bounce}} = 4L/c = 2 t_{\mathrm{cell}}` | D / C1 | full pump cycle |
| 19 | `f_b = 1/T_{\mathrm{bounce}} = m_e c²/(2h)` | D | matches `aethos_physics.f_bounce` |
| 20 | Seed: `A_0² = B_0² = C_0²/2` (`C_0=\sqrt2`) | I | 45°–45°–90° seed |
| 21 | `A_{k+1} = C_k/2`, `B_{k+1} = 1-\sqrt{1-A_{k+1}^2}` | P1-L / D | `pi/constructive_pi.py` |
| 22 | `(1-A_k)² + B_k² = 1` (unit-circle coords) | D | book Ch 1 notation |
| 23 | `N_k C_k \to 2\pi`, `π = \lim 2^{k+1} C_k` | D | `pi/README.md` |

---

## 1.9 Section 1 — Postulate list (minimal)

1. **P1** — Space is sea S (ontology).
2. **P1a** — Global states on S allowed; signaling still causal.
3. **P1-v** — Sea light is vapor of h-quanta (modes `\lambda`); full-spectrum = populated band structure.
4. **P1-L** — Physical address structure is a π-lattice generated by right-triangle bisection (book Ch 1–2; `pi/` engine).
5. **P2** — Free photon = structureless excitation (no coin/spring engine).
6. **P3** — `ψ` is disturbance amplitude; `|ψ|²` gives rates.
7. **P4** — Interaction selects outcome (effective collapse).

Everything else in Section 1 is **D** or **E**.

---

## 1.10 Lattice cell ontology (book Chapter 1)

Maps *Packets and Strings* Ch 1 to sea ontology (Secs 1.1–1.7) and geometry mandates **C1**.

### 1.10.1 Three scales (do not conflate)

| Object | Symbol | Definition | Tag |
|--------|--------|------------|-----|
| **Lattice cell** (address width) | `λ_C` | `h/(m_e c)` — full Compton wavelength | **GEOMETRY** |
| **Coin half-width** (cavity) | `L` | `λ_C/2 = h/(2 m_e c)` — **C1** single anchor | **GEOMETRY** |
| **Reduced Compton** (QM texts) | `ƛ_C` | `ħ/(m_e c) = λ_C/(2π)` | **I** |

**Rule:** Book diagrams may label the **cell** by `λ_C`. Section 2 bounce geometry always uses **`L`**, never a second fitted length.

Cross-link: `aethos_physics.coin_half_width`, `compton_wavelength`, `bounce_period`.

### 1.10.2 Cell tick and light speed

Define the **cell crossing time**:

\[
t_{\mathrm{cell}} \equiv \frac{\lambda_C}{c} = \frac{h}{m_e c^2}
\]

**Derive (PROVEN):**

\[
\frac{\lambda_C}{t_{\mathrm{cell}}} = \frac{h/(m_e c)}{h/(m_e c^2)} = c
\]

**Interpretation (MODEL):** one lattice update advances one cell-width at speed `c`. **ANCHORED** identity once `λ_C`, `h`, `c` are **E**.

Full electron **pump period** (C1):

\[
T_{\mathrm{bounce}} = \frac{4L}{c} = \frac{2\lambda_C}{c} = 2\, t_{\mathrm{cell}}
\qquad
f_b = \frac{1}{T_{\mathrm{bounce}}} = \frac{m_e c^2}{2h}
\]

Do **not** identify `t_{\mathrm{cell}}` with `T_{\mathrm{bounce}}` without the factor 2.

### 1.10.3 Cell spacetime 2-volume and `E = mc²`

Natural **spacetime cell measure** (space width × time height in `ct` units):

\[
\mathcal{A}_{\mathrm{cell}} = \lambda_C \cdot (c\, t_{\mathrm{cell}}) = \lambda_C^2
= \frac{h^2}{m_e^2 c^2}
\]

In lattice language (book Ch 1.3):

\[
E = m\, c^2 \quad \Leftrightarrow \quad \text{energy} = \text{mass fullness} \times \mathcal{A}_{\mathrm{cell}}
\]

when mass is measured in natural cell units. This is **MODEL** interpretation on **ANCHORED** `E=mc²` (**E**).

**Not claimed:** every `v²` in physics is a literal spatial area at this pass (see **O1-6**).

### 1.10.4 Sea tension and Higgs vev (preview)

Book Ch 1.4 identifies sea intrinsic tension with Higgs vev `v \approx 246\,\mathrm{GeV}`.

| Statement | Tag | Status |
|-----------|-----|--------|
| Sea has mechanical tension / stiffness | **MODEL** | Sec 2 spring chain |
| `v = 246` GeV as sea tension scale | **E** anchor | SM input |
| Identification tension ≡ Higgs vev | **MODEL** | full mechanism **Ch 11** |

### 1.10.5 π-lattice recurrence (sync with `pi/`)

**Seed** (`k=0`): `A_0 = B_0 = 1`, `C_0 = \sqrt{2}`, `N_0 = 4`.

**Seed identity (eq. book 1.1 — seed only):**

\[
A_0^2 = B_0^2 = \frac{C_0^2}{2}
\]

**Recurrence** (`constructive_pi.pi_recurrence`):

\[
A_{k+1} = \frac{C_k}{2}, \quad
B_{k+1} = 1 - \sqrt{1 - A_{k+1}^2}, \quad
C_{k+1} = \sqrt{A_{k+1}^2 + B_{k+1}^2}, \quad
N_{k+1} = 2 N_k
\]

**Unit-circle rule** (book coordinates `(1-A,\, B)`):

\[
(1 - A_k)^2 + B_k^2 = 1
\]

**π closure:**

\[
N_k\, C_k \to 2\pi \quad (k\to\infty)
\qquad\Rightarrow\qquad
\pi = \lim_{k\to\infty} 2^{k+1}\, C_k
\]

(since `N_k = 4\cdot 2^k`).

**Book ↔ code notation map**

| Book Ch 1 label | `pi/` / `constructive_pi` | Role |
|-----------------|---------------------------|------|
| `B` (halving leg) | `A` | bisected chord / half-angle sine |
| `A` (radial complement) | `1 - \sqrt{1-A_{\mathrm{code}}^2}` chain | sagitta / versine coordinate |
| `C` | `C` | hypotenuse |

Full iteration tables and π limits: `derivations/book_ch02_pi_lattice_derivations.md`.

**Status:** recurrence arithmetic **PROVEN** in `pi/`; physical identification "particles follow lattice addresses" = **P1-L MODEL**.

### 1.10.6 Microcausality (required with "connected sea")

**P1a (reprise):** One global state on S does **not** permit controllable superluminal signals.

\[
[O_A,\, O_B] = 0 \quad \text{for spacelike-separated observables}
\]

Disturbances propagate on the light cone (**D4**, Sec 1.5.6). Entanglement = correlated sea state, not message channel.

**T1 (reprise):** Bell violation + no-signaling tests.

### 1.10.7 Free photon vs cell pattern (book §1.4a)

| Property | Free photon (**P2**) | Electron pattern (Sec 2) |
|----------|----------------------|---------------------------|
| Structure | none | coin + spring + trapped photon |
| Internal clock | `d\tau = 0` | `T_{\mathrm{bounce}}` finite |
| Spatial extent | none (packet) | one cell, cavity `L` |
| Sea role | propagating vapor quantum | localized pump in S |

---

## 1.11 What Section 2 must import from Section 1

| Import | Used for |
|--------|----------|
| `E = hf`, `p = h/λ` | Inner photon energy in coin |
| `dτ = 0` for free photon | Contrast with trapped photon clock |
| `ψ`, `ρ`, continuity | Spread of inner photon in coin |
| `E = pc` bridge | Relativistic kinematics of bounce |
| **P1-v** `\{a_\lambda\}`, `\phi_B` | Shed/recapture as vapor ↔ trapped mode (Secs 7, 5) |
| `λ_C`, `L`, `t_{\mathrm{cell}}`, `T_{\mathrm{bounce}}`, `f_b` | C1 coin geometry; no second length FIT |
| **P1-L** | Lattice address language (book Ch 1–2) |
| **P1a** microcausality | Entanglement without signaling (Sec 6) |

### Forward-links from P1-v

- **Section 2:** phase ladder vapor (sea) → liquid/gas (electron) → solid (proton); Born from `\|a\|^2`.
- **Section 5:** compression selects which sea modes couple to coin.
- **Section 7:** soft shred = vaporize inner photon; recapture = trap one mode quantum.

---

## 1.12 Section 1 — Open problems (explicit)

| ID | Claim | Status |
|----|-------|--------|
| O1-1 | Derive Born rule from sea mechanics alone | **PARTIAL** (sea `T_S^2` deposition map; Gleason/uniqueness still open) |
| O1-2 | “Instant” connectivity without violating causality | **PARTIAL** (D4 + T3) |
| O1-3 | Unique vacuum energy density of S | **PARTIAL** (`rho_S=Pi_vac u_S/c^2` cutoff bookkeeping; unique `L_cell` calibration open) |
| O1-4 | Geometric “packet size” = h | **REFRAMED** as E=hf |
| O1-5 | Calibrate `B_full`, `\phi_B` for sources (full vs partial packet) | **OPEN** |
| O1-6 | All squared quantities in physics = geometric areas | **OPEN** (conjecture; Ch 1.3 bullet deferred to Ch 13) |
| O1-7 | Physical particle trajectories follow π-lattice addresses | **MODEL** (P1-L); needs zeptosecond **T** |

---

## 1.13 Section 1 — Review checklist

- [x] Every paragraph in `section_01_photon_sea.md` mapped
- [x] `E=pc` derived from stated key math
- [x] Proper time proof for photon
- [x] ψ/Born/continuity linked
- [x] Causality caveat on “connected ocean”
- [x] Falsifiers listed (T1–T4)
- [x] Honest limits on “path revealed at emission”
- [x] **P1-v** vapor spectrum / full packet (Sec 1.3.4)
- [x] **P1-L** lattice cell ontology synced with C1 + `pi/` (book Ch 1, v1.2)

**Section 1 derivations: COMPLETE (v1.2).**

Next file: `section_02_derivations.md` (electron coin, trapped photon, pump, γ, uncertainty).
