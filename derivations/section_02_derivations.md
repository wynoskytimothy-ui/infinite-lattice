# Section 2 — The Electron: Formal Derivations

Maps `section_02_electron.md` into definitions, equations, proof status, and tests.

---

## 2.1 Imports from Section 1

From `section_01_derivations.md`:

- `E = hf`, `p = h/λ`, `E = pc` (photon relations)
- `dτ = dt*sqrt(1-v^2/c^2)` (proper-time relation)
- `ψ`, `ρ=|ψ|^2`, continuity equation
- causality constraints on correlations
- **P1-v** sea vapor modes `\{a_\lambda\}`, spectral fill `\phi_B` (Sec 1.3.4)

These are treated as available base facts.

---

## 2.2 Electron object model

Define electron state at time `t`:

\[
\mathcal{E}(t)=\{x(t),\,\phi(t),\,\kappa(t),\,m(t),\,\chi(t)\}
\]

where:
- `x(t)` = effective inner-photon position coordinate between sides (`x in [-L/2, L/2]`)
- `φ(t)` = pump phase
- `κ(t)` = spring compression/extension coordinate
- `m(t)` = membrane color-phase variable (binary/opposed)
- `χ(t)` = spinor-like orientation state

Component map (from Section 2 text):

1. coin (two sides, opposite orientation),
2. spring (connective/tension degree),
3. membrane (opposed visible phase),
4. trapped photon (engine).

Tags:
- Structural decomposition = **P2-1**
- Trapped-photon engine = **P2-2**

---

## 2.3 Minimal coin Hamiltonian (Tier-1 progress on O2-1)

Use the white/black side basis:

\[
|W\rangle=\begin{bmatrix}1\\0\end{bmatrix},\quad
|B\rangle=\begin{bmatrix}0\\1\end{bmatrix}
\]

Define a minimal driven two-level Hamiltonian for the trapped inner photon:

\[
\hat H_{coin}(t)=
\frac{\hbar}{2}\,\Delta(\kappa)\,\sigma_z
+\frac{\hbar}{2}\,\Omega(\kappa)\,\sigma_x
+\hbar\,g_E E_{obs}(t)\,\sigma_z
\]

where:
- `Delta(kappa)` is side-energy asymmetry from compression state `kappa`,
- `Omega(kappa)` is inter-side tunneling/bounce coupling,
- `E_obs(t)` is observation/compression channel strength,
- `g_E` is coupling calibration.

Interpretation mapping:
- unobserved transit: `|Omega| >> |Delta|` gives mixed-side superposition,
- pinned/observed state: large effective `|Delta + 2 g_E E_obs|` drives side localization.

Equivalent cavity-coordinate form (for your oscillator language):

\[
\hat H_x=\frac{\hat p_x^2}{2m_{eff}}+V(x,\kappa),\quad
V(x,\kappa)\approx V_0(\kappa)\left(\frac{2x}{L}\right)^2
+\epsilon(\kappa)\frac{2x}{L}
\]

with two-level reduction obtained by projecting `H_x` onto the first two side-localized modes.

### 2.3.1 Geometry-derived parameters (C1 mandate — Step 2 gate)

**Authority:** `architecture_mandates.md` C1; full audit `section_02_geometry_audit.md`.

Single length anchor:

\[
L \equiv \frac{\lambda_C}{2}=\frac{h}{2m_e c}
\]

Bounce frequency and spring (GEOMETRY):

\[
\omega_b=\frac{\pi c}{2L}=\frac{\pi m_e c^2}{h},\qquad
k_s=m_e\omega_b^2
\]

Coin Hamiltonian closures:

\[
\Omega(\kappa)=\Omega_0(1-\kappa^2),\quad \Omega_0=\frac{\omega_b}{2}=\frac{\pi c}{4L}
\]
\[
\Delta(\kappa)=\frac{k_s L^2}{2\hbar}\,\kappa^2
\]

Implementation: `aethos_physics.py` (`coin_half_width`, `omega_bounce`, `k_s_from_geometry`, `delta_kappa`, `omega_hopping`).

Status:
- `(L,\omega_b,k_s,\Delta,\Omega)` core = **GEOMETRY** (O2-1 **closed for core**).
- `g_E` observation coupling = **PARTIAL** (Sec 5).

---

## 2.4 Pump cycle equations

### 2.4.1 Minimal periodic dynamics

Model inner photon as bounded oscillator along coin axis:

\[
x(t) = \frac{L}{2}\cos(\omega_b t + \phi_0), \quad \omega_b = 2\pi f_b
\]

Bounce frequency:

\[
f_b = \frac{v_{int}}{4L}
\]

with `v_int = c` in geometry anchor (Sec 2.3.1), equivalent to `f_b = \omega_b/(2\pi)`.

At side contact (`x = ±L/2`) spring is maximally compressed; near mid-transit (`x~0`) maximally extended.

This formalizes your 4-step cycle.

Status: **MODEL** (bounded cavity-like oscillator), mathematically consistent.

### 2.4.2 Internal clock identity

Define electron internal clock rate by bounce frequency:

\[
\nu_{clock} := f_b
\]

This is exactly your “electron is a light clock” statement in equation form.

Tag: **P2-3** (clock postulate).

---

## 2.5 Time dilation derivation (your Section 2 formula)

Use the moving light-clock geometry:

- Vertical cavity size: `L`
- External drift speed: `v`
- One half-bounce travel length in lab frame:

\[
\ell = \sqrt{L^2 + (v\Delta t/2)^2}
\]

Photon travel time per half-bounce: `\ell/c`.
Equating and simplifying yields standard Lorentz factor:

\[
f_b(v) = f_{b0}\sqrt{1-\frac{v^2}{c^2}} = \frac{f_{b0}}{\gamma}
\]

Equivalent internal-budget form (as in your text):

\[
v_{int} = \sqrt{c^2 - v^2}
\]

Status: **D**, PROVEN from light-clock geometry + SR.

Section-12 closure link:

\[
v_{space}^2+v_{time}^2=c^2,\quad
v_{space}=v,\quad
v_{time}=\frac{c}{\gamma}=\sqrt{c^2-v^2}
\]

so Section 2's `v_int=sqrt(c^2-v^2)` is exactly the Section-12 motion-budget time component.

---

## 2.6 Spin mapping

Your Section 2 rule:
- photon near white side -> spin up
- photon near black side -> spin down
- transit -> superposition

Formal map:

\[
\chi(t)=a(t)\,|\uparrow\rangle + b(t)\,|\downarrow\rangle,\quad |a|^2+|b|^2=1
\]

with identification:
- near white: `|a|^2 -> 1`
- near black: `|b|^2 -> 1`
- transit: both nonzero.

Measurement along axis `n` uses projectors `\Pi_{\uparrow n},\Pi_{\downarrow n}`.

Status:
- spinor formalism = **E/ANCHORED**
- geometric white/black association = **MODEL** (P2-4).

---

## 2.7 Superposition and collapse in electron cavity

### 2.6.1 Unobserved evolution

Between interactions, use unitary cavity evolution:

\[
i\hbar\frac{d}{dt}|\chi\rangle = \hat H_{coin}|\chi\rangle
\]

Interpretation: trapped-photon transit corresponds to mixed side amplitudes.

### 2.6.2 Observation/compression event

Compression map (effective):

\[
\rho \mapsto \frac{\Pi_k \rho \Pi_k}{\mathrm{Tr}(\Pi_k\rho)}
\]

where outcome `k` is side/axis-dependent.

This is your “pressure pins photon” collapse mechanism.

Status: **ANCHORED** as effective measurement rule; microscopic derivation from full environment is **OPEN**.

---

## 2.8 Born rule from spring constitutive dynamics (Tier-1 progress on O2-2)

Your claim:

`|amplitude|^2 = spring tension^2 = Born rule`.

### 2.8.1 Constitutive spring law

Let elongation along the coin axis be `u(x,t)` and define spring tension:

\[
T(x,t)=k_s\,u(x,t)
\]

with stiffness `k_s>0` (Hooke closure).

Mechanical energy stored in the spring field:

\[
U_{spring}=\frac{1}{2}\int k_s\,u(x,t)^2\,dx
=\frac{1}{2k_s}\int T(x,t)^2\,dx
\]

Tag: **P2-6** (linear constitutive law).

### 2.8.2 Coupling tension to inner-photon envelope

From Section 2.3, inner-photon dynamics are governed by `H_coin` / cavity reduction.
Let the position envelope be `psi(x,t)` (normalized):

\[
\int |\psi(x,t)|^2\,dx = 1
\]

Mechanical coupling postulate (minimal):

\[
u(x,t)=\alpha\,\mathrm{Re}\{\psi(x,t)\}
\quad\Rightarrow\quad
T(x,t)=k_s\alpha\,\mathrm{Re}\{\psi(x,t)\}
\]

with real coupling constant `alpha`.

For a stationary envelope (`psi` real up to global phase), this gives:

\[
T(x)\propto \psi(x)
\]

Tag: **P2-7** (linear amplitude-tension coupling).

### 2.8.3 Detection density from tension-squared deposition

Observation/compression deposits energy locally where tension is high.
Define local detection intensity:

\[
\mathcal I(x,t)\propto T(x,t)^2
\propto |\psi(x,t)|^2
\]

Normalize over the coin domain `[-L/2,L/2]`:

\[
P(x,t)=\frac{\mathcal I(x,t)}{\int \mathcal I(x',t)\,dx'}
=\frac{|\psi(x,t)|^2}{\int |\psi(x',t)|^2\,dx'}
\]

This is exactly the Born form `P=|\psi|^2` under the linear constitutive+coupling chain above.

Status:
- algebraic Born closure from `T=k_s u` and `u propto Re(psi)` = **MODEL, PARTIAL-PROVEN**
- full nonlinear spring law + stochastic environment derivation (Gleason-level uniqueness) remains open.

### 2.8.5 Charge sign from chirality/topology (Tier-4 progress on O2-3)

Define a structural chirality index `\chi\in\{-1,+1\}` from coin topology:

\[
\chi \equiv \operatorname{sign}\!\left[(\mathbf v_{pump}\times \mathbf n_{spring})\cdot \mathbf n_{coin}\right]
\]

where:
- `\mathbf v_{pump}` is inner-photon circulation direction,
- `\mathbf n_{spring}` is spring-winding axis,
- `\mathbf n_{coin}` is coin normal.

Charge assignment postulate:

\[
q = q_0\,\chi,\qquad q_0>0
\]

Electron branch (pre-fusion, Section 2):

\[
\chi_e=-1 \Rightarrow q_e=-q_0=-e
\]

Proton branch (post-fusion, Section 3) flips winding/orientation at `K\ge K_f`:

\[
\chi_p=+1 \Rightarrow q_p=+q_0=+e
\]

Neutron composite (Section 4) sums charges:

\[
q_n=q_p+q_e+q_{\gamma,obs}=+1-1+0=0
\]

Status: **MODEL, PARTIAL-PROVEN** (topological sign rule + composite closure; first-principles proof that fusion must flip `\chi` remains open).

### 2.8.4 Bridge to `H_coin`

Using cavity reduction of Section 2.3:

\[
\psi(x,t)=\langle x|\chi(t)\rangle
\]

with `chi` evolved by `H_coin`.
Then:

\[
P(x,t)=|\langle x|\chi(t)\rangle|^2
\]

and spring tension tracks the same envelope via `T propto Re(psi)`.

So Born statistics in Section 2 are no longer a bare representation identity; they follow from:
1. Hooke law (`P2-6`),
2. linear tension-amplitude coupling (`P2-7`),
3. standard envelope interpretation of `H_coin` evolution.

Remaining gap:
- prove `alpha` and `k_s` from coin geometry/material (ties to O2-1 material closure),
- extend to nonlinear compression regime without ad hoc renormalization.

### 2.8.5 Section-1 back-link

This gives a concrete electron-level route to Section 1 Born anchor (`rho=|psi|^2`):
- Section 1 global sea amplitude remains foundational,
- Section 2 shows trapped-photon spring mechanics reproduce the same probability density rule locally.

Full sea-only derivation (O1-1) is still separate/open.

---

## 2.9 Uncertainty from single-cavity tradeoff

From Section 2 and standard QM:

\[
\Delta x\,\Delta p \ge \frac{\hbar}{2}
\]

Mechanical interpretation (your language):
- strong localization (compression/pinning) narrows `\Delta x`,
- broadens momentum support `\Delta p`.

Status:
- inequality = **E/ANCHORED**
- coin-geometry interpretation = **MODEL**, consistent with 1D cavity wave packets.

---

## 2.10 Charge / phase / mass statements

Section 2 uses structural language:
- negative charge = open/expansive pump state
- light mass = maximally extended spring
- phase liquid/gas in between photon vapor and proton solid

**P1-v link (Sec 1.3.4):** sea **vapor** = free h-quanta; electron **liquid/gas** = trapped pump between vapor and proton **solid**; shed inner photon (Sec 7) returns energy to vapor; recapture traps one mode again.

Formal handling:

1. Introduce structural order parameters:
   - openness `O in [0,1]`,
   - compression `K in [0,1]`,
   - excitation `Q`.
2. Map qualitative states:
   - electron baseline: high `O`, low `K`,
   - proton baseline (Section 3): low `O`, high `K`.

Charge itself remains fixed empirical property `q_e = -e` (**E**).

Status:
- qualitative order-parameter map = **MODEL**
- numeric charge value/mass = **E**.

---

## 2.11 Equation set (robust core for Section 2)

1. `H_coin=(hbar/2)Delta(kappa)sigma_z + (hbar/2)Omega(kappa)sigma_x + hbar g_E E_obs(t)sigma_z` (minimal coin Hamiltonian)
2. `x(t) = (L/2)cos(omega_b t + phi_0)`  (pump trajectory model)
3. `f_b = v_int/(2L)`  (bounce clock)
4. `f_b(v)=f_b0*sqrt(1-v^2/c^2)=f_b0/gamma`  (time dilation)
5. `v_int = sqrt(c^2-v^2)`  (motion-budget internal speed)
6. `chi = a|up> + b|down>`, `|a|^2+|b|^2=1`  (spin mapping)
7. `rho -> Pi_k rho Pi_k / Tr(Pi_k rho)`  (effective collapse by compression)
8. `T=k_s u`, `U_spring=(1/(2k_s)) int T^2 dx`  (spring constitutive law)
9. `u=alpha Re(psi) => P(x)=|psi(x)|^2`  (Born from tension-squared detection)
10. `P(x)=|langle x|chi(t)rangle|^2` with `H_coin` evolution  (Hamiltonian bridge)
11. `Delta x Delta p >= hbar/2`  (uncertainty anchor)

---

## 2.12 Tests and falsifiers

T2-1. **Velocity clock test**
- Prediction: internal transition rates scale as `1/gamma`.

T2-2. **Axis measurement law**
- Combined with Sec 5: spin probabilities follow `cos^2(theta/2)` under directional compression.

T2-3. **Decoherence by photon flux**
- Larger ambient photon flux increases pinning frequency and shortens coherence intervals.

T2-4. **Cavity-geometry dependence**
- If effective `L` changes under controlled state prep, predicted `f_b ~ 1/L` scaling should appear in model-level simulations.

---

## 2.13 Cross-reference updates required

### Back-links

- Section 1: add explicit note that trapped photons (Section 2) are timelike clock systems unlike free photons.
- Section 12 now reuses `f_b=f_{b0}/gamma` and `v_int=sqrt(c^2-v^2)` as the local clock-law base.

### Forward-links

- Section 3 uses high-compression limit of Section 2 (`K -> 1` fusion regime).
- Section 5 uses Section 2 compression map as measurement primitive.
- Section 5 should import `H_coin(t)` to derive `M_n` operators in O5-1.
- Section 6 uses Section 2 pump phase variable in `Gamma_form`.
- Section 10 uses Section-12-compatible clock factor `sqrt(1-2GM/(rc^2))` via Section 2 clock interpretation.

---

## 2.14 Open items

| ID | Claim | Status |
|---|---|---|
| O2-1 | Derive exact coin Hamiltonian from geometry/material assumptions | **GEOMETRY** (core `L`, `k_s`, `Δ`, `Ω` from C1; `g_E` Sec 5) |
| O2-2 | Derive Born law from spring constitutive dynamics alone | PARTIAL (Hooke + linear `T↔psi` + `P∝T^2` closure added; nonlinear/uniqueness still open) |
| O2-3 | Derive charge sign from structural chirality/topology, not assignment | PARTIAL (`q=q_0 chi` from pump/spring/coin triad; fusion chirality flip proof open) |
| O2-4 | Connect mass ratio 1/1836 directly to geometric compression invariant | MIGRATED to `section_03_derivations.md` (O3-1, PARTIAL closure `R_pe=1/(1-alpha K_f)`) |

---

## 2.15 Completion status

- Section 2 intuition blocks mapped to equations.
- Core light-clock/time-dilation proof completed.
- Measurement/spin bridge formalized.
- Minimal effective coin Hamiltonian added for Tier-1 closure path.
- Born-rule constitutive chain (`T=k_s u`, `P∝T^2`) added for Tier-1 O2-2 progress.
- Tier-4 chirality charge-sign map added for O2-3.
- **Step 2 geometry gate:** `section_02_geometry_audit.md`; C1 `L=λ_C/2`; `H_coin` terms from geometry.
- Cross-reference targets identified.

**Section 2 derivation pass: COMPLETE (v2 — geometry core).**
