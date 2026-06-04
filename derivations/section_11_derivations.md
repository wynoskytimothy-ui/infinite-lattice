# Section 11 — Dark Matter and Dark Energy: Formal Derivations

Maps `section_11_dark_matter_dark_energy.md` into equations/proofs/tests with cosmology constraints.

---

## 11.1 Imports

From Section 2:
- electron structural decomposition (coin/spring/membrane/trapped-photon)

From Section 10:
- gravity and cosmological acceleration equations (`r_s`, Friedmann, `w`)

From Section 5:
- observation as physical interaction/measurement channel

---

## 11.2 Structural split hypothesis

Your core hypothesis:

\[
\text{Normal matter unit} = (\text{spring} + \text{inner photon})
\]
\[
\text{Dark matter unit} = (\text{spring only})
\]
\[
\text{Dark-energy contribution} = (\text{inner photon freed into sea})
\]

Tag this as postulate set:
- **P11-1** spring-without-inner-photon as dark-matter ontology
- **P11-2** freed inner-photon pressure as dark-energy source channel

Status: **MODEL** (not standard anchor).

---

## 11.2.1 Postulate P11-3: connective DM mesh (ripple fill)

**Intuition:** dark-matter spring units (no pump) are **everywhere**, pass through barriers that block EM/coins, can **stretch** into thin filaments and **back-fill** ripples until a continuous tension path exists between distant **pumped** sites (normal matter).

This is **not** electron-style entanglement among DM units alone (`f_{clock}^{DM}\approx 0`). It is the **substrate** on which:

- Sec 6 pump phase-lock modes propagate (`\ell_c` = mesh reach before snap/decoherence);
- Sec 7 soft tunneling spring modes transit (Sec 7.3.4).

Define **fill fraction** on path `\mathcal P` between sites A and B:

\[
\phi_{AB}=\frac{1}{L_{\mathcal P}}\int_{\mathcal P}\phi(s)\,ds,\qquad \phi\in[0,1]
\]

- `\phi=0`: empty channel; `\phi=1`: fully filled DM-backed ripple (continuous string).

Minimal fill dynamics (MODEL):

\[
\frac{d\phi}{dt}=\Gamma_{fill}(1-\phi)-\Gamma_{snap}\,\phi
\]

Formation rate modifier for Sec 6 (cross-link):

\[
\Gamma_{form}\leftarrow \Gamma_{form}\,\phi_{AB}\,e^{-d/\ell_c}
\]

**Properties claimed (MODEL):**
- barrier-transparent (no inner photon to pin);
- stretchable filaments with finite energy cost (not infinitely thin without limit);
- halo `\rho_{DM}` supplies fill material.

Status: **P11-3 MODEL**; `\Gamma_{fill}`, `\Gamma_{snap}`, `\ell_c(\rho_{DM})` calibration **OPEN** (O11-5).

---

## 11.3 Property map equations

### 11.3.1 Gravitating but electromagnetically dark

If dark component carries mass density `rho_DM` and negligible EM coupling:

\[
\nabla^2 \Phi = 4\pi G(\rho_b + \rho_{DM})
\]

but interaction cross-section with photons approximately suppressed:

\[
\sigma_{\gamma DM} \approx 0 \quad (\text{effective})
\]

This captures your “has gravity, no light interaction.”

Status:
- Poisson/gravity relation = **ANCHORED**
- near-zero EM coupling assumption = **MODEL/PHENOMENOLOGICAL**.

### 11.3.2 No chemical clumping channel

If no inner-photon-mediated bonding channel exists, then non-gravitational binding term is absent:

\[
U_{bind}^{DM}\approx 0,\qquad U_{total}^{DM}\approx U_{grav}
\]

Hence halo-like diffuse gravitational clustering is expected.

Status: **MODEL**, observationally compatible at large scale.

### 11.3.3 Quantitative `sigma_{gamma DM}` from spring microdynamics (Tier-4 progress on O11-1)

Normal electron interaction (Sections 5–6) uses inner-photon-enabled channel:

\[
\sigma_{obs,e}(\omega)=A_{eff}\,\eta_{geom}\,S_{res,e}(\omega),
\qquad
S_{res,e}(\omega)=\frac{(\Gamma_2/2)^2}{(\omega-\omega_0)^2+(\Gamma_2/2)^2}
\]

with `omega_0` set by trapped-photon pump frequency.

For dark-matter spring-only units (no inner photon), define:

\[
S_{res,DM}(\omega)\equiv 0
\]

because there is no internal two-level resonance mode to couple.

Residual non-resonant elastic coupling remains through spring displacement only:

\[
H_{int,DM}=-q_{spring}\,u\,E_\gamma,\qquad
q_{spring}\sim \frac{e}{L_{spring}}
\]

This gives Rayleigh-like suppression:

\[
\sigma_{\gamma DM}(\omega)
\le
\sigma_{geom}\left(\frac{\hbar\omega}{E_{spring}}\right)^{\!4}
\]

with

\[
\sigma_{geom}=\pi R_{spring}^2,\qquad
E_{spring}=\hbar\omega_{spring}=\hbar\sqrt{\frac{k_s}{m_{spring}}}
\]

Equivalent compact form used in derivations:

\[
\sigma_{\gamma DM}(\omega)=\sigma_{geom}\,\mathcal K_{sup}(\omega),
\qquad
\mathcal K_{sup}(\omega)=\left(\frac{\hbar\omega}{E_{spring}}\right)^{\!4}
\]

### 11.3.4 Observational bound interface

Direct-detection and photon-scattering null results impose:

\[
\sigma_{\gamma DM}(\omega)\;<\;\sigma_{\max}^{exp}(\omega)
\]

Model constraint:

\[
\pi R_{spring}^2\left(\frac{\hbar\omega}{E_{spring}}\right)^{\!4}
<\sigma_{\max}^{exp}(\omega)
\]

which fixes allowed `(R_spring, k_s, m_spring)` combinations.

Typical consequence in this model:
- for eV-scale probe photons and electron-scale spring parameters, `\sigma_{\gamma DM}` is many orders below current experimental thresholds, matching "dark" behavior.

Status:
- explicit microscopic suppression law for `\sigma_{\gamma DM}` = **MODEL, PARTIAL-PROVEN**
- first-principles calibration of `(q_spring, m_spring, R_spring)` from coin material constants remains open.

---

## 11.4 Halo and rotation constraints

For circular velocity in galaxy:

\[
v_c^2(r)=\frac{G M(<r)}{r}
\]

If `M(<r)` includes dominant DM halo term, flat rotation curves can emerge.

Required consistency set:

1. galaxy rotation curves
2. weak/strong lensing mass maps
3. cluster collision offsets
4. CMB + large-scale structure growth

Status: **ANCHORED** observables; your ontology must fit all simultaneously.

---

## 11.5 Dark energy bridge

Cosmic acceleration equation (Section 10):

\[
\frac{\ddot a}{a}=-\frac{4\pi G}{3}\left(\rho+\frac{3p}{c^2}\right)
\]

Acceleration requires

\[
w=\frac{p}{\rho c^2}<-\frac{1}{3}
\]

If your freed-inner-photon sea-pressure mechanism is valid, it must produce an effective `w(z)` consistent with data (near `-1` today).

Section-10 closure (O10-3):

\[
P_{sea,DE}=\frac{1}{3}u_{\gamma,free},\quad
p_{DE}=-\rho_{DE}c^2+\Pi_s(z),\quad
w(z)=-1+\frac{\Pi_s(z)}{\rho_{DE}c^2}
\]

CPL fit interface:

\[
w(z)=w_0+w_a\frac{z}{1+z}
\]

Status:
- acceleration condition = **ANCHORED**
- specific spring/photon source interpretation = **MODEL**
- explicit `w(z)` bridge imported from Section 10.10.1 = **MODEL/PARTIAL**.

---

## 11.6 “Does not dilute” claim precision

Your section states dark energy does not dilute.

Formal continuity equation for component `i`:

\[
\dot{\rho_i}+3H\left(\rho_i+\frac{p_i}{c^2}\right)=0
\]

If `w_i=-1`, then `rho_i` constant with expansion.

So your claim is valid only if effective source behaves close to cosmological-constant equation of state.

Status: **ANCHORED** equation + **MODEL** requirement on mechanism.

---

## 11.7 Coupled-origin model (one event -> two sectors)

Represent conversion channel:

\[
\mathcal{N} \;\rightarrow\; \mathcal{D}_m + \mathcal{D}_e
\]

where
- `N` = normal unit (spring+inner photon),
- `D_m` = dark-matter spring remnant,
- `D_e` = dark-energy sector contribution.

Effective bookkeeping:

\[
\dot{\rho}_{NM} = -Q,\quad \dot{\rho}_{DM}=f_m Q,\quad \dot{\rho}_{DE}=f_e Q,\quad f_m+f_e=1
\]

with model-dependent transfer rate `Q`.

This formalizes your “one event, two results.”

Status: **MODEL** interacting-dark-sector style parameterization.

### 11.7.1 Microscopic transfer law `Q` (Tier-4 progress on O11-2)

Model one separation event on a normal unit as energy bookkeeping:

\[
\mathcal E_N = \mathcal E_{spring} + \mathcal E_{\gamma,in}
\]

If inner photon decouples at rate `\Gamma_{sep}` (per unit volume), define number density `n_{NM}` and mean event energy `\bar{\mathcal E}_\gamma`:

\[
Q = n_{NM}\,\Gamma_{sep}\,\bar{\mathcal E}_\gamma
\]

Using mass-energy density `\rho_{NM}=n_{NM} m_N c^2`:

\[
Q = \rho_{NM}\,\frac{\Gamma_{sep}}{m_N c^2}\,\bar{\mathcal E}_\gamma
\]

Partition fractions follow event energy split:

\[
f_e = \frac{\bar{\mathcal E}_\gamma}{\mathcal E_N},\qquad
f_m = \frac{\bar{\mathcal E}_{spring}}{\mathcal E_N},\qquad
f_m+f_e=1
\]

so

\[
\dot{\rho}_{DE}=f_e Q,\qquad
\dot{\rho}_{DM}=f_m Q,\qquad
\dot{\rho}_{NM}=-Q
\]

### 11.7.2 `Gamma_sep` from spring–photon microdynamics

Import Section-2/5 pin/unpin competition:

\[
\Gamma_{sep}=\Gamma_{unpin}-\Gamma_{pin}
\]

with microscopic closures

\[
\Gamma_{pin}\propto g_E^2 E_{obs}^2 S_{res,e}(\omega),
\qquad
\Gamma_{unpin}\propto \Gamma_{shake}(\kappa,T_{env})
\]

and Arrhenius-like barrier form for rare separation:

\[
\Gamma_{unpin}=\nu_0 \exp\!\left(-\frac{\Delta_{sep}}{k_B T_{env}}\right)
\]

where `\Delta_{sep}` is the spring–inner-photon binding scale (set by pump frequency `hbar omega_b`).

Low-temperature / low-shake limit:

\[
\Gamma_{sep}\approx 0 \Rightarrow Q\approx 0
\]

High-shake astrophysical environments can raise `Q` locally (candidate source for sector conversion bursts).

### 11.7.3 Cosmological interface

Insert into continuity equations (Section 10/11):

\[
\dot{\rho}_{NM}+3H\rho_{NM}=-Q,
\qquad
\dot{\rho}_{DM}+3H\rho_{DM}=f_m Q,
\qquad
\dot{\rho}_{DE}+3H(\rho_{DE}+p_{DE}/c^2)=f_e Q
\]

Late-time limit with `w_{DE}\approx -1` gives nearly constant `\rho_{DE}` only if `Q` is small today:

\[
Q(z=0)\ll 3H_0\rho_{DE}
\]

Status:
- explicit `Q` from `(rho_NM, Gamma_sep, bar E_gamma)` = **MODEL, PARTIAL-PROVEN**
- first-principles `\Delta_{sep}, nu_0, Gamma_shake` calibration from coin geometry still open.

---

## 11.8 Observation/detectability claim bounds

Your strong claim: direct detection cannot work if detector requires pinning inner photon.

Formalized conditional statement:

If all available detector channels require coupling to absent degree `X` (inner-photon mode), then

\[
\sigma_{det}(DM)\to 0 \quad \text{for that detector class}
\]

Important precision:
- this is a detector-class theorem, not yet universal theorem over all conceivable interactions.

Status: **MODEL**, logically valid under stated detector assumption.

### 11.8.1 No-detection theorem by channel class (Tier-4 progress on O11-3)

Partition detector couplings into channel classes relative to dark-matter Hilbert space
`H_DM = H_spring` (no inner-photon factor `H_gamma`).

**Class A — pin/readout channels (Sections 5–6).**

If interaction requires inner-photon operator `O_gamma` on `H_gamma`:

\[
H_{int}^{(A)} = g_A\,O_\gamma \otimes O_{env}
\]

and DM state satisfies `\rho_{DM}\in\mathcal S(H_{spring})`, then

\[
\mathrm{Tr}_{spring}\!\left[\rho_{DM}\,H_{int}^{(A)}\right]=0
\]

hence

\[
\Gamma_{det}^{(A)}(DM)=0,\qquad \sigma_{det}^{(A)}(DM)=0
\]

**Class B — resonant EM channels.**

Use Section-11.3.3:

\[
\sigma_{\gamma DM}(\omega)=\sigma_{geom}\mathcal K_{sup}(\omega),\quad
S_{res,DM}=0
\]

so all resonant absorption/emission amplitudes vanish; only heavily suppressed off-resonant elastic terms remain.

**Class C — nuclear-recoil / ionization channels requiring two-level coin readout.**

If recoil signal extraction requires `sigma_z` (or equivalent coin) matrix element between spring states without inner mode:

\[
\mathcal M_{recoil}\propto \langle s'|\sigma_z|s\rangle \times (\text{spring overlap})
\]

DM spring-only ground sector has no coin transition channel, giving

\[
\sigma_{SI}^{DM}\le \sigma_{geom}\mathcal K_{sup}(E_{recoil})
\]

**Class D — gravitational channels.**

Mass coupling remains:

\[
\nabla^2\Phi = 4\pi G(\rho_b+\rho_{DM})
\]

This is detection of gravity, not null; excluded from "direct particle detection" null-claim scope.

### 11.8.2 Scope statement (what is and is not proven)

Proven under model assumptions:
- **Class A** exact null coupling.
- **Class B/C** parametrically suppressed cross-sections with explicit upper bounds.

Not yet proven universally:
- detectors whose leading operator is purely spring displacement with large coherent amplification (e.g. engineered metamaterial resonators tuned to `E_spring`) could in principle produce nonzero response.

Therefore the correct theorem form is:

\[
\sigma_{det}^{(A)}=0,\qquad
\sigma_{det}^{(B,C)}\le \sigma_{geom}\mathcal K_{sup}
\]

not a claim of mathematical impossibility over all Hamiltonians.

Status: **MODEL, PARTIAL-PROVEN** (channel-class theorem complete; universality over all interaction bases remains open).

---

## 11.9 Entropy/time statements for dark matter

Your section says dark matter has no internal clock and limited relaxation channels.

Formal placeholder:

\[
f_{clock}^{DM}\approx 0\quad (\text{model postulate})
\]

and reduced non-gravitational thermalization channels:

\[
\Gamma_{therm}^{DM} \ll \Gamma_{therm}^{baryon}
\]

Need caution:
- cosmological entropy accounting must still obey global thermodynamic constraints.

Status: **MODEL/OPEN**.

Section-12 consistency constraint:
- keep `f_{clock}^{DM}\approx 0` as a dark-sector microdynamics postulate,
- do not identify it with SR photon null condition (`d\tau=0`) unless additional derivation is provided.

---

## 11.10 Equation set (Section 11 robust core)

1. `nabla^2 Phi = 4pi G (rho_b + rho_DM)`
2. `sigma_gammaDM = sigma_geom K_sup(omega)`, `K_sup=(hbar omega/E_spring)^4`, `S_res,DM=0`
3. `sigma_geom=pi R_spring^2`, `E_spring=hbar sqrt(k_s/m_spring)`
4. `U_total^DM ~ U_grav` (no bonding channel model)
5. `v_c^2 = G M(<r)/r`
6. `ddot a / a = -(4pi G/3)(rho + 3p/c^2)`
7. `w = p/(rho c^2) < -1/3` for acceleration
8. `dot rho + 3H(rho + p/c^2)=0`
9. `Q=rho_NM (Gamma_sep/(m_N c^2)) bar E_gamma`, `f_e=bar E_gamma/E_N`, `f_m=bar E_spring/E_N`
10. `Gamma_sep=Gamma_unpin-Gamma_pin`, `Gamma_unpin=nu_0 exp(-Delta_sep/(k_B T_env))`
11. coupled continuity: `dot rho_NM+3H rho_NM=-Q`, `dot rho_DM+3H rho_DM=f_m Q`, `dot rho_DE+...=f_e Q`
12. **P11-3:** `\phi_{AB}\in[0,1]`, `d\phi/dt=\Gamma_{fill}(1-\phi)-\Gamma_{snap}\phi`, `\Gamma_{form}\propto\phi_{AB}e^{-d/\ell_c}`

---

## 11.11 Tests and falsifiers

T11-1. **Halo+lensing joint fit**
- One parameter set must fit rotation + lensing + cluster offsets simultaneously.

T11-2. **Structure growth**
- Model must match CMB + LSS growth constraints.

T11-3. **Dark-energy equation-of-state fit**
- Derived effective `w(z)` must be observationally viable.

T11-4. **Direct-detection class test**
- Show detector families tied to inner-photon-pin mechanism yield null coupling as predicted.

T11-5. **Coupled-sector signatures**
- If `Q != 0`, look for redshift-dependent deviations from vanilla ΛCDM expansion/growth.

---

## 11.12 Cross-reference updates required

### Back-links
- Section 10 cosmology equations now explicit hard constraints on Section 11 interpretation.
- Section 2 decomposition is reused for spring/photon split ontology.

### Forward-links
- Section 12 now references Section 11 clockless-dark-sector claim as model-dependent (not SR null-particle identity).
- Section 6 `\Gamma_{form}`, `\ell_c` import `\phi_{AB}` fill (Sec 6.13).
- Section 7 soft tunneling imports DM transit channel (Sec 7.3.4).

---

## 11.13 Open items

| ID | Claim | Status |
|---|---|---|
| O11-1 | Derive quantitative `sigma_gammaDM` from explicit spring microdynamics | PARTIAL — **Step 11 gate:** `\sigma_{geom}\mathcal K_{sup}` from `k_s`, `L_0`; material calibration OPEN |
| O11-2 | Derive `Q` transfer law from spring-photon separation mechanism | PARTIAL — **Step 11 gate:** `Q`, `\Gamma_{sep}` structure; `\nu_0`, `\Delta_{sep}` calibration OPEN |
| O11-3 | Prove universal no-detection theorem across all interaction channels | PARTIAL — **Step 11 gate:** Classes A–D; universality OPEN |
| O11-4 | Obtain observationally consistent `w(z)` from freed-inner-photon pressure model | PARTIAL — **Step 11 gate:** `calibrate_pi0_for_w0`; joint dataset fit OPEN |
| O11-5 | Calibrate `\Gamma_{fill}`, `\Gamma_{snap}`, `\ell_c(\rho_{DM})` for P11-3 mesh | PARTIAL — **Step 11 gate:** `\ell_c(\rho_{DM})` proxy; fill rates OPEN |

---

## 11.14 Completion status

- Section 11 intuition mapped to gravity/cosmology constraint equations.
- Strong claims rephrased into testable conditional statements.
- Joint-fit requirements to established datasets explicitly imposed.
- Tier-4 microscopic `\sigma_{\gamma DM}` suppression law added for O11-1.
- Tier-4 transfer law `Q` and `\Gamma_{sep}` closure added for O11-2.
- Tier-4 channel-class no-detection theorem added for O11-3.

**Section 11 derivation pass: COMPLETE (v1).**
