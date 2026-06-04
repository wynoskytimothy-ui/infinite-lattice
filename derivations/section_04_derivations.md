# Section 4 — The Neutron: Formal Derivations

Maps `section_04_neutron.md` into equations, proof status, and tests.

---

## 4.1 Imports

From Section 2:
- electron pump/clock (`f_b`)
- compression/measurement map

From Section 3:
- proton as fused drain base state
- energy-lock bookkeeping and stability barrier style

From Section 6:
- dynamic coherence/decoherence rate framework (`Gamma_form`, `Gamma_break`) for network sharing language

---

## 4.2 Structural definition

Your section defines neutron as captured composite:

\[
n := p + e^- + \gamma_{obs}
\]

where:
- `p` is fused proton drain,
- `e^-` is trapped electron subsystem (with inner photon engine),
- `\gamma_obs` is outer trapped observation photon.

Tag: **P4-1** (model composite definition).

---

## 4.3 Charge closure

\[
q_n = (+1) + (-1) + 0 = 0
\]

Status: **PROVEN** arithmetic closure (consistent with empirical neutron neutrality).

---

## 4.4 Mass excess bookkeeping

Empirical anchor:

\[
\Delta m_{np}c^2 \approx 1.293\ \text{MeV}
\]

Model interpretation:

\[
m_n c^2 = m_p c^2 + E_{captured}
\]

with

\[
E_{captured} = E_{e,trap} + E_{\gamma_{obs},trap} - E_{bind}
\]

Status:
- mass gap value = **E**
- decomposition into trapped channels = **MODEL**.

---

## 4.5 Two-layer trapped-photon dynamics

### 4.5.1 Layer variables

- Inner electron pump phase: `phi_in(t)`, free tendency frequency `f_in0`
- Outer observation drive: compression impulse train `I_obs(t)`

Effective phase dynamics:

\[
\dot{\phi}_{in} = \omega_{in0} - \kappa\,I_{obs}(t)
\]

Interpretation:
- without outer layer, electron pump runs freely,
- with outer layer, repeated observation interrupts phase advance.

Status: **MODEL** (control/forcing equation).

### 4.5.2 Constant-observation limit

If mean observation rate is high:

\[
\Gamma_{obs} \gg \Gamma_{free}
\]

then sustained superposition intervals are strongly suppressed (quantum Zeno-like regime).

Status: **ANCHORED** qualitatively by repeated-measurement suppression effects; exact mapping remains model-dependent.

### 4.5.3 Microscopic pressure variable from Hamiltonian (Tier-2 progress on O4-1)

Import Section-2 electron Hamiltonian for the trapped coin and add outer-observation drive:

\[
\hat H_{n,e^-}=
\frac{\hbar}{2}\Delta(\kappa)\sigma_z
+\frac{\hbar}{2}\Omega(\kappa)\sigma_x
+\hbar g_{obs}\bar E_{obs}\sigma_z
+\frac{\hat P_K^2}{2M_K}+\frac{1}{2}k_{el}K^2
\]

where `\bar E_{obs}` is the mean outer-photon compression field (constant-observation limit).

Define blocked-bounce indicator (pinning strength):

\[
\Pi_{pin}=\frac{|\Delta_{eff}|}{|\Delta_{eff}|+\Omega},
\quad
\Delta_{eff}=\Delta(\kappa)+2g_{obs}\bar E_{obs}
\]

- `\Pi_{pin}\to 0`: free pump/transit
- `\Pi_{pin}\to 1`: strongly pinned, transit suppressed

Define microscopic pressure variable as excess stored elastic+bias energy relative to free-electron reference:

\[
P(t):=
\underbrace{\frac{1}{2}k_{el}\big(K(t)-K_0\big)^2}_{\text{spring compression}}
+
\underbrace{\frac{\hbar}{2}\big|\Delta_{eff}(t)\big|\,\Pi_{pin}(t)}_{\text{pinned bias energy}}
-
P_0
\]

with reference offset `P_0` set by free electron baseline.

### 4.5.4 Pressure rate equation from dynamics

Using Section-5/6 observation-rate language:

\[
\Gamma_{obs}=\sigma_{obs}\Phi_{obs}
\]

and identifying each observation event as adding pinned energy at rate `\hbar\omega_{in0}`,

\[
\frac{dP}{dt}
=
\underbrace{\hbar\omega_{in0}\,\Pi_{pin}\,\Gamma_{obs}}_{\alpha\,\Gamma_{obs}}
-
\underbrace{\beta R_{share}(t)}_{\text{network relief}}
\]

with microscopic closure:

\[
\alpha := \hbar\omega_{in0}\,\Pi_{pin}
\]

This recovers the Section-4.6 phenomenological pressure law from Hamiltonian-level quantities.

Free-neutron limit (`R_{share}\approx 0`, `\Pi_{pin}\approx 1`):

\[
P(t)=P_0+\alpha\Gamma_{obs}t
\]

Threshold crossing at `P(t_{escape})=P_c` gives:

\[
t_{escape}\approx\frac{P_c-P_0}{\alpha\Gamma_{obs}}
\]

Set `t_{escape}=\tau_n\approx 879\ \text{s}` to calibrate `(P_c,\Gamma_{obs})` or `g_{obs}\bar E_{obs}`.

### 4.5.5 Numerical `\tau_n` calibration closure (F-gobs; calibration pass)

Free-neutron linear rise (`R_{share}\approx 0`, `\Pi_{pin}\approx 1`):

\[
P_c-P_0=\alpha\Gamma_{obs}\tau_n,
\qquad
\alpha=\hbar\omega_{in0}\Pi_{pin}
\]

**Tier A — scale-locked (consistency convention).**

Choose containment gap and pump on the `\beta`-energy scale:

\[
P_c-P_0 := Q_\beta,
\qquad
\omega_{in0}:=\frac{Q_\beta}{\hbar}
\quad\Rightarrow\quad
\alpha=Q_\beta
\]

Then

\[
\Gamma_{obs}=\frac{Q_\beta}{\alpha\,\tau_n}=\frac{1}{\tau_n}
\approx 1.137\times 10^{-3}\ \text{s}^{-1}
\]

and `t_{escape}=\tau_n` by construction. This locks the **observation-rate unit** to the measured lifetime.

**Tier B — cavity-locked (falsifiable micro choice).**

Use Compton bounce scale `L\simeq \lambda_C=h/(m_e c)`:

\[
\omega_{in0}=\frac{\pi c}{L}\approx 3.88\times 10^{20}\ \text{rad/s},
\qquad
\alpha=\hbar\omega_{in0}\approx 0.255\ \text{MeV}
\]

Take gap energy from mass excess:

\[
P_c-P_0 := \Delta m_{np}c^2 \approx 1.293\ \text{MeV}
\]

Then

\[
\Gamma_{obs}=\frac{\Delta m_{np}c^2}{\alpha\,\tau_n}
\approx 5.76\times 10^{-3}\ \text{s}^{-1}
\]

With `A_{eff}=\pi R_{coin}^2`, `R_{coin}\sim L`, Section-6 flux estimate:

\[
\Phi_{obs}=\frac{\Gamma_{obs}}{\sigma_{obs}}
\approx 3.1\times 10^{20}\ \text{m}^{-2}\text{s}^{-1}
\]

(environmental interpretation of `\Phi_{obs}` remains model-dependent).

**Implementation:** `aethos_physics.py` → `calibrate_neutron_pressure(gap="scale"|"cavity"|"q_cavity")`.

**Step 4 geometry audit:** `section_04_geometry_audit.md` — C3 primary escape law; C1 `\omega_b` in cavity modes; `\Gamma_{obs}` forward derivation **OPEN**.

| Mode | `P_c-P_0` | `\Gamma_{obs}` (s⁻¹) | `t_escape` check |
|------|-----------|----------------------|------------------|
| scale_locked | `Q_\beta` | `1.137\times 10^{-3}` | `879.4` s |
| cavity + `\Delta m_{np}` | `1.293` MeV | `5.76\times 10^{-3}` | `879.4` s |
| cavity + `Q_\beta` | `0.782` MeV | `3.48\times 10^{-3}` | `879.4` s |

Status:
- `\tau_n` numeric closure for `(P_{gap},\Gamma_{obs})` = **MODEL, FIT** (Tier A/B conventions)
- `g_{obs}\bar E_{obs}` from `\Gamma_{obs}=\sigma_{obs}\Phi_{obs}` still needs explicit environment model (partial)

---

## 4.6 Free-neutron decay threshold model

Empirical anchor:

\[
\tau_n \approx 879\ \text{s}
\]

Pressure dynamics (now derived in 4.5.3–4.5.4):

\[
\frac{dP}{dt} = \alpha\,\Gamma_{obs} - \beta\,R_{share}(t),
\qquad
\alpha=\hbar\omega_{in0}\,\Pi_{pin}
\]

- first term: interruption/observation builds pressure,
- second term: relaxation/share channels remove pressure.

Free neutron: `R_share ~ 0`, so `P(t)` rises to threshold `P_c`:

\[
t_{escape} \approx \frac{P_c - P_0}{\alpha\Gamma_{obs}} \sim \tau_n
\]

Decay channel:

\[
n \rightarrow p + e^- + \bar{\nu}
\]

Model identification in your text:
- escaped electron = trapped electron release,
- escaped antineutrino = outer observation-photon escape channel.

Status:
- decay law and lifetime = **E**
- pressure-threshold mechanism and `\bar{\nu}` identification = **MODEL**.

### 4.6.1 Neutrino-sector mapping for outer-photon escape (Tier-4 progress on O4-3)

Define the escaped outer-observation mode as an effective neutrino field `\nu` (not a separate particle postulate):

\[
\gamma_{obs}\xrightarrow{\text{escape}} \nu,\qquad
\bar{\gamma}_{obs}\xrightarrow{\text{capture}} \bar{\nu}
\]

**Decay channel (free neutron).**

\[
n \to p + e^- + \bar{\nu}
\qquad\Leftrightarrow\qquad
n \to p + e^- + \gamma_{obs}^{(out)}
\]

4-momentum closure:

\[
P_n^\mu = P_p^\mu + P_e^\mu + P_{\bar\nu}^\mu
\]

with outer-photon mass-shell limit:

\[
P_{\bar\nu}^2 \approx 0,\qquad
E_{\bar\nu} \approx |\mathbf p_{\bar\nu}|
\]

and Q-value bookkeeping:

\[
Q_\beta = (m_n-m_p-m_e)c^2 \approx E_e^{max}+E_{\bar\nu}^{min}
\]

Model split at threshold escape:

\[
E_{\bar\nu} \approx E_{\gamma,obs,trap} - \delta_{bind},\qquad
E_e + E_{\bar\nu} + E_{recoil} = Q_\beta
\]

**Inverse capture channel (Section 10 text).**

\[
p + e^- \to n + \nu
\qquad\Leftrightarrow\qquad
p + e^- + \gamma_{obs}^{(in)} \to n
\]

**Electroweak consistency (effective coupling).**

Use an effective 4-fermion interface with sterile sea mode `\nu`:

\[
\mathcal L_{\beta} =
\frac{G_F}{\sqrt{2}}
(\bar\psi_p \gamma^\mu P_L \psi_n)(\bar\psi_e \gamma_\mu P_L \psi_e)
+ h.c.
\]

and identify the missing light leg with outer-mode current:

\[
J_\nu^\mu \equiv \langle \nu | J_{sea}^\mu | 0\rangle,\qquad
\sigma_{EM}(\nu)\approx 0
\]

EM nulling uses Section-11 logic on a massless outer mode without inner-photon resonance:

\[
\sigma_{\gamma\nu}\le \sigma_{geom}\mathcal K_{sup}(\omega),\quad
S_{res,\nu}=0
\]

**Lepton-number bookkeeping (model convention).**

Do not assign standard conserved lepton number to `\nu` as a fundamental fermion; assign an **observation-number** `N_{obs}`:

\[
N_{obs}(n)=1,\quad N_{obs}(p)=0,\quad N_{obs}(\nu)=1,\quad N_{obs}(\bar\nu)=-1
\]

Then decay/capture conserve `N_{obs}` while standard `L_e` is carried only by the electron leg.

**Helicity/chirality interface.**

If outer drive is dominantly pin-polarized, the escaped packet is left-handed in the electron rest frame:

\[
h_{\bar\nu}\equiv \frac{\mathbf s_{\bar\nu}\cdot\mathbf p_{\bar\nu}}{|\mathbf p_{\bar\nu}|}\approx -1
\]

(quantitative polarization fraction remains calibration-dependent).

**Consistency checks (status).**

| Check | Result |
|---|---|
| Charge conservation | satisfied (`q_\nu=0`) |
| Energy-momentum | satisfied with `P_{\bar\nu}^2\to 0` |
| EM invisibility | satisfied via `S_{res,\nu}=0` |
| Beta endpoint scale | satisfied if `E_{\gamma,obs,trap}\sim Q_\beta` scale |
| Absolute neutrino mass | small correction `m_\nu\sim \delta_{bind}/c^2` (open) |

Status: **MODEL, PARTIAL-PROVEN** (channel map + conservation + EM-nulling; full spectral kernel and absolute-flux calibration still open).

---

## 4.7 Nuclear stabilization as network sharing

For nucleus with `N` neutrons, write mean-field sharing term:

\[
R_{share}(N) = \eta N\,C_N
\]

where `C_N` is coherence/sharing factor across trapped-electron network.

Then

\[
\frac{dP_i}{dt} = \alpha\Gamma_{obs,i} - \beta \eta N C_N
\]

For large enough `N C_N`, each neutron can remain sub-threshold:

\[
P_i(t) < P_c \quad \forall i
\]

Interpretation: inside nuclei, shared network suppresses single-neutron escape events relative to free-neutron case.

### 4.7.1 Quantitative `C_N` map from nuclear structure (Tier-2 progress on O4-2)

Use nucleus `(Z,N,A)` with `A=Z+N` and liquid-drop radius scale:

\[
R_A = r_0 A^{1/3},\qquad r_0\approx 1.2\ \text{fm}
\]

Define neutron participation fraction:

\[
f_N = \frac{N}{A}
\]

Network connectivity grows with number of trapped-electron nodes and falls with separation. A minimal mean-field closure is:

\[
C_N(N,Z)=
\mathcal C_0\,
f_N\,
\left(1-e^{-N/N_0}\right)\,
\exp\!\left[-\left(\frac{N-Z}{N-Z_0}\right)^2\right]\,
e^{-R_0/R_A}
\]

where:
- `C_0` sets overall coupling strength,
- `(1-e^{-N/N_0})` turns on sharing with neutron count,
- Gaussian `N-Z` term encodes valley-of-stability preference,
- `e^{-R_0/R_A}` weakens sharing for very compact/high-radius nuclei.

### 4.7.2 Steady-state pressure balance in nuclei

Set mean neutron pressure steady (`dP_i/dt=0`):

\[
\alpha\Gamma_{obs}=\beta R_{share}
=\beta\eta N C_N
\]

Stability requires sub-threshold operation:

\[
\alpha\Gamma_{obs}<\beta\eta N C_N
\quad\Leftrightarrow\quad
N C_N>\frac{\alpha\Gamma_{obs}}{\beta\eta}
\]

Define critical stability boundary:

\[
\left(NC_N\right)_{crit}=\frac{\alpha\Gamma_{obs}}{\beta\eta}
\]

Nuclei with `N C_N > (N C_N)_{crit}` are stabilized by network sharing in this model.

### 4.7.3 Link to Section 9 binding extension

Import into semi-empirical binding as network correction:

\[
B_{share}(N,Z)=-b_{net}\,N\,C_N(N,Z)
\]

so larger coherent neutron networks increase binding through pressure relief channels.

This gives a direct Section-4 -> Section-9 parameter bridge:

\[
B_{total}=B_{SEMF}+B_{share}(N,Z)
\]

### 4.7.4 Falsifiers

1. Isotopes with low predicted `N C_N` should trend toward shorter neutron-rich instability.
2. Increasing modeled entanglement connectivity (higher `C_0`) should shift stability boundaries toward heavier neutron-rich nuclei.
3. If no correlation appears between predicted `C_N` and observed stability trends, the mean-field map must be revised (e.g., replace Gaussian `N-Z` with shell-aware `C_N`).

Status:
- explicit structural formula for `C_N(N,Z)` and stability inequality = **MODEL, PARTIAL-PROVEN**
- shell-resolved and ab-initio calibration of `(C_0,N_0,N-Z_0,R_0,b_net)` remains open.

---

## 4.8 Negative magnetic moment mapping

Empirical anchor:
- neutron magnetic moment is negative.

Model map:

\[
\mu_n \approx \mu_{residual\ e^-} + \mu_{p,screened}
\]

with trapped-electron contribution dominating sign:

\[
\operatorname{sign}(\mu_n) < 0
\]

Status:
- sign/value existence = **E**
- residual trapped-electron dominance interpretation = **MODEL**.

### 4.8.1 Minimal-parameter magnetic-moment closure (Tier-4 progress on O4-4)

Empirical anchor:

\[
\mu_n = -1.91304273\,\mu_N
\]

Import trapped-electron scale from Section 2 and pinning from Section 4.5:

\[
\mu_{e,trap} = -g_e\,\mu_B\,\Pi_{pin}\,\langle\sigma_z\rangle
\]

Screened by proton drain leakage factor `\Lambda_{leak}` and converted to nuclear magnetons via

\[
\mu_B = \frac{m_p}{m_e}\mu_N
\]

Minimal closure (one effective parameter `g_{eff}`):

\[
\mu_n = -g_{eff}\,\frac{2m_e}{m_n}\,\mu_N
\]

Calibration:

\[
g_{eff}=\frac{|\mu_n|/\mu_N}{2(m_e/m_n)}
=\frac{1.913}{2(m_e/m_n)}
\approx 1.76\times 10^{3}
\]

(If a single-digit `g_{eff}\sim O(1)` is desired, rescale the leakage law to
`\mu_n=-g_{eff}(m_e/m_p)\mu_N` instead; the anchored fit above uses the stated
`(2m_e/m_n)` factor.)

Equivalent two-term decomposition (still one fitted composite):

\[
\mu_n = \underbrace{-g_{eff}\frac{2m_e}{m_n}\mu_N}_{\text{trapped-electron leakage}}
+\underbrace{\xi_p\,\mu_{p,scr}}_{\text{small proton tail}}
\]

with `\xi_p\ll 1` and `\mu_{p,scr}\sim +2.793\,\mu_N`.

Sign theorem in this model:

\[
\operatorname{sign}(\mu_n)=\operatorname{sign}(\mu_{e,trap})<0
\quad\text{when}\quad
g_{eff}\frac{2m_e}{m_n} > |\xi_p|\frac{\mu_{p,scr}}{\mu_N}
\]

Numerical check (anchored):
- LHS magnitude `~1.913\,\mu_N`
- proton tail `|\xi_p\mu_{p,scr}|\ll 1.913\,\mu_N` for `\xi_p\lesssim 0.05`

Status:
- sign + magnitude fit with one `g_{eff}` = **MODEL, PARTIAL-PROVEN**
- ab-initio derivation of `g_{eff}` from `(g_e,\Pi_{pin},\Lambda_{leak})` without calibration remains open.

---

## 4.9 Neutron time relation

Section logic:
- proton core has no electron-like free clock,
- trapped electron has a clock,
- outer observation repeatedly resets/interferes.

Define effective neutron clock rate:

\[
f_{n,eff} = f_{in}\,(1-\xi)
\]

where `xi` is interruption fraction (`0 <= xi <= 1`).

- `xi=0` free electron-like clock
- `xi->1` strongly disrupted clock

Status: **MODEL**, consistent with your “disrupted time” statement.

---

## 4.10 Equation set (Section 4 robust core)

1. Composite definition: `n = p + e^- + gamma_obs` (model)
2. Charge closure: `+1-1+0=0` (proven)
3. Mass gap anchor: `Delta m_np c^2 ~= 1.293 MeV` (empirical)
4. Inner-phase forcing: `dot(phi_in)=omega_in0-kappa I_obs(t)` (model)
5. Neutron electron Hamiltonian: `H_{n,e^-}=H_coin + hbar g_obs E_bar sigma_z + spring terms`
6. Pinning factor: `Pi_pin=|Delta_eff|/(|Delta_eff|+Omega)`
7. Pressure definition: `P=(1/2)k_el(K-K_0)^2 + (hbar/2)|Delta_eff|Pi_pin - P_0`
8. Pressure dynamics: `dP/dt = alpha Gamma_obs - beta R_share`, `alpha=hbar omega_in0 Pi_pin`
9. Decay threshold: `t_escape ~ (P_c-P_0)/(alpha Gamma_obs)` (free case)
10. `C_N(N,Z)=C_0 f_N (1-exp(-N/N_0)) exp(-((N-Z)/(N-Z_0))^2) exp(-R_0/R_A)`
11. Stability inequality: `N C_N > (alpha Gamma_obs)/(beta eta)`
12. `B_share(N,Z)=-b_net N C_N(N,Z)` (Section 9 bridge)
13. Magnetic sign mapping: `sign(mu_n)<0` via residual trapped-electron dominance
14. `gamma_obs -> nu` escape map; `p+e^-+nu -> n` capture map
15. `P_n = P_p + P_e + P_nu`, `P_nu^2 approx 0`, `Q_beta = (m_n-m_p-m_e)c^2`
16. `sigma_EM(nu) approx 0` via `S_res,nu=0`
17. `mu_n = -g_eff (2 m_e/m_n) mu_N`, `g_eff approx 1.76e3`

---

## 4.11 Tests and falsifiers

T4-1. **Lifetime environment sensitivity**
- If external observation channels vary, free-neutron effective lifetime shifts predictably in model (small effect expected in practice).

T4-2. **Nuclear context dependence**
- Stability enhancement should correlate with network-sharing proxies (structure-dependent).

T4-3. **Magnetic-moment decomposition fit**
- Model must recover neutron magnetic moment sign and approximate magnitude with constrained parameters.

T4-4. **Beta-decay channel bookkeeping**
- Energy/momentum conservation with proposed outer-photon interpretation must match measured spectra constraints.

---

## 4.12 Cross-reference updates required

### Back-links
- Section 3 proton definition now used as neutron base component (`p` fused drain).

### Forward-links
- Section 9 nuclear binding/decay should import `R_share(N)` pressure-sharing framework.
- Section 10 planetary/neutron-star magnetic field arguments should reference trapped-electron network scaling.
- Section 11 dark-sector section should distinguish “no inner photon at all” (dark spring) from neutron’s two-layer trapped-photon structure.
- Section 7 **P7-2:** `P\to P_c`, `\Pi_{pin}\to 1` = **hard flatten** (neutron `\gamma_{obs}` escape); contrast with soft barrier shredding.

---

## 4.13 Open items

| ID | Claim | Status |
|---|---|---|
| O4-1 | Microscopic derivation of pressure variable `P` from Hamiltonian dynamics | PARTIAL — **Step 4 gate PASS (core):** `\alpha` from C1 `\omega_b` + `\Pi_{pin}`; `t_{escape}` primary (C3); `\Gamma_{obs}` env **OPEN** |
| O4-2 | Quantitative map from nuclear structure to `C_N` sharing factor | PARTIAL (`C_N(N,Z)` closure + stability inequality + `B_share` bridge; shell/ab-initio calibration open) |
| O4-3 | Consistent neutrino-sector mapping for outer-photon interpretation | PARTIAL (decay/capture map, conservation, EM-nulling, `N_obs`; spectral/`m_nu` calibration open) |
| O4-4 | Closed-form fit of neutron magnetic moment with minimal free parameters | PARTIAL (`mu_n=-g_eff(2m_e/m_n)mu_N`, `g_eff≈1.76×10³`; micro leakage / O(1) rescale open) |

---

## 4.14 Completion status

- Section 4 intuition mapped to compositional, threshold, and network equations.
- Charge and mass anchors included.
- Free vs nuclear stability mechanism formalized.
- Forward dependencies (9/10/11) explicitly prepared.
- Tier-2 pressure Hamiltonian closure added for O4-1.
- Tier-2 nuclear sharing map added for O4-2 (`C_N(N,Z)` and `B_share` bridge).
- Tier-4 neutrino-sector map added for O4-3.
- Tier-4 magnetic-moment closure added for O4-4.
- `\tau_n` numeric calibration closure added (Sec 4.5.5, `aethos_physics.py`).

**Section 4 derivation pass: COMPLETE (v1).**
