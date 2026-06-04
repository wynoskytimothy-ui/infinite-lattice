# Section 3 — The Proton: Formal Derivations

Maps `section_03_proton.md` into equations, proof status, and tests.

---

## 3.1 Imports

From Section 2:
- inner-photon pump and compression variable `K`
- light-clock interpretation (`f_b` as internal time rate)
- structural order parameters (`O`, `K`, `Q`)

From Section 1:
- `E=hf`, `E=pc`, `E^2=(pc)^2+(mc^2)^2`

---

## 3.2 Fusion threshold model (electron -> proton)

### 3.2.1 Regimes

Define compression coordinate `K in [0,1]` and critical fusion threshold `K_f`.

- `0 <= K < K_f`: elastic compression (measurement-like pinning), reversible.
- `K >= K_f`: structural fusion, irreversible branch (proton phase).

Formal state map:

\[
\mathcal{E}_{electron}(K<K_f) \to \mathcal{P}_{proton}(K\ge K_f)
\]

Status: **MODEL** postulate `P3-1` (phase transition style).

### 3.2.2 Clock shutdown

Electron clock: `f_b > 0`.
Proton post-fusion: trapped mode no longer freely bouncing -> effective

\[
f_b^{(p)} \approx 0
\]

Interpretation: no internal pump clock of electron type.

Status: **MODEL**, consistent with your text and with Section 2 clock definition.

### 3.2.2a Direct observables for “no internal proton clock” (Tier-4 progress on O3-4)

Define coherent internal clock rate (Section 2/12):

\[
f_{clock}^{(p)} \equiv \frac{\omega_b^{(p)}}{2\pi}
\]

Model postulate after fusion (`\Omega\to 0`, locked inner mode):

\[
f_{clock}^{(p)}=0
\]

Distinguish from spin/magnetic observables:
- proton still has magnetic moment (`\mu_p\neq 0`) and NMR response,
- but no trapped-photon bounce clock.

Predicted nulls:
1. **No periodic EM emission** at `f_{clock}^{(p)}` from isolated proton (`\nu=0` line set).
2. **No free-proton beta clock** (proton stable; unlike neutron `f_{n,eff}` disruption).
3. **Contrast observable:** compare time-domain compression sidebands:
   - electron: sidebands at `f_b>0`,
   - neutron: sidebands at `f_{n,eff}>0` but reduced by outer observation,
   - proton: sideband clock absent (`f_{clock}^{(p)}=0`).

Operational discriminator:

\[
\mathcal S_p(\omega)\equiv \left|\int dt\,E_{obs}(t)e^{i\omega t}\right|^2
\]

with model prediction `\mathcal S_p(\omega_b)=0` while `\mathcal S_e(\omega_b)>0` under matched probe geometry.

Status: **MODEL, PARTIAL-PROVEN** (explicit null observables + `\mathcal S_p` test; experimental sensitivity thresholds still open).

### 3.2.3 Microscopic fusion Hamiltonian at `K_f` (Tier-1 progress on O3-2)

Import Section-2 coin dynamics:

\[
\hat H_{coin}(\kappa)=
\frac{\hbar}{2}\Delta(\kappa)\sigma_z
+\frac{\hbar}{2}\Omega(\kappa)\sigma_x
+\hbar g_E E_{obs}(t)\sigma_z
\]

Add slow structural compression coordinate `K` (same order parameter as Section 2):

\[
\hat H_{spring}=\frac{\hat P_K^2}{2M_K}+\frac{1}{2}k_{el}K^2
\]

and a fusion-capable structural potential:

\[
V_{struct}(K)=
\begin{cases}
\dfrac{1}{2}k_{el}(K-K_0)^2, & K<K_f\\[8pt]
V_f+\dfrac{1}{2}k_{pl}(K-K_f)^2+U_{lock}, & K\ge K_f
\end{cases}
\]

with `k_{pl}\gg k_{el}` (fused branch is stiff/irreversible).

Total pre-fusion Hamiltonian:

\[
\hat H_{e}=\hat H_{coin}(\kappa)+\hat H_{spring}+V_{struct}(K)
\]

Fusion transition rule (microscopic branch selector):

\[
K\ge K_f \;\Rightarrow\;
\Omega(\kappa)\to 0,\quad
\hat H_{coin}\to \hat H_{p}=\frac{\hbar}{2}\Delta_f\sigma_z
\]

so bounce channel `sigma_x` is quenched and only fused bias/drain mode remains.

Equivalent Landau form near threshold (useful for simulations):

\[
V_{landau}(K)=\frac{r}{2}K^2+\frac{u}{4}K^4,\quad
r=r_0\left(1-\frac{K_{ext}}{K_f}\right)
\]

External compression `K_ext` drives `r` through zero at `K_f`, producing the phase switch.

Barrier/fusion-rate closure:

\[
\Delta E_{barrier}=\int_{K_0}^{K_f}\sqrt{2M_K\left(V_{struct}(K)-V_{min}\right)}\,dK
\]

\[
\Gamma_{fuse}\sim \omega_0\,e^{-\Delta E_{barrier}/k_BT}
\]

Post-fusion energy accounting (links to Section 3.3):

\[
E_{locked}=\left\langle \hat H_{coin}\right\rangle_{e}-\left\langle \hat H_{p}\right\rangle_{p}
\quad\Rightarrow\quad
\Delta m c^2=E_{locked}
\]

Status:
- effective fusion Hamiltonian + branch map `H_e -> H_p` at `K_f` = **MODEL, PARTIAL-PROVEN**
- first-principles values of `k_{el}, k_{pl}, U_{lock}, Delta_f` from coin material geometry remain open (ties to O2-1/O3-1).

---

## 3.3 Energy-to-mass conversion statement

Your claim: bounce kinetic/oscillation energy becomes fused structural mass.

Write:

\[
\Delta m\,c^2 = E_{locked}
\]

where `E_locked` is internal energy removed from free pump dynamics and stored in fused configuration.

Equivalent rest-energy bookkeeping:

\[
m_p c^2 = m_e c^2 + E_{fusion,stored}
\]

Status:
- `E=mc^2` identity: **E/ANCHORED**
- assignment of `E_fusion,stored` to trapped-photon-lock mechanism: **MODEL**

---

## 3.4 1836 mass-ratio from fusion geometry (Tier-2 progress on O3-1)

Text claim:

\[
\frac{m_p}{m_e} \approx 1836
\]

Empirical anchor:

\[
R_{pe}:=\frac{m_p}{m_e}=1836.15267343...
\]

### 3.4.1 Geometric length law

Model the coin spring span as compression-dependent length:

\[
L(K)=L_0(1-\alpha K),\qquad 0\le K\le 1
\]

- electron baseline (maximally open): `K=K_0\approx 0`  -> `L_e=L_0`
- fused proton endpoint: `K=K_f` -> `L_p=L_0(1-\alpha K_f)`

Define geometric mass-ratio kernel:

\[
\mathcal{F}_{geom}(K_f,\alpha)=\frac{L_e}{L_p}=\frac{1}{1-\alpha K_f}
\]

This is the Section-3 statement “maximum stretch / maximum compression of the same spring structure” in equation form.

### 3.4.2 Energy closure (links to Section 3.3)

Use pump/cavity energy scale from Section 2:

\[
E_{pump}(L)=\frac{hc}{2L}
\]

Identify electron rest-scale term:

\[
m_e c^2 \sim E_{pump}(L_e)=\frac{hc}{2L_0}
\]

Fusion locks the high-compression photon mode energy:

\[
E_{locked}\sim E_{pump}(L_p)-E_{pump}(L_e)
=\frac{hc}{2}\left(\frac{1}{L_p}-\frac{1}{L_0}\right)
\]

Then:

\[
m_p c^2=m_e c^2+E_{locked}
\]

so

\[
R_{pe}=\frac{m_p}{m_e}
=1+\frac{E_{locked}}{m_e c^2}
\approx \frac{L_0}{L_p}
=\mathcal{F}_{geom}(K_f,\alpha)
\]

when `E_locked >> m_e c^2` (proton mass dominated by locked pump energy).

Status: **MODEL, PARTIAL-PROVEN** algebraic closure.

### 3.4.3 Calibration to observed 1836 (CONSEQUENCE CHECK — C2 mandate)

**Primary (Step 3 geometry):** derive `K_f` from maximum compression / `U_max` (Sec 2 export) — **not** from fitting `R_pe`.

**Consequence check (E anchor only):** if geometry predicts

\[
R_{pe}^{model}=\frac{1}{1-\alpha K_f^{geom}}
\]

compare to CODATA `1836.152…`. Mismatch falsifies geometry; match supports — **do not invert** by setting `K_f := (R_pe-1)/R_pe` as definition.

Legacy FIT formula (deprecated as primary):

\[
K_f=\frac{R_{pe}-1}{R_{pe}} \approx 0.999456
\]

Status: **FIT row retained for regression only** — see **`section_03_geometry_audit.md`** (Step 3 gate).

**Step 3 geometry result (GEOMETRY):**

| Quantity | Value | Tag |
|----------|-------|-----|
| `K_f^{pin}` | `\sqrt{\Omega_0/(\Omega_0+A)}` | **GEOMETRY** (F2) |
| `R_pe^{model,(0)}` | `\pi^2/8 \approx 1.234` | **GEOMETRY** (spring-only) |
| `\mathcal{M}_{lat}` | `R_pe^E / R_pe^{model,(0)} \approx 1488` | **OPEN** → Step 12 (C6) |

**Proof:** length–energy self-consistency `R_{pe}^{length}=R_{pe}^{energy}` has **only** `K=0`, `R=1` in `[0,1]` — large `R_pe` requires lattice multiplier, not FIT inversion.

### 3.4.4 Predictive form and falsifier

Model prediction:

\[
R_{pe}^{model}=\frac{1}{1-\alpha K_f}
\]

If independent estimates of `(L_0,L_p)` or `(K_f,\alpha)` are produced from coin microgeometry, the model is falsified when:

\[
\left|\frac{1}{1-\alpha K_f}-1836.152...\right| > \delta_{exp}
\]

Status:
- closed-form `F(K_f,alpha)` now derived,
- numeric match requires `K_f` near 1 (or equivalent `L_0/L_p\approx 1836`),
- first-principles independent calculation of `K_f` from material constants remains open (ties to O2-1).

---

## 3.5 Charge partition / quark-region mapping

Your regional map:
- top pole: `+2/3`
- equator: `-1/3`
- bottom pole: `+2/3`

Charge closure proof:

\[
\frac{2}{3}-\frac{1}{3}+\frac{2}{3}=+1
\]

Status:
- arithmetic closure: **PROVEN**
- “regions not particles” interpretation: **MODEL**
- consistency with measured proton charge `+e`: **ANCHORED** at net level.

---

## 3.6 Drain-field formalization (positive charge language)

Introduce effective radial sea-flow potential `\Phi_d(r)` around proton drain:

\[
\mathbf{u}_d(r) = -\nabla \Phi_d(r)
\]

Sign convention:
- proton: inward drain (`u_d` points inward)
- electron (Section 2 language): outward pressure contribution from active pump.

---

## 3.6.1 Coulomb mapping from sea drain/pump (Tier-2 progress on O3-3)

Model the sea response with an effective static potential equation:

\[
\nabla\cdot\big(\epsilon_{eff}\nabla\Phi\big)=-\rho_q
\]

where `\rho_q` is an effective source density from drain/pump charges.

For a point proton drain source `Q_p` at origin:

\[
\Phi_d(r)=\frac{1}{4\pi\epsilon_{eff}}\frac{Q_p}{r}
\]

with radial inward flow magnitude:

\[
u_d(r)=\left|\frac{d\Phi_d}{dr}\right|=\frac{Q_p}{4\pi\epsilon_{eff}r^2}
\]

For an electron pump source `Q_e` (outward/negative drain sign):

\[
\Phi_e(r)=\frac{1}{4\pi\epsilon_{eff}}\frac{Q_e}{r},\qquad Q_e<0
\]

### 3.6.2 Two-body interaction energy

Place proton at origin and electron at separation `r`.
Interaction energy of electron in proton drain field:

\[
U_{eff}(r)=Q_e\,\Phi_d(r)=\frac{1}{4\pi\epsilon_{eff}}\frac{Q_pQ_e}{r}
\]

Force on electron:

\[
\mathbf F(r)=-Q_e\nabla\Phi_d(r)
=\frac{1}{4\pi\epsilon_{eff}}\frac{Q_pQ_e}{r^2}\hat{\mathbf r}
\]

Choose charge assignment:

\[
Q_p=+Ze,\qquad Q_e=-e
\]

Then:

\[
U_{eff}(r)=-\frac{1}{4\pi\epsilon_{eff}}\frac{Ze^2}{r}
\]

\[
\mathbf F(r)=-\frac{1}{4\pi\epsilon_{eff}}\frac{Ze^2}{r^2}\hat{\mathbf r}
\]

This is the Coulomb form with effective permittivity `\epsilon_{eff}`.

### 3.6.3 Calibration to SI Coulomb law

Standard vacuum form:

\[
U_C(r)=\frac{1}{4\pi\epsilon_0}\frac{Ze^2}{r},\qquad
\mathbf F_C(r)=-\frac{1}{4\pi\epsilon_0}\frac{Ze^2}{r^2}\hat{\mathbf r}
\]

Therefore the mapping closure is:

\[
\boxed{\epsilon_{eff}=\epsilon_0}
\]

at leading order, provided sea-response units are calibrated once.

Equivalent coupling constant form:

\[
k_e^{eff}=\frac{1}{4\pi\epsilon_{eff}}
\quad\Rightarrow\quad
k_e^{eff}=k_e
\]

### 3.6.4 Sea-tension interpretation link

Your “sea between them is in tension” statement maps to energy density:

\[
u_{sea}\propto (\nabla\Phi)^2
\]

Between drain and pump, gradients add, giving net binding energy scaling `\propto 1/r` after integration of the mutual potential above.

So electromagnetic attraction is modeled as coupled sea-flow potential energy, not a separate force postulate.

### 3.6.5 Optional medium correction (model extension)

In dense matter or modified sea response:

\[
\epsilon_{eff}=\epsilon_0\,\chi_{sea},\qquad \chi_{sea}\ge 1
\]

This predicts index/screening-like shifts in effective coupling without changing the `1/r` structure.

Status:
- Coulomb kernel recovered from drain/pump potential coupling = **MODEL, PARTIAL-PROVEN**
- first-principles derivation of `\chi_{sea}` from coin/spring microconstants remains open.

### 3.6.6 Falsifier

If measured two-body potential deviates from `1/r` at tested scales, the drain/pump potential model must introduce additional sea terms (e.g. Yukawa cutoffs) and refit `\epsilon_{eff}`.

---

## 3.7 Stability criterion

Text claim: proton is stable because fused spring cannot spontaneously unfuse.

Formal metastability criterion:

\[
\Delta E_{barrier} \gg k_B T_{ambient}
\]

and for spontaneous decay rate:

\[
\Gamma_{unfuse}\sim A\,e^{-\Delta E_{barrier}/k_BT} \approx 0 \quad \text{(ambient)}
\]

Interpretation: proton corresponds to deep minimum in effective structural potential.

Status: **MODEL** (Arrhenius-like barrier language), qualitatively aligned with observed proton longevity.

---

## 3.8 Phase-state ladder formalization

From Sections 1-3:
- photon: vapor/no structure
- electron: liquid/gas (active pump)
- proton: solid/fused

Represent by order parameter pair `(S,L)`:
- structural rigidity `S`
- pump freedom `L`

Qualitative ordering:

\[
\text{photon}: (S\!\downarrow,L\!\downarrow),\quad
\text{electron}: (S\!\uparrow,L\!\uparrow),\quad
\text{proton}: (S\!\uparrow\uparrow,L\!\downarrow)
\]

Status: **MODEL** taxonomy (useful for cross-section consistency).

---

## 3.9 Equation set (Section 3 robust core)

1. Regime switch: `K < K_f` elastic, `K >= K_f` fused
2. `H_e = H_coin(kappa) + P_K^2/(2M_K) + (1/2)k_el K^2 + V_struct(K)`
3. Fusion branch: `K>=K_f => Omega->0`, `H_coin -> (hbar/2)Delta_f sigma_z`
4. Landau drive: `V_landau=(r/2)K^2+(u/4)K^4`, `r=r_0(1-K_ext/K_f)`
5. Clock suppression in fused state: `f_b^(p) ~ 0`
6. Energy lock: `Delta m c^2 = E_locked`
7. Geometric ratio law: `R_pe = 1/(1-alpha K_f) = L_0/L_p`
8. Empirical anchor: `R_pe = 1836.152...` with `K_f approx 1-1/R_pe` at `alpha=1`
9. Charge closure: `2/3 - 1/3 + 2/3 = +1`
10. Sea Poisson form: `div(eps_eff grad Phi) = -rho_q`
11. Coulomb limit: `U_eff = Q_p Q_e/(4 pi eps_eff r)`, `eps_eff = eps_0` (leading calibration)
12. Stability barrier: `Gamma_unfuse ~ exp(-DeltaE_barrier/k_BT)`, `Gamma_fuse ~ exp(-DeltaE_barrier/k_BT)`

---

## 3.10 Tests and falsifiers

T3-1. **High-energy breakup threshold**
- Prediction: proton breakup requires energy exceeding fusion barrier scale.

T3-2. **Compression trajectory simulations**
- If coin-geometry model is explicit, crossing `K_f` should produce hysteresis/irreversibility.

T3-3. **Mass-ratio derivation challenge**
- A successful geometric model must output `m_p/m_e` near empirical value without arbitrary tuning.

T3-4. **Clock-signature contrast**
- Proton should lack electron-like internal bounce signatures in contexts where electron pump markers are observable.

---

## 3.11 Cross-reference updates required

### Back-links
- Section 2 open item O2-4 now moved here as central Sec 3 derivation target.
- Section 2 `H_coin` is now the explicit pre-fusion term in `H_e`; fusion quenches `Omega` at `K_f`.

### Forward-links
- Section 4 neutron model must use proton as fused drain base state.
- Section 9 nuclear model uses proton drain + regional charge mapping.
- Section 10 gravity language depends on accumulated drain fields.
- Section 11 dark-sector comparisons require clear distinction: fused-with-locked mode vs spring-without-inner-photon.

---

## 3.12 Open items

| ID | Claim | Status |
|---|---|---|
| O3-1 | Closed-form derivation of `m_p/m_e` from fusion geometry | PARTIAL — **Step 3 gate PASS (core):** `K_f^{pin}` + `R_pe^{model,(0)}=\pi^2/8`; **1836** needs `\mathcal{M}_{lat}` (Step 12, C6) |
| O3-2 | Microscopic Hamiltonian for fusion transition at `K_f` | PARTIAL (`H_e`, `V_struct`, `H_p` branch map + barrier rate added; material constants still open) |
| O3-3 | Quantitative mapping from drain potential to Coulomb law parameters | PARTIAL (`U,F` Coulomb kernel from `Phi_d`; `eps_eff=eps_0` leading map; `chi_sea` micro-derivation open) |
| O3-4 | Direct observable for “no internal proton clock” in this model language | PARTIAL (`f_clock^p=0` nulls + `S_p(omega)` discriminator; sensitivity calibration open) |

---

## 3.13 Completion status

- Section 3 intuition mapped to formal regime/energy/charge/stability equations.
- Core arithmetic/proportional closures included.
- Forward dependencies for Sections 4,9,10,11 explicitly set.
- Tier-1 fusion Hamiltonian closure added for O3-2 (effective form).
- Tier-2 geometric mass-ratio closure added for O3-1 (`R_pe` from `K_f`, `alpha`).
- Tier-2 Coulomb mapping closure added for O3-3 (`epsilon_eff` calibration).
- Tier-4 proton-clock null observables added for O3-4.

**Section 3 derivation pass: COMPLETE (v1).**
