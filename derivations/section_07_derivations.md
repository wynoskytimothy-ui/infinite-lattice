# Section 7 — Tunneling: Formal Derivations

Maps `section_07_tunneling.md` into equations, proof status, and tests.

---

## 7.1 Imports

From Section 2:
- coin/pump state variables (`x(t)`, `f_b`, compression)

From Section 5:
- observation/dephasing channel language (`M_n`, `D(rho)`)

From Section 6:
- decoherence environment rates (`Gamma_break`, photon-flux effects)

From Section 3/4:
- heavier fused structures have larger shredding threshold scale

---

## 7.2 Barrier model

Standard 1D barrier setup:

- barrier height `V0`
- incident particle energy `E < V0`
- barrier thickness `L`

Transmission amplitude in WKB form:

\[
T \sim e^{-2\kappa L}, \qquad \kappa=\frac{\sqrt{2m(V_0-E)}}{\hbar}
\]

This is the quantitative anchor behind your thickness/mass statements.

Status: **E/ANCHORED**.

### 7.2.1 Microscopic `kappa` from coin/spring Hamiltonian (Tier-2 progress on O7-1)

Import Section-2 cavity Hamiltonian in barrier region:

\[
\hat H_x=\frac{\hat p_x^2}{2m_{eff}}+\underbrace{\frac{1}{2}k_s u^2}_{\text{spring tension}}
+\underbrace{\frac{\hbar}{2}\sqrt{\Delta(\kappa)^2+\Omega(\kappa)^2}}_{\text{coin bias+bounce scale}}
+\underbrace{V_{bar}(\kappa)}_{\text{barrier compression drive}}
\]

with barrier drive from environmental compression (Section 5):

\[
V_{bar}(\kappa)=\hbar g_E E_{bar}(x)\,\sigma_z
\quad\Rightarrow\quad
\Delta_{eff}=\Delta(\kappa)+2g_E E_{bar}(x)
\]

Define effective local barrier energy scale:

\[
U_{bar}(x)=\frac{\hbar}{2}\sqrt{\Delta_{eff}(x)^2+\Omega(\kappa)^2}
+\frac{1}{2}k_{el}\big(K(x)-K_0\big)^2
\]

For incident channel energy `E`, WKB exponent uses:

\[
\kappa(x)=\frac{1}{\hbar}\sqrt{2m_{eff}(x)\,\big[U_{bar}(x)-E\big]_+}
\]

### 7.2.2 Shredding map to `m_eff` and `Omega`

Your shredding stage corresponds to reduced bounce and increased sea-coupling:

\[
\Omega(\kappa)\to \Omega_0(1-\xi_{shred}),\qquad
m_{eff}(x)\to m_e\,\mathcal M(\xi_{shred})
\]

with minimal closure:

\[
\xi_{shred}=\frac{|E_{bar}(x)|}{|E_{bar}(x)|+E_{ref}},
\qquad
\mathcal M(\xi)=1+\lambda_m\xi_{shred}
\]

- electron-like states: small `\lambda_m`, easier shredding, smaller `kappa`
- proton/neutron-like fused states: large `\lambda_m`, harder shredding, larger `kappa`

This reproduces the Section-7 mass-scaling trend `\kappa\propto\sqrt{m}` when `\mathcal M` is identified with effective tunneling mass.

### 7.2.3 Thickness law recovery

For approximately uniform barrier segment of length `L`:

\[
T\sim \exp\!\left(-2\int_0^L \kappa(x)\,dx\right)
\approx \exp(-2\bar\kappa L)
\]

with

\[
\bar\kappa=\frac{1}{L}\int_0^L \frac{1}{\hbar}\sqrt{2m_{eff}(x)[U_{bar}(x)-E]}\,dx
\]

So the anchored law `T~e^{-2\kappa L}` is retained, but `\kappa` is now computed from coin/spring/barrier fields instead of being postulated alone.

Status:
- microscopic identification of `\kappa` from `H_x`, `\Delta_{eff}`, spring term = **MODEL, PARTIAL-PROVEN**
- ab-initio field-to-parameter map for `E_{bar}(x)`, `\lambda_m`, and `E_{ref}` remains open.

---

## 7.3 “Shredding” as effective coherence-loss / delocalization channel

Your mechanism (coin -> vapor-like spread -> recondense) can be formalized as a staged open-system process:

1. **Entry (deformation):** barrier coupling increases dephasing and mode mixing
\[
\dot{\rho} = -\frac{i}{\hbar}[H,\rho] + \mathcal{D}_{barrier}(\rho)
\]

2. **Transit (evanescent support):** amplitude inside barrier decays with `e^{-\kappa x}`
\[
\psi(x)\propto e^{-\kappa x},\quad 0<x<L
\]

3. **Exit (recoherence / recapture):** outside barrier, free propagation channels resume and electron recovers compact pump behavior.

Interpretation link:
- “inner photon dissolves into sea” = strong reduction of confined-mode participation during barrier segment.

Status:
- evanescent law = **ANCHORED**
- specific “inner-photon dissolve/recapture” ontology = **MODEL**.

### 7.3.1 Recoherence/recondensation after barrier exit (Tier-4 progress on O7-2)

After exit at `x=L`, define a compactness order parameter `\chi(t)\in[0,1]` (`\chi=1` fully recondensed coin, `\chi=0` shredded/vapor-like).

Phenomenological recovery law:

\[
\dot{\chi} = \Gamma_{rec}(1-\chi) - \Gamma_{sh}\chi
\]

Steady state:

\[
\chi_{ss}=\frac{\Gamma_{rec}}{\Gamma_{rec}+\Gamma_{sh}}
\]

Microscopic closures from Section-2 spring relaxation and Section-5 dephasing:

\[
\Gamma_{rec}=\frac{k_s}{m_{eff}}\,\frac{1}{2\pi}\,\Pi_{pin}^{-1},
\qquad
\Gamma_{sh}=\Gamma_{break}+\Gamma_{obs}
\]

Recovery timescale:

\[
\tau_{rec}=\frac{1}{\Gamma_{rec}+\Gamma_{sh}}
\]

Transmission through barrier with partial recoherence:

\[
T_{eff} = T_{WKB}\,\chi_{ss}
= e^{-2\bar\kappa L}\,\frac{\Gamma_{rec}}{\Gamma_{rec}+\Gamma_{sh}}
\]

Status: **MODEL, PARTIAL-PROVEN** (explicit `\tau_{rec}` and `T_eff`; field-calibrated `\Gamma_{obs}` in barrier exit region still open).

### 7.3.2 Entanglement-cost sharing and partner decoherence shift (Tier-4 progress on O7-3)

If tunneling event shares entanglement cost with a partner mode (Section 6), write partner decoherence increment:

\[
\Delta\Gamma_{break}^{(partner)} = \eta_{share}\,\Gamma_{sh}
\]

with sharing fraction `\eta_{share}\in[0,1]`.

Predicted visibility shift for partner channel:

\[
\Delta V \approx -\frac{\partial V}{\partial \Gamma_{break}}\Delta\Gamma_{break}^{(partner)}
\approx V_0\,\tau_{coh}\,\eta_{share}\,\Gamma_{sh}
\]

where `\tau_{coh}` is partner coherence time.

Operational discriminator:
- compare which-path insertion experiments with and without pre-established partner entanglement;
- nonzero `\Delta V` at matched barrier settings supports `\eta_{share}>0`.

Status: **MODEL, PARTIAL-PROVEN** (measurable shift law; ab-initio `\eta_{share}` from coin graph still open).

### 7.3.3 Postulate P7-2: partial shred vs full flatten (compression regimes)

**Intuition (book language):** an electron can pass a barrier by **shedding its inner-photon packet** while the **spring** remains extended on a sea/DM-backed ripple — but **full flattening** (no bounce, no mid-transit, permanent pin or escape) requires **extreme** compression comparable to **nuclear**, **stellar-core**, or **black-hole** limits.

This is **not** a new WKB postulate; it classifies **when** shredding stays partial vs when Sec 4–5 collapse/escape laws take over.

#### Two regimes

| Regime | Name | Pinning | Inner photon | Spring | Typical outcome |
|--------|------|---------|--------------|--------|-----------------|
| **Soft** | partial shred / tunnel | `\Pi_{pin} \ll 1` | packet **shed** (delocalized into sea) | **travels** on extended mode | `T_{eff} = T_{WKB}\,\chi_{ss}`; **recapture** possible |
| **Hard** | full flatten / collapse | `\Pi_{pin} \to 1` | **pinned** or **escaped** (`\gamma_{obs}`) | **no bounce** | decay, fusion, capture; **no** ordinary tunnel recapture |

Define a **flattening indicator**:

\[
\Pi_{pin}=\frac{|\Delta_{eff}|}{|\Delta_{eff}|+\Omega}
\]

- **Soft barrier:** `\Pi_{pin}` rises locally but stays below unity; `\xi_{shred}<1`.
- **Hard environment:** `\Pi_{pin}\to 1` and/or pressure `P\to P_c` (Sec 4) — package cannot spring-travel.

#### Energy / pressure threshold table (order-of-magnitude anchors)

| Scale | Symbol / anchor | Order | AETHOS reading | Tag |
|-------|-----------------|-------|----------------|-----|
| Chemical / lab barrier | `V_0 - E`, `E_{bar}` | eV–keV | soft shred; `\xi_{shred}` finite | **E** |
| Compton package | `m_e c^2` | `\sim 0.511` MeV | coin size / bounce reference; not yet full flatten | **E** |
| Nuclear / Fermi | `P_c`, `\Delta_{sep}\sim\hbar\omega_b` | MeV / nucleon scale | hard pin; sharing `R_{share}`; `\gamma_{obs}` channel | **E/FIT** (Sec 4) |
| Stellar core | `\rho`, `P_{core}` | `\sim 10^{16}` Pa (order) | sustained hard compression; fusion tunneling still **soft** for **incident** proton branch | **E** |
| Black hole / GR | horizon scale | extreme geometry | **limit** where extended spring picture breaks — **MODEL** until metric coupling derived | **MODEL** |

**Rule P7-2:** ordinary tunneling operates in the **soft** row; **hard** row explains neutron escape, nuclear binding, and “cannot spring through” without replacing `T\sim e^{-2\kappa L}`.

#### Shed → transit → recapture (mechanical sequence)

1. **Entry:** barrier field raises `\xi_{shred}`; inner-photon compactness `\chi` drops (Sec 7.3.1).
2. **Transit:** evanescent spring mode on path with optional **DM fill** (Sec 7.3.4); `\psi(x)\propto e^{-\kappa x}`.
3. **Exit:** passing sea/probe photon delivers compression impulse `\hbar g_E E_{obs}(t)` (Sec 5); `\Gamma_{rec}` restores `\chi\to\chi_{ss}` — **recapture** of one vapor mode into coin (Sec 1.3.4 **P1-v**).

**Contrast (hard):** when `P\ge P_c`, outer `\gamma_{obs}` escapes (Sec 4); recapture requires capture channel `p+e^-+\nu\to n`, not barrier exit alone.

Status: **P7-2 MODEL**; consistent with **PARTIAL** O7-2 recoherence law and Sec 4 pressure threshold.

### 7.3.4 DM ripple transit channel (cross-link P11-3)

Dark-matter spring units (no inner photon) form a **barrier-transparent connective mesh** (Sec 11 **P11-3**). During **soft** tunneling:

- electron **spring** extends along a thinned, filled ripple path;
- **fill fraction** `\phi_{path}\in[0,1]` on the A→B segment (Sec 6.13);
- `\phi_{path}=1` means continuous DM-backed tension line; `\phi_{path}=0` means no mesh-assisted link.

Effective shredding/transit modifier (phenomenological):

\[
\xi_{shred}\to \xi_{shred}\,(1-\eta_{DM}\phi_{path}),\qquad
\bar\kappa \to \bar\kappa\,(1+\eta_\kappa(1-\phi_{path}))
\]

with `\eta_{DM},\eta_\kappa` fit constants. Interpretation: filled DM channel **eases** shred transit (lower effective `\bar\kappa`); unfilled path reverts to bare WKB.

Status: **MODEL**; `\eta_{DM}` calibration **OPEN** (O7-4).

---

## 7.4 Thickness scaling proof

Given `T ~ exp(-2kappa L)`, log-transform:

\[
\ln T = -2\kappa L + const
\]

Hence linear dependence of `ln T` on `L` with slope `-2kappa`.

Status: **PROVEN** from WKB form.

---

## 7.5 Mass scaling proof

Because

\[
\kappa \propto \sqrt{m}
\]

we get

\[
\ln T \propto -L\sqrt{m}
\]

So increasing mass suppresses tunneling exponentially at fixed `L, V0-E`.

This matches your “heavier -> harder to tunnel.”

Status: **PROVEN** from same anchor.

---

## 7.6 “Between atoms” geometric interpretation

Your text emphasizes barrier emptiness and path-through-gaps.

Formal statement:
- in condensed matter, effective barrier potential is spatially heterogeneous
- dominant tunneling pathways correspond to lower effective action trajectories through potential channels.

Action form:

\[
S_{eff} = \int \sqrt{2m(V(\mathbf{r})-E)}\,ds
\]

Most probable paths minimize `S_eff` (instanton-like/path-integral view), which can align with inter-atomic gap routes.

Status:
- least-action tunneling path formalism = **ANCHORED**
- literal “always between atoms, never through effective potential” = **MODEL phrasing** (keep as interpretation).

---

## 7.7 Observation inside barrier and detectability

Your claim: particles seem absent inside barrier.

Formal reading:
- local probability density in classically forbidden region is exponentially small,
- detector coupling in that region often insufficient for robust click rates.

\[
P_{detect}(x) \propto |\psi(x)|^2 \propto e^{-2\kappa x}
\]

So “not seen in barrier” emerges from very low occupancy/coupling, not logical nonexistence.

Status: **ANCHORED** with model-compatible interpretation.

---

## 7.8 Timescale / “instant tunneling” caution

Your text says tunneling seems nearly instant.

Robust statement:
- measured traversal-time proxies are subtle and setup-dependent,
- no reliable controllable superluminal signaling arises from tunneling.

Keep causality constraint:

\[
\text{No signaling speed } > c
\]

Status: **ANCHORED** (causal consistency requirement).

---

## 7.9 Real-world anchors

1. **STM:** tunneling current
\[
I \propto e^{-2\kappa d}
\]
with tip-sample gap `d`.

2. **Tunnel diode:** thin barrier enables high transmission branch.

3. **Alpha decay / Gamow factor:** decay rate exponential in barrier action.

4. **Stellar fusion:** finite proton tunneling probability enables pp-chain ignition.

Status: **E/ANCHORED** phenomena.

---

## 7.10 Equation set (Section 7 robust core)

1. `H_x = p_x^2/(2m_eff) + (1/2)k_s u^2 + (hbar/2)sqrt(Delta^2+Omega^2) + V_bar`
2. `kappa(x)=(1/hbar)sqrt(2 m_eff(x)[U_bar(x)-E])`
3. `U_bar=(hbar/2)sqrt(Delta_eff^2+Omega^2)+(1/2)k_el(K-K_0)^2`
4. shredding: `Omega->Omega_0(1-xi)`, `m_eff->m_e M(xi)`, `xi=|E_bar|/(|E_bar|+E_ref)`
5. `T ~ exp(-2kappa L)`
6. anchored limit: `kappa = sqrt(2m(V0-E))/hbar`
7. `ln T = -2kappa L + const`
8. `ln T ~ -L sqrt(m)` (fixed `V0-E`)
9. inside barrier: `psi(x) ~ exp(-kappa x)`
10. detectability proxy: `P_detect ~ |psi|^2 ~ exp(-2kappa x)`
11. path-action form: `S_eff = int sqrt(2m(V(r)-E)) ds`
12. `dot chi = Gamma_rec(1-chi)-Gamma_sh chi`, `chi_ss=Gamma_rec/(Gamma_rec+Gamma_sh)`
13. `tau_rec=1/(Gamma_rec+Gamma_sh)`, `T_eff=T_WKB chi_ss`
14. `Delta Gamma_break^(partner)=eta_share Gamma_sh`, `Delta V approx V_0 tau_coh eta_share Gamma_sh`
15. **P7-2:** soft shred (`Pi_pin<<1`, `xi<1`) vs hard flatten (`Pi_pin->1`, `P->P_c`)
16. `T_eff = T_WKB chi_ss` (soft); recapture via `Gamma_rec` + passing photon compression
17. DM transit: `phi_path`, `xi_shred -> xi_shred(1-eta_DM phi_path)` (P11-3 link)

---

## 7.11 Tests and falsifiers

T7-1. **Thickness scan**
- plot `ln T` vs `L`; expect near-linear trend.

T7-2. **Mass scan**
- compare isotopic/particle mass variants under matched barriers; expect exponential suppression with `sqrt(m)`.

T7-3. **Barrier-shape dependence**
- structured materials should show preferred transmission channels matching lower `S_eff` paths.

T7-4. **STM calibration**
- verify `I(d)` exponential decay and extract `kappa`.

T7-5. **Soft vs hard compression discriminator**
- below nuclear/`P_c` scale: expect recoherence `\chi_{ss}>0` and `T_{eff}=T_{WKB}\chi_{ss}`;
- at/above `\Pi_{pin}\to 1` or `P\to P_c`: expect escape/capture channels (Sec 4), not ordinary recapture.

T7-6. **DM-column / fill proxy (MODEL)**
- if `\phi_{path}` proxy increases (halo density, low-Z barrier), predict modest increase in `T_{eff}` or `\tau_{rec}` at matched `\bar\kappa L`.

---

## 7.12 Cross-reference updates required

### Back-links
- Section 5 observer/dephasing channel now used to formalize “shredding” entry stage.
- Section 6 environment/decoherence language consistent with barrier-coupled channel increase.
- Section 4 pressure threshold `P_c`, `\gamma_{obs}` escape = **hard flatten** limit of **P7-2**.
- Section 11 **P11-3** DM mesh = transit substrate for soft shredding (Sec 7.3.4).

### Forward-links
- Section 8 interference visibility under which-path detection should reuse channel-strength language.
- Section 9 alpha decay section should reference Gamow-style barrier exponent.
- Section 10 stellar fusion section should explicitly reference proton tunneling exponent.

---

## 7.13 Open items

| ID | Claim | Status |
|---|---|---|
| O7-1 | Derive explicit coin/spring micro-Hamiltonian yielding effective `kappa` | PARTIAL — **Step 7 gate:** `kappa_wkb_from_h_x`, `u_bar_from_h_coin`; `g_E E_{bar}(x)` map **OPEN** |
| O7-2 | Quantify recoherence/recondensing dynamics after barrier exit | PARTIAL — **`gamma_rec_from_geometry`**, `T_eff` pipeline; exit `\Gamma_{sh}` **OPEN** |
| O7-3 | Map “entanglement cost sharing” to measurable partner decoherence shifts | PARTIAL (`Delta Gamma_break`, `Delta V` law; `eta_share` micro-derivation open) |
| O7-4 | Calibrate DM fill modifier `eta_DM`, `phi_path` effect on `T_eff` / `\xi_{shred}` | PARTIAL — structure in Step 7 gate; `\eta_{DM}`, `\eta_\kappa` **OPEN** (P11-3) |

---

## 7.14 Completion status

- Section 7 intuition mapped to WKB/Gamow-compatible equations.
- Thickness/mass suppression proofs included.
- Ontology language aligned with causal/measurement constraints.
- Forward links to Sections 8/9/10 prepared.
- Tier-2 microscopic `kappa` closure added for O7-1.
- Tier-4 recoherence dynamics added for O7-2.
- Tier-4 entanglement-cost decoherence shift added for O7-3.
- **P7-2** partial shred vs full flatten + energy threshold table (Sec 7.3.3).
- DM ripple transit cross-link (Sec 7.3.4 → P11-3, Sec 6.13).

**Section 7 derivation pass: COMPLETE (v1.1).**
