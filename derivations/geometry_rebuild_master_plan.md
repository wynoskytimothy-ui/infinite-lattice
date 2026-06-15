# AETHOS Geometry-First Rebuild — Master Plan

**Goal:** Walk Sections 1→12 **one at a time**. At each step, every claim is tagged and either **derived from your geometry**, **anchored to standard physics**, **calibrated (FIT)**, or explicitly **GAP / OPEN**. Nothing left as silent “MODEL prose.”

**Rule:** Do not start section `N+1` until section `N` passes its **Geometry Gate** (below).

---

## What “resolved” means

| Status | Meaning | Allowed in text? |
|--------|---------|------------------|
| **GEOMETRY** | Follows from coin/spring/sea/DM dimensions + stated postulates only | “Derived from geometry” |
| **PROVEN** | Pure math from prior rows | Yes |
| **ANCHORED** | Matches standard QM/SR/GR/experiment when parameters chosen | “Same as textbook X” |
| **FIT** | One or more parameters fixed from data (`calibration_sheet.md`) | “Calibrated to …” |
| **PARTIAL** | Structure in place; geometry or data still missing | “Open: …” |
| **OPEN** | No acceptable closure yet | Must list in gate |
| **GAP** | Physics has no answer; AETHOS does not claim one | “Not addressed” |
| **REJECT** | Sketch falsified or inconsistent | Do not use |

**Geometry primitives (your stack):**

```text
Sea S  →  h-quanta modes (P1-v)
Coin   →  radius L ~ λ_C/2, two sides, σ_z spin map
Spring →  k_s, κ, bounce ω_b = πc/L
Pump   →  trapped inner photon, H_coin, Π_pin
DM     →  spring-only mesh, φ_AB fill (P11-3)
Network→  R_share, C_N, B_share (nucleus / entanglement)
```

---

## Geometry Gate (per section)

Before marking section **N** done:

1. [ ] Every equation in §“robust core” has a tag column.
2. [ ] Every **OPEN** item has either a geometry path or “needs user clarification #___”.
3. [ ] No conflict with `conflict_log.md` (or conflict updated).
4. [ ] At least one **test or number** in `aethos_physics.py` or `test_aethos_physics.py` if section has FIT geometry.
5. [ ] `symbol_registry.md` updated for new symbols.
6. [ ] One paragraph: **what this section imports** / **what it exports** to N±1.

---

## 12-step order (dependency-safe)

| Step | Section | Intuition file | Derivation file | Geometry focus | Gate priority |
|------|---------|----------------|-----------------|----------------|---------------|
| **1** | Photon sea | `section_01_photon_sea.md` | `section_01_derivations.md` | P1-v vapor modes; Born from T_S²; causality | O1-1 uniqueness; O1-5 φ_B |
| **2** | Electron | `section_02_electron.md` | `section_02_derivations.md` | **L, k_s, H_coin, ω_b** — foundation for all | **O2-1 geometry of H_coin** |
| **3** | Proton / fusion | `section_03_*.md` | `section_03_derivations.md` | **K_f, L_p, fusion barrier** | O3-1: K_f from geometry vs FIT |
| **4** | Neutron | `section_04_*.md` | `section_04_derivations.md` | P, P_c, Π_pin, γ_obs escape | τ_n story (FIT vs weak) |
| **5** | Measurement | `section_05_*.md` | `section_05_derivations.md` | **Λ_n, M_n from H_coin** | O5-3 Bell geometry |
| **6** | Entanglement | `section_06_*.md` | `section_06_derivations.md` | φ_AB, Γ_form, ℓ_c | O5-3 + O11-5 |
| **7** | Tunneling | `section_07_*.md` | `section_07_derivations.md` | κ from H_x; P7-2 shred | O7-4 η_DM |
| **8** | Double slit | `section_08_*.md` | `section_08_derivations.md` | wakes A_L,A_R; Γ_partner | ³He/⁴He Λ |
| **9** | Atom | `section_09_*.md` | `section_09_derivations.md` | A_lm, E_bond, C_b | H₂ C_b |
| **10** | Cosmic | `section_10_*.md` | `section_10_derivations.md` | B field, w(z) bridge | f_part, τ_flip |
| **11** | Dark sector | `section_11_*.md` | `section_11_derivations.md` | σ_γDM, Q, P11-3 | Π_vac, halo joint fit |
| **12** | Time / Zeno | `section_12_*.md` | `section_12_derivations.md` | motion budget, descent | GR equivalence scope |

**Parallel track:** `aethos_active.py`, lattice, anchor sequences — **physical per C6**; active nodes may use **any** `SequenceKind`, not primes only.

---

## Step-by-step checklist (current snapshot)

### Step 1 — Sea ✅ v1.1 (review pass needed)

| Item | Status | Geometry next action |
|------|--------|----------------------|
| P1, P1a, P2, P3, P4 | P / MODEL | P1a ↔ conflict #1 wording |
| P1-v vapor spectrum | MODEL | O1-5: fit φ_B on real spectra |
| E=hf, Born, continuity | ANCHORED / PARTIAL | O1-1: Gleason or nonlinear spring |
| O1-3 ρ_Λ | PARTIAL | derive Π_vac or FIT L_cell |

### Step 2 — Electron ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| H_coin effective | **GEOMETRY (core)** | O2-2 Born uniqueness; O2-3 chirality |
| Born T=k_s u | PARTIAL | close O2-2 |
| d = λ_C/2 | **GEOMETRY (C1)** | — |
| O2-3 charge χ | PARTIAL | chirality from pump×spring×coin |

### Step 3 — Proton ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| K_f^{pin} | **GEOMETRY** | F2 pin balance ~0.41 |
| R_pe^{model,(0)} | **GEOMETRY** | `\pi^2/8` spring-only |
| 1836 full match | **OPEN** | `\mathcal{M}_{lat}` Step 12 (C6) |
| Fusion H_p branch | PARTIAL | barrier at K_f from same L, k_s |

### Step 4 — Neutron ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `t_escape` law | **GEOMETRY** | Primary C3 narrative |
| `\alpha`, `\omega_b` | **GEOMETRY** | C1 coin imports |
| `P_c` candidates | **GEOMETRY / E** | G1–G3 hierarchy |
| `\Gamma_{obs}` forward | **OPEN** | Sec 6 / Step 12 Zeno |
| Weak rate | **ANCHORED** | comparison only |

### Step 5 — Measurement ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `\Lambda_n` from `L_0`, `H_{coin}` | **GEOMETRY** | `lambda_n_from_coin_gradient` |
| Kraus / pin `p` | **GEOMETRY** | `\exp(-\Lambda_n)` |
| O5-3 sign sketch | **REJECT** | falsifier in tests |
| Bell full kernel | **PARTIAL** | `\phi_{AB}` Stage B (C7) |
| SG `g_E` calibration | **FIT** | reference apparatus row |

### Step 6 — Entanglement ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `C(t)` ODE | **PROVEN** | — |
| `\ell_c^{geom}=\lambda_C` | **GEOMETRY** | C7 Stage A |
| `\sigma_{obs}`, `\Gamma_{break}` | **GEOMETRY** + env | `\Phi_{env}` OPEN |
| `\Gamma_{form}` | **GEOMETRY** structure | `k_{lock}\sim f_b` |
| Bell via `\phi_{AB}` | **CONTRACT** | P11-3 fill ODE OPEN |

### Step 7 — Tunneling ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `\kappa` from `H_x` | **GEOMETRY** | `kappa_wkb_from_h_x` |
| P7-2 soft/hard | **GEOMETRY** | `\Pi_{pin}(\kappa)` |
| `T_{eff}` pipeline | **PARTIAL** | `\Gamma_{sh}` OPEN |
| `\eta_{DM}`, `\phi_{path}` | **MODEL** | P11-3 (O7-4) |

### Step 8 — Double slit ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `A_0` wake scale | **GEOMETRY** | `a0_wake_scale` |
| Wake kernels | **GEOMETRY** | `\sigma_{wake}\sim L_0` |
| `\Lambda_{^3He}/\Lambda_{^4He}` | **MODEL** | discriminator ratio |
| `\sigma_{e,i}` | **OPEN** | O8-1 |

### Step 9 — Atom ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `E_bond`, `C_b` chain | **GEOMETRY** + E check | `\eta_{AB}` proxy OPEN |
| `k_{nl}`, `A_{lm}` | **PARTIAL** | l≤1 map |
| `C_N`, `B_{share}` | **MODEL** | O9-1 graph OPEN |

### Step 10 — Cosmic ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `B(R)`, `B_NS` dipole | **GEOMETRY** + E check | `f_part`, `f_NS` OPEN |
| `\tau_{flip}` bridge | **PARTIAL** | Earth-core calibration OPEN |
| `w(z)`, CPL | **PARTIAL** | O11-4 dataset fit |

### Step 11 — Dark sector ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| `\sigma_{\gamma DM}` | **GEOMETRY** structure | `m_{spring}` OPEN |
| P11-3 `\phi_{AB}` | **PARTIAL** | `\Gamma_{fill}` OPEN |
| `Q`, channel classes | **PARTIAL** | joint fit OPEN |
| `w(z)` bridge | **PARTIAL** | O11-4 datasets |

### Step 12 — Time / Zeno / lattice ✅ **CORE GATE PASS**

| Item | Status | Geometry next action |
|------|--------|----------------------|
| Zeno / no instant | **PROVEN** | `P(p)` micro OPEN |
| Motion budget | **ANCHORED** | Kerr map OPEN |
| `\mathcal{M}_{lat}` | **GEOMETRY** | proton anchor species OPEN |
| `R_{pe}^{pred}` | **E-check** ~1840 vs 1836 | species-dependent |

**Rebuild Steps 2–12: CORE GATES COMPLETE.**

## Conflicts to resolve during rebuild

| # | Issue | Resolve in step |
|---|--------|-----------------|
| 1 | Ocean vs causality | 1, 6 |
| 2 | Collapse micro-derivation | 2, 5 |
| 3 | Path at emission | 1, 8 |
| 4 | DM “never detectable” | 11 |
| 5 | Limit vs physical instant | 12 |

---

## Artifacts to maintain (one pass per step)

1. `section_XX_derivations.md` — tags + gate checklist at end  
2. `section_XX_geometry_audit.md` — optional per-step audit (create when starting step)  
3. `derivations/section_XX_math_crosswalk.md` — if external math exists  
4. `aethos_physics.py` — one function group per section  
5. `test_aethos_physics.py` — regression for that section  
6. `calibration_sheet.md` — F-* rows  
7. `conflict_log.md` — move items to RESOLVED when gated  

---

## Recommended execution (your “one step at a time”)

```text
FOR N = 1 TO 12:
    Read section_N intuition + section_N derivations
    List every OPEN/PARTIAL → geometry path or GAP
    Ask user only blocking clarifications for N
    Implement geometry closure + code + test
    Run Geometry Gate
    STOP until gate passes
```

**Start here:** **Step 2 (Electron)** — not Step 1. Step 1 is anchored; **all geometry flows from L, k_s, H_coin**.

---

## Mandates (LOCKED)

See **`architecture_mandates.md`** for C1–C7. Summary:

| ID | Locked decision |
|----|-----------------|
| C1 | `L = λ_C/2` only |
| C2 | `K_f` derived (Step 3); 1836 = **E** check |
| C3 | Pressure escape primary |
| C4 | Network only; no gluons |
| C5 | Joint DM ripple; sign sketch **REJECT** |
| C6 | 3D complex plane / lattice formula (any SequenceKind; Sec 12) |
| C7 | Stage A `ℓ_c` → Stage B `φ_AB` |

**Step 2:** **CORE GATE PASS** — `section_02_geometry_audit.md`.  
**Step 3:** **CORE GATE PASS** — `section_03_geometry_audit.md` (`K_f^{pin}`, `R_pe^{model,(0)}`; 1836 via `\mathcal{M}_{lat}` Step 12).  
**Step 4:** **CORE GATE PASS** — `section_04_geometry_audit.md`.  
**Step 5:** **CORE GATE PASS** — `section_05_geometry_audit.md`.  
**Step 6:** **CORE GATE PASS** — `section_06_geometry_audit.md`.  
**Step 7:** **CORE GATE PASS** — `section_07_geometry_audit.md`.  
**Step 8:** **CORE GATE PASS** — `section_08_geometry_audit.md`.  
**Step 9:** **CORE GATE PASS** — `section_09_geometry_audit.md`.  
**Step 10:** **CORE GATE PASS** — `section_10_geometry_audit.md`.  
**Step 11:** **CORE GATE PASS** — `section_11_geometry_audit.md`.  
**Step 12:** **CORE GATE PASS** — `section_12_geometry_audit.md`. **12-step geometry rebuild complete.**

---

## What we cannot promise to “fully resolve”

Even with your layout, these stay **GAP** unless you add new postulates:

- Baryon asymmetry, three generations, Higgs hierarchy  
- Full Einstein equations from motion budget alone  
- BH information, unique measurement interpretation  
- QCD proton radius / parton structure  

Mark them **GAP** in each section gate so they are not smuggled in as geometry.
