# AETHOS Architecture Mandates (Locked)

**Authority:** User-locked rebuild parameters for Geometry Gate passes.  
**Rule:** No `derivations/*.md` or physics code may contradict these without explicit amendment here.

**Locked:** 2026-06-02 (Step 2 trigger)

---

## C1 — Coin half-width

| Field | Value |
|-------|--------|
| **Mandate** | **Single anchor:** `L ≡ λ_C/2 = h/(2m_e c)` |
| **Scope** | Cavity, bounce path, pump, `H_x` well, `ω_b`, `k_s` closure, Sec 5–7 |
| **Forbidden** | Secondary coin lengths, fitted `L` in electron core |
| **Tag** | **GEOMETRY** (from `m_e`, `h`, `c` only) |

---

## C2 — Fusion compression `K_f` and 1836

| Field | Value |
|-------|--------|
| **Mandate** | **`K_f` derived** from maximum mechanical compression of coin geometry — **not FIT to 1836.15** as primary closure |
| **Chain** | Step 3: `K_f` from spring/coin limit → `L_p/L_0` → `R_pe` prediction compared to **E** anchor 1836.152… |
| **Allowed** | Document **E** mass ratio as experimental check on geometry, not as input to `K_f` |
| **Forbidden** | `K_f := (R_pe−1)/R_pe` as *definition* without geometric derivation |
| **Tag** | **GEOMETRY** (Step 3 gate); Step 2 exports `L`, `k_s`, `U_max` |

---

## C3 — Neutron lifetime `τ_n`

| Field | Value |
|-------|--------|
| **Mandate** | **Primary:** pressure escape / outer-photon threshold (`P → P_c`, `γ_obs` escape) |
| **SM weak rate** | Document as **observable consequence** / comparison layer — **not equal partner** |
| **Repo** | Sec 4.5.5 FIT remains **calibration check**; narrative priority = geometry escape |
| **Tag** | **GEOMETRY** + **E** cross-check |

---

## C4 — Strong force

| Field | Value |
|-------|--------|
| **Mandate** | **Network pressure + entanglement sharing** (`R_share`, `C_N`, `B_share`) across lattice-linked nodes |
| **Forbidden** | Gluons, QCD partons as fundamental in AETHOS narrative |
| **Tag** | **GEOMETRY/NETWORK** (Secs 4, 6, 9) |

---

## C5 — Bell / entanglement

| Field | Value |
|-------|--------|
| **Mandate** | **Joint mechanical ripple** on DM-backed string; compression events at A and B |
| **Forbidden** | `sgn(cos(θ))` half-plane hidden-variable sketch (**REJECT** — falsified) |
| **Status** | O5-3 **PARTIAL** — sign sketch **REJECT**; joint ripple linear projection half-scale; full kernel needs `\phi_{AB}` (C7 Stage B) |
| **Tag** | **GEOMETRY** target (Secs 5–6) |

---

## C6 — 3D complex plane (`aethos_active`, Sec 12)

| Field | Value |
|-------|--------|
| **Mandate** | **Physical + computational unified** — the **3D complex plane** Ψ=(z,ζ) is the arena; lattice formula on anchor chains generates coordinates |
| **Active nodes** | **Any countable anchor set** may define active nodes — **not limited to primes**. Primes are one species (`SequenceKind.PRIMES`); evens, powers of two, Fibonacci, sqrt-scaled, custom chains, etc. are equally valid |
| **Correlations** | Different species → different (X,Y,ζ) on the same 8×4 wing topology (`aethos_sequences.make_chain`, `ActiveNetwork100.bootstrap(chain_species=…)`). |
| **Naming** | **Use:** 3D complex plane / lattice formula. **Deprecated:** prime lattice, infinity lattice, φ-prime lattice. See repo `ONTOLOGY.md`. **π lattice** (Part I) is a **separate** construction. |
| **Rule** | 3D complex plane modules must **import** Sec 2–12 geometry constants (`L`, `ω_b`, …), not replace them |
| **Step** | Step 12 gate wires plane ↔ `v_space²+v_time²=c²`; `\mathcal{M}_{lat}` (Step 3 gap) may depend on **which species** is physically selected |
| **Tag** | **GEOMETRY** (Sec 12); code track **PHYSICS-LATTICE** |

---

## C7 — DM mesh (staging)

| Field | Value |
|-------|--------|
| **Mandate (this pass)** | **Stage A:** effective **`ℓ_c`** in `Γ_form`, `Γ_break`, tunneling modifiers |
| **Mandatory forward contract** | **Stage B (Step 6/11 gate):** replace `ℓ_c` with **`φ_AB` fill + filaments** (P11-3); Bell **`E=-φ_{AB}\cos(a-b)`** at full fill (**CONTRACT** in Step 6 gate) |
| **Forbidden** | Claiming DM mesh is fully derived while only `ℓ_c` FIT exists |
| **Tag** | **PARTIAL → GEOMETRY** (two-stage) |

---

## Documentation rules (no silent hand-waving)

1. **ANCHORED** = “matches standard X when …” — never “derived from geometry.”
2. **FIT** = numeric row in `calibration_sheet.md` with procedure — never default parameter.
3. **GEOMETRY** = only `(L, k_s, m_e, c, h, κ, …)` chain from mandates.
4. **REJECT** = must cite falsifier (e.g. O5-3 sign test).
5. Every section **exports** symbols the next section **imports** (see `geometry_rebuild_master_plan.md`).

---

## Gate status

| Step | Status |
|------|--------|
| Mandates C1–C7 | **LOCKED** |
| Step 2 | **CORE GATE PASS** (`section_02_geometry_audit.md`) |
| Step 3 | **CORE GATE PASS** (`section_03_geometry_audit.md`) |
| Step 4 | **CORE GATE PASS** (`section_04_geometry_audit.md`) |
| Step 5 | **CORE GATE PASS** (`section_05_geometry_audit.md`) |
| Step 6 | **CORE GATE PASS** (`section_06_geometry_audit.md`) |
| Step 7 | **CORE GATE PASS** (`section_07_geometry_audit.md`) |
| Step 8 | **CORE GATE PASS** (`section_08_geometry_audit.md`) |
| Step 9 | **CORE GATE PASS** (`section_09_geometry_audit.md`) |
| Step 10 | **CORE GATE PASS** (`section_10_geometry_audit.md`) |
| Step 11 | **CORE GATE PASS** (`section_11_geometry_audit.md`) |
| Step 12 | **CORE GATE PASS** (`section_12_geometry_audit.md`) |
