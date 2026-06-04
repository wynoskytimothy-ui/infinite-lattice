# AETHOS Formal Derivations (Sections 1–12)

One derivation file per section. Each intuition block gets:

| Tag | Meaning |
|-----|---------|
| **P** | Postulate (model assumption) |
| **D** | Derived (follows from prior P/D + math) |
| **I** | Identity (definition or algebraic tautology) |
| **E** | Empirical anchor (standard physics, experimentally fixed) |
| **T** | Test / falsifier (how to check) |

## Status legend

- **PROVEN** — mathematical proof complete in-file
- **ANCHORED** — matches established physics when parameters chosen
- **MODEL** — follows from AETHOS postulates; needs independent test
- **OPEN** — not yet derived; listed for next pass

## File index

| Section | Intuition source | Derivation file | Status |
|---------|------------------|-----------------|--------|
| 1 | Photon sea | `section_01_derivations.md` | **DONE** (this pass) |
| 2 | Electron / coin | `section_02_derivations.md` | **DONE** (v1) |
| 3 | Proton / fusion | `section_03_derivations.md` | **DONE** (v1) |
| 4 | Neutron | `section_04_derivations.md` | **DONE** (v1) |
| 5 | Measurement | `section_05_derivations.md` | **DONE** (v1) |
| 6 | Entanglement | `section_06_derivations.md` | **DONE** (dynamic form/break pass) |
| 7 | Tunneling | `section_07_derivations.md` | **DONE** (v1) |
| 8 | Double slit | `section_08_derivations.md` | **DONE** (v1) |
| 9 | Atom | `section_09_derivations.md` | **DONE** (v1) |
| 10 | Cosmic scales | `section_10_derivations.md` | **DONE** (v1) |
| 11 | Dark matter/energy | `section_11_derivations.md` | **DONE** (v1) |
| 12 | Zeno + time | `section_12_derivations.md` | **DONE** (v1) |

## Dependency chain

```
Sec 1 (sea, photon math, ψ as disturbance)
  → Sec 2 (trapped photon, pump, clock)
    → Sec 5,6,7,8 (measurement, entangle, tunnel, slit)
  → Sec 3,4 (proton, neutron)
    → Sec 9 (atom)
  → Sec 10,11 (cosmos, dark sector)
  → Sec 12 (Zeno + dilation unifies 1–10)
```

Work rule: finish and review each `section_XX_derivations.md` before starting the next.

## Geometry-first full rebuild

**`architecture_mandates.md`** — locked C1–C7. **`geometry_rebuild_master_plan.md`** — 12-step gates. Step 2 core: **`section_02_geometry_audit.md`**.

## External math crosswalks

User-provided “Section N Mathematics” blocks are indexed in `external_math_index.md`. Section 5 has a full merge/hold/reject table in `section_05_math_crosswalk.md`. Calibration APIs live in `aethos_physics.py`; tests in `test_aethos_physics.py`.

## Cross-Reference Rule (always-on)

From now on, every section pass uses this loop:

1. **Derive current section (`N`)**
   - Add full formulas/proofs/tests for section `N`.
2. **Back-link update**
   - Update earlier sections `1..N-1` if new notation/constraints affect them.
3. **Forward-link prep**
   - Add import requirements for sections `N+1..12`.
4. **Consistency table refresh**
   - Keep symbol definitions, assumptions, and identities synchronized.
5. **Conflict log**
   - Record any claim that is MODEL vs ANCHORED vs OPEN.

This keeps all sections cross-referenced and continuously updated as we move.

## Global cross-reference files

- `derivations/cross_reference_matrix.md` — section-to-section dependency map
- `derivations/symbol_registry.md` — single source of truth for symbols/units
- `derivations/conflict_log.md` — unresolved assumptions or contradictions
- `derivations/open_items_rollup.md` — prioritized PARTIAL/OPEN tracker (active OPEN: 0)
- `derivations/calibration_sheet.md` — empirical anchors + fit procedures for phenomenological parameters
- `derivations/physics_questions_map.md` — **master map**: all equation cores vs open physics questions (what we answer, what remains GAP)

## Next pass (post–Tier 4)

1. Work through **calibration queue** in `calibration_sheet.md` (`F-gobs` done — run `python aethos_physics.py`; next: `C_b`, `\Lambda`).
2. Close uniqueness gaps (Born Gleason route, O11-3 universality).
3. Wire calibrated constants into simulation (`aethos_active.py` / codec stack per Section 12 hook table).
