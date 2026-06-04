# External Section 5 Mathematics ↔ AETHOS Crosswalk

Maps user-provided “Section 5: Measurement and Observation” math blocks to `section_05_derivations.md`, `aethos_physics.py`, and status tags.

**Rule:** External prose is pedagogy unless tagged **PROVEN** or **FIT** here. Do not upgrade **MODEL** claims to **PROVEN** without a derivation or calibration row.

---

## Core questions (8)

| # | Question | External answer | Repo location | Tag | Action |
|---|----------|-----------------|---------------|-----|--------|
| 1 | Compression math on inner photon | `Ĉ = n̂·σ̂`, eigenstates `\|±n̂⟩` | §5.3 `M_{s,n}`, §5.5 | **E** spin; **PARTIAL** Kraus from `H_coin` (O5-1) | **Merge** — rename `Ĉ` → spin component; collapse = Kraus/`Π` |
| 2 | Why measurement creates state | `⟨S_z⟩=0` → post `±1`; K-S; `ρ` | §5.4, P5-1 | **ANCHORED** QM; **MODEL** interpretation | **Hold** — label “creates” as P5-1, not new theorem |
| 3 | Where `cos²(θ/2)` comes from | `\|⟨+n̂\|ψ⟩\|²` | §5.5 | **PROVEN** | **Merge** — no change |
| 4 | Why 50/50 symmetric | `t_white/T = 1/2` | §5.6 | **PROVEN** for `ρ=I/2`; bounce = **MODEL** | **Hold** — general law is Born, not time fractions |
| 5 | Bell violation mechanism | Hidden λ in transit; “one ocean” | §5.8, §6.7, `conflict_log` #1 | **ANCHORED** `E=-cos(α−β)`; locality wording **reject** | **Hold** — no signaling; O5-3 geometry proof **OPEN** |
| 6 | Sequential measurements | `\|W⟩=(\|+x⟩+\|-x⟩)/√2` | §5.7 | **PROVEN** | **Merge** |
| 7 | Stern–Gerlach ↔ coin | `F=μ∂B/∂z`, two spots | §5.5, `aethos_physics.lambda_n_from_sg` | **ANCHORED** qual.; **FIT** `g_E` (O5-2) | **Strengthen** — calibrate `Λ_n` |
| 8 | Observation requires motion | `Δt≥d/c`, Zeno | §5.10, §12 | **ANCHORED** `H_int≠0`; Zeno **MODEL** | **Hold** — do not derive `τ_n` from Zeno |

---

## Block-by-block

### Math Block 1 — Compression operator

| External | Repo | Verdict |
|----------|------|---------|
| `\|ψ⟩=α\|W⟩+β\|B⟩` | §5.4.1 `|χ⟩` | **ANCHORED** |
| Matrix `Ĉ(n̂)` | `σ_n`, Kraus `M_{±,n}` | **ANCHORED** — same object, different name |
| `P(+n̂)=cos²(θ/2)` for `\|W⟩` | §5.5 | **PROVEN** |
| “Comes from coin geometry” | §5.5 + O5-1 | **PARTIAL** — geometry enters via `Λ_n(B,L,g_E)` |

### Math Block 2 — Creates state

| External | Repo | Verdict |
|----------|------|---------|
| `⟨S_z⟩=0` before, `±1` after | §5.4 | **ANCHORED** |
| `ρ` off-diagonal → 0 | §5.4.3 | **ANCHORED** |
| K-S “obvious from mid-transit” | P5-1 + conflict #2 | **MODEL** |

### Math Block 3 — Stern–Gerlach

| External | Repo | Verdict |
|----------|------|---------|
| Dipole force | Standard | **ANCHORED** |
| `F_compress = e v × B` | — | **MODEL** sketch only |
| Two spots = binary pin | §5.5 + channel | **MODEL**, consistent |
| 50/50 from `d/c` symmetry | §5.6 | **MODEL** for `\|+x⟩` only; use `ρ=I/2` in docs |

### Math Block 4 — Sequential

| External | Repo | Verdict |
|----------|------|---------|
| `P(±x\|z up)=1/2` | §5.7 | **PROVEN** |
| `τ_random ~ h/mc²` | Sec 2 `L~λ_C` | **MODEL** timescale |

### Math Block 5 — Bell

| External | Repo | Verdict |
|----------|------|---------|
| `E(a,b)=-cos(a-b)` | §5.8, §6.7 | **ANCHORED** |
| `\|S\|=2√2` | §5.8 CHSH | **ANCHORED** |
| “Ocean nonlocal” | `conflict_log` #1 | **Reject** as physics claim |
| Integral over `θ_A` | §6.12 (new) | **REJECT** sign-half-plane sketch (fails vs QM at π/4); O5-3 **OPEN** |

### Math Block 6 — Motion / Zeno

| External | Repo | Verdict |
|----------|------|---------|
| `Δt_obs=d/c>0` | §5.10 | **ANCHORED** |
| `Δt_min ~ h/mc²` | Sec 2 bounce | **MODEL** |
| `λ_Zeno ∝ (f₀/R)²` | Sec 12 | **ANCHORED** phenomenology |
| `τ_n=879 s` Zeno | Sec 4 FIT | **Separate** — not Sec 5 theorem |

### Math Block 7 — Wigner

| External | Repo | Verdict |
|----------|------|---------|
| Collapse at `t_compress` | P5-1 | **MODEL** + decoherence **ANCHORED** |
| `τ_decohere ~ 10⁻³⁰ s` | Standard scale | **ANCHORED** order |

---

## Master equations crosswalk

| External master | Repo §5.12 | Code (`aethos_physics.py`) |
|-----------------|------------|----------------------------|
| `Ĉ(n̂)` | `σ_n`, `M_{s,n}` | `kraus_m_lambda`, `measurement_pin_probability` |
| `P=cos²(θ/2)` | eq 5 | `spin_projection_probability` |
| `P(up)=1/2` bounce | eq 6 | only if `prep='mixed'` |
| `ρ` before/after | §5.4 | — |
| `E=-cos(a-b)` | eq 8 | `bell_correlation_coin_geometry` |
| `Δt_min=h/mc²` | Sec 2 link | `bounce_period(m_e)` |
| `λ_Zeno` | Sec 12 | not fitted here |

---

## Strengthening checklist (Section 5)

- [x] Crosswalk file (this document)
- [x] `Λ_n` / `g_E` API in `aethos_physics.py` (O5-2 **PARTIAL**)
- [x] O5-3 falsifier: sign-half-plane ≠ QM (§6.12, `test_aethos_physics.py`)
- [x] Tests `test_aethos_physics.py` (T5-1, T5-3 sanity)
- [ ] Full `Γ_n(κ, geometry)` from material constants (O5-1 finish)
- [ ] Analytic O5-3 proof without `sign(cos)` import (still **OPEN**)

---

## Related files

- `derivations/section_05_derivations.md`
- `derivations/section_06_derivations.md` §6.12 (Bell geometry)
- `derivations/calibration_sheet.md` — **F-gE**, **F-LambdaN**
- `derivations/conflict_log.md` — #1 locality, #2 collapse
- `derivations/external_math_index.md` — index for Sections 1–5
