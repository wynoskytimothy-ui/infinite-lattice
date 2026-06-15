# Why the E-Check Calibration Values Work — Hidden Patterns

**Question:** Why `n=80`, `depth=3`, `f_coin,3=0.405`, `f_coin,4=0.5`?

This is **not** a first-principles proof — it is a decomposition of the formulas already in the repo.

---

## Pattern A — Mass ratio is a **two-factor product**

\[
R_{pe} = \underbrace{\frac{\pi^2}{8}}_{\text{spring shrink } L_p/L_0^{-1}} \times \underbrace{\mathcal{M}_{lat}}_{\text{address cascade}}
\]

| Factor | Source | Value |
|--------|--------|-------|
| `π²/8` | `l_p_min_spring`: `L_p = (8/π²)L_0` | ~1.234 |
| `M_lat` | `m_lat_from_active_network()` | E-check ~1497 |
| Product | `r_pe_model_reference_bootstrap()` | ~1847 (0.6% vs CODATA) |

**Hidden reciprocal:** length uses `8/π²`; mass uses `π²/8`. Same geometry, inverted leg.

**E gap name in code:** `lattice_mass_multiplier() = R_pe^E / R_pe^(0) ≈ 1488` — the multiplier the lattice must carry between spring-only fusion and measured mass.

---

## Pattern B — `M_lat` is almost **linear in node count**

\[
\mathcal{M}_{lat} = \frac{W_{wing}\sum_i \mu_i}{N_{origins}+N_{branch}+N_{vector}}
= \frac{32\sum_i \mu_i}{40+4+8}
= \frac{32}{52}\sum_i \mu_i
\]

For `origin_max_depth=3` (fixed in E-check):

| Quantity | Value | Why fixed |
|----------|-------|-----------|
| `N_origins` | **40** | Origin tree: `1+3+9+27` (3-ary meet tree to depth 3) |
| `N_branch + N_vector` | **12** | Topology constants in `aethos_active` (4+8) |
| `W_wing` | **32** | `8 vectors × 4 branches` per room |

So for depth 3, **only** `count` (number of active nodes) moves `M_lat` — roughly linearly.

Sweep (primes, depth=3):

| `count` | `Σμ` | `M_lat` | `R_pe^pred` | err% |
|---------|------|---------|-------------|------|
| 75 | 2352 | 1447 | 1786 | 2.8% |
| **80** | **2433** | **1497** | **1847** | **0.6%** |
| 85 | 2744 | 1689 | 2083 | 13.5% |
| 100 | 2809 | 1981 | 2444 | 33.1% |

Target: `Σμ ≈ 1488×52/32 ≈ 2419`. At `n=80`, `Σμ=2433`.

---

## Pattern C — **80 = 16 × 5** role ledger closure

`ActiveNetwork100._assign_role` cycles `i % 5`:

| `i%5` | Role | Chain length |
|-------|------|--------------|
| 0 | SOLO | 1 |
| 1 | PAIR | 2 |
| 2 | TRIPLE | 3 |
| 3 | K_CHAIN | 3–6 |
| 4 | FOUR_WAY | 2–4 |

When `count` is a multiple of **5**, each role appears equally (balanced ledger).

- `n=80` → **16** of each role (perfect balance)
- `n=100` → 20 of each (still balanced but **overshoots** mass because longer K_CHAIN tails add weight)

**Interpretation (MODEL):** proton mass may correlate with a **complete 16-fold repetition** of the 5-role address cycle on the 40-origin cosmic tree — not yet derived, but this is why `80` beats `100` without being an arbitrary fit to 1836 alone.

Also: `80/40 = 2` active nodes per origin room on average.

---

## Pattern G — **1280 wing slots → 80 active = 1/16**

`OriginTree.lattice_count_estimate()` at depth 3:

\[
N_{wings} = N_{origins}\times 32 = 40\times 32 = 1280
\]

Each origin **room** has 32 wings (`8 vectors × 4 branches`).

E-check proton bootstrap: `count=80` on 40 origins:

| Quantity | Value |
|----------|-------|
| Nodes per origin | `80/40 = **2**` |
| Fraction of wings used per origin | `2/32 = **1/16**` |
| Global activation | `80/1280 = **1/16**` |
| Role cycles | `80/5 = **16**` |

**Hidden pattern:** the mass-ratio E-check sits at **one sixteenth** of the cosmic wing address space — the same **16** as the balanced role-ledger repetition count.

Code: `wing_activation_analysis()` in `aethos_physics.py`.

**MODEL interpretation:** proton mass may load **2 wing-addresses per origin room** (or 1/16 of total wing budget) — not yet derived from fusion geometry alone.

---

## Pattern D — He ratio is **mass vs structure cancellation**

At matched temperature, thermal speed cancels:

\[
\frac{\Lambda_{3He}}{\Lambda_{4He}}
= \frac{f_{coin,3}}{f_{coin,4}}\cdot\frac{m_{4He}}{m_{3He}}
\]

| Term | Value | Effect |
|------|-------|--------|
| `m_4/m_3` | **1.327** | Mass-only → 32.7% split |
| SM expectation | ~**<1%** | Structure nearly cancels mass |
| Target band | **1.05–1.10** | Net ~5–10% after partial cancellation |
| Required `f_3/f_4` | `1.075/1.327 ≈ **0.81**` | Structure cancels ~19% of mass bias |

E-check pair `f_3=0.405`, `f_4=0.5` → `f_3/f_4=0.81` → ratio **1.075**.

**Hidden pattern:** the discriminator is not "isotopes differ" (mass alone predicts 33%) — it is whether **coin-coupling structure** nearly cancels mass leaving a **residual 5–10%**. That is the lattice story vs SM.

Deriving `0.405` from isotope electron shell microphysics — **OPEN** (O8 `σ_{e,i}`, `f_{coin,i}`).

---

## Pattern E — Same spine as Chapter 17

| Calibration | Pattern spine |
|-------------|---------------|
| `M_lat` | **Address descent** (Pattern 5) — π spring factor × cascade |
| `f_coin` | **Partner/compression** (Pattern 3) — internal structure vs mass |
| `n=80` | **5-role + 40-origin** topology — C6 active network |

---

## Pattern F — Material blob path (C6)

`m_lat_from_material_blob(ElectronBlob)` varies species per node from density/coupling.

At `n=80`, `depth=3`, uniform-prime bootstrap remains the best R_pe match in sweep; blob mixes species and shifts `Σμ` — run `scripts/pattern_why_discriminators.py` §E.

**Status:** material selection for proton fusion anchor set — **OPEN** (which blob is the proton sea?).

---

## What is still NOT explained

1. Why nature selects `count=80` not `count=79` (no role balance).
2. Why `f_coin,3=0.405` from helium electron structure (only ratio constraint `0.81`).
3. Which `ElectronBlob` (if any) is the proton-fused sea — blob bootstrap does not beat uniform primes yet.

---

## Reproduce

```powershell
python scripts/pattern_why_discriminators.py
```

---

*v1.0 — 2026-06-05*
