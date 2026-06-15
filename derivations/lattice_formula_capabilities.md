# AETHOS 3D complex plane — lattice formula audit

The **lattice formula** generates coordinates on the **3D complex plane** Ψ=(z,ζ). Not the π lattice (see `ONTOLOGY.md`).

Audit of `aethos_lattice`, `aethos_recursive`, `aethos_sequences` (verified by probes, Jun 2026).

## What the four formulas actually are

Only **VA1–VA4** exist. Depth `k` does not add formula families — it adds **segments** (regimes).

For sorted anchors `P = (p₁,…,pₖ)`, transgressor `n`, segment `s = segment_index(P,n)`:

| Segment | Meaning |
|---------|---------|
| `s = 0` | `n < p₁` (before first anchor) |
| `s = i` | `pᵢ ≤ n < pᵢ₊₁` |
| `s = k` | `n ≥ pₖ` (after last anchor) |

**VA1 canonical** (before vector wing):

- `s=0`: `(p₁+pₖ, p₁, Z)`
- `0<s<k`: `(pₖ+n, n, Z)`
- `s=k`: `(pₖ+n, pₖ, Z)`

VA2–VA4 use the same segment breakpoints with different `(X,Y)` algebra (sign flips, `2n` couplings).  
**32 wings** = apply one of 8 vectors (VA or VB Y-swap + sign flips) to that canonical triple.

### Z accumulator (`z_depth`) — underused capability

For `k > 2`:

- **Ends** (`s=0` or `s=k`): `Z = sum(P) + n`
- **Interior** (`0 < s < k`): `Z = sum(P)` — **plateau** (composition lock)

Example `P=(3,5,7,11)`: for `n ∈ {3,…,10}`, `Z = 26` constant while `X,Y` still change every time `n` crosses an anchor.

**Latent use:** interior segments are a **stable “bulk” phase** between anchor crossings; ends are **boundary phases** coupled to `n`. Toggle via `lock_interior=False` in `canon_on_chain` for alternate physics (plateau disappears).

---

## Capabilities we already use

| Capability | Where |
|------------|--------|
| 32 independent wings per bank | `LatticeBank32`, `LatticeBank32K` |
| k velocity boundaries = anchors | `velocity_boundaries()` |
| Solo swap meet `p@n=q ⟺ q@n=p` (all 32 wings) | tests, `swap_meet_solo_all_wings` |
| k=2 matches PDF tables | `verify_matches_spec_k2` |
| Triple compose `(3,5)@7 = (3,5,7)@5` (4 branches) | `try_compose_triple` |
| Countable species (primes, evens, 2^n, sqrt, …) | `aethos_sequences` |
| Origin tree ×3 per node | `aethos_origins` |
| k! permutation side-channel | `aethos_permutation` |
| Odd-prime default chain | `chain_primes` |

---

## Capabilities built-in but not fully exploited

### 1. Regime finite-state machine (k+1 states)

Each bank is an **explicit FSM**: labels `case1..case3` (pair) or segment index (k).  
**Project use:** emit regime on each `n` step for tunneling, measurement, Zeno (section docs) without new math.

```python
lat.regime_label(n)   # per wing
segment_index(P, n)   # global segment
```

### 2. Same-n wing collisions (degeneracy classes)

At fixed `n`, different wings can land on the **same coordinate** (`find_same_n_collisions`).  
Example: `p=5, n=7` → **8** collision groups across 32 wings.

**Latent use:** **equivalence classes of wings** → store one witness, map 8 wings to one address (compression). Or treat as **gauge redundancy** in physics labeling.

### 3. Cross-bank routing table (one meet per wing)

For solo banks `p` and `q`, each wing has (empirically) **one** dominant cross-meet in bounded `n`: the swap `(n_left=q, n_right=p)`.

**32 wings → 32 deterministic routers** between two prime worlds.

**Latent use:** build a **meet index** for networking origins, sharding, or “which n on bank B hits bank A’s address”.

### 4. Pair-bank endpoint swap (scale-independent)

`(a,p) @ n=p'` meets `(a,p') @ n=p` on same wing — verified for `(3,541)` vs `(3,5)`.

**Latent use:** **renormalization** — change UV anchor `541→5` without moving the semantic coordinate at the meet witness.

### 5. Related-pair cross meets

`(3,5)` vs `(3,7)` banks meet at e.g. `(12,5,15)` with `(n₁,n₂)=(7,5)` — not the solo swap pattern.

**Latent use:** **family webs** — words/chains sharing prefix anchor `3` form a predictable meet graph.

### 6. Extension witnesses (shallow ⊂ deep)

`extension_witness(P)` searches `(P[:-1])` vs `P` for coordinate equality.  
**Not** the same as solo swap; `swap_like` hits are rare for `k≥4`.

**Latent use:** justified **inductive chain extension** `P → P∪{p}` only when a witness exists (your “promote depth when lattice forces it”).

### 7. Branch quadrants (4 phases per point)

At fixed `(P,n)`, VA1–VA4 give **4 distinct** canonical triples (related by sign/branch algebra).  
After wings: up to **32** distinct images.

**Latent use:** **polarization / isospin / flavor** without new primes — same anchors, four branch “phases”.

### 8. VA vs VB mirror rooms

VB applies `yxz_to_xyz` before sign flips → **mirror chamber** paired to each VA wing.

**Latent use:** parity doubling, CPT-style twin rooms, entanglement sections.

### 9. Species-local swap meets (not cross-species)

- Evens: `{2}@4 = {4}@2` ✓  
- Powers of 2: `(2,4)@8 = (2,8)@4` ✓  
- Primes vs evens at same `n`: **no** meet in small search window

**Latent use:** each **IntersectionType** is its own geometry; cross-species **bridges** require explicit composed banks, not assumption.

### 10. Non-integer anchors (`sqrt_scaled`, float chains)

`canon_on_chain` supports float anchors and optional `lock_interior`.

**Latent use:** **irrational / continuous** ladder (phonon sea, field scales) without leaving the 4-formula framework.

### 11. Permutation capacity = k! side states

On sorted multiset, **k!** distinguishable order dots via ε-offset spiral.

**Latent use:** order encodes **history / path**; multiset encodes **content** — already in words/codec, applicable to any k-chain project.

### 12. `anchor()` reference points

Each lattice has `anchor()` at `n = p` (pair: at `p` endpoint).

**Latent use:** fixed **rest frames** per wing for comparing perturbations.

### 13. Origin tree meets only use VA1 on parent chain

`OriginTree._expand_from` spawns children from `canon_on_chain(VA1, chain, witness_n)` only.

**Latent use (gap):** spawn **3 children from VA1, VA2, VA3** or from **three different meet witnesses** → richer dimensionless tree.

### 14. Active network addressing

100 nodes × 32 wings × unbounded `n` × origin depth × role-mixed chains.

**Latent use:** **deterministic simulation seeds** for cosmic/planetary scales (section 10).

---

## Structural laws (proved in code/tests)

1. **Four formulas forever** — depth = more segments, not more recurrence types.  
2. **Solo swap meet** — for distinct odd primes `p,q`, all 32 wings: `solo(p)@q = solo(q)@p`.  
3. **Triple lock** — `(p₁,p₂)@p₃ = (p₁,p₂,p₃)@p₂` at witness (compose test).  
4. **Interior Z lock** — for `k≥3`, plateau between first and last anchor crossings.  
5. **k boundaries** — `velocity_boundaries() == list(P)` on every wing in `LatticeBank32K`.

---

## Common misconceptions (avoid)

| Myth | Reality |
|------|---------|
| `solo(p)@q` equals pair `(p,q)@p` | **False** (0/32 wings for 3,5) |
| Any two species meet at same n | **False** (primes vs evens) |
| More anchors = same addresses | Changing `P` changes entire trajectory |
| Shallow/deep always swap at endpoints | Only reliable for **solo k=1** swap; use `extension_witness` for k>2 |

---

## Suggested next implementations

1. **`MeetIndex`** — precompute per-wing `(n_a,n_b)` routers between banks.  
2. **`RegimeTrace`** — stream `(n, segment, regime_label, coord)` for diagnostics.  
3. **`SpeciesBridge`** — explicit composed bank for cross-species (not automatic).  
4. **`OriginSpawnPolicy`** — VA1–VA4 or witness-based child origins.  
5. **`DegeneracyQuotient`** — collapse wings by `find_same_n_collisions`.  
6. **`PlateauDetector`** — flag interior Z segments for “stable vacuum” intervals.

Run live audit: `python aethos_discover.py`
