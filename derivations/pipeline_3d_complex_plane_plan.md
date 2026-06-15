# 3D Complex Plane Pipeline — Bit-by-Bit Build Plan

**Purpose:** Build accuracy, speed, and compression **one bit at a time** on **your formulas only** — with math worked out, known cons, and mitigations **before** each code step.

**Status:** PLAN (Jun 2026). No bit is “done” until its gate tests pass.

**Companion code (exists today):**

| Layer | Module | Role |
|-------|--------|------|
| Geometry | `aethos_complex_plane.py`, `aethos_lattice.py`, `aethos_recursive.py` | canon + 32 wings |
| Spring ℂ | `aethos_spring_complex.py`, `aethos_complex_rotation.py` | `i_act`, Klein-4 |
| Cell | `aethos_physics.SpacetimeCell` | Ψ = (z, ζ, n) |
| Meets | `aethos_intersection_nodes.py` | witnesses, entangled pairs |
| Attractors | `aethos_attractor_index.py` | (z, ζ) buckets |
| Memory | `aethos_hub_signature.py`, `aethos_notch_encoder.py` | hub + notches |
| Eval | `eval_beir.py`, `diagnose_failures.py` | BEIR harness |

**Rule:** Each **BIT** adds one invariant, one storage field, one benchmark gate. **Never skip gates.**

---

## 0. Native object (already closed — reference)

**State:**

```
Ψ = (z, ζ)   z = X + iY ∈ ℂ,   ζ ∈ ℝ
α = (A, b, w, n)   chain, branch VA1..4, wing 1..8, transgressor
```

**Pipeline:**

```
token → prime chain A
(A, b, w, n) ──canon_on_chain + apply_vector──► (X,Y,ζ) ──► z, ζ
```

**32 chambers:** one canonical `(X,Y,ζ)` per (A,b,n), then wing `w` only transforms.

**Tags:** PRIMITIVE/READOUT **DERIVED**; physics reading **MODEL**.

---

## Pipeline overview (12 bits)

```
BIT 0  Layer-0 certificate (n+ni, |z|²=2n²)
BIT 1  Word → SpacetimeCell at anchor n
BIT 2  Quantized attractor key κ(z,ζ)
BIT 3  Doc attractor set (multi-hub)
BIT 4  Attractor candidate router (pre-BM25)
BIT 5  |z| band template (4 bands × hub)
BIT 6  Notch fingerprint bind to cell
BIT 7  Meet witness index (solo + promotion)
BIT 8  Path fiber bit (same z, different chain)
BIT 9  Query cell profile (multi-word Ψ)
BIT 10 Scoring fusion (BM25 gate + signals)
BIT 11 Compression ledger (bytes + lossless check)
```

Each bit below: **Math → Storage → Algo → Con → Mitigation → Gate test**.

---

## BIT 0 — Layer-0 certificate

**Math (PROVEN):**

```
|A| = 0  ⇒  z₀(n) = n + ni,  ζ₀(n) = n
|z₀(n)|² = 2n²
```

**Role:** Sanity anchor — all coords must be **consistent with** Pythagorean factor 2 at origin, not arbitrary floats.

**Storage:** none (runtime check only).

**Con:** Integer coords at large n overflow float32 in wire format.  
**Mitigation:** Store hub coords as **float32** or **int16** with documented range; reject |coord| > 2¹⁵ in pack.

**Gate:** `tests/test_complex_plane.py` + assert `imaginary_start(n).modulus_squared == 2*n*n` for n ∈ {1,3,7,100}.

---

## BIT 1 — Word → SpacetimeCell

**Math:**

```
chain(w) = parent_primes(w) sorted unique  (or (prime(w),) if solo)
Ψ(w) = wing_transform(VA1, chain(w), n₀, wing=1)
cell(w) = SpacetimeCell.from_psi(Ψ, n₀, chain, branch, wing)
```

Default `n₀ = 7` (match hub `anchor_n` today).

**Storage (per hub word, extends HubEntry):**

| Field | Bytes | Meaning |
|-------|-------|---------|
| `z_re`, `z_im` | 4+4 | spring (or derive from coord) |
| `zeta` | 4 | depth |
| `n` | 2 | rail at evaluation |
| `band` | 1 | \|z\| band id 0..3 (BIT 5) |

**Con:** `coord (x,y,z)` from registry may **not** carry chain metadata — cell rebuild can pick wrong A.  
**Mitigation:** Always use `chain_for_word(registry, w)` + `lattice_composite(w)` for chain, never coord alone.

**Con:** Single wing (L01) misses 32-wing consensus.  
**Mitigation:** BIT 1 uses VA1/w1 for **primary key**; BIT 5 adds consensus bands.

**Gate:** For 100 registry words, `cell(w).z` matches `complex(*coord[:2])` at same n₀, wing=1.

---

## BIT 2 — Quantized attractor key

**Math:**

```
κ(z, ζ; q) = (round(Re z / q), round(Im z / q), round(ζ / q))  ∈ ℤ³
```

Default `q = 1` (integer spring coords at most anchors).

**Neighborhood (query):**

```
N(κ; r) = { κ' : ||κ' − κ||_∞ ≤ r }
```

**Complexity:** O(1) hash lookup; O((2r+1)³) neighbor buckets.

**Con:** **Bucket collision** — many unrelated words share κ.  
**Mitigation:** BIT 3 stores **set of keys per doc**; BIT 7 adds meet prime factor; BIT 10 BM25 gate.

**Con:** Quantize too coarse → false merges.  
**Mitigation:** Calibrate q on dev set: minimize collision rate at fixed recall.

**Gate:** Triple witness `(3,5,7)` → κ = (12, 5, 15); query radius r=0 hits only docs with that hub.

---

## BIT 3 — Doc attractor set

**Math:**

```
K(doc) = { κ(cell(w)) : w ∈ hubs(doc), strength(w) ≥ τ }
```

Store inverted index: `κ → [doc_id]`.

**Storage:**

| Structure | Size (order) |
|-----------|----------------|
| Per doc | \|K(doc)\| × 12 B (3×int32 key) |
| Inverted | Σ_docs \|K(doc)\| × (key + doc_id) |

With K=12 hubs, **~144 B/doc** attractor keys (upper bound).

**Con:** Hubs ⊂ doc — miss body tokens.  
**Mitigation:** Keep BM25 on full `doc_tokens`; attractors only **route candidates**.

**Con:** Duplicate κ in one doc.  
**Mitigation:** Store **unique** κ per doc; keep **max strength** word as witness label.

**Gate:** Build index from `build_all_hub_signatures`; random 50 docs — each doc retrievable by its top hub κ.

---

## BIT 4 — Attractor candidate router

**Math:**

```
C(q) = ⋃_{w ∈ query} N(κ(cell(w)), r)
```

**Pipeline position:**

```
query → C(q)  (small candidate set)
      → rank_with_hub_signatures(C(q))  (existing)
```

**Complexity:** O(|Q|·(2r+1)³ + |C(q)|·K) vs O(|AllDocs|·K).

**Con:** Zero lexical overlap queries — C(q) empty.  
**Mitigation:** **Fallback:** if |C(q)| < M_min, union MeetIndex (BIT 7) or full corpus BM25 top-N.

**Con:** Radius r too small → miss relevant docs (near attractor, different κ).  
**Mitigation:** r=1 default; tune on BEIR nDCG@10.

**Gate:** On SciFact dev, compare recall@100 of C(q) vs full corpus — target ≥ 90% of full recall before enabling in prod path.

---

## BIT 5 — |z| band template (compression + consensus)

**Math:**

At fixed (A, n), branch fan gives **4 distinct |z|** values (w=1):

```
band(w) = argmin_b | |z_b| − |z| |   b ∈ {VA1..VA4}
```

Store **band id** (2 bits) + **optional** VA1+VA2 real sum:

```
z_obs = Re(z_VA1 + z_VA2)   (Im cancels when mirror pair)
```

**Storage savings:** 4 bands × shared template per (A,n) class vs 4 full complex floats.

**Con:** Band collision — different semantics, same |z|.  
**Mitigation:** Band is **pre-filter** only; full coord in hub entry for scoring.

**Gate:** At (3,5,7), n=5: exactly 4 bands, 8 wings per band (verified).

---

## BIT 6 — Notch fingerprint bound to cell

**Math:**

```
M(w) = { conj(z_i)·z_j : i,j ∈ {VA1..VA4} }
Store top-K peaks by |M_ij|, K=10
Pack: 10 × 10 B = 100 B (aethos_notch_encoder)
```

**Bind rule:**

```
notch doc fingerprint ↔ top hub word w* with same κ(cell(w*))
```

**Con:** Cross-branch peaks ambiguous without branch indices.  
**Mitigation:** Notch already stores `(branch_a, branch_b)` in 10 B peak.

**Con:** Notch similarity ignores ζ.  
**Mitigation:** Require **pool_factor Jaccard ≥ θ** before notch score (existing `doc_notch_score`).

**Gate:** Same doc, two builds → identical notch bytes; similar docs Jaccard > random docs.

---

## BIT 7 — Meet witness index

**Math (solo swap):**

```
meet(p, q)  ⇔  solo(p)@n=q  coord = solo(q)@n=p   (all 32 wings)
```

**Index:**

```
PrimeFactor p → { doc_id : p ∈ pool_factors(doc) }
Query: factors(q_words) → union docs → rank by witness κ
```

**Promotion (triple):**

```
(3,5)@7 = (3,7)@5 = (5,7)@3  →  κ* = (12,5,15)
```

**Con:** Meet index **blows up** on dense corpora (high factor overlap).  
**Mitigation:** Cap candidates per factor; require **≥2 factor hit** or **κ match** for promotion.

**Con:** False meet from float coord noise.  
**Mitigation:** Integer canon coords only in meet probe; use `wing_transform` not cached float drift.

**Gate:** `probe_solo_swap(3,5)` activates; `triple_equalization` single κ; eval recall vs MeetIndex-only routing.

---

## BIT 8 — Path fiber bit (entanglement without collapse)

**Math:**

Same κ, different witness:

```
(z_a, ζ_a) = (z_b, ζ_b)  but  (A_a, n_a) ≠ (A_b, n_b)
```

Store **witness label**:

```
fiber(doc, κ) = (chain_prefix, n_witness, meet_kind)
```

**Scoring:** boost docs sharing κ **and** compatible fiber; do **not** merge states.

**Con:** Storage explosion if all fibers kept.  
**Mitigation:** Keep **top-1 witness per κ per doc** only.

**Gate:** `find_entangled_meet_pairs` on seed network → count pairs; index stores distinct fibers.

---

## BIT 9 — Query SpacetimeCell profile

**Math:**

```
Query profile Q:
  words → { cell(w_i) }
  κ_q = { κ(cell(w_i)) }
  z_obs_q = Σ_i IDF(w_i) · Re(z_VA1 + z_VA2)_i   (optional vector)
```

**Con:** Multi-word query κ spread — no single bucket.  
**Mitigation:** Jaccard on κ sets: `|κ_q ∩ K(doc)| / |κ_q ∪ K(doc)|` (already in `CorpusAttractorIndex.score_doc_overlap`).

**Gate:** Two-word query with shared factor → overlap score > unrelated doc.

---

## BIT 10 — Scoring fusion (do not break what works)

**Existing signals** (`aethos_hub_signature.score_document`):

| Signal | Role |
|--------|------|
| 1 BM25 | lexical backbone |
| 2 Coord meet (32-wing consensus) | geometric |
| 3 Neighbor | L4–L6 expansion |
| 4 Morph meet | letter GCD |
| 5b Pool Jaccard | composite overlap |

**Add (one at a time):**

| Signal | Formula | λ |
|--------|---------|---|
| **8a** Attractor Jaccard | `\|κ_q ∩ K(doc)\| / \|κ_q ∪ K(doc)\|` | λ_κ (start 0.3) |
| **8b** Notch sim × Jaccard | `notch_sim · J_pf` | λ_n (start 0.2) |
| **8c** Meet witness hit | +1 if promotion κ* ∈ K(doc) | λ_m (start 0.4) |

**BM25 gate (keep):** lattice signals off when `BM25 < 0.15 × max_bm25` **unless** query floor (existing).

**Con:** λ tuning overfits one dataset.  
**Mitigation:** Train λ on **train qrels only**; report test separately; save brain (`eval_beir` pattern).

**Gate:** A/B on SciFact: nDCG@10 ≥ baseline hub-only before merging λ to main.

---

## BIT 11 — Compression ledger

**Math (bytes per doc budget):**

| Component | Target |
|-----------|--------|
| Hub entries K=12 | ~240 B (20 B × 12) |
| Attractor keys unique | ~144 B |
| Notch fingerprint | ~120 B |
| Fiber witness | ~16 B |
| **Total lattice layer** | **~520 B/doc** |

**Lossless check:**

```
reconstruct: κ → top hub word → coord → compare to hub.coord
notch: unpack → matrix peaks → within ε of live encode
```

**Con:** “99% compression” on **raw corpus** requires **rule layer** (CompressionEngine) — not from BIT 11 alone.  
**Mitigation:** Report **two ratios:** (A) index bytes vs inverted index, (B) rule+exception vs raw logs.

**Gate:** Random 1000 docs — ledger < 600 B/doc avg; reconstruct coords exact on integers.

---

## Build order (strict)

| Week | Bit | Deliverable | Do NOT start until |
|------|-----|-------------|-------------------|
| 1 | 0–1 | `word_to_cell(registry, w)` + tests | — |
| 1 | 2–3 | `build_attractor_index_from_hubs(sigs)` | BIT 1 gate |
| 2 | 4 | `candidates_from_attractors(q)` in eval path | BIT 3 gate + recall@100 |
| 2 | 5 | band field on HubEntry | BIT 1 |
| 3 | 6 | wire notch to κ | BIT 2 |
| 3 | 7 | meet router integration | BIT 4 fallback tested |
| 4 | 8 | fiber witness optional field | BIT 7 |
| 4 | 9–10 | signal 8a only, then 8b, 8c one at a time | each λ gate |
| 5 | 11 | compression report script | all prior |

**File to create per bit:** `pipeline/bit_XX_<name>.py` + `tests/test_pipeline_bit_XX.py` — **one PR per bit**.

---

## Master con list (honest)

| Con | Severity | Mitigation bit |
|-----|----------|----------------|
| κ collision | High | BM25 gate + multi-key doc + meet |
| Chain metadata loss | High | BIT 1 chain_for_word |
| Attractor misses body terms | Medium | BM25 fallback |
| 32-wing cost | Medium | Consensus 4 wings + bands |
| Meet candidate explosion | Medium | factor caps + 2-hit rule |
| λ overfit | Medium | train/test split + brain |
| ζ ignored in notch | Low | pool Jaccard gate |
| Not GA/Clifford confusion | N/A | document as separate (Part II) |
| Physics overclaim | High | math-first tag; MODEL on readings |

---

## First code task (when you say “build BIT 1”)

```python
# pipeline/bit_01_word_cell.py  (planned)
def word_to_spacetime_cell(registry, word: str, *, n: int = 7) -> SpacetimeCell:
    ...
```

Single test file, single eval hook in `diagnose_failures.py` — **no** full eval_beir change until BIT 4 gate.

---

## Sign-off checklist before “pipeline complete”

- [ ] Each BIT 0–11 gate test green
- [ ] SciFact nDCG@10 ≥ baseline at each signal add
- [ ] Candidate set |C(q)| p50 / p95 logged
- [ ] Bytes/doc ledger ≤ target
- [ ] `derivations/pipeline_3d_complex_plane_plan.md` updated with measured numbers

**This document is the contract.** Code follows bits; bits do not jump ahead of gates.
