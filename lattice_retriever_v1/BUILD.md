# Lattice Retriever v1 — gated build

**Read [`PRINCIPLES.md`](PRINCIPLES.md) first.** One main branch + transgressor meets + 32-way rotation — not BM25, not rerankers, not “32 separate formulas.”

This directory is a **clean assembly** of the 8-stage pipeline in `docs/prime_lattice_pipeline.html`.
It does **not** import PAM v27, `MultiCorpusBrain`, rerankers, embeddings, or BM25 until Stage 08 explicitly allows a bounded lexical score.

## Rules (hold the line)

1. **One stage at a time.** Do not start stage N+1 until stage N gate passes.
2. **No rerankers** until Stage 08. Stages 01–07 produce addresses, pools, and touch weights only.
3. **No BM25 / dense vectors** until Stage 08 — and then only on the **bounded pool** from Stage 08 step 1.
4. **Allowed imports:** `aethos_words`, `aethos_promotion`, `aethos_core`, `plane3d/*`, `pipeline/bit_01` … `bit_04` as each stage unlocks. See `deny.py`.
5. **Every stage has a synthetic gate test** — no BEIR until Stage 08 integration test.
6. **Glass box:** every stage returns inspectable reasons (JSON/dict), not a single float.

## Stage map

| Stage | Module | Existing primitive (reference only) | Gate test |
|-------|--------|-------------------------------------|-----------|
| **01** | `stage01_symbols.py` | `letter_to_prime`, prime order | `tests/test_stage01.py` |
| **02** | `stage02_intersections.py` | promotion pair meet, `number_intersection` | `tests/test_stage02.py` |
| **03** | `stage03_rotation.py` | `bit_01` word cell, 32 wings | `tests/test_stage03.py` |
| **04** | `stage04_promote.py` | `PromotionRegistry`, L2 promote | `tests/test_stage04.py` |
| **05** | `stage05_free_token.py` | `min(p,q)*max(p,q)`, `kappa_pair_meet` | `tests/test_stage05.py` |
| **06** | `stage06_composites.py` | multi-prime chain, compound keys | `tests/test_stage06.py` |
| **07** | `stage07_semantic_light.py` | rare-neighbor 3-way, idf-weighted | `tests/test_stage07.py` |
| **08** | `stage08_retrieve.py` | lazy branch + touch lift + **bounded** score | `tests/test_stage08.py` |

## Stage gates (pass/fail)

### Stage 01 — Symbols → primes

**In:** raw string  
**Out:** ordered tuple of letter-primes per symbol; prime 2 = first letter prime in chain  

**Pass when:**
- `a→3, b→5, …` deterministic (first 26 odd primes; **2 reserved** for hub axis)
- Same string → same prime sequence every time
- `tas` and `sat` → **different** prime order (anagram separation at symbol level)

```bash
python -m pytest lattice_retriever_v1/tests/test_stage01.py -q
```

### Stage 02 — Intersections → subwords (32-lattice nodes)

**Spec:** AETHOS φ-Prime Lattice — `LatticeBank32` (4 branches × 8 vectors = 32 lattices).
Reference: `AETHOS_Complete_Technical_Specification_FINAL.pdf`, `aethos_lattice.py`.

**In:** prime pair or triple  
**Out:** `lattice_signature` — 32 `(X,Y,Z)` coords at transgressor `n`; `composite` = P×Q for pairs

**Pass when:**
- `73+23` and `69+27` both sum to **96** but **0/32** wing coords match
- `intersect(t,h)` stable (same 32-coord signature every call)
- 3-way `ing` ≠ 2-way `in` on lattice signature

**NOT the node:** sum-only anchor. **IS the node:** all 32 lattice coordinates.

### Stage 03 — Frequency → rotation / quadrant

**In:** token + corpus doc-frequency  
**Out:** wing/quadrant id (1–32) + SpacetimeCell

**Pass when:**
- Same letters, **different df** → different quadrant (task vs sate fixture)
- Rotation is stable (same input → same quadrant)

### Stage 04 — Promote frequent subwords

**In:** stream of text, promotion threshold  
**Out:** registry with new L2 primes for `ing`, `th`, etc.

**Pass when:**
- After enough `…ing` occurrences, `ing` has its own prime
- Old intersections still resolve (no retrain — append only)
- **Distinct parents, not raw spam:** high count in **one** parent word does **not** promote (e.g. `running`×6 → `ing` stays intersection-only; `thethethe`×N → `th` not promoted)
- **Deterministic replay:** `promote_from_stream(same_corpus)` twice → identical promotion records (text, prime, order)
- **Idempotent re-ingest:** second `observe_stream(same_corpus)` on one registry → no new primes, no reorder

### Stage 05 — Free token = P×Q

**In:** two primes (promoted or letter) + optional quadrant (Stage 03)  
**Out:** composite meet identity **without** registry row

**Order policy (locked):**
- `meet_composite = min(P,Q) × max(P,Q)` — order-free FTA identity (inverted-index key)
- `invoke_order` + `quadrant` (1–32) — order re-enters **outside** the product
- Do **not** bake rotation into the integer product (breaks factor-back to {P,Q})

**Pass when:**
- `meet_composite(p,q) == meet_composite(q,p)` always
- `meet_composite(p,q) != meet_composite(q,r)` for distinct pairs
- **Factor-back:** `factor_pair_composite(P×Q) → {P,Q}` exactly (semiprime only)
- **Corridor regeneration:** address from primes → discard → re-address → bit-identical lattice + corridor_key
- **Composite-only regeneration:** factor product + quadrant → same address (no cache row)
- No vocabulary table required

### Stage 06 — Multi-way composites (Section 5)

**Structure:** anchor pair `a ≤ p` + transgressor `n` — not a flat set, not free nesting.
Cases 1/2/3 = where `n` falls relative to `a`, `p` (`prime_pair_case`).

**Transgressor rule (locked):**
- **Read-order default:** first two symbols → anchors `(a≤p)`, third symbol → `n`
- **Pool prime present:** promoted pool prime is **always** `n`; the other two → `a≤p`
  (Stage 04 promotions are first-class Section 5 atoms)

**Identity (inherits Stage 05):**
- `meet_composite = ∏ distinct primes` — order-free; factor-back recovers exactly `k` distinct factors
- `anchor_sum = a+p+n` — placement metadata only (th/ri degeneracy); **never identity**

**Repeated primes (locked):**
- k-way product requires **k distinct** primes; repeats → `RepeatedPrimeError`
- Explicit fallback: 2-way meet on min/max **distinct** pair (Stage 02 route)

**Pass when:**
- `ing` = {i,n,g}: read-order roles → Case 1, VA1A coord `(76,29,95)`, product `19×29×47`
- `factor_k_composite(product, 3)` → exactly `{19,29,47}`; rejects squared factors
- Corridor regeneration from composite + read_order → bit-identical
- `thing` decomposes from letter product; `thinking` from promoted `th`+`ing` when registry has them
- `all` / `ll` repeats routed to pair fallback, not k-way

### Stage 07 — Semantic light-up (+ L4–L6 wing cage)

**Promotion ladder (free tokens — formula-built, no vocab rows):**
```
L1 symbol → letter prime
L2 2-way meet → th, ing (pool prime)
L2 3-way meet → hing, ing (pool prime on Section 5 triple)
L3 compound → th×ing, t×hing, symbol×subword×word (products stack)
```

**Disambiguation:** `thing` ≠ `th+ing` ≠ `t+hing` — different products even when letters overlap.

**Wing cage at each 3-way anchor:**
- Inverted index: `anchor_composite → WingCage` (small)
- L4/L5/L6 = dim4/dim5/dim6 + rotation quadrant 1–4 per correlated term
- 2-way strength = co-occurrence count → drift_weight (stays close if frequent)
- Sliding 3-way windows up to **6 tokens** per doc observation

**Pass when:**
- `th`, `ing`, `hing` promoted separately; `thing` paths all distinct composites
- `cat`/`pet`/`purr` cage lights on rare query terms; `the` hub ≈ 0
- Hub contribution &lt; 5% of touch weight
- Cage adds correlations without changing base promoted prime (lazy lift)

### Stage 08 — Retrieve (lazy branch + corridor tokens)

**Two-layer gate (do not conflate):**
1. **Synthetic** (`test_stage08.py`) — lazy loop, glass-box trace, miss policy, hub-safe routing
2. **SciFact** (`test_stage08_scifact.py` + `scripts/bench_lattice_retriever_v1.py`) — honest baseline, no tuning

**Miss policy (locked — zero-recall / Course 1 §7):**
1. Primary: rarest selective pin intersect (min-df pin per term, not union of pins)
2. Widen: rarest single-term union if intersect empty
3. **FTA letter fallback:** L1 letter-prime postings for OOV query terms
4. **Empty:** if still no corridor — trace explains miss (`MissPolicy.EMPTY` skips step 3)

**Pin selectivity (narrow vs lift — locked):**
- Each corridor pin carries **pin-level document frequency** (`pin_doc_freq`)
- **Rarest-filter narrows only on selective pins:** whole-word identity always qualifies; promoted subword pins qualify only if pin df ≤ selectivity threshold (~5% corpus)
- **Hub-class pins** (high pin df) stay in the index for drift-layer lift — they never narrow
- **Standalone hub terms** (high term df) skip narrowing but are not deleted; compounds carry their own identity pin and df
- Route on **min-df selective pin**, never union of all pins
- Adjacent query bigrams may route on **compound pins** (own df) before standalone hub terms

**Glass-box trace (`retrieve_with_trace`):**
- Query prime factors per term
- Route mode + filter steps
- Cages considered + rare dots (drift-weight, L4–L6, quadrant)
- Per-hit reasons (corridor keys, wing-cage lift)

**SciFact starting line (honest, no tuning):** record before touching weights.
- Pure-lattice reference ~0.367 nDCG@10; BM25-hybrid ~0.776 — first clean number may be below both
- Day-one pass: geometry retrieves above chance + trace explains ranks

**Synthetic pass when:**
- Rarest prime filter reduces pool (log steps)
- No full corpus scan
- Lazy corridor keys (Stage 05)
- 8/10 fixture top-1
- OOV fixture defines FTA fallback; empty policy returns trace-only miss
- `retrieve_with_trace().explain()` complete

**SciFact pass when:** bench run recorded; pool recall > chance on smoke subset

```bash
python -m pytest lattice_retriever_v1/tests/ -q
```

## Integration (after Stage 08 passes)

Only then:
- Run on SciFact with same harness as `scripts/bench_supervised_bridges.py`
- Compare to κ-primary baseline — expect pool recall first, nDCG second
- Add glass-box rules from audits — **not** new rerankers

## What NOT to do (Cursor drift list)

- Do not import `aethos_multi_corpus`, `eval_beir`, `routed_search`, `bridge_search` before Stage 08
- Do not add `cross_encoder`, `sentence_transformers`, `rank_bm25` at any stage
- Do not "helpfully" merge stages — each file stays one stage
- Do not skip synthetic tests for "real data looks fine"

## Current status

| Stage | Status |
|-------|--------|
| 01 | **implemented** — `stage01_symbols.py` |
| 02 | **implemented** — `stage02_intersections.py` (32-lattice `LatticeBank32`) |
| 03 | **implemented** — `stage03_rotation.py` |
| 04 | **implemented** — `stage04_promote.py` |
| 05 | **implemented** — `stage05_free_token.py` |
| 06 | **implemented** — `stage06_composites.py` |
| 07 | **implemented** — `stage07_semantic_light.py` |
| 08 | **implemented** — `stage08_retrieve.py` |
