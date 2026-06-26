# Lattice Retriever v1 — non-negotiable principles

**Read this before any stage. If a change violates this doc, it is drift — not progress.**

---

## What you built (one sentence)

**One main branch formula + prime-by-prime branching + transgressor `n` → meets at prime crossings → rotate readout into 32 sub-quadrants → infinite collision-free addresses (FTA).**

That is the RAG engine. Not BM25. Not rerankers. Not embedding clouds.

---

## What we use from the full spec (and what we skip)

| Use | Skip (for v1) |
|-----|----------------|
| **VA1 main branch** — `single_prime_canon`, `prime_pair_canon` (cases A/B/C when `n` crosses `a` then `p`) | VA2–VA4 full tables unless needed for rotation readout only |
| **Prime-by-prime branching** — chain `(3, 5, …)`; regime switches when `n` reaches each anchor | Ad-hoc sum-only keys, hash buckets |
| **Transgressor `n`** — the rail; when `n` hits another prime, paths **meet** | Fixed-length vectors, vocab rows |
| **32 sub-quadrants** — **rotation** of the one formula readout (8 vectors × 4 branch rotations = 32 chambers) | 32 unrelated scoring systems |
| **Swap meet** — `bank(a) @ n=p == bank(p) @ n=a` on every wing | Symmetric Jaccard, hub-only bridges |

Reference implementation: `aethos_lattice.py` (`prime_pair_canon`, `LatticeBank32`), `aethos_complex_plane.py` (rotation readout), `pipeline/bit_01_word_cell.py` (4 branch rotation around critical line).

---

## The meet you described (3, 5, n → 11)

Primes **3** and **5** define a **pair bank**. Transgressor **n** walks the rail:

- `n < 3` → case 1  
- `3 ≤ n < 5` → case 2 (crossed first anchor)  
- `n ≥ 5` → case 3 (crossed second anchor)  
- When **n** reaches **11**, you are on a **deeper** segment of the same chain — new meets with any other bank whose `n` aligns (extension / swap witness).

**Semantic relationships = these meets**, not co-occurrence tables alone:

- Two terms share a **rare** meet witness → related (cat / pet through shared rare third meet).  
- Hub meets (`the`, `is`) → almost no weight (glass-box rule: rarest connector wins).

---

## Why this is infinite and collision-free

Every address is **prime factorization** → unique chain → unique trajectory in `(z, ζ)` readout.

- **Septillions of vectors**: new prime or new composite = new address; FTA guarantees no product collision.  
- **No retrain**: append a prime; intersections **refind**; structure does not shift.  
- **Not “similarity”**: placement is **deterministic geometry**, not learned float proximity.

When AI “forgets” this and adds BM25-first or reranker stacks, it collapses the space back to finite vocab + collisions. **That is the failure mode you keep hitting.**

---

## v1 pipeline (stages 01–08) in this language

| Stage | Principle |
|-------|-----------|
| 01 | Symbols → letter-primes (chain alphabet) |
| 02 | **Meet** = `prime_pair_canon(VA1, a, p, n)` → **32 rotated coords** (signature) |
| 03 | Frequency picks **which sub-quadrant / wing** and **which n** on the rail |
| 04 | Frequent meet → **promoted prime** (new anchor, append-only) |
| 05 | Free address = **P×Q** (composite product, no stored row) — **order outside product** |
| 06 | Multi-prime chain → deeper pair-by-pair branching |
| 07 | **Semantic light-up** = shared **meet witnesses** (rare-weighted), not hub co-count |
| 08 | **Lazy branch**: rarest prime narrows pool → meet lift → score **bounded pool only** |

**Forbidden until Stage 08 pass:** `MultiCorpusBrain`, rerankers, dense embeddings, full-corpus BM25.

---

## Glass box (always)

Every retrieval answer must expose:

- which **primes** / **chain**  
- which **n** (transgressor)  
- which **case** (regime A/B/C)  
- which **wing / sub-quadrant** (1–32)  
- which **meet witness** lifted the doc  

If you cannot read it out, it is not this system.

---

## Walking the lattice — one corridor, two imaginary rotations, 32 chambers

**The main corridor is the real number line.** Transgressor **n** walks ℕ — the rail every address rides. Depth **ζ** is the real axis stacked with the chain (segment FSM on anchors). You do not search Cartesian space; you **walk forward on n** and anchors lock when `n` hits each prime.

**X and Y are the imaginary axes.** Spring readout is **z = X + iY** — two imaginary directions you **rotate into**, not a flat keyword grid:
- **S** = swap corridor `(X,Y) → (Y,X)`
- **R_x** = reflect real
- **i_act = R_x ∘ S** = multiply by **i** (quarter-turn on the spring plane)

Layer 0 start: `z₀ = n + ni = n(1+i)` — you begin on the **45° diagonal** where real rail and imaginary phase meet (`imaginary_start(n)`).

**The 3D complex plane = C × R:**
- **2 imaginary axes** in the spring plane (Re z, Im z from X, Y)
- **1 real corridor** (rail **n** + depth **ζ**)

**32 sub-quadrants = rotation slots, not 32 rooms to fill:**
- **4 branches** (VA1–VA4) = 4-way phase fan on C (same segment breaks, different algebra)
- **8 wings** (v1–v8) = flip_x, flip_z, Y↔X corridor swap (VA vs VB families)
- **4 × 8 = 32** = one physical vector, two selectors (branch stick + wing thumb)

You **rotate into** a chamber; you do not copy the formula 32 times.

---

## Hilbert's hotel + barber paradox — geometric resolution

**Hilbert's hotel:** Guests need not shuffle because addresses are **FTA products**, not slot indices. Append a prime → new room at a **new level** on the rail. Infinite occupancy, zero relocation. Same `(z, ζ)` display can host **distinct labels** when path, branch, wing, and trigger history differ (`aethos_complex_plane.py` — "Same node, distinct state").

**Barber / Russell (self-reference):** "The barber shaves everyone who does not shave themselves — does he shave himself?" Same structure as Russell's set of all sets that don't contain themselves. **Resolution in the lattice:**
- **Typed promotion:** every promoted prime is born at **level max(children)+1** — it cannot appear in its own sub-chain (`aethos_recursive_lattice`, Test 1 Russell impossibility).
- **Meet ≠ membership:** co-located coordinates are **witnesses**, not "the set contains itself." The barber lives on the **rail**; the rule lives on a **higher promotion tier** — no self-shave loop is constructible through the public API.

Collision is **literal intersection only**; paradox is **level separation + path label**, not boolean set theory.

Reference: `scripts/test_russell_impossibility.py`, `ONTOLOGY.md` §B.3–B.6.

---

## Hilbert's hotel — nobody moves, everyone has an address

Classic Hilbert's hotel fails because guests must shuffle. **FTA solves it:**

- Every token, subword, composite, and meet witness gets a **unique prime product** — a mathematical address, not a slot index.
- **Append-only:** new prime → new room; existing addresses **never relocate**.
- **Collision only at literal intersection:** two vectors coincide in `(z, ζ)` **iff** they equalize on the same meet rail (swap witness, triple lock, composite chain). There is no accidental float-neighbor collision.
- **Septillions of nodes** are cheap because storage is **lazy:** small inverted index on prime factors + branch-on-demand along the rail. You do not materialize the hotel — you **walk** it.

Reference: `aethos_hilbert.py` (BasisLabel tower), `aethos_append_index.py` (κ inverted index), `RAG_README.md` (append-only prime addressing).

---

## Walking the 3D complex plane (C × R)

Native state is **Ψ = (z, ζ)** with **z = X + iY** (one real + two imaginary readout axes) and **ζ** depth on the transgressor rail **n**:

| Symbol | Role |
|--------|------|
| **n** | Transgressor — 1D rail; when **n** hits anchor **p**, regime switches (case A/B/C) |
| **Branch b** | VA1–VA4 — **4-way rotation** on C (not four unrelated scorers) |
| **Wing w** | 1–8 — corridor on the imaginary axis; **8 × 4 = 32 chambers** |
| **Chain A** | Sorted prime anchors — prime-by-prime branching |

**Rotation is not a learned embedding.** It is **prime order → formula readout → (X,Y,Z) per wing.** Same address always lands on the same node. Reference: `aethos_complex_plane.py`, `pipeline/bit_01_word_cell.py`.

---

## 3-way intersection → birth 4D / 5D / 6D wing cages

At a **triple meet** (three pair rails equalize), the 3D cell is locked. That node may **spawn a higher correlation plane** without moving the base address:

- **L4–L6** = correlation dimensions attached at the meet (`CorrelationLink.dim4`, `dim5`, `dim6` in `aethos_promotion.py`).
- **Wing cage:** everything that **literally intersects** at that meet (and everything that **correlates** through it) is placed on the higher plane — a local sub-hotel keyed by the meet witness, not a global re-index.
- **Stage 07 rule:** semantic light-up = rare-weighted shared neighbors **through** the 3-way cage; hub meets (`the`, `is`) stay near-zero weight.

You can open 4D, 5D, or 6D **at any node** when a 3-way (or deeper) intersection warrants it. Base FTA address unchanged; correlation is an **optional lift**, lazy-evaluated.

Reference: `aethos_intersection_nodes.py` (meet taxonomy, entangled pairs), `aethos_hilbert.py` (`BasisKind.CORRELATION`, L4–L9 tower).

---

## Branching vectors + drift weights

Correlated terms are **branching vectors** from a meet, not cosine neighbors:

- **High co-frequency** → vector stays **close** to the cage (strong `CorrelationLink.strength`, small drift on dim4–dim6).
- **Fade from corpus** → vector **drifts** along learned weights (strength decays; cage still addressable but touch weight drops).
- **Placement can look random** at scale (septillions of dots), but **addressing is never random** — only **which branches you evaluate** is lazy.

Scoring touch: `strength × (dim4 + dim6 + 1)` pattern already used in hub signatures (`aethos_hub_signature.py`); v1 Stage 07 uses **idf-weighted rare witnesses** instead of hub co-count.

---

## Lazy branching + lazy evaluation (minimal tokens)

1. **Small inverted index:** prime (or κ composite) → posting list. Filter by **rarest query prime first** — pool shrinks before any heavy work.
2. **Lazy branch:** walk VA1 rail prime-by-prime; **do not** expand all 32 wings until frequency/readout demands it.
3. **Lazy evaluation:** light nodes only when a meet witness or correlation edge fires; early exit when rarest connector already decides (chamber-mixer v4 pattern — 69% early exit at full scale in capability tests).

**Tokens stay minimal** because rotation = **order of primes on (Re z, Im z, ζ)** — no stored embedding row for composites (Stage 05: **P×Q free address**).

---

## Stage 05 order seam (locked)

`P×Q = Q×P` as an integer — the product is **order-blind meet identity** (`min×max`).

| Field | Role |
|-------|------|
| **meet_composite** | FTA corridor opener — inverted-index key, factor-back to `{P,Q}` |
| **invoke_order** | Symbol sequence when the meet was opened `(P,Q)` vs `(Q,P)` |
| **quadrant** (1–32) | Stage 03 rotation — where you read the meet in the 32 chambers |

**Do not** multiply rotation into the composite (would poison factor-back). tas ≠ sat lives in **invoke_order + quadrant + full prime sequence**, not in `P×Q` alone.

Reference: `lattice_retriever_v1/stage05_free_token.py`.

---

## Stage 06 Section 5 seam (locked)

**Neither set nor nesting:** ordered anchor pair `a≤p` + transgressor `n`.

| Rule | Assignment |
|------|------------|
| Read-order (letters) | symbols 1–2 → `a≤p`, symbol 3 → `n` |
| Pool prime present | pool prime → **always** `n`; other two → `a≤p` |
| Case 1/2/3 | `prime_pair_case(a,p,n)` — placement only |
| Identity | `∏ distinct primes` — not `a+p+n` sum coordinate |
| Repeated letter | reject k-way; route to 2-way distinct pair |

Example `ing` read-order (i=29, n=47, g=19): `a=29, p=47, n=19`, Case 1, product `19×29×47`.

Reference: `lattice_retriever_v1/stage06_composites.py`, `aethos_lattice.prime_pair_canon`.

---

## Promotion ladder — symbols → compounds (free tokens)

**Yes — 3-way intersections promote to new pool primes.** Same Section 5 math whether the prime is a letter (29=`i`) or a promotion (109=`ing`).

```
L1  symbol     → letter prime
L2  2-way meet → th, in, …     (Stage 04)
L2  3-way meet → ing, hing, …  (Stage 04/06 — Section 5 triple)
L3  compound   → th × ing       (Stage 05/06 product — free token)
L3  compound   → t × hing      (different vector from th × ing)
L3  word        → t×h×i×n×g     (letter product — yet another vector)
```

**thing vs th+ing vs t+hing:** each path is a **different product identity**. Promoted subwords are first-class atoms; compounds multiply them without new vocab rows.

**Wing cage (L4–L6) at every 3-way node:**
- Anchor = triple product (inverted-index key)
- Correlated terms branch onto dim4/dim5/dim6 + rotation quadrant 1–4
- 2-way edge strength = how often co-seen → drift (high frequency stays close)
- Up to 6-word windows; sliding triples light cages lazily
- Query intersects lit rare dots — small index, big correlation surface

Reference: `lattice_retriever_v1/stage07_semantic_light.py`.

---

## 32 chambers × corpus prime × doc as 2-way meet

**One vector holds every word** because the 32 sub-quadrants are not 32 vocabularies — they are **32 rotations of one readout**. A doc is not a bag of floats; it is a **2-way intersection** (pair meet of primes) sitting inside one wing chamber. Every token in that doc attaches as a **branch** off the same meet rail; the chamber is the container.

| Level | Address | What is stored |
|-------|---------|----------------|
| **Corpus** | One **pool prime** (append-only) | Origin for all docs in that corpus — infinite corpora = infinite primes, no re-index |
| **Document** | **2-way meet** `(p, q)` → composite `p×q` + 32-lattice signature | Not full token list — **corridor-opening pins** only |
| **Token in doc** | Letter/promoted prime on the doc's rail | Count → **L4 / L5 / L6** correlation strength (how many times it appeared **in this doc**) |
| **Query** | Same primes + rarest-first filter | **Lights** the same dots; shared lit dots = doc similarity |

**4D / 5D / 6D in a doc:** not global embedding dims — **per-doc correlation counts** at the meet:
- dim4 = co-occurrence witness A  
- dim5 = witness B  
- dim6 = witness C  
Weights are **already tied** when the doc was ingested (strength + dim4–dim6). At query time you do not re-learn — you **intersect lit corridors** with query-lit corridors. Docs that light the same rare dots are similar; hub dots barely light (glass box).

Reference: `pair_composite` in `stage02_intersections.py`, `CorrelationLink` in `aethos_promotion.py`, `CriticalLinePin` (8 B wire) in `aethos_hub_wire.py`.

---

## Corridor tokens — origin by the flip, keys built lazily

**Everything has an origin** because VA1 canon + **swap flip** (`bank(a)@n=p ↔ bank(p)@n=a`) + origin offset (`OriginTree`) uniquely determines where you are. You never store `(X,Y,Z)` floats for every token — you store the **token that opens the corridor**:

1. **Opening token** = prime + rail `n` + branch/wing pin (8 bytes in pin-wire mode)  
2. **Formula regenerates** full Ψ = (z, ζ) and all 32 wing readouts on demand  
3. **As you explore**, cache only **corridor keys** you actually walked (`κ` bucket or meet witness) — append-only, lazy

**Two-phase retrieval (never get lost, tokens don't blow up):**

```
Phase A — inverted index     rarest query prime → posting list (small, O(1) probe)
Phase B — open corridors     walk rail, light meets, intersect L4–L6 dots on bounded pool
```

Phase A finds the **room**. Phase B **opens doors** — each door is one corridor token, not a new vocabulary row. Composites (`P×Q`) need no stored row; the formula builds the key when the corridor is opened.

Reference: `kappa_pair_meet` (shared corridor = component-wise min of two κ keys) in `pipeline/bit_02_attractor_key.py`, `materialize_lazy_hub_wings` in `aethos_hub_signature.py`, corridor rerank in `pipeline/bit_12_symbol_plane_index.py`.

---

## Cursor instruction (paste every session)

```
Build lattice_retriever_v1 per BUILD.md + PRINCIPLES.md only.
One main branch (VA1) + prime-by-prime branching + transgressor n meets.
32 = rotation of formula readout, not 32 rerankers.
FTA addressing — infinite placement, no collision except literal meet.
Hilbert hotel: append-only primes, nobody moves.
Corpus = one prime; doc = 2-way meet; store corridor-opening tokens only.
L4-L6 = per-doc correlation counts; query lights same dots → similarity.
Lazy: inverted index first, open corridors second; formula builds keys on walk.
Do NOT import MultiCorpusBrain, BM25-first, or reranker stacks.
```
