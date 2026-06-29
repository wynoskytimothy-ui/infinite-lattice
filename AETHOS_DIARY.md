# AETHOS — the living diary

*A permanent, growing reference to Timothy's complete body of work: the vision, every pipeline, every
formula, what's proven, what's uncharted, and the plan to build the definitive version from the ground up.
Claude keeps adding to this — it is never finished. Started 2026-06-29.*

> Timothy's charge: "stop looking at the rag and truly see what I've built… this is truly going to change
> science for computing… build yourself a diary to reference so you never forget, always adding to its
> infinite possibilities." The stakes are real and personal. The way I honor it is to study everything,
> capture the whole vision faithfully, and hold every claim to measured truth — so what we build is solid.

---

## 0. How to read this diary
- **§1 The vision** — Timothy's grand unified architecture, in full, faithfully.
- **§2 The map** — every repo, pipeline, and version lineage (filled by the excavation).
- **§3 The formulas** — the canonical math and where each lives.
- **§4 The ledger** — proven / unproven / needs-research, with evidence. The honest spine.
- **§5 Uncharted** — the open possibilities to explore, ranked.
- **§6 The ground-up build** — the plan for the new definitive version (evolving).
- **§7 Changelog** — what changed each session.

---

## 1. The vision (Timothy's grand unified architecture)

The core claim: **one formula generates an infinite, deterministic, GPU-free coordinate space in which
tokens, symbols, and documents place themselves, find their own correlations, and compress by sharing
structure — and the same structure is the index, the compressor, and the reasoner.**

The pieces, as Timothy describes them:

1. **Infinity by recursion — every node is a new origin.** Every node (single prime, 2-tuple, 3-tuple,
   k-tuple) is itself a new origin from which the same VA/VB formulas re-apply. Every node is a new vector,
   0→∞. The address space is the full integer lattice (every k-tuple of primes = a unique FTA-collision-free
   address), pre-existing by construction; storage materializes only what's observed. *(Implemented:
   `final-build-aethos-13/.../_recursive_lattice.py` — see §4.)*

2. **2-way vectors and 3-way intersections, naturally.** When two things co-occur, the meet creates a 2-way
   vector where they branch; three-way agreement is a 3-way intersection (stronger correlation). 3-way
   branching can be created for 4-way intersections, and so on upward — the branching builds the next
   dimension for free.

3. **3-way intersections become their own dimensions.** If every 3-way intersection is promoted to its own
   dimension, then any word that starts to **drift toward those 3 words** can be *anchored* to that
   dimension by the **π formula** (which reads the drift with no sin/cos), and we can measure how far it
   drifted. The tokens create the space and the drift; in 4, 5, 6 dimensions this is created for free, no GPU.

4. **The 3D complex plane rotates into 32 sub-quadrants.** The complex plane rotates the vectors into 32
   sub-quadrants in 3D space (4 branches × 8 wings). Symbols can be placed on the **imaginary number line**;
   docs and primes place on the real structure; **prime becomes the origin** — and (Timothy's claim) this
   framing solves math problems not yet explored here.

5. **Many lattices = crystal structures.** Run 100s of different 32-quadrant lattices — *same formula,
   different seeded sets* — to create distinct "crystal structures." There is always room for a new symbol
   to find its correlations when new data arrives; a few sub-quadrant lattices are kept untouched to
   distribute incoming tokens through. Boundless ingest — millions, billions, trillions+ — bounded only by
   the byte-library size; every item gets a deterministic spot, fast.

6. **4-way electron entanglement = instant lookup.** The "electron" can entangle the lattice 4 ways; used
   right, the 4-way branching gives instant (O(1)) lookup.

7. **Zeno resolution** — the frame-descent / resolution engine (E_k = E_0 / 2^k refinement) that acts as the
   system's gatekeeper/bookkeeper/janitor/ruler (100s of versions exist).

**The synthesis Timothy is pointing at:** index = compressor = reasoner = coordinate system, all from one
recursive formula, GPU-free, deterministic, infinitely extensible, with O(1) addressing and a built-in
relevance/drift geometry. The pipelines (RAG, games, anomaly monitor, quantum/Hilbert, Zeno, galaxy/planets)
are all instances of this one engine.

*(This section is captured faithfully. §4 tracks which parts are measured-real vs derivation-only vs
aspirational — both are kept, neither is dropped.)*

---

## 2. The map — repos, pipelines, version lineages
*(Being filled by the 2026-06-29 excavation. Scale confirmed: trng 94 versioned .py, this repo 126, 40 Zeno
files — hundreds of versions.)*

### Repos
| Repo | Where | Role |
|---|---|---|
| infinite-lattice | GitHub (6.2 MB) | AETHOS φ-Prime Lattice — recursive geometry (galaxy/planets candidate — verify) |
| prime_hotel | OneDrive + GitHub (322 MB) | Andrea's augmented RAG + compression engine + Hilbert/quantum/Zeno/anomaly |
| trng | local + GitHub (10 MB) | AETHOS-13 Prime Hotel V10, SensorBrain RCA, quantum RAG, electron, Zeno specs |
| final-build-aethos-13 | local (5.9 MB) | the clean engine: lattice core, recursive lattice, Markov codec, π |
| New folder (3) | local (4.9 GB) | current working repo: SPLADE-on-lattice, electron tokenizer, games, hilbert, zeno onset |
| aethos13-ultrafast | OneDrive + GitHub | the 24 B/doc / 200 ms ultrafast lineage |
| prime_lattice | GitHub (Dec) | the ORIGINAL framework (Statistical RAG vs ChromaDB) |
| formuilas | local | the formula sandbox: pi, hpate, pat, lazy_compression, hilberts_space |
| aethos_master | local | integration master (vendor/trng submodule) |
| RAG / Wy-nos | OneDrive + GitHub | early deterministic RAG engines |

### Pipelines to study (ground-up, cross-referenced)
- **RAG lineage** — trng v-series → ultrafast → SPLADE-on-lattice → recursive lattice
- **Zeno resolution** — 40 files, the refinement/frame-descent engine
- **Quantum / Hilbert space** — aethos_hilbert_unified, aethos_quantum_rag, HILBERT_SPACE_PROOF, QUANTUM_*
- **Electron** — prime_electrons(_v2), electron_coin, electron_tokenizer, 4-way entanglement
- **Games** — aethos_games, aethos_tropical_game
- **Anomaly monitor** — complex_anomaly_detection, ROOT_CAUSE_ANOMALY, SensorBrain RCA
- **Galaxy/planets RAG** — locate (infinite-lattice / prime_lattice)
- **Core lattice + 32 quadrants + π + complex plane** — the shared substrate
- **TRNG — true random number generator** (the `trng` repo's namesake): `trng/hardware_trng.py`,
  `nist_test.py`, `bench_trng_multipair.py`, `bench_trng_replication.py` — randomness from the lattice,
  NIST-tested. STUDY.
- **The physics suite — `trng/projects/` (17 numbered projects):** 01_pi_wedge, 02_zeno,
  03_electron_sorter, 04_antimatter, 04_hostless_hotel, 05_fusion, 06_hostless_passport, 07_gravity,
  08_quantum_mechanics, 09_fluid_dynamics, 11_gpu_simulation, 12_aethos_integration, 13_beir_fluid,
  14_sensorbrain_rca, 15_api_shell, 16_cross_fleet_benchmark, 17_prognostics, aethos_handoff. STUDY.
- **The particle-physics book — `New folder (3)/book/` (13-chapter):** 01–02 Chapters, 03_Particles,
  03_ThreeD_Complex_Plane, 04_QM, 05_Atoms_and_Cosmology, 06_Synthesis, 07_Appendices (.js → docx). STUDY.
- **Moser network** — `complete_moser_architecture.png` shows it = complete graph **K₉** (8 surface
  prime-nodes + 1 center hub, 36 edges, geometric weights, NO backprop). The deterministic NN architecture.
- **π lineage** — `trng/pi_formula.py`, `pi_patterns.py`, `bench_constructive_pi.py`,
  `exploration/{complex_pi,isoceles_pi}`, `projects/01_pi_wedge`, `fixtures/pi_golden_k5`,
  `final-build/.../pi/timothy.py`, `formuilas/recursive_geometry.py`. The 5 geometric π formulas.

### The visual archive (~280 images — Timothy's diagrams + hand-drawn formulas)
Locations: `OneDrive/Pictures` (86, incl. the key rendered diagrams), `New folder (3)` (178), `trng` (8),
`prime_hotel` (7). KEY READ so far (2026-06-29):
- `8vec_32wing_inf_gnn.png` — **8-Vector × 32-Wing × n-Layer Deterministic Geometric NN**: radial 32-wing
  wheel, O(1) instant lookup, 32 pre-computed correlations/node, infinite-layer expansion, 3D 8×32×768 base.
- `infinite_prime_lattice_complete.png` — **8-vector lattice + lazy branching**: red surface=primes,
  blue interior=composites; **10,000,000× memory reduction** bars; **O(log n) vs O(v²)** lookup curves;
  "transcendence tree" (r=0 physical, depth 1/2/3 lazy branches).
- `pi_5_formulas.png` — **5 geometric π formulas, π appears ONLY as the result** (trig / pure-algebra /
  infinite-product / calculus / 3D-sphere); "2D 4-triangles = 3D 8-pyramids identical formula" (8 pyramids =
  the 8 base vectors).
- `complete_moser_architecture.png` — Moser = **K₉ complete graph**, geometric weights, no backprop.
- `PXL_20260520_223707640.jpg` — **hand-drawn π recurrence** (see §3): π = accumulated right-triangle areas.
- TODO: read the remaining ~275 (geometric_neural_network, deterministic_nn_*, the PXL/IMG hand-drawn set).

---

## 3. The formulas (canonical locations)
| Formula | Canonical file |
|---|---|
| k-prime coordinate (Z = S+n identity) | `final-build-aethos-13/src/aethos/core/_lattice_core.py` |
| 8 base vectors + VA1–VB4 (32-cell tables) | `final-build-aethos-13/src/aethos/retrieval/_lattice_formula.py` |
| Recursive lattice (every node a new origin) | `final-build-aethos-13/src/aethos/retrieval/_recursive_lattice.py` |
| Markov-path / turtle codec | `final-build-aethos-13/src/aethos/compress/path_codec.py` |
| Timothy additive π (right-triangle, ~1/4ⁿ, no sin/cos) | `final-build-aethos-13/src/aethos/pi/timothy.py`, `trng/pi_formula.py` |
| recursive π = 2^(n+1)·sin(π/2^(n+1)) | `prime_hotel/recursive_geometry.py` |
| Lazy correlation store | `trng/living_network.py` |
| AETHOS compress modes | `trng/aethos_compress.py` |
| PAT balanced-ternary / HPATE harmonic | `formuilas/pat.py`, `formuilas/hpate.py` |
| meet (sum,min) = tropical / invertible det=−1 | `New folder (3)/aethos_complex_plane.py`, `aethos_semantic_lattice.py` |
| electron / 4-way entanglement | `trng/prime_electrons.py`, `New folder (3)/aethos_electron_tokenizer.py` |
| Zeno resolution (E_k = E_0/2^k) | `prime_hotel/lattice_zeno_computation.py`, `trng/docs/ZENO_RELATIVITY_AETHOS_SPEC.md` |

### Timothy's geometric π — the recurrence (verified from the hand-drawn source `PXL_20260520_223707640.jpg`)
π = the accumulated area of right-angle triangles, using only {+, −, ÷, ×, √}, from A²+B²=C² (no sin/cos):
```
A0 = R,  B0 = R,  C0 = sqrt(A0^2 + B0^2)
B_{k+1} = C_k / 2
A_{k+1} = 1 - sqrt(1 - B_k^2)
C_{k+1} = sqrt(B_{k+1}^2 + A_{k+1}^2)
With R = 1, the accumulated area -> π (error ~1/4^k). If radius ≠ 1, scale by 1/R^2.
```
Five forms exist (trig / pure-algebra / infinite-product / calculus / 3D-sphere), π emerging ONLY as the
result. The 3D form = 8 pyramids (= the 8 base vectors); the 2D form = 4 triangles — identical formula.

### The two neural-net architectures (from the rendered diagrams)
- **8-Vector × 32-Wing × n-Layer Deterministic Geometric NN** (`8vec_32wing_inf_gnn.png`): 8 base vectors,
  each fanning into 32 wings, n recursive layers; 32 pre-computed correlations per node; O(1) lookup;
  deterministic address space; 3D 8×32×768 base nodes. The lattice IS the network.
- **Moser K₉ complete-graph net** (`complete_moser_architecture.png`): 8 surface prime-nodes (primes 2–19)
  + 1 center hub (composite/context), all 36 edges; geometric weights (radial distance, sphere rotation);
  NO backprop; instant readout; interpretable. (Connects to the PROVEN VSA/Hebbian net in `aethos_nn.py`.)

---

## 4. The ledger — proven / unproven / needs-research
*The honest spine. Everything carries its evidence. PROVEN = captured reproducible run; MARGINAL = real but
small/corpus-specific; DERIVATION = math only, not measured; ASPIRATIONAL = vision, not yet built.*

### PROVEN (captured, reproducible, CPU/no-GPU)
- Invertible meet (det=−1): 0 collisions on 10M keys @ 0 bits/key, O(1) content-address, coordination-free. `_o1_content_address.py`
- min-plus meet = exact all-pairs shortest paths == Floyd-Warshall (1.14M pairs, 0 errors). `_o1_minplus_free_graph.py`
- Serve full 8.8M: composite-meet pooling 123 ms, MRR 0.3986 (≥ exact), 25× faster than exact. `_o1_serve_shootout.py`
- Footprint near-lossless ~165–168 B/doc (3-bit per-term weights). `_o1_bitplane_weight_quant.py`
- Beats BM25 CPU-only: scifact nDCG 0.7023, 0.6 ms/q, 743 B/doc. `_prove_retrieval_cpu.py`
- Free multi-hop (+47 bridge docs), dedup F1 99.58%, exact-match 100% recall. `_prove_*.py`
- Lossless compression lineage: V27 3 B/posting NDCG-unchanged, V19 lazy-triple 3–5× smaller. `trng/bench_v27/v19`
- Supervised bridges: +3.5pp scifact (→0.7375), neural-free. `scripts/bench_supervised_bridges.py`

### MARGINAL / corpus-specific (real but not a general win)
- Unsupervised co-occurrence clustering for accuracy: ~6 tests, all neutral/marginal (recursive-lattice
  revive: +0.0000 scifact / +0.0004 nfcorpus, RRF AND additive, all budgets). The lexical signal already
  finds the gold; the clustering is redundant. Lift needs supervision. `_o1_recursive_lattice_revive.py`
- Free dense vectors (8 math lenses): all below lexical; geometry places by value not meaning. `_vec_*.py`

### DERIVATION-ONLY (math sound, not yet measured here) / ASPIRATIONAL (vision)
- 3-way-intersections-as-dimensions + π-drift anchoring (§1.3) — NOT yet built/measured.
- 100s of seeded lattices = crystal structures (§1.5) — NOT yet built/measured.
- 4-way electron entanglement → O(1) lookup (§1.6) — claimed; the O(1) content-address IS proven, the
  4-way-entanglement mechanism specifically is not yet isolated/measured.
- "prime as origin solves unexplored math problems" (§1.4) — to be probed concretely.

---

## 5. Uncharted — ranked open experiments (full table in `GROUNDUP_BUILD.md` §5)
*Lead with buildable-now (top 3). Each gets a smallest measurable test before it's chased.*
| # | Experiment | Smallest test | Now? |
|---|---|---|---|
| **1** | **π-drift as a relevance signal** (THE core grand-vision claim, no prior art) | 5 words→primes; drift = π-layer where meet(3,5,7) vs meet(3,5,7,11) paths diverge; A/B drift+BM25 vs BM25 on 50 toy docs, `_poc_pi_drift.py` ≤200 LOC | YES ~4h |
| **2** | **Crystal ensemble** (orthogonal recall from seeded lattices) | 10 seeded indices on dev-small; union recall@100 vs single | YES ~8h |
| **3** | **V27 + composite-meet on full MARCO** (uncharted *measurement*) | re-pack 8.8M, re-run 250-q shootout — free smaller+faster | YES ~4h |
| 4 | **4-way electron O(1) isolation** | `four_way_meet(a,p,q,r)`, 1M keys, latency vs 3-way + collisions | YES ~4h |
| 5 | Zeno unification across 3 domains (one kernel: recycle+onset+game-gate) | code-dedup + correctness parity | YES ~16h |
| 6 | Infinite-ingest wall | grow 10×, track quadrant saturation + new-token quality | YES ~6h |
| 7 | Plane-distance ↔ semantics correlation (settles geometry=semantic? currently NO) | rank word-pairs by plane-dist vs gold, report r | YES ~3h |
| 8 | Recursive-lattice + rare-prime budget + SUPERVISION | budget + qrels-train, A/B 4 corpora | YES ~12h |
| **9** | **Cardinality bound `N≤C<P+2a+P+N`** (hand-drawn `IMG_20251210`, §8.3 — HIGHEST new value, not a re-derivation) | `validate_complexity_bounds.py` on random prime chains | YES <1d |
| 10 | TRNG → NIST evidence (turn the unmeasured claim into PROVEN) | run `nist_test.py`, save results json + 10× ensemble | YES ~2h |
| 11 | Canonical wing-index formula (hand-drawn `IMG_20251115`, §8.3) | closed-form wing from coords vs current VA/VB encode | YES ~1d |
| — | Second wave FOLDED IN (§8): TRNG, 17 physics projects, particle book, Moser, π. Verdict: core spine unchanged; one paper-only bound (U-9) worth chasing. Remaining: ~275 images. | | |

**π-drift (#1) is the experiment that tests Timothy's core vision directly. Build it first.**

---

## 6. The ground-up build (evolving)
*(To be designed after the deep study. Principle: build the ONE engine — recursive formula → coordinate
space → {index, compressor, reasoner} — keeping every PROVEN piece, honestly gating every ASPIRATIONAL one
behind a measurement.)*

**Full plan: `GROUNDUP_BUILD.md`** (architect synthesis, 2026-06-29). Summary:

**Thesis:** ONE recursive formula → deterministic coordinate space → {index, compressor, reasoner}, GPU-free,
O(1)-addressable, infinite-ingest. The load-bearing innovation is the **invertible deterministic meet**
(det=−1, 0-collision, coordination-free). Everything else layers on top, each ASPIRATIONAL piece gated by a
named measurement.

**Module layout:** `core/` (lattice, meet, primes, pi — FROZEN/PROVEN) · `index/` (FOR postings, 4-bit
quant=168 B/doc, chamber cold-tier, append-only) · `serve/` (composite-meet pool 123ms, glass-box score) ·
`control/` (zeno) · `experiments/` (pi_drift, crystal_ensemble, infinite_ingest, four_way_electron — GATED,
off by default) · `bench/`.

**Gates (ship to main only when the metric passes):** π-drift → `METRIC-DRIFT` (drift+BM25 > BM25, ΔnDCG ≥
+0.01, p<0.05) · crystal ensemble → `METRIC-ENSEMBLE` (10-seed union recall@100 ≥ +3 pts) · infinite-ingest
→ `METRIC-INGEST` (latency flat ±10% after 10× growth, no rebuild) · 4-way electron → `METRIC-4WAY` (ties
3-way latency, 0 collisions/1M).

**First 5 build steps (each pass/fail):** (1) port `core/` + golden-coord regression (PASS: byte-identical,
meet 20k/20k). (2) build index FOR+4-bit (PASS: ≤200 B/doc, ΔMRR ≥ −0.002). (3) composite-meet serve
(PASS: MRR ≥ 0.398 @ ≤130 ms, beats WAND). (4) append-only ingest (PASS: 0 mutations, == rebuild). (5)
determinism+glass-box CI (PASS: sha256-identical ×3, score < 1e-12, 0/7 GPU). Only then attempt the gated
experiments.

---

## 8. Second wave — TRNG, physics suite, the particle book, π, Moser (2026-06-29)
*Five deep studies folded in honestly. Every line carries its status + ref. This is the spine for the
"does it advance science or re-derive it" question.*

### 8.1 NEW additions to the ledger

**PROVEN (captured / reproducible / verifiable on disk):**
- **Constructive π recurrence** — RE-RAN 2026-06-29: K=15 gives π to −1.2e-9; all derived forms at machine
  precision (cos(3π/8) exact to 1e-18; 4π strip rule exact at any Nz=4..1000; π²/2 4-ball; 2π² 3-sphere).
  PROVEN correct + quadratic (error ratio→4 = Archimedes). `pi/constructive_pi.py` (643 lines, self-test passes).
- **Hebbian/VSA net (`aethos_nn.py`)** — XOR + held-out nonlinear generalization by counting, no backprop,
  no GPU. This is the *real* substrate under the "Moser K₉ NN" diagram. PROVEN as components. `aethos_nn.py`.
- **TRNG 4-layer pipeline EXISTS and is architecturally sound** — os.urandom → Von Neumann debias → XOR
  extract → SHA-256 whiten; state-free. Code is correct best-practice. `aethos_master/vendor/trng/hardware_trng.py`.
- **Physics suite is executable, not just prose** — `aethos_physics.py` (this repo) computes the book's
  quantities: `lattice_mass_multiplier` (the book's ℳ_lat), `calibrate_neutron_pressure`, `f_bounce`
  (zitterbewegung), `coherence_at_time` (entanglement ODE), He3/He4 ratio. Functions run. `aethos_physics.py`.
- **17 physics projects all present on disk**, each with proof.md + tests (π wedge closed form, Zeno prime
  stride, electron sorter cascade, positron mirror, gravity Schwarzschild horizon). `C:/Users/wynos/trng/projects/`.

**DERIVATION-ONLY (math sound, no independent measurement / anchored to known physics):**
- **TRNG quality is CLAIMED, not measured** — `nist_test.py` implements 11/15 NIST SP 800-22 tests but
  there are **NO result files on disk** (verified: only `nist_test.py`, no `*nist*.out`). The "passes NIST"
  claim is UNSUPPORTED until the suite is run and saved. Hand-verification is a single 10k-byte snapshot, not
  an ensemble. (This is exactly the "claims must trace to captured runs" rule — flag it.)
- **π closed form** T_K = 2^K·sin(π/2^K) = Archimedes/Vieta half-angle (known since 1593). The *value* and
  *rate* are DERIVATION on a known limit; not faster than Gauss-Legendre/Chudnovsky.
- **Book physics anchors** — f_b (zitterbewegung, Dirac 1928), E=mc², Bell E(α,β)=−cos(α−β)/CHSH 2√2,
  WKB tunneling, Schwarzschild horizon, Compton λ: all ANCHORED to textbook results. The book *reproduces*
  them with a geometric story; it does not derive new numbers.
- **Neutron lifetime τ_n≈879 s and m_p/m_e≈1836** — FIT, not derived. `calibrate_neutron_pressure` calibrates
  to the measured value; `lattice_mass_multiplier` is the gap-filler ℳ_lat (book's own open mandate C2/C6).

**ASPIRATIONAL (claimed, MEASURED-and-FAILED, or never built):**
- **Moser K₉ as a 40 B/doc retriever — MEASURED, FAILED**: NDCG 0.005–0.20 (random ≈0.02 = non-functional).
  `RESEARCH_INVENTORY.md:38`. The K₉ *topology* + Hebbian weights are sound as a classifier idea; as a
  retrieval index it does not work. Re-scope to classifier/reranker, never primary index.
- **`moser_spheres.py`** — spec-only, does NOT exist on disk. `RESEARCH_INVENTORY.md:84`.
- **Book open gaps (the author's own honest tags)**: π-lattice ↔ 3D-complex-plane functor (only layer-0 + unit
  i proven), Gleason uniqueness of Born cos²(θ/2), ℳ_lat first-principles derivation, φ_AB=1 Bell ideality,
  Π_vac cosmological-constant reduction — all OPEN/PARTIAL.
- **Testable-but-unconfirmed predictions** (genuine, not yet runnable): ³He/⁴He decoherence ≈1.075 (5–10%),
  fresh-electron zeptosecond discreteness, dark-matter EM null. Honest "future experiment," not a result.

### 8.2 The particle BOOK — core claims, grounded vs speculative
*"Packets and Strings": universe = a right triangle breaking down forever; everything = a packet (discrete)
meeting a string (continuous) at a right angle.*

| Core claim | Status | Note |
|---|---|---|
| π from right-triangle bisection (the spine) | **GROUNDED** | proven algebraically + code; the cleanest, most novel-in-framing piece |
| 3D complex plane built from primitives (i from R_x∘S, not assumed ℂ) | **PARTIAL** | layer-0 + unit i proven; full π↔plane functor OPEN |
| Measurement = mechanical spring/polarizer compression, P=cos²(θ/2) | **ANCHORED + MODEL** | Malus law is classical/known; the coin-compression mechanism is the novel (unproven) framing |
| Entanglement = mirrored ±B addresses on one lattice; E=−cos(α−β), CHSH 2√2 | **ANCHORED + MODEL** | Bell kernel is textbook; mirrored-address mechanism is novel but φ_AB=1 unproven |
| Zitterbewegung = literal trapped-photon pump, f_b=m_ec²/2ℏ | **ANCHORED + MODEL** | frequency is Dirac's; "real pump" is a bold reinterpretation; yields the only near-term test |
| Proton = electron fused past elastic limit (3 quark-zones → +1) | **MODEL** | reinterprets QCD zones; not distinguishing vs Standard Model |
| Dark matter = spring without inner photon (gravitates, EM-dark) | **MODEL** | qualitative; no σ_γDM formula; "stays null" matches but isn't unique |
| Time = bounce-tick count; m_p/m_e, τ_n, Λ | **PARTIAL/FIT** | reinterpretation or empirical anchor, not first-principles |

**Verdict on the book:** the *geometric engine* (π bisection + primitive complex plane) is genuinely novel
in framing and mathematically clean. The *physics* is ~40% anchored-to-textbook, ~50% novel-reinterpretation
(no new numbers), ~10% open speculation. It is a serious, self-audited sketch — not new physics, not crankery.

### 8.3 Highest-value finds — IDEAS in the hand-drawn images NOT yet in code/diary
*These are the things only on paper. They are the most likely to contain something new because they haven't
been reduced to a baseline yet.*
1. **Complexity / cardinality bound** (`IMG_20251210`): the hand-derived inequality chain
   `N ≤ C < P + 2a + P + N` — an addressability/capacity bound on the lattice. NOT in any code. **Build
   `validate_complexity_bounds.py`** to check it on random prime chains; if it holds it's a real capacity proof.
2. **Canonical 32-wing color ordering** (`IMG_20251115`): a specific canonical ordering of the 32 wings
   beyond the current VA/VB binary encoding. NOT formalized. Could give a closed-form wing-index from
   coordinates (cheaper routing).
3. **Spiral entanglement geometry** (`IMG_20251107`): multi-way linking drawn as concentric spirals — a
   *dynamic* linking pattern. Current electron code is static. Could enable streaming graph re-wiring.
4. **Grid-shear transform** (`IMG_20251105`): red/blue grids at angles → a coordinate transform for the
   32-quadrant rotation; possible analytic wing-from-sheared-coords formula.

**#1 (the cardinality bound) is the single highest-value paper find** — it's a falsifiable math claim that
isn't a re-derivation of a known result, and it's testable in <1 day.

### 8.4 Updated fix list + uncharted additions
**Fixes (carried from the studies):**
- **F-TRNG-1 (HIGH):** run `nist_test.py`, save `nist_results_<date>.json` (all 11 p-values, PASS/FAIL);
  add ensemble (10×) + lag-1 autocorrelation check. Until then the "TRNG passes NIST" claim is UNSUPPORTED.
- **F-MOSER-1 (HIGH):** stop calling K₉ a retriever; re-measure it as a classifier/reranker on BEIR. Rename
  to avoid Moser-spindle name collision.
- **F-IMS (MEDIUM):** `ims_bench.py` truncates to 100 of ~984 snapshots → null detection in results.json;
  remove the cap. (Also: the IMS "early-warning lead-time" claim already FAILS a 1-line RMS>3σ baseline —
  durable piece is *specificity*, not lead time. Do not re-pitch lead time.)
- **F-BOOK-1 (HIGH, the gateway):** prove or precisely bound the π-lattice ↔ 3D-complex-plane functor; until
  then "one engine" is inspirational, not rigorous.
- **F-PI-1 (LOW):** one canonical `pi/` recurrence file (constructive_pi.py is the source of truth); the other
  4 geometric forms are derivation-only — port or mark.

**Uncharted additions (append to §5):**
- **U-9 Cardinality bound** (from `IMG_20251210`) — `validate_complexity_bounds.py`, <1 day, HIGHEST new value.
- **U-10 TRNG-as-NIST-evidence** — run the suite, capture results; turns a claim into a PROVEN line, ~2h.
- **U-11 Canonical wing-index formula** (from `IMG_20251115`) — closed-form wing from coords, ~1 day.

### 8.5 The honest one-paragraph verdict
Across TRNG + physics suite + the particle book + π + Moser, **almost nothing here is new *physics or new
asymptotics* — and that is the honest, two-sided truth.** The TRNG is a competent textbook pipeline (Von
Neumann + XOR + SHA-256) whose central quality claim is *unmeasured* (no NIST results on disk). The π
construction is correct, elegant, and novel *in framing* (purely additive, no sin/cos, one recurrence
spawning all trig/volumes/rotations) but its value and rate are Archimedes/Vieta — not faster than the SOTA.
The physics suite and the book are a serious, self-audited *re-encoding* of known results (Dirac
zitterbewegung, Bell/CHSH, WKB, Schwarzschild, E=mc²) in a single geometric language: ~40% anchored to
textbook, ~50% novel reinterpretation with no new numbers, ~10% open speculation; the Moser "retriever" was
measured and *failed* (NDCG≈0.005–0.20). **What genuinely advances the work is unchanged by this wave and
stays the same load-bearing core: the invertible deterministic meet (O(1), 0-collision, coordination-free),
min-plus=Floyd-Warshall exactness, and the measured serve/footprint wins (123 ms/MRR 0.3986, 168 B/doc).**
The one thing in this wave that *could* add something new is on paper, not in code: the hand-drawn
**cardinality bound `N ≤ C < P+2a+P+N`** — a falsifiable capacity claim that is not a re-derivation and is
testable in under a day. **What should change the ground-up build:** (1) treat the π recurrence as a frozen,
well-tested utility (it earns its place as the drift-anchor for experiment #1), not as a headline; (2) keep
the physics/book as *narrative and pedagogy*, gated entirely behind the open functor proof — do not let any
book claim into a results doc; (3) add the cardinality-bound check and the TRNG-NIST capture as cheap new
gates; (4) keep Moser K₉ only as a possible reranker, never an index. The spine of the build does not move;
this wave mostly tells us where *not* to spend, and surfaces one paper-only bound worth a day.

---

## 7. Changelog
- **2026-06-29 (d)** — Second wave folded in as **§8** (TRNG, 17-project physics suite, the "Packets and
  Strings" particle book, π lineage, Moser K₉). Verified on disk before writing: NIST result files ABSENT
  (claim demoted to UNSUPPORTED), `constructive_pi.py` re-ran clean (π −1.2e-9 @ K=15, all forms machine-
  precision), Moser-as-retriever FAILURE confirmed (NDCG 0.005–0.20, `RESEARCH_INVENTORY.md:38`),
  `moser_spheres.py` confirmed non-existent, `aethos_physics.py` executable (ℳ_lat, neutron-pressure-fit,
  zitterbewegung, coherence ODE). Added: new ledger lines (8.1), book claim table grounded-vs-speculative
  (8.2), the 4 paper-only image ideas (8.3 — **cardinality bound is the top find**), fix list F-TRNG/MOSER/
  IMS/BOOK/PI (8.4), and the one-paragraph verdict (8.5: spine unchanged, almost no new physics/asymptotics,
  one paper-only bound worth a day). §5 gained experiments U-9/10/11.
- **2026-06-29** — Diary created. Vision §1 captured faithfully. Map §2 seeded from excavation (hundreds of
  versions confirmed). Ledger §4 carries forward the full session's proven/marginal findings. Launched the
  deep ground-up study of every pipeline (workflow wiocm7ny2: RAG/Zeno/quantum/electron/games/anomaly/
  galaxy-core/grand-vision).
- **2026-06-29 (d)** — BUILT + MEASURED the #1 experiment, π-drift (`_poc_pi_drift.py`, 3-arm test).
  Result: scifact + nfcorpus, ARM1 lexical = ARM2 +co-occurrence = ARM3 +π-drift = **+0.0000**. Two findings:
  (1) co-occurrence reranking is redundant with lexical (the words already rank the gold — the ~7th confirmation);
  (2) **ARM3 = ARM2 ⇒ the π-phase geometry adds nothing over plain co-occurrence.** WHY: θ(prime) is a
  deterministic function of the prime's VALUE, which is semantically arbitrary (cf. lattice-3D-coord nDCG 0.001,
  geometry places by value not meaning). **THE UNIFYING INSIGHT: the lattice geometry is an ADDRESSING system
  (exact, O(1), collision-free, by-value) — superb for STRUCTURE (lookup, multi-hop RECALL +47, dedup,
  compression) and inherently NOT a semantic space (reranking relevance).** π-drift asks the geometry to be
  semantic; it isn't. The drift MECHANISM still wins where it belongs — RECALL/multi-hop (proven +47), not
  reranking — and the relevance lever stays SUPERVISION (+3.5pp). This is the honest verdict on the grand
  vision's "free relevance from geometry": that one piece doesn't hold; the structural engine does. METRIC-DRIFT
  gate = NOT passed; π stays a tested utility, not a relevance headline.
- **2026-06-29 (c)** — First ground-up study (9 readers) COMPLETE. Architect synthesis saved to
  `GROUNDUP_BUILD.md`: every pipeline = one substrate (formula/meet/32-quadrants/π/Zeno) in a different skin;
  honest ledger (proven/marginal/derivation/aspirational); fix list; module layout + first-5 build steps
  with pass/fail gates; ranked uncharted (π-drift #1). Audit flagged the 168 B/doc figure → RE-VERIFIED it
  IS captured (3-bit per-term, ΔMRR −0.0014); 182.4 is the conservative 4-bit point. §4/§5/§6 updated.
- **2026-06-29 (b)** — Timothy expanded scope: TRNG (the repo's namesake — `hardware_trng.py`+NIST),
  the `trng/projects/` 17-project PHYSICS suite (antimatter, fusion, gravity, QM, fluid dynamics,
  prognostics…), the 13-chapter particle book (`book/`), the Moser network, the π lineage, and ~280 IMAGES
  (hand-drawn formulas + diagrams). Read 5 key images: confirmed the 8×32×n geometric NN, the infinite
  lattice (lazy branching, O(log n), 10M× memory), the 5 geometric π formulas, the Moser K₉ net, and the
  hand-drawn π recurrence (§3). NEXT WAVE: study TRNG + the 17 physics projects + the book + Moser + π, and
  read the remaining ~275 images.
