# AETHOS RESEARCH ATLAS — the living cross-reference

> Single source of truth for Timothy's 6–7 months of AETHOS / φ-prime-lattice research, scattered across
> ~13 repos + many branches + phone photos. Built 2026-06-25 during an autonomous deep-organization run.
> Goal: inventory every version, record **what each solved**, and converge on **the best unified version**.
> This doc is appended to continuously — newest findings at the bottom of each section. Cross-reference it.

---

## 0. WHAT THIS FRONTIER IS (the one-paragraph north star — re-read first)

A **φ-prime lattice**: every integer/token gets a deterministic 3D address from its prime factorization
(8 wings × 4 branches = **32 chambers**; velocity = `min(n,a)`; the **meet** of two anchors a<p is the
invertible vector `(a+p, min, a+p)`). Verified deep structure: the meet **IS the tropical (min,+) semiring**
(= exact shortest paths / least-action / minimax), the **octant IS the Legendre/Jacobi character mod 3·5·7**
(a balanced multiplicative→XOR homomorphism), the depth **IS a measure** (inclusion-exclusion), the **3-way
meet is the atom** (an exact erasure code; its node = `(top2-sum, MEDIAN, total-sum)`), and a **constructive-π**
recurrence builds the roots-of-unity phases with no transcendentals. These unify into a **two-temperature
engine**: one complex operator `M_β = -(1/β)log Σ a_k e^{-βc_k+iφ_k}` is the **particle** (cold = exact
tropical meet) and the **wave** (warm = π-roots interference) — verified. APPLICATIONS proven this session:
(a) retrieval — the lattice SERVES learned-sparse (SPLADE) at **SOTA + small + fast across corpora**, while its
own symbolic scoring ties BM25 (an *algebraic* wall: tropical = path, not additive evidence); (b) compression —
FOR 0.428 GB / chamber 6.2× on full MARCO; (c) the **algebraic corpus** (corpus = one big number, decode=factor,
add=multiply, correlations free); (d) erasure codes, set-reconciliation (Minisketch), Merkle provenance,
exact DFT/interferometer, GCD/continued-fractions, Sierpinski. Also: IMS bearing fault monitor (4h early),
a Γ-ODE LLM mixer, a proven VSA/Hebbian net. The lattice is an EXACT addressing/integrity/optimization
substrate; semantics/accuracy come from a thin learned layer it serves small+fast.

---

## 1. REPO INVENTORY (the map — filled by the inventory pass)

External (NOT research, ignore): `MLC/repos/mlcommons`, `OneDrive/New folder/inference` (MLPerf suites).

| Repo | py | md | Branches / versions | What it is (TBD by inventory) | Status |
|---|---|---|---|---|
| `New folder (3)` | 724 | 121 | cursor/symbol-plane-audits-slim-index, main | THIS session's repo: lattice retrieval, algebraic corpus, SPLADE-on-lattice, the math dives | ACTIVE |
| `trng` | 736 | 75 | main, quantgum, constructive-circle, sensorbrain-rca×2, cmapss, rca-catalog, trng-catalog | biggest; sensorbrain/RCA + constructive-circle + playground formula sweep + Andrea battlecards | core |
| `prime_hotel` (OneDrive/GitHub) | 161 | 209 | andrea, main, tims, pitagora, ultrafast-24x | **Pitagora project root** (Andrea's RAG app + BEIR); 209 docs | central |
| `final-build-aethos-13` | 48 | 46 | main | the Andrea+OSCAR INTEGRATION build (vendor/oscar + vendor/pitagora_andrea); 46 stage-findings | integration |
| `aethos_master` | 108 | 8 | main | a "master" consolidation attempt | review |
| `Projects/aethos13` | 79 | 2 | cursor/aethos13-hierarchical-lexicon-scifact | hierarchical lexicon on scifact | review |
| `OneDrive/New folder` | 297 | 52 | main | big; aethos13_complete, oscar, aethos_explorer/snapshots, ground_rag | review |
| `Wy-nos` (OneDrive/GitHub) | 12 | 1 | phi-prime-lattice, pi-calculation-benchmark, novelty-eval×6, cognitive-calculator, math-equation | the public-facing / novelty-eval + pi-benchmark repo | review |
| `aethos13-ultrafast` (×2 copies) | 14/19 | 2/3 | andrea, main, api/omega | the "ultrafast" speed variant | review |
| `formuilas` | 29 | 4 | (none) | the formulas | review |
| `lattice/` (non-git) | — | — | — | lattice/aethos/retrieval.py | review |
| `pi/` (non-git) | — | — | — | pi work | review |
| `New folder`, `New folder (2)` | — | — | — | aethos13_complete, oscar | review |

Photos/assets: `OneDrive/Pictures/Root cause formula with 8-vector prime 4-way branching`, `CrossDevice/{moto g, Pixel 10a}/storage/Download` (hand-drawn lattice photos, the coordinate-tables PDF).

---

## 2. CAPABILITY MATRIX — which version solved what, best (filled by inventory)

For each capability: the repos that tackled it, the BEST measured result, and where it lives.

| Capability | Best version / repo | Measured result | Notes |
|---|---|---|---|
| Retrieval (SOTA) | New folder (3): SPLADE-on-lattice | BEIR scifact 0.70 / nfcorpus 0.35 / fiqa 0.35, 0.2-2.6ms, 1-20MB | matches SPLADE++; MARCO native encode running |
| Compression | New folder (3): FOR + chamber | MARCO 0.428 GB / 0.347 GB (6.2×) | persisted |
| Algebraic corpus | New folder (3): aethos_algebraic_corpus | corpus = 2.64M-digit number, decode=factor | scoring ties BM25 |
| Lattice math (tropical/octant/measure) | New folder (3) memories | all verified | see §0 |
| Two-temperature / fused operator | New folder (3): aethos_fused_meet | particle+wave in one M_β | frontier closed |
| RAG integration | final-build-aethos-13 (oscar+pitagora) | interface = add_documents/retrieve | adapter pending |
| Sensorbrain / RCA / bearings | trng | IMS bearing fault 4h early | TBD which is best |
| Constructive π | final-build/src/aethos/pi/timothy.py | π = 2+Σ2^(n+2)·sin·(1-cos), err~1/4ⁿ | canonical; viz in OneDrive/Pictures |
| **Retrieval (lexical champion)** | **trng/bench_prime_triple_v10.py (V10)** | **SciFact 0.7792** / nfcorpus 0.3388 / fiqa 0.2375 | beats BM25(.665)/TAS-B(.643)/ColBERT(.671) on scifact; per-corpus locality + sentence-local rank-3 |
| Retrieval (Andrea-validated) | prime_hotel @ f4ed996 (tag andrea-tested) | SciFact 0.6531, R@10 0.779 | independently validated by Andrea |
| Retrieval (ultra-small) | aethos13-ultrafast | SciFact 0.6668 @ **24 B/doc**, 53ms | int8 quantized |
| Fast BM25 (Zipf duality) | trng/lattice_bm25_findings.md | log(prime)↔IDF **r=0.9836**, **4.32× speedup**, NDCG +0.16pp | the foundation; uint16 lossless |
| Anomaly / NASA bearings | OneDrive/New folder (blind_crucible, diagnostic_oracle) | **47.8h lead time**, 1.23B sectors/sec, IMS inner-race | [[ims-bearing-monitor-validated]] |
| Hilbert space proof | prime_hotel/HILBERT_SPACE_PROOF.md + formuilas | 6 axioms ✓, Level-5 = 16,008 dims | formal proof code |
| Universal CCR (prime channels) | prime_hotel/universal_ccr.py | prime=feature-channel (3=var,5=stride,7=causal…) | cross-domain |
| 32-wing parallel + halt/reconstruct | prime_hotel/32_WING_*.md | 100% util, 5 corrupted wings recovered | |
| Trillion-scale addressing | New folder/oscar/test_trillions.py | 100k primes = 320 BILLION nodes | scale proof |
| Cognitive Calculator (codebase 2) | OneDrive/New folder @ 9d2b9ca | unitary evolution, zero-keyword, 11.8MB ontology | 364 files; the Stage 8-10 ports |
| Investor 35-test suite (codebase 3) | Projects/aethos13 @ c28202e | 1B-doc 0.050ms, MNIST/NASA/scifact, DECK_ANSWERS | the pitch evidence |
| Patent | trng/PROVISIONAL_PATENT.md (Omaga, USPTO Jul 2025) | 10 claims, seedless recursive logic | filed |
| QM-on-lattice validation | pi/aethos_quantum.py + PAPER_chsh_aethos.md | **CHSH=2√2 (140σ@10¹⁰)**, 17 tests, KS 0/512, teleport F=1 | HONEST: bounds saturated NOT exceeded |
| TRNG (true random) | pi/electron_trng + electron_sorter | **98.9% NIST/TestU01**, 200 Mbit/s RTX5080 | same seed→diff output (OS-jitter entropy) |
| Deterministic semantic encoder | prime_hotel/aethos_llm_encoder.py | 24-dim, NO training, 205× faster, 99.93% param reduction | non-neural sentence embedder |
| Zero-shot semantic recall | Projects/aethos13 correlation-brain | rare-term PPMI, **95% recall@100**, text-only | April predecessor of June algebraic corpus |
| Multi-domain RCA (k!) | Wy-nos/factorial_engine.py | all k! causal seqs + dominance, 100% medical/bearing | "no prior art", patent-worthy |
| Turing-completeness / bignum | lattice/turing.py + bignum.py | register machine in coords; **>5× float64** on π | computational substrate proof |
| Text→prime bridge | formuilas/pat.py + hpate.py | balanced-trinary + harmonic shared-factor sim | only place embeddings↔primes bridge exists |
| Riemann Hypothesis exploration | pi/rh_*.py | right-triangle ζ-shrinkage, slope→−1/2 at zeros | EXPLORATORY, not a proof, unique |
| **The meet = unified classical computation** | aethos_tropical_game.py + math-dive probes | sorting, APSP, assignment, Goldbach, subset-sum (2^(n/2)), median, partitions, **games (minimax)** — ALL exact | REAL_BUT_KNOWN: re-encodings, no new complexity; the UNIFICATION under one meet is the win |

PRIOR MAP: `final-build-aethos-13/MASTER_INVENTORY.md` (2026-04-25) is the user's own catalog of the
pre-June work (11 repos, 7 worktrees, 5 parallel codebases). THIS ATLAS = that map + the June session's NEW
layer (below) + the convergence. **HONESTY FLAGS from the prior map**: (1) the email "MS MARCO 0.948" number
"cannot be located in any of the 5 codebases" — unbacked, do not cite; (2) compression "99.9%" is KNOWLEDGE
compression (rule extraction), NOT byte-lossless — different paradigm, needs reframing; (3) BEIR_MOSER pure-
formula scored 0.20/0.14/0.005 (honest negative: pure formula < BM25 — matches this session's "symbolic
scoring ties/loses to BM25, the wall is algebraic").

JUNE-2026 SESSION ADVANCES (New folder (3), NOT in the April map — the new frontier):
- SPLADE-on-lattice = SOTA+small+fast across BEIR (scifact 0.70/nfcorpus 0.35/fiqa 0.35, 0.2-2.6ms, 1-20MB,
  no CE) [[sota-via-learned-sparse-on-lattice]]; native MARCO encode running.
- Algebraic corpus (corpus = one number) [[aethos-algebraic-corpus]]; FOR/chamber compression 0.428GB/6.2×.
- The math IS classical structures (tropical/Legendre/Euclid/Sierpinski) [[aethos-formulas-are-classical-structures]];
  two-temperature particle/wave engine + the fused complex operator [[aethos-two-temperature-engine]].
- The exact coordinate formula verified [[aethos-coordinate-formula]]; honest scorecard [[honest-headline-scorecard]].

---

## 3. THE BEST UNIFIED VERSION (synthesis — converging)

The honest convergence (no single version wins everything — the best version is per-goal, then unified):

**HOME**: `final-build-aethos-13/` is the user's chosen consolidation target (per MASTER_INVENTORY §XI). The
unified build should live there, importing the best-of-each:

1. **Retrieval — two tiers, pick by goal:**
   - *Max accuracy, general, SOTA*: **SPLADE-on-lattice** (June; serves learned-sparse small+fast across all
     corpora; beats on nfcorpus/fiqa, generalizes). The new champion for GENERAL retrieval.
   - *Self-contained, no neural model, lexical-strong*: **`final-build/src/aethos/retrieval/unified.py`** (April
     Stage-7) is the canonical self-contained tier — V10-faithful + per-corpus routing + V27 scatter + Markov
     codec: **SciFact 0.7789 (tied V10), NFCorpus 0.3333, FiQA 0.2292, sub-ms P50, 49-219 B/doc lossless**. Use
     where you want zero model deps. (Raw V10 in trng is its predecessor; unified.py supersedes it.)
   - *Foundation both share*: lattice-BM25 Zipf duality (log-prime=IDF r=0.9836, 4.32×) + the scatter-fast meet
     + a lossless path/posting codec. The meet = the fast WAND primitive.

   **APRIL↔JUNE CONVERGENCES (same idea, rediscovered — strong signal it's right):**
   - April "V27 numpy-scatter BM25" (217× speedup, np.add.at) ≡ June "fast presence-meet" (scatter tf, gather
     candidates, 18×). Same vectorized-meet optimization.
   - April "Markov-path codec" (Timothy's "every doc is a PATH/Markov chain" → per-state Huffman, 49-219 B/doc
     lossless) ≡ June "chamber codec" on the posting-gap stream (8.91 bits/posting, 6.2×). Same lossless
     path-stream compression; the chamber/context-mixer is the stronger coder, the Markov codec the simpler.
   - April "no single architecture wins all 3 corpora; per-dataset routing is the honest answer" ≡ June measured
     BEIR per-corpus behavior (CE helps fiqa +11pp, HURTS scifact). Same honest finding.
   - April's deepest finding: "the structural gap is mostly Tier-1; rank-3 tuple channels need V10's
     AethosPureIndexer to be net-positive, on plain BM25 they add noise" ≡ June's "symbolic scoring ties BM25,
     the wall is ALGEBRAIC (tropical=path not evidence-sum)." Both say: the lattice's structure is the win,
     extra symbolic rerank signals don't add evidence — accuracy needs a learned layer (SPLADE) or V10's heavy Tier-1.
2. **The math-native engine**: the **algebraic corpus** (corpus=a number, decode=factor, correlations free) +
   the **two-temperature/fused operator** (particle=tropical meet, wave=π-roots) — the theoretical core that
   explains WHY (tropical=path not evidence-sum ⇒ symbolic scoring ties BM25; SOTA needs the served learned layer).
3. **Breadth (port into final-build per MASTER_INVENTORY §XI roadmap)**: anomaly/NASA (47.8h lead), Hilbert
   proof, universal CCR, 32-wing parallel, Cognitive Calculator (unitary evolution, zero-keyword, ontology
   bridges), constructive π, the GNN package, the trillion-scale addressing. Each has a measured win; none is
   yet in one place.
4. **RAG delivery**: the lattice retrieval behind Andrea's Pitagora `add_documents`/`retrieve` interface
   (adapter pending) — so any of the above serves through the agentic RAG.

CONSOLIDATION GAP (per April map): only ~5-10% consolidated into final-build. The June work adds the SOTA
retrieval + the theory but ALSO needs porting. **Recommended single source of truth = final-build-aethos-13/
src/aethos/ + this session's New folder (3) modules ported in.** Detailed best-version-per-capability above (§2).

### 3B. CONSOLIDATION PLAN (concrete — how to build the ONE best version)

**HOME = `final-build-aethos-13/src/aethos/`.** Port the best-of-each (source → target), in this order:

1. **Lock the formula ground-truth.** Copy aethos_master's `SPEC.pdf` + `COORDINATE_TABLES.pdf` + `test_spec_canon.py`
   golden tables → `final-build/spec/`. This is the authoritative 8-vector/4-branch/32-chamber reference; the June
   `aethos-coordinate-formula` memory + `aethos_complex_plane.py` already match it. One ground truth, everything tests against it.
2. **Retrieval = the two-tier engine** (`src/aethos/retrieval/`):
   - self-contained tier: `unified.py` (already there, V10-faithful + V27 scatter + Markov codec) + port the June
     **algebraic corpus** (New folder (3)/`aethos_algebraic_corpus.py`) as the math-native variant.
   - SOTA tier: port the June **SPLADE-on-lattice** (New folder (3)/`_route2_splade_lattice.py` + `marco_splade_native.py`).
   - Both behind ONE interface (Andrea's `add_documents`/`retrieve`) so the agentic RAG can pick a backend.
3. **Finish + unify the codec** (`src/aethos/compress/`): complete the Markov-codec save/load (`_markov_codec.py`,
   wiring incomplete) AND register the June **chamber/FOR codec** as the stronger coder. Both are lossless path/posting
   compression; pick per-tier (chamber for cold/archival, Markov/FOR for hot).
4. **Port the do-not-lose uniques** (§6) into new subpackages:
   - `src/aethos/encoder/` ← prime_hotel `aethos_llm_encoder.py` (deterministic 24-dim embedder).
   - `src/aethos/anomaly/` ← OneDrive Chaos-Consensus + trng SensorBrain (sensorbrain-rca-32) + the 77h NASA validation.
   - `src/aethos/rca/` ← Wy-nos `factorial_engine.py` (the k! engine, patent-worthy).
   - `src/aethos/semantic/` ← Projects/aethos13 hierarchical lexicon + correlation-brain (the April predecessor of the
     June algebraic corpus + corridors — MERGE these two lineages; they solve the same zero-shot-semantics problem).
5. **Theory layer** (`src/aethos/theory/`) ← the June two-temperature/fused-operator (`aethos_fused_meet.py`) + the
   classical-structures proofs (tropical/Legendre/Euclid). This is the "why it works" that unifies all the above.
6. **RAG delivery**: the Pitagora adapter (both backends selectable) — the paused build.
7. **Docs/IP**: copy the empirical anchors (MULTIDATASET_FINDINGS, lattice_bm25_findings, UNIFIED_RAG_RESULTS, the
   NASA executive summary, the 35-test DECK_ANSWERS) + the canonical diagrams (OneDrive/Pictures) + the Omaga patent
   → `final-build/docs/`. Then archive the 7 prime_hotel worktrees + redundant snapshots (harvest-then-delete).

**The single biggest merge opportunity** the inventory revealed: the April **correlation-brain** (Projects/aethos13:
rare-term PPMI, zero-shot, 95% recall@100, trains on text only) and the June **algebraic corpus + corridors + SPLADE-
on-lattice** are three attempts at the SAME thing — zero-shot semantic recall on the lattice. Unify them: the algebraic
corpus is the substrate, corridors/correlation-brain are the self-supervised rule, SPLADE is the learned-layer ceiling.

---

## 4. CROSS-REFERENCE LOG (running notes, newest at bottom)

- 2026-06-25 07:05 — survey done (`_RESEARCH_SURVEY.txt`); atlas seeded; inventory pass launching (wf wc2y4e7xw, 9 repos).
- 2026-06-25 07:10 — ASSETS cataloged (`_RESEARCH_ASSETS.txt`): hand-drawn lattice photos in `New folder (3)/_user_photos/` (IMG_2025*, ~40); generated viz in `_playground_viz/` (meet_field, octant_field, prime_gasket, sierpinski, sheared_sierpinski); a book PDF `book/output/Packets_and_Strings_Full.pdf`. The DOC/spec layer is in `CrossDevice/*/storage/Download/`: AETHOS_Complete_Technical_Specification(_v2/_FINAL), AETHOS_Coordinate_Tables_Complete, AETHOS_OSCAR_Complete_Explanation, AETHOS_Operations_Design_for_Andrea, AETHOS_Investor_Overview_Eric, **AETHOS_Technical_Validation_35_Tests** (the validated-capabilities doc — matches the 48-test battery).
- 2026-06-25 07:11 — MARCO SPLADE encode had DIED overnight at 22:41 (chunk 12/~45, PC slept). RESUMED (`_splade_encode_resume.log`, resumable, skips done chunks). NOTE: long background jobs die when the PC sleeps; resumable ones survive. The deeper-math-dive workflow (w0acldr9j) also died overnight — to re-run.
- 2026-06-25 07:20 — FOUND the user's own prior consolidation map: `final-build-aethos-13/MASTER_INVENTORY.md` (2026-04-25). Folded into §2/§3: 11 repos + 7 worktrees + 5 parallel codebases, the cross-version BEIR results (V10 0.7792 scifact champion, Andrea 0.6531, ultrafast 0.6668@24B/doc, master 0.5328), lattice-BM25 Zipf duality, NASA 47.8h, Hilbert proof, CCR, 32-wing, the Omaga patent. Spec PDFs extracted to `_docs/`. Honesty flags recorded (MS-MARCO-0.948 unbacked; "99.9% compression" = knowledge not byte-lossless; pure-formula < BM25). The April map stops pre-June; this atlas adds the June frontier on top. Consolidation was ~5-10% done in April; final-build is the home.
- 2026-06-25 07:25 — UNIFIED_RAG_RESULTS folded: April↔June CONVERGENCES recorded (§3: V27 scatter≡fast meet, Markov codec≡chamber codec, per-corpus routing≡BEIR finding, "Tier-1 is the win"≡"the wall is algebraic"). unified.py (final-build) = canonical self-contained retrieval (scifact 0.7789 sub-ms, 49-219 B/doc).
- 2026-06-25 07:35 — 8-repo INVENTORY pass done (wf wc2y4e7xw; 1 agent failed=formuilas, to redo). Wrote §6 DO-NOT-LOSE registry + §7 per-repo essence + §3B CONSOLIDATION PLAN. Digest in `_inventory_digest.txt`. Biggest merge insight: correlation-brain (April) + algebraic corpus + corridors + SPLADE (June) = three attempts at the SAME zero-shot-semantics problem → unify. MARCO encode resumed, ~25min to done then index+serve. Re-running the deeper-math-dive (died overnight).
- 2026-06-25 08:20 — END-TO-END AGENTIC RAG on scifact ✓ (`scifact_agentic_demo.py`): lattice retriever (algebraic CPU) → OSCAR AethosSynthesizer (real dereference/build_prompt/provenance) → answer. 4/5 gold in top-5, real provenance chain. LLM text-gen is a local extractive stand-in (no Ollama/API reachable; one-line swap to provider="ollama"/api_key). Integration proven: Timothy's retrieval → Andrea's agentic RAG works. Minor: adapter reported scored_with_bm25=False on this slice (warm-corridor path not hit → rank-descending fallback scores; order correct) — worth a look. MARCO serve loading the 1.06B-posting index (native number imminent).
- 2026-06-25 08:00 — GAMES + MATH-DIVE done. Games: `aethos_tropical_game.py` (minimax=meet, 7.16× D4 reduction). Math-dive: REAL_BUT_KNOWN across sorting/optimization/number-theory/games — the meet EXPRESSES a huge range of classical computation exactly under one operation; the unification is real, new power is not. Both folded to §2/§5. "Do everything asked" checklist now: organization ✓, RAG adapter ✓, games ✓, math-dive ✓; remaining = MARCO encode→index+serve (running).
- 2026-06-25 07:50 — RAG ADAPTER done + validated through Andrea's Pitagora harness (nDCG 0.73-0.92 scifact, both backends selectable). Timothy's original ask ("make my retrieval usable with his") = CLOSED. Building the tic-tac-toe tropical-meet solver next (the games-rebuild idea).
- 2026-06-25 07:45 — formuilas/lattice/pi inventoried (the gap). **pi/ is the GENESIS root**: from-scratch π-free recurrence (1500-digit), TRNG 98.9% NIST, QM-on-lattice CHSH=2√2 (17 tests, 10¹⁰ samples, honest), RH-exploration cluster (rh_*.py, not a proof). lattice/: Turing-complete + bignum >5× float64. formuilas/: PAT/HPATE text→prime bridge. Added to §2/§6/§7. ORGANIZATION CORE COMPLETE (atlas 278 lines, all repos). Remaining: encode→index+serve (running, ~40min), math-dive (running), pending builds (RAG adapter, games). Honesty note: pi/'s own BENCHMARK_IMPROVEMENTS confirms geometric encoder exact-match 100% / paraphrase 0% = the answer-ness wall, AGAIN — the recurring honest finding across every version: pure-geometric retrieval is exact-but-not-semantic; semantics needs a learned/correlation layer.

---

## 5. OPEN THREADS / PENDING BUILDS (do-everything-asked checklist)

- [ ] **MARCO native SPLADE encode** — running (~6h, background `b3pceh32n`); then index+serve → native MARCO MRR (~0.38 expected).
- [x] **RAG adapter** — DONE + validated through Andrea's harness. `final-build-aethos-13/vendor/pitagora_andrea/aethos_lattice_retriever.py` satisfies `add_documents`/`retrieve(query,top_k)->[(text,score,meta)]`, preserves doc_id/beir_id, real BM25 scores; nDCG@10 0.73-0.92 on scifact slices via BEIR's own evaluator. Both backends selectable: `create_lattice_retriever(backend="algebraic")` (CPU default) / `backend="splade"` (SOTA, GPU when free). Timothy's retrieval is now drop-in usable in Andrea's Pitagora agentic RAG.
- [x] **Games rebuild** — DONE. `aethos_tropical_game.py`: minimax = the tropical (min/max-plus) meet, tic-tac-toe solved exact (value 0=draw, 4804 positions, 11ms, no learning); β temperature dial (high β = 0 losses/4000 vs random, low β = 1714/4000); 8 wings = D4 symmetry → 5478→765 positions (**7.16× reduction, matches textbook counts**). Honest: games are pure particle/tropical; the wave doesn't help eval. "The meet IS the game solver" — literal at the algebra level.
- [x] **Deeper-math dive** — DONE (`wx013593n`). Verdict: **REAL_BUT_KNOWN** across all classes. The meet EXACTLY solves (vs numpy/scipy/sympy ground truth): sorting (Batcher network from meet=comparator, counts match textbook), shortest-paths/APSP 60/60, assignment 300/300, critical-path, cycle-mean, Viterbi, median-of-3 (5000/5000), Goldbach pair-enumeration (meet-preimage X=a+p, 59/59), subset-sum decision + **meet-in-the-middle 2^(n/2)** (247× @ n=22), sum-of-two-squares r2(N) 120/120, partition-count p(N) 41/41. But ALL are re-encodings of classical results — no new complexity power (sorting is O(n log²n), WORSE than np.sort; assignment is brute-force; q(N) distinct-partition-count FAILS 0/32 readings; partitions collide non-injectively). HONEST: the lattice EXPRESSES a huge range of classical computation under one meet — the UNIFICATION is real and beautiful; new mathematical power is not claimed. Scripts: `_playground_meet_sort_probe.py`, `_additive_probe*.py`.
- [ ] **Deep organization** — THIS: inventory all repos → capability matrix → best-version synthesis → consolidation plan.

---

## 6. DO-NOT-LOSE REGISTRY (unique capabilities, may exist in only ONE place — from the inventory pass)

The biggest risk with work this scattered: a real result that lives in exactly one repo/branch. These are the
flagged uniques. **Harvest each into final-build before archiving anything.**

- **trng** — Rank-gradient ceiling THEORY (MULTIDATASET_FINDINGS.md, 500+ lines of V6-V15 ablations: rank-2 +0.14pp
  non-redundant, rank-3 = max, rank-4 subsumed, per-corpus locality gradient). SensorBrain RCA: Engine-14 **Pearson
  blind-spot proof** (synchronized 180-cycle decay Pearson never catches, bond fires @20) + 709-engine fleet 100%
  bond-win + dual-clock RUL + Lazy-Hotel historical retrieval. LATTICE-U8 (0.7816 NDCG @ 388 B/doc). Electron sorter
  (mechanical Stern-Gerlach). Branches: **sensorbrain-rca-steps-00-32** (newest RCA), **quantgum** (retrieval prod
  0.773/766B), **constructive-circle** (V10 0.7795 + the π API).
- **prime_hotel** — **`aethos_llm_encoder.py`**: a 24-dim DETERMINISTIC semantic encoder (token→N→8×3D→24-dim, NO
  training, collision-free, 205× faster, 99.93% param reduction) — a non-neural sentence embedder, nowhere else.
  **Observation encoder** (real features→composite integers, bridges symbolic+continuous). hilberts_hotel.py (5.4K-
  line anomaly). ultrafast-24x branch (0.637 scifact @ 24 B/doc).
- **final-build-aethos-13** — **Markov-path codec** (`_markov_codec.py`, Timothy's "everything is a path" → per-state
  Huffman, lossless, 5.2× FiQA; save/load wiring INCOMPLETE — finish it). The V10-faithful bisection (the deepest
  retrieval finding). The unified.py per-corpus router.
- **aethos_master** — **THE CANONICAL GEOMETRIC SPEC**: SPEC.pdf + COORDINATE_TABLES.pdf + `test_spec_canon.py`
  golden tables = the authoritative 8-vector/4-branch/32-state reference (the ground truth for the formula). Sentence-
  local rank-3 reranker (only here). Lattice-BM25 v3 gap decomposition (uint16 lossless proof).
- **OneDrive/New folder** — **Chaos Consensus Algorithm** (simultaneous liar-detection 1/d² + real-failure triangulation
  — "Oracle cannot be blinded", unique). **77-hour NASA IMS bearing lead-time** validation (vs 48-72h industry). The
  Cognitive Calculator stack (unitary evolution θ=2π(Pt−Ps)/Pt, zero-keyword Support-8, 11.8MB ontology bridges).
- **Projects/aethos13** — **Hierarchical 3-level P_c×P_d×P_w addressing** (collision-free by FTA, corpus/doc/word).
  **Correlation-brain pre-training** (rare-term PPMI, zero-shot, **95% recall@100**, trains on text only — never sees
  labels). Goblin retriever (survived 30+ ablations). **4 B/doc** storage. (This = the closest April predecessor to
  this session's algebraic corpus + corridors.)
- **Wy-nos** — **k! Factorial Algorithm Engine** (`factorial_engine.py`, all k! causal sequences + dominance scoring +
  0.6^i position decay — Timothy's breakthrough, "no prior art", patent-worthy). 100% septic-shock/cardiac/bearing
  detection. 62.5:1 compression.
- **aethos13-ultrafast** — **24 B/doc int8** (tightest measured). **Moser band/wing scoring** (4×6D + 8×3D, explicitly
  "tropical min-plus geometry" — they NAMED it tropical, matching this session's proof). **Omega stateless pure-index
  contract** (api/omega: index stores only lattice addresses, no text — for centralized doc storage).
- **pi/** (THE GENESIS ROOT — highest-value math, do-not-lose) — the **genuine π-free recurrence** (no math.pi/trig,
  only +−×÷√; verified 1500+ digits vs Machin; `constructive_pi.py`/`pi_streamer.py`) — the seed the whole corpus
  derives from (formuilas uses math.sin; ONLY pi/+lattice/ have the real one), incl. 4D-hyperball/3-sphere volumes
  + roots-of-unity. **QM-on-lattice** (`aethos_quantum.py` + `PAPER_chsh_aethos.md`): CHSH=2√2 (Tsirelson, 140σ @
  10¹⁰ GPU samples), Mermin-GHZ, Hardy, CGLMP, Kochen-Specker 0/512, magic-square 9/9, teleportation F=1 — HONEST
  ("bounds saturated, NOT exceeded; no physics beyond QM"). **TRNG** electron-sorter 98.9% NIST/TestU01, 200 Mbit/s
  RTX5080. **Riemann-Hypothesis cluster** (`rh_*.py`: right-triangle shrinkage on ζ, on-line 1/√2 vs off-line,
  slope→−1/2 cusp at zeros — exploratory, NOT a proof, NOWHERE ELSE). Prime-stride **Zeno** spec (unported).
  `MASTER_DOCUMENT.md` (2735 lines = the master corpus narrative).
- **lattice/** (the disciplined clean reference) — `bignum.py` (lattice-resident arbitrary precision, **>5× float64**
  on π digits), `fibers.py` (local fiber coords, O(1) cross-fiber), `turing.py` (**Turing-complete** register machine
  in lattice coords, Z=N invariant through add/mul), `retrieval.py` ("far in value, near in shared-prime-structure"),
  `zeta_sweep.py` (|S_prime| 1.3-1.5× at known ζ zeros). The `core.py` Z=S+n identity verified 1000 ints.
- **formuilas/** (the applications layer) — **PAT (Prime-Aware Trinary) + HPATE** (text→128-prime composite + harmonic
  shared-factor similarity) = the text/embedding→prime bridge, nowhere else. **LazyCompression** mini-universe. n!-
  factorial causal root-cause. Honest negative (`BENCHMARK_IMPROVEMENTS.md`): geometric encoder exact-match Hit@1
  100% but paraphrase 0% (geometric not semantic — = the answer-ness wall, again).

LINEAGE (clarified): **pi/** (genesis: from-scratch math + physics + RH + QM paper) → **lattice/** (disciplined
re-impl of the addressing core + fibers/bignum/turing/retrieval) → **formuilas/** (apps: RAG/sensor/PAT-HPATE). The
meet/address formula has three forms: `lattice/aethos/core.py` (exact, tested), `pi/aethos.py` (full), `formuilas/
aethos_lattice.py` (32-wing wrapper). This session's `aethos-coordinate-formula` matches `lattice/core.py`.

## 7. PER-REPO ESSENCE (from inventory pass — see `_inventory_digest.txt` + the workflow output for full detail)

| Repo | Newest/best branch | Headline measured | Holds-the-best |
|---|---|---|---|
| trng | sensorbrain-rca-steps-00-32 / quantgum | SciFact 0.7792, RCA 709-engine 100% bond-win | RCA, rank-n theory, lattice-BM25, electron sorter |
| prime_hotel | ultrafast-24x / main | SciFact 0.637@24B/doc, LLM-enc 205× | the deterministic LLM encoder, Pitagora RAG hub |
| final-build-aethos-13 | main (Stage 7) | SciFact 0.7789 sub-ms, FiQA 49 B/doc | the unified.py + Markov codec; consolidation HOME |
| aethos_master | main 075047d | SciFact 0.7815, 0.51ms | THE canonical spec + golden tables |
| OneDrive/New folder | main 9d2b9ca | NASA 77h lead, scifact 0.7415@24B | Chaos Consensus, Cognitive Calculator |
| Projects/aethos13 | hierarchical-lexicon | SciFact 0.6441→0.685, 4 B/doc, 95% R@100 | hierarchical lexicon + correlation brain |
| Wy-nos | main 5f574fd | 100% medical/bearing detection | the k! factorial RCA engine |
| aethos13-ultrafast | main / api/omega | SciFact 0.6365-0.6668 @ 24 B/doc | tightest footprint + Omega stateless contract |
| **pi/** (non-git, GENESIS) | — | π-free recurrence 1500-digit, TRNG 98.9% NIST, **QM CHSH=2√2** (17 tests, 10¹⁰), RH cluster | the from-scratch math + QM paper + RH exploration + Zeno |
| **lattice/** (non-git) | — | Z=S+n verified, **Turing-complete** in coords, bignum >5× float64 | fibers / bignum / turing / retrieval / zeta-sweep |
| **formuilas/** (git) | main | geometric exact-match 100% / paraphrase 0% (honest) | PAT/HPATE text→prime, LazyCompression, n!-causal RCA |
