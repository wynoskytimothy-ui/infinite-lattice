# AETHOS — master index

A prime-lattice symbolic engine derived from a physics framework (Ψ=(z,ζ) 3D complex
plane, constructive-π rotation, Zeno frame-descent kernel, FTA prime addressing). The
same engine is used as a **retrieval system**, a **neural-network substrate**, a
**root-cause-analysis monitor**, and a **capability testbed**. ~150 Python files at the
root + ~600 in subdirs + a physics derivation set.

**Status legend:** ✅ validated (measured, honest numbers) · ◐ built/exploratory (code
runs, results partial or not independently re-verified here) · ○ theory/derivation.

---

## 0. Start here — the validated headline results

| result | number | where | status |
|---|---|---|---|
| Bearing RCA on real NASA/IMS data | caught fault **~4h early**, root-caused the source | `aethos_bearing_*.py`, `aethos_meet_rca.py` | ✅ |
| AETHOS is a neural network (VSA/Hebbian) | XOR **100%**, **92%** held-out nonlinear generalization | `aethos_nn.py`, `aethos_brain.py` | ✅ |
| Γ-ODE mixer beats attention (LM) | char **−28%** ppl · word **−35/−53%** · text8 **−7.2%** (hybrid **−10.4%**) | `gpu_gamma_scan*`, `gpu_scale_text8.py` | ✅ |
| Γ-ODE long-context speed | **5.2× faster** than flash attention @16k | `gpu_gamma_scan4.py` | ✅ |
| Retrieval representation (MARCO) | conservative stemming **+6.6%** over BM25 | `marco_lab.py`, `stem_safe.py` | ✅ |
| Supervised bridges / knowledge injection (BEIR) | **+6.5pp** scifact / **+1.6pp** | `aethos_bridges.py`, `*_glossary.py` | ✅ |
| Recall boundary | SSM exact-recall **17% vs 100%**; **hybrid fixes it (100%)** | `gpu_mqar*.py` | ✅ |

Honest negatives (also validated): unsupervised correlations do **not** improve retrieval
ranking (similarity ≠ relevance); a naive deep SSM needs the full Mamba recipe to train.

---

## 1. The lattice engine (core substrate)
The append-only prime address book: text → primes → composites → coords.
- **Core:** `aethos_core.py`, `aethos_lattice.py`, `aethos_tokenize.py`, `aethos_token_levels.py`, `aethos_composite.py`, `aethos_compound.py`, `core/` (17 files)
- **Symbol family** (~40): `aethos_symbol_{morph,meets,entangle,knowledge,markov,subjects,synthesis,composite,word,...}.py`
- **Indexing/scale:** `aethos_append_index.py`, `aethos_sharded_index.py`, `aethos_scale.py`, `aethos_pool_tiers.py`

## 2. Retrieval / RAG  (the biggest arc)
- **BEIR engine:** `eval_beir{,_lattice,_symbol}.py`, `lattice_rag.py`, `aethos_lattice_retrieval.py`, `aethos_cascade_retrieval.py`
- **Correlations/bridges:** `aethos_bridges.py` (supervised + gold-doc), `verify_bridges.py`, `verify_golddoc.py`
- **Knowledge injection:** `{scifact,nfcorpus,fiqa}_glossary.py`, `wiki_teacher.py`
- **MARCO ground-up build (this thread):** `marco_lab.py` (harness) → `marco_baseline.py` → `rungs_climb{,2}.py` → `stem_audit.py`/`stem_safe.py` (the +6.6% rung) → `corridor_audit.py`/`cascade_audit{,2}.py` (verified correlations) → `marco_{bridges,corridor,integrated,meet,gated,harder}.py` (every integration wiring tested)
- **Finding:** representation + supervision move ranking; unsupervised similarity doesn't. Full arc in memory `msmarco-retrieval-goal.md`, `rag-signals-beir-finding.md`.

## 3. Neural network / brain / Γ-ODE mixer
- **Proof it's a net:** `aethos_nn.py`, `aethos_brain.py`, `aethos_silicon_brain.py`, `brains/` (20)
- **Γ-ODE LLM mixer:** `gpu_gamma_scan{,2,3,4}.py` (parallel scan + speed), `gpu_mqar{,2,3,4}.py` (recall), `gpu_scaleup*` + `gpu_scale_text8.py` (scale), `gpu_mamba_lite.py`, `gpu_scan_grad.py`. Writeup: `GAMMA_MIXER_FINDINGS.md`.

## 4. Bearing RCA (validated)
`aethos_bearing_monitor.py`, `aethos_bearing_multisignal.py`, `aethos_bearing_primes.py`, `aethos_brain_bearings.py`, `aethos_meet_rca.py`. Data: `C:\Users\wynos\Downloads\{1st,2nd}_test`. Memory: `ims-bearing-monitor-validated.md`.

## 5. Hidden-capability audits  ◐
Test 1–48 capability probes (Russell-paradox impossibility, perfect hash via FTA,
compositional CRDT, graph coloring, proof checker, reversible computing, tunneling/
double-slit). Drivers: `aethos_discover.py`, `audit_100.py`, `pattern_audit.py`,
`aethos_games.py` (games). *Built/run; not all independently re-verified in this thread.*

## 6. The physics formula & constructs
- **Derivations:** `section_01…12_*.md` (photon sea → electron → entanglement → tunneling → Zeno), `derivations/` (51), `aethos_complete_{11,12}_sections.md`, `aethos_proof_extensions.md`
- **Constructs:** `aethos_complex_plane.py`, `aethos_complex_rotation.py`, `aethos_pi_{bridge,phase}.py`, `aethos_quantum.py`, `aethos_qubit_node.py`, `aethos_zeno_onset.py`, `aethos_hilbert.py`, `plane3d/` (24), `pi/` (8)
- **Implemented-vs-described status:** memory `aethos-constructs-implemented-vs-described.md`, `zeno-kernel-is-the-os.md`. ○/◐

## 7. Unified design (this thread's synthesis)
`UNIFIED_ENGINE.md` — multi-level primes → complex-plane rotation → corridors → meet →
composites-as-exact-memory (factoring = unbinding). The one open question: does the
complex-plane geometry encode semantics, or must corridors come from co-occurrence.

---

## Data on disk
- `text8` (95 MB) + `text8.zip` — Γ-ODE scale-test corpus
- `marco_pool.pkl` (100 MB) — cached MARCO eval pool
- MARCO full: `C:\Users\wynos\trng\marco_data\` (collection 2.9 GB + qrels + queries)
- Bearing: `C:\Users\wynos\Downloads\{1st,2nd}_test`
- BEIR: `C:\Users\wynos\.cursor\worktrees\prime_hotel\bbl\beir_datasets\`

## Persistent memory (the durable conclusions — survive across sessions)
`...\.claude\projects\C--Users-wynos-New-folder--3-\memory\` — 15 files indexed by
`MEMORY.md`: bearing validation, neural-net proof, Γ-ODE mixer, BEIR/RAG findings, the
MARCO arc, distributional-semantics correction, lattice-training paradigm, and the
working principles (measurement rigor, explore-don't-say-can't, honest two-sided tests).

## Top-level docs
`ARCHITECTURE.md`, `ONTOLOGY.md`, `PARADIGM.md`, `README.md`, `RAG_README.md`,
`FINDINGS.md`, `GAMMA_MIXER_FINDINGS.md`, `UNIFIED_ENGINE.md`, this `INDEX.md`.

---

## Honest scorecard
- **Validated (numbers, reproduced):** bearing RCA, neural-net proof, Γ-ODE beats attention (small–mid scale), retrieval representation +6.6%, supervised bridges/knowledge injection, the recall boundary + hybrid fix.
- **Built, not fully re-verified here:** the 92-file engine breadth, the Test 1–48 audits, the constructs, games.
- **Open / not working:** unsupervised correlations for *ranking* (definitively no on the easy pool); deep-SSM at scale needs the Mamba recipe; complex-plane-geometry-as-semantics untested.
- **Caveat across the board:** small-to-mid scale; single-seed at the largest sizes; the whole tree is **uncommitted** (327 untracked entries).
