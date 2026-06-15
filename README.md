# AETHOS — π lattice & 3D complex plane

Two constructions — **not** the same object. See **[`ONTOLOGY.md`](ONTOLOGY.md)**.

| Construction | What it is | Code |
|--------------|------------|------|
| **π lattice** | Unit-circle bisection → π, cells, ±B branches | `pi/constructive_pi.py` |
| **3D complex plane** | **Ψ = (z, ζ)** — spring plane + depth; 32 chambers; lattice formula on anchor chains | `aethos_complex_plane.py`, `aethos_lattice.py`, `aethos_sequences.py` |

Primes name a **species** (`SequenceKind.PRIMES`), not the arena. Unbounded **n/k/depth** is extension, not a separate "infinity lattice."

Token promotion, constructive π, physics derivations, and BEIR retrieval share the same repo.

## Core stack (`core/`)

| Module | Purpose |
|--------|---------|
| [`core/primes.py`](core/primes.py) | Odd-prime chains, promotion pool tiers |
| [`core/l1_characters.py`](core/l1_characters.py) | Letter → prime (L1) |
| [`core/l2_subwords.py`](core/l2_subwords.py) | PMI subword promotion (L2) |
| [`core/phi_lattice.py`](core/phi_lattice.py) | **3D complex plane engine** (legacy filename) — coordinates, swap meet |
| [`core/bridge_registry.py`](core/bridge_registry.py) | Sync L2 into production registry |

## Production (`aethos_*.py`)

- **3D complex plane:** `aethos_lattice`, `aethos_recursive`, `aethos_sequences`, `aethos_origins`, `aethos_complex_plane`
- **Tokens L1–L9:** `aethos_promotion`, `aethos_pipeline`, `aethos_crossmeaning`
- **Retrieval:** `aethos_hub_signature`, `eval_beir.py`
- **Physics / quantum:** `aethos_physics.py`, `aethos_quantum.py`, `aethos_qubit_node.py`, `derivations/`, `section_*.md`
- **π lattice:** `pi/constructive_pi.py`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`core/README.md`](core/README.md).

## Capability suite (33 tests)

Beyond retrieval, the formula has been stress-tested for load-bearing
properties — paradox resolution, a context-mixing compressor that beats
zlib/bz2/lzma (0.83 bits/byte, native-JIT, 32 parallel lanes), a halting
predictor, byte-exact suspend/resume, qubit simulation (Bell 2√2, GHZ), and
the **Zeno kernel** that unifies termination, addressing, GC, safety, and
timing into one prime frame-descent.

```bash
python scripts/run_capability_suite.py          # fast tier (~45s)
python scripts/run_capability_suite.py --all     # + full-corpus codec tests
```

Full write-ups: [`derivations/formula_capability_tests_results.md`](derivations/formula_capability_tests_results.md).

## Quick start

```bash
python scripts/show_3d_complex_plane.py   # formula -> Psi=(z,zeta) demo
python -m pytest tests/ -q
python run_aethos.py
python -m pytest test_aethos.py -q
```

BEIR eval (requires local `beir_datasets` or `BEIR_DATA_DIR`):

```bash
python eval_beir.py --datasets scifact --max-docs 500
```

## Key formulas (3D complex plane)

- **State:** `Ψ = (z, ζ)`, `z = X + iY`; label `α = (A, b, w, n)`
- **Chambers:** 4 branches × 8 wings = 32; `wing_transform(branch, chain, n, wing)`
- **Z plateau:** interior segments `ζ = sum(A)` when `k > 2`
- **Swap meet:** `bank(p) @ n=q` meets `bank(q) @ n=p`

## License

Add your license here if publishing publicly.
