# Infinite Lattice

AETHOS **φ-Prime Lattice** — recursive geometric engine (VA1–VA4, 32 wings, unbounded transgressor depth) with token promotion, constructive π, physics derivations, and BEIR retrieval.

## Core stack (`core/`)

| Module | Purpose |
|--------|---------|
| [`core/primes.py`](core/primes.py) | Odd-prime chains, promotion pool tiers |
| [`core/l1_characters.py`](core/l1_characters.py) | Letter → prime (L1) |
| [`core/l2_subwords.py`](core/l2_subwords.py) | PMI subword promotion (L2) |
| [`core/phi_lattice.py`](core/phi_lattice.py) | φ-lattice coordinates, swap meet, `prime_factor_similarity` |
| [`core/bridge_registry.py`](core/bridge_registry.py) | Sync L2 into production registry |

## Production lattice (`aethos_*.py`)

- **Lattice math:** `aethos_lattice`, `aethos_recursive`, `aethos_sequences`, `aethos_origins`
- **Tokens L1–L9:** `aethos_promotion`, `aethos_pipeline`, `aethos_crossmeaning`
- **Retrieval:** `aethos_hub_signature`, `eval_beir.py`
- **Physics:** `aethos_physics.py`, `derivations/`, `section_*.md`
- **Constructive π:** `pi/constructive_pi.py`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`core/README.md`](core/README.md).

## Quick start

```bash
python -m pytest tests/ -q
python run_aethos.py
python -m pytest test_aethos.py -q
```

BEIR eval (requires local `beir_datasets` or `BEIR_DATA_DIR`):

```bash
python eval_beir.py --datasets scifact --max-docs 500
```

## Key formulas

- **Coordinates:** 4 canonical branches × 8 vectors = 32 wings; `compute_coordinates(chain, n, wing)`
- **Z plateau:** interior segments `Z = sum(P)` when `k > 2`
- **Swap meet:** `bank(p) @ n=q` meets `bank(q) @ n=p`
- **Retrieval:** scale-invariant `prime_factor_similarity` on composite prime factors (not raw Euclidean on large products)

## License

Add your license here if publishing publicly.
