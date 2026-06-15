# AETHOS architecture

**Ontology:** [`ONTOLOGY.md`](ONTOLOGY.md) — **π lattice** (bisection, `pi/`) vs **3D complex plane** (Ψ=(z,ζ), lattice formula). Do not use "prime lattice" or "infinity lattice."

Two layers: **3D complex plane core** (pure formulas) and **token processor** (semantics).

```
┌─────────────────────────────────────────────────────────────┐
│  Projects: physics sections, codec, custom anchor species   │
│                    aethos_core.py                           │
│  32 wings · VA1–VA4 · k-depth · origins · meets · active    │
└───────────────────────────┬─────────────────────────────────┘
                            │ formula_coord(chain, n, wing)
┌───────────────────────────▼─────────────────────────────────┐
│  NLP / vocabulary: promotion L1–L9, natural reading         │
│              aethos_token_processor.py                      │
└───────────────────────────┬─────────────────────────────────┘
                            │ overlay, pipeline
┌───────────────────────────▼─────────────────────────────────┐
│  Optional: codec bytes, word order dots, full pipeline      │
└─────────────────────────────────────────────────────────────┘
```

## 3D complex plane core (`aethos_core.py`)

**No tokens.** Use for any project that only needs your formulas.

| Capability | API |
|------------|-----|
| 32 wings per bank | `bank(chain)`, `LatticeProject.all_wings_at(n)` |
| 4 branches × 8 vectors | `formula_coord`, `canon_recursive`, `canon_on_chain` |
| Countable species | `SequenceKind`: PRIMES (odd), EVENS, POWERS_OF_2, FIBONACCI, … |
| k-anchor recursion | `LatticeBank32K`, `segment_index`, `z_depth` |
| Dimensionless origins | `OriginTree`, 3 children per node, 32 wings per room |
| Solo swap meet | `solo_swap_meet(p, q)` |
| Active nodes | `active_network(100)` |
| Named projects | `open_project("electron", origin_depth=2)` |

```python
from aethos_core import AethosLatticeCore, SequenceKind

core = AethosLatticeCore()
electron = core.open_project("electron", chain_len=10, origin_depth=3)
print(electron.coord(n=7))           # one wing
print(electron.all_wings_at(7))    # all 32
print(core.solo_swap_meet(3, 11))
```

Implementation modules (core only): `aethos_lattice`, `aethos_recursive`, `aethos_sequences`, `aethos_origins`, `aethos_active`, `aethos_permutation`, `aethos_golden_coords`.

## Token processor (`aethos_token_processor.py`)

**Vocabulary and reading** on top of core coordinates.

| Layer | Role |
|-------|------|
| L1–L3 | Letters, sub-words, words (intersection vs dedicated prime) |
| L4–L6 | Float correlations |
| L7–L9 | Natural clusters / Markov |

`registry.lattice_address(...)` builds an anchor chain from tokens, then calls **`formula_coord`** from core (single math path).

```python
from aethos_token_processor import TokenProcessor

proc = TokenProcessor()
proc.ingest("apple phone chip", "apple fruit pie")
print(proc.resolve("apple", ["phone", "chip"]))
print(proc.lattice_address("apple"))
```

## Combined pipeline (`aethos_pipeline.py`)

```python
from aethos_pipeline import AethosPipeline, smoke_corpus

pipe = AethosPipeline()
pipe.ingest(*smoke_corpus())
pipe.open_lattice_project("photon", origin_depth=2)  # core-only side
```

## Anchor policy

| Chain | Values |
|-------|--------|
| `PRIMES` | 3, 5, 7, 11, … (2 skipped) |
| Letters a–z | First 26 odd primes |
| Other species | Independent chains (evens may use 2) |

## Application lanes (optional)

| Lane | Module | Uses |
|------|--------|------|
| Codec payload dots | `aethos_codec` | core + origins |
| Word order dots | `aethos_words` | core + permutation |
| Semantic overlay | `aethos_overlay` | tokens + codec local |

## Capability suite & derived systems

The formula's load-bearing properties are stress-tested in 33 capability
tests (`scripts/test_*.py`, results in
[`derivations/formula_capability_tests_results.md`](derivations/formula_capability_tests_results.md)).
Several grew into standalone systems on the same core math:

| System | Module(s) | What it is |
|--------|-----------|------------|
| Recursive lattice | `aethos_recursive_lattice` | Promoted-prime hierarchy; `walk_down`/`walk_up`, level invariant (Russell-safe), provenance |
| Context-mixing codec | `scripts/test_chamber_mixer_v*` | FTA-addressed chambers + logistic mixer; 0.83 bits/byte (beats zlib/bz2/lzma), native-JIT, 32 parallel quadrant lanes; byte-exact suspend/resume |
| Quantum / qubit | `aethos_quantum`, `aethos_qubit_node`, `aethos_ocean_graph` | Exact statevector; any node = qubit; ocean-fill dial (Werner 1/√2); Bell 2√2, GHZ |
| Zeno kernel | `section_12` + `scripts/test_zeno_*` | One prime frame-descent = gatekeeper (terminate) + bookkeeper (address) + janitor (bounded resource) + security (no singular state) + ruler (clock); proven substitutable for the recycler |

**The Zeno frame-descent is the shared substrate:** termination certificates,
prime-base addressing, bounded-resource recycling, illegal-state safety, and
the tick schedule are all the same descent (`w_n = 1/∏p_k`, positive-width
floor) in different domains — see Tests 32–33.

## Entry points

```bash
python aethos_core.py              # lattice-only demo
python aethos_token_processor.py   # token demo
python run_aethos.py               # full regression
python scripts/run_capability_suite.py   # 33 capability tests (fast tier)
python aethos_quantum.py           # 2-qubit Bell/CHSH demo
```

## Golden & persistence

- `fixtures/golden_coords.json` — formula regression
- `aethos_persist.py` — token reader state only (not core projects)
