# Lazy Correlation Brain — Query → Gold Doc → Deepening Corpus

**Status:** Active (Jun 2026). Implements compound learning on `SymbolKnowledgeIndex`.

---

## Core idea

Your formulas support **lazy evaluation with branching**:

1. **First pass** — corpus scan builds only what co-occurs (direct window links).
2. **Branch** — morph families and bridge links are *derived*, not stored naively.
3. **Gaps** — unlinked signal words are the lazy frontier (`gap_words()`).
4. **Gold doc pretrain** — feed a small explanatory document *before* the main corpus
   to teach correlations the big corpus never states explicitly.
5. **Compound learn** — each new corpus pass deepens the brain without forgetting
   prior pair counts (`compound_learn()`).

```
query ──► gold doc (pretrain) ──► saved brain ──► main corpus (SciFact)
                │                      │                    │
                └──── teaches ─────────┴──── deepens ──────┘
                              new direct + morph + bridge
```

---

## Example: quantum + inductive biometrics + zero dimension

SciFact alone rarely links `quantum` ↔ `zero` ↔ `dimension` (biomedical abstracts).

**Pretrain gold doc** (`gold_quantum_biometrics`):

> Quantum inductive biometrics links measurement to zero dimensional Hilbert
> space boundaries. Quantum states projected to zero dimension carry inductive
> biometric signatures…

After `compound_learn({gold_quantum_biometrics: text})`:

| Pair | Before SciFact | After pretrain |
|------|----------------|----------------|
| `quantum` + `zero` | missing | **direct** |
| `quantum` + `dimension` | missing | **direct** |
| `inductive` + `biometrics` | missing | **direct** |
| `zero` + `dimension` | missing | **direct** |

SciFact then adds **new** links (millions) on top — pretrain links persist in
`_cooccur_pairs` counts and `cross_links`.

---

## Lazy evaluation layers

| Layer | When computed | Storage |
|-------|---------------|---------|
| **Direct** | Doc scan (window co-occurrence) | pair count in brain |
| **Morph** | Branch when suffix families exist | shared root prime |
| **Bridge** | Branch from direct + morph siblings | `via` word |
| **κ plane** | BIT 12 index build | `scifact_plane.pkl` |
| **Gaps** | On demand | words with no edge |

Each corpus run only **extends** `_cooccur_pairs` for new docs; `_build_cross_links()`
re-branches morph and bridge from the full pair set.

---

## Workflow

### 1. Link query to gold doc (pretrain)

```python
from aethos_symbol_knowledge import SymbolKnowledgeIndex, PRETRAIN_QUANTUM_GOLD

brain = SymbolKnowledgeIndex.build_from_corpus(PRETRAIN_QUANTUM_GOLD, dataset="pretrain")
brain.save()  # brains/symbol_knowledge/pretrain.pkl
```

### 2. Load main corpus brain and compound learn

```python
brain = SymbolKnowledgeIndex.load("scifact")
report = brain.compound_learn(PRETRAIN_QUANTUM_GOLD)
brain.save()  # overwrites or save as scifact_compound.pkl
```

### 3. Verify memory

```python
brain.remembers("quantum", "dimension")   # True after pretrain
brain.query_gold_links(
    ["quantum", "zero", "dimension", "inductive", "biometrics"],
    "gold_quantum_biometrics",
)
```

### 4. Fast search (BIT 12)

```python
from pipeline import build_symbol_plane_index, route_symbol_plane_candidates

plane = build_symbol_plane_index(brain)
route = route_symbol_plane_candidates(brain, plane, ["quantum", "zero", "dimension"])
# gold_quantum_biometrics should rank high if keys overlap
```

---

## Test

```bash
python scripts/test_pretrain_brain_memory.py
python scripts/test_pretrain_brain_memory.py --full   # uses full scifact.pkl
```

**Gate:**

- [ ] Pretrain teaches `quantum`+`dimension` (missing in SciFact-only)
- [ ] `compound_learn` adds links without dropping prior SciFact pairs
- [ ] Save/load round-trip preserves pretrain links
- [ ] Plane router returns gold doc for quantum query

---

## Relation to 12-bit universe

| Bit | Role in lazy brain |
|-----|-------------------|
| 1–2 | Word → Ψ → κ for each new token |
| 3–4 | Inverted κ index for fast recall |
| 12 | Symbol knowledge + correlation meet κ |
| Brain file | `brains/symbol_knowledge/*.pkl` (~24 B witness → millions of meets) |

Pretrain is the **smallest corpus that closes the gap** between a query and a gold
answer before the expensive main corpus teaches the rest.

**Next layer:** Markov correlation cascade — `derivations/markov_correlation_cascade.md`
and `aethos_symbol_markov.py` (predict next word at each node; strengthen on miss).
