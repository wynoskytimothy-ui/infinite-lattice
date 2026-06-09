# Compound Corpus Stacking — Stronger Correlations, Smarter Disambiguation

**Status:** Active (Jun 2026)

---

## Your model (correct)

> Download every corpus and train on top. It should not hurt — only make it
> smarter. 3-way intersections distinguish contexts: `apple+phone` is technical;
> `apple+pie+fruit` is a different rare-word bundle.

**Yes.** That is exactly how the brain is designed.

---

## Why stacking does not hurt

| Mechanism | What happens when corpus B stacks on A |
|-----------|----------------------------------------|
| `stack_corpus()` / `compound_learn()` | **Adds** co-occurrence counts; never deletes A |
| Pair links | Strength **accumulates** (`count += new`) |
| Morph / bridge | **Re-derived** from full pair set (richer, not replaced) |
| Pretrain / gold triples | **Persist** in saved `.pkl` |
| Membrane filler | Still blocked — `the`, `and` never pollute |

Corpus B does not overwrite corpus A. It **deepens** the same brain file.

```
Brain v1  (SciFact)     →  4.8M links
Brain v2  (+ NFCorpus)  →  4.8M + new (stronger counts, new bridges)
Brain v3  (+ pretrain)  →  + gold triples (quantum, zero, dimension)
Brain v4  (+ qrels 3-way train)  →  + context bundles per query
```

---

## 3-way intersections disambiguate meaning

`apple` alone is ambiguous. **3-way bundles** lock the context:

### Technical cluster

```
apple + phone + chip
apple + phone + software
apple + silicon + processor
```

Rare words: `chip`, `silicon`, `processor`, `software` — **do not** correlate with pie/fruit.

### Fruit cluster

```
apple + pie + fruit
apple + fruit + orchard
apple + pie + cinnamon
```

Rare words: `pie`, `fruit`, `orchard`, `cinnamon` — **do not** correlate with chip/silicon.

The **same anchor** (`apple`) has **different promoted triples** per context.
Query `apple phone` → promotes / predicts technical triples only.
Query `apple pie` → promotes / predicts fruit triples only.

This is the 3-way intersection insight from `aethos_discriminative.py` applied to
the symbol knowledge brain.

---

## Training stack (recommended order)

1. **Download corpus** → `build_from_beir("scifact")` → save brain
2. **Stack next corpus** → `brain.stack_corpus(nfcorpus, name="nfcorpus")`
3. **Pretrain gold gaps** → `compound_learn(PRETRAIN_QUANTUM_GOLD)`
4. **Qrels trinary** → `TrinaryTrainer.train_from_qrels()` per dataset
5. **Markov walk** → strengthen on prediction misses
6. **Plane index** → rebuild κ for fast search

Each step only adds signal. Conflicts are rare because:
- Weak false pairs stay weak (low count)
- Strong true pairs accumulate
- 3-ways require **query overlap** — random cross-context triples are not promoted together

---

## Code

```python
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_trinary_train import TrinaryTrainer

brain = SymbolKnowledgeIndex.load("scifact")
brain.stack_corpus({"d_new": "..."}, name="nfcorpus")
brain.save()

trainer = TrinaryTrainer(knowledge=brain)
trainer.train_query("q1", "apple phone chip", ["gold_tech_doc"])
trainer.predict_triple_completion("apple phone", ["apple", "phone"])
# → chip, software, silicon (not pie, fruit)
```

Demo:

```bash
python -m pytest tests/test_symbol_trinary_train.py::TestAppleDisambiguation -q
```

---

## Honest limit

If corpus B is **huge** and shares ambiguous anchors (`apple`, `bank`, `cell`), pair
counts for the ambiguous word grow in **both** directions. 3-way promotion and
query-conditioned trinary training are what **keep contexts apart** — not the
pair layer alone. Always train with query + gold (or category tag) for ambiguous
anchors.
