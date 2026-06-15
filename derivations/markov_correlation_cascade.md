# Markov Correlation Cascade — Predicting Text Through Node Transitions

**Status:** BIT 13 design + `aethos_symbol_markov.py` (Jun 2026)

---

## Your question

> Can the formula be Markovian — predict words that *should* come next; when
> words don't match, build stronger correlation? Each correlation doesn't need
> the full sequence — only **transitions at nodes** monitored through primes,
> intersections, and composites.

**Yes.** That is the right architecture for this stack.

---

## Key insight

A full sentence does **not** need one global model.  Each **node** (word at a
prime/intersection address) only needs:

1. **Outgoing transitions** — who followed this node in corpus text?
2. **Correlation neighbors** — who co-occurs / bridges / shares morph root?
3. **Prime overlap** — who shares chain primes on the 3D plane?

Reading text = walking node → node.  Prediction = local cascade at each step.

```
  node_t (word, chain, imag)
       │
       ├── Markov:     P(w_{t+1} | w_t)     from bigram counts
       ├── Correlation: neighbors in brain   from cross_links / plane adjacency
       └── Prime:      Jaccard(chain_t, chain_{t+1})
       │
       ▼
  predicted top-k words
       │
       ├── HIT  → transition already known
       └── MISS → strengthen (bump cooccur + direct link)
```

---

## Three cascade layers

| Layer | State | Transition data | Formula |
|-------|-------|-----------------|---------|
| **L1 Markov** | word token | `bigram[(w₁,w₂)]` | `P(w₂\|w₁) = count / Σ count` |
| **L2 Correlation** | cross_link edge | direct / morph / bridge | strength × kind boost |
| **L3 Prime** | `chain(w)` primes | intersection overlap | `\|chain₁ ∩ chain₂\| / \|chain₁ ∪ chain₂\|` |

Score blend (default):

```
score(w₂ | w₁) = 1.0·P_markov + 0.6·corr + 0.35·prime_overlap
```

Each layer is **lazy** — computed from saved brain, not re-parsed corpus.

---

## Mismatch → stronger correlation

```python
brain.observe_step("quantum", "cell")  # predicted zero, dimension, ...
```

If `cell` not in top-k:

1. `mismatch_strengthen += 1`
2. `_cooccur_pairs[("quantum","cell")] += 2`
3. `_add_link(..., kind="direct")` — brain deepens

This is **online compound learn** one transition at a time — same philosophy as
`compound_learn()` but driven by prediction error.

---

## Relation to 12-bit universe

| Bit | Markov role |
|-----|-------------|
| 1–2 | Node address: word → Ψ → κ |
| 12 | Correlation adjacency for cascade |
| **13** | Markov transitions + predict + strengthen |

The **24-byte witness** (κ / hub pin) identifies the node; transitions are
counts and correlation strengths — millions of paths from compact storage.

---

## Usage

```python
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_symbol_markov import build_markov_brain

knowledge = SymbolKnowledgeIndex.load("scifact_compound")
brain = build_markov_brain(knowledge)

# predict
brain.predict_next("quantum", top_k=5)

# walk text — monitor transitions, strengthen misses
brain.walk_text("quantum zero dimension biometrics analysis")

# accuracy
brain.accuracy()  # top1, top5, mismatch_strengthen
```

---

## What this can and cannot do (honest)

**Can:**
- Predict likely **next content word** in-domain (SciFact + pretrain)
- Detect **surprising transitions** and deepen correlations automatically
- Cascade through **saved** correlations without re-scanning full corpus each step
- Run in **milliseconds** per step with plane adjacency attached

**Cannot yet:**
- Generate fluent long prose (no char-level LM)
- Predict across membrane filler words (the, and — excluded by design)
- Beat neural LMs on raw perplexity without more training passes

**Next improvements:**
- Category-conditioned Markov (reuse `MarkovCrossLattice` themes)
- κ-meet transition: predict via `correlation_meet_keys` not just word bigram
- Perplexity eval on held-out SciFact abstracts

---

## Trinary gold training (query → rarest 3-way)

When training with qrels, promote the **rarest 3-way correlation** in each gold doc:

```python
from aethos_symbol_trinary_train import TrinaryTrainer, load_beir_qrels_train

trainer = TrinaryTrainer(knowledge=knowledge)
trainer.train_query(qid, query_text, gold_doc_ids)
# multiple golds → picks rarest doc, promotes shared 3-ways across all golds
trainer.predict_triple_completion(query, ["quantum", "zero"])  # 3rd word
```

```bash
python scripts/train_trinary_qrels.py --max-queries 30
```

## Test

```bash
python -m pytest tests/test_symbol_markov.py tests/test_symbol_trinary_train.py -q
python aethos_symbol_markov.py
```
