# The paradigm: a retriever that learns by appending data, not by retraining weights

This is the framing document for the engine in this repo. The README has the numbers
and the API; [FINDINGS.md](FINDINGS.md) has the method and the negative results. This
file says what it all *means*, and stays honest about the boundaries — because an
honest claim is the only kind worth making.

---

## The one-line shift

A neural retriever stores everything it knows inside a frozen block of trained
weights. To change what it knows, you retrain it. **This engine stores everything it
knows as open, append-only data — so it learns the way a database grows: by adding
rows.** Three different kinds of learning, one mechanism (append), zero retraining:

| what it learns | how, in a neural retriever | how, here |
|---|---|---|
| a new **document** | re-embed it, rebuild the index | append a posting list — O(1), no reindex |
| a new **relevance** signal | fine-tune the encoder on it | count it into a bridge — no gradient descent |
| what a new **word means** | retrain + re-embed the whole corpus | append a definition — one editable line |

Every score traces back to a prime address; every bridge to the training queries that
formed it; every definition to a line you can read, verify, and correct. It is a glass
box, not a sealed artifact.

## Why "store it as open data" doesn't fall apart

The obvious objection is that open, editable knowledge should be slow, huge, and
brittle. We measured each worry and it isn't:

- **Speed.** Lexical query **0.55 ms**, full stack (with the learned bridges) **~1 ms** —
  numpy-vectorized, 67× faster than where we started, every step lossless.
- **Memory.** **743 B/doc** on disk. And new knowledge is *nearly free*: the engine stores
  shared subwords (char-trigrams), whose address space saturates — so 28 definitions
  (3,716 subword tokens) allocated **14 new primes, zero new trigrams**. The marginal
  cost of a new document falls from 59 primes to 6 as the corpus fills. Continual
  learning is memory-bounded.
- **Scale.** Exact sharding to any N (a fan-out + merge equals one index, verified), and
  champion lists keep queries sub-millisecond at any corpus size.
- **Determinism.** Identical regardless of ingest order; reload is lossless and stays
  appendable. No training run to reproduce, no seed to chase.

## What it actually achieves (measured, held-out, full corpus)

scifact, the journey: **BM25 0.665 → our lexical 0.700 → +bridges 0.759 → +knowledge 0.779**
(nDCG@10). That last number is **+0.108 over zero-shot ColBERT (0.671)** — and every step
after lexical was *added as data*, not trained.

Across BEIR: scifact **0.779**, nfcorpus 0.335, fiqa 0.244, trec-covid 0.606, touché 0.355.
Beats or matches BM25 on most; beats ColBERT where the corpus is lexically clean.

## The honest boundary

This is not "we beat neural retrieval." We don't, everywhere — and the shift doesn't
need us to:

- On large **question→answer semantic gaps**, a trained dense encoder still wins (fiqa:
  ColBERT 0.317 vs our 0.244). Its weights encode distributional meaning we can't match
  by counting alone.
- **Knowledge injection works when the definition's vocabulary overlaps the gold's** —
  it recovered 12 of 27 gap queries on scifact, not all. Coverage and definition quality
  are the levers.
- **Unsupervised distributional semantics hit a ceiling** here: second-order rare-anchored
  similarity finds genuine synonyms (`curcumin~turmeric`, `prp~prion`) but barely moves
  retrieval. Meaning that isn't in the co-occurrence statistics has to come from outside.

The shift is *orthogonal* to raw semantic strength. It isn't about having the strongest
frozen representation — it's about being **the system that never has to freeze**: it learns
continuously, forgets nothing, costs almost nothing to extend, and can be corrected on the
spot. And the one place counting hit a wall — rare words with no statistics — is closed by
the very mechanism this paradigm makes cheap: *tell it what the word means*, one appendable
definition at a time.

## Why you can trust the live path: we named every dead end

This engine is credible because the work that produced it was honest about what failed:

- the **0.78** we set out to chase was a 500-document subset artifact; at full corpus it
  was ~0.65, and our append index already beat it
- the **letter-prime ψ-encoder** scored 0.10 — surface geometry, not meaning
- **unsupervised PPMI / manifold / second-order** expansion drifted or barely helped
- **champion lists** are a speed/accuracy knob, not lossless; we labelled them so
- **hyperparameter tuning** on a small dev set *overfit* and hurt held-out test

Every negative is recorded in [FINDINGS.md](FINDINGS.md) with the script that proves it.
The positives are trustworthy because the negatives weren't hidden.

## In one paragraph

Retrieval, rebuilt as append-only counting on prime addresses: a BM25-class engine whose
documents, relevance judgments, and world-knowledge are all editable, inspectable,
append-only data instead of a sealed block of weights. It is fast, compact, deterministic,
shards exactly, learns continuously without retraining, extends for almost no memory, and
can be read and corrected line by line. It does not replace a trained dense retriever on
the hardest semantic gaps — it is a different kind of system entirely: **knowledge as
infrastructure you edit, not an artifact you re-grow.**

---

*Everything here is reproduced by a script in `scripts/` and summarized in the README. Start
with `python scripts/run_beir.py scifact 1`, then `knowledge_bridges.py`, `vocab_saturation.py`,
`bench_sharded.py`.*
