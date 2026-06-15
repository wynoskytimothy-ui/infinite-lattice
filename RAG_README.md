# lattice-rag

**An append-only, multi-view lattice retrieval engine — BM25-class accuracy by pure counting on prime addresses. No neural weights, no training loop, O(1) append, no reindex, deterministic and verifiable.**

*(The retrieval engine extracted from the AETHOS repo; see [`README.md`](README.md) for the broader project.)*

Each token is a *particle address*: a `(view, token)` pair gets a unique prime on first sight and keeps it forever. Documents are products of those addresses; the inverted index is a posting list per prime. Scoring is multi-view BM25; the optional accuracy layer learns query→document term *bridges* by counting relevance judgements (qrels) — supervised retrieval with no gradient descent. The whole thing is built to **grow forever by appending**: new documents are new addresses, never a reindex, never a retrain, never forgetting.

> **Read [`PARADIGM.md`](PARADIGM.md) for what this all means** — a retriever that learns by *appending data* (documents, relevance, *and* world-knowledge) instead of retraining weights — and [`FINDINGS.md`](FINDINGS.md) for the method and the honest negative results.

---

## Headline numbers (BEIR, held-out test, nDCG@10)

| corpus | docs | lexical | **+ bridges** | BM25 | ColBERT |
|---|---:|---:|---:|---:|---:|
| **scifact** | 5,183 | 0.700 | **0.759** | 0.665 | 0.671 |
| **nfcorpus** | 3,633 | 0.321 | **0.335** | 0.325 | 0.344 |
| **fiqa** | 57,638 | 0.239 | 0.244 | 0.236 | 0.317 |
| trec-covid | 171,332 | 0.606 | — † | 0.656 | 0.677 |
| touché | 382,545 | 0.355 | — † | 0.367 | 0.347 |

† test-only corpora (no train qrels → bridges can't be trained).

**Honest reading:** beats or matches BM25 on 3/5, beats zero-shot ColBERT on the lexically-clean corpus (scifact, **+0.09 nDCG**), and loses to neural dense retrieval where the question→answer semantic gap is large (fiqa) or the text is clean-lexical at scale (trec-covid). The supervised bridges help most where there is training data *and* exploitable structure (scifact **+0.06**); modestly elsewhere. See [FINDINGS.md](FINDINGS.md) for the full story, including the negative results.

## Speed / footprint / scaling (measured)

| | result |
|---|---|
| **query** | ~1 ms full stack (lexical + bridges), 0.14–0.55 ms lexical — numpy-vectorized |
| **ingest** | ~2,100–3,600 docs/s (3.3× after a cached-sieve prime pool + gear cache) |
| **footprint** | **743 B/doc** on disk (delta+zlib), ~2 KB/doc dense in-RAM |
| **persistence** | lossless `save()` / `load()`, stays appendable |
| **scaling** | sub-ms at any N via champion lists; `ShardedIndex` is distributed-**exact** |

Every speed and footprint gain is either strictly lossless or carries a quantified, opt-in tradeoff.

---

## Install

```bash
pip install -e .          # needs numpy
```

## Quick start

```python
from lattice_rag import AppendOnlyLatticeIndex, RelevanceBridges, bridge_search

idx = AppendOnlyLatticeIndex()
for doc_id, text in corpus.items():
    idx.add(doc_id, text)            # O(1) append — no reindex
idx.finalize()                       # build the numpy fast path (~15x, lossless)

hits = idx.search("0-dimensional biomaterials", k=10)

# optional supervised accuracy layer — learned by COUNTING qrels (no SGD)
br = RelevanceBridges(idx, len(idx.alive), min_pairs=1).learn(queries, train_qrels, corpus)
hits = bridge_search(idx, br, "0-dimensional biomaterials", k=10)

idx.save("index")                              # compact, lossless
idx2 = AppendOnlyLatticeIndex.load("index")    # reloads appendable
```

### Scale out (distributed-exact)

```python
from lattice_rag import ShardedIndex
sh = ShardedIndex(n_shards=8)        # hash-routed shards, shared vocabulary
for doc_id, text in corpus.items():
    sh.add(doc_id, text)
sh.finalize()                        # gathers global stats (N, df, avgdl)
hits = sh.search("query", k=10)      # fan-out + merge == one big index, exactly
```

### Sub-ms at any size (one machine)

```python
idx.finalize(champion_m=500)         # bounded top-M per term: query work is O(1) in N
```

---

## Architecture

| module | what |
|---|---|
| `aethos_append_index.py` | the engine: multi-view tokenization (word + char-trigram + prefix), prime-addressed inverted index, BM25 scoring, numpy dense fast path, champion lists, compact persistence |
| `aethos_bridges.py` | supervised relevance bridges (learn by counting qrels) + the rerank/expand search |
| `aethos_sharded_index.py` | `ShardedIndex` — distributed-exact sharding with global stats |
| `core/primes.py` | the prime engine (cached sieve) |

## Reproduce the numbers

```bash
python scripts/run_beir.py scifact 1       # lexical + bridges, held-out test
python scripts/run_beir.py fiqa 2
python scripts/run_beir.py trec-covid      # large, test-only → lexical
```

The method studies (speed, footprint, scaling, the author-a-doc loop, the negative results) are each isolated in a `scripts/bench_*.py` / `scripts/diagnose_*.py` and referenced from [FINDINGS.md](FINDINGS.md).

## What it is and isn't

- **Is:** a strong, transparent, append-only lexical+supervised retriever; every score traces to named primes and every bridge to named training queries; grows forever with no reindex; scales horizontally with exact sharding.
- **Isn't:** a dense neural retriever. It does not learn distributional embeddings, so it cannot close a large question→answer semantic gap the way a trained encoder does. Use it where lexical signal + relevance feedback is strong, or fuse it with a dense model.

## License

MIT.
