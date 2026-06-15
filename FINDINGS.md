# Retrieval by counting on prime addresses — method, results, and honest negatives

A retrieval engine that scores documents with multi-view BM25 over a prime-addressed,
append-only inverted index, and adds accuracy by *counting* relevance judgements into
query→document term bridges (supervised retrieval, no gradient descent). This document
records what worked, what didn't, and why — every claim traces to a runnable script.

---

## 1. How this started, and the first honest correction

The work began as "can the lattice beat BM25 / reach the 0.78 nDCG an older int8 version
reported on scifact?" The first finding was a correction, not a trick:

**The 0.78 was a 500-document subset artifact.** The old "UltraFast" numbers (scifact
0.78–0.89) were all measured with `max_docs=500` (35 queries), where the top-10 is 2% of
the corpus. On the full scifact corpus (5,183 docs, top-10 = 0.2%) that same configuration
lands at ~0.65 — the repo's own `ACCURACY_CORPUS_SIZE_CLARIFICATION.md` already flagged it.
So there was no full-corpus 0.78 to chase. Meanwhile a plain append-only multi-view index
already scored **0.700** on full scifact — above BM25 (0.665), ColBERT (0.671), and the
old version's true full-corpus number. *Lesson: always check corpus size before comparing
BEIR numbers; subset runs inflate by 0.10–0.20 nDCG.*

## 2. Method

**Multi-view lattice index** (`aethos_append_index.py`). Every term flows through three
tokenization *gears* — word, character-trigram, prefix — each on its own prime namespace.
A `(view, token)` gets a unique prime on first sight (`core/primes.py`, a cached sieve) and
keeps it forever; documents are products of those primes; the index is a posting list per
prime. Scoring is multi-view BM25 with positional weighting on the word gear. Because idf is
read from the live document frequency at query time, **adding a document only appends to
posting lists** — no reindex, no retrain, no forgetting.

**Supervised relevance bridges** (`aethos_bridges.py`) — the accuracy layer. For each
(query, gold-doc) pair in the training qrels, count which doc-words co-occur with each
query-word in a *relevant* pair. A bridge `qt → dt` kept across ≥ `min_pairs` distinct
relevant pairs (and above an idf gate) is a learned vocabulary link. At query time the
bridges rerank and *expand* the candidate pool: a document reachable through learned
partners of the query's words enters even with no query word in it. This is information
BM25 has never seen (human relevance), injected by **counting** — so it stays deterministic,
append-only, and verifiable (every bridge traces to named training queries). It is the
classic IR-as-translation idea (Berger–Lafferty 1999) done as counting, not SGD.

**Dense fast path** (`finalize()`). The same arithmetic, vectorized: posting lists become
`(uint16 doc-id, float16 tf)` arrays, scoring is a numpy scatter-add. The Python loop drops
from millions of postings to ~30 query terms — ~15× faster, metric-lossless.

**Scaling.** *Champion lists* (`finalize(champion_m=M)`) keep only each term's top-M docs,
so query work is O(query_terms · M), independent of N — sub-ms at any size, ~1% accuracy
cost. *Sharding* (`aethos_sharded_index.py`) hash-routes docs to shards that share one
vocabulary and score with global stats; a fan-out + merge equals a single index **exactly**.

**Active learning** (`scripts/bench_author_doc_loop.py`). When a query's gold doc shares no
words with it (so no rerank can reach it), the learned bridges name the missing concept; a
grounded note connecting the two vocabularies is authored and appended in O(1); the query is
then answered directly and the original gold doc becomes reachable via one feedback round.

## 3. Results

### Accuracy (BEIR, held-out test, nDCG@10) — `scripts/run_beir.py`

| corpus | docs | lexical | + bridges | BM25 | ColBERT |
|---|---:|---:|---:|---:|---:|
| scifact | 5,183 | 0.700 | **0.759** | 0.665 | 0.671 |
| nfcorpus | 3,633 | 0.321 | **0.335** | 0.325 | 0.344 |
| fiqa | 57,638 | 0.239 | 0.244 | 0.236 | 0.317 |
| trec-covid | 171,332 | 0.606 | — | 0.656 | 0.677 |
| touché | 382,545 | 0.355 | — | 0.367 | 0.347 |

Beats/matches BM25 on 3/5; beats ColBERT on lexically-clean scifact; loses to neural where
the question→answer gap is large (fiqa) or the text is clean-lexical at scale (trec-covid).
The bridge lift tracks how much training data and exploitable structure a corpus has
(scifact +0.06; +0.01 elsewhere).

### Speed / footprint (scifact, measured)

- Lexical query 37 ms → **0.55 ms** across the optimization arc (df-cap → numpy), lossless.
- Full stack (lexical + bridges) **~1.35 ms**, recall 0.866.
- Ingest **8.2 s → 2.5 s** per 5 K docs (cached-sieve prime pool was 41% of build time; gear cache).
- Footprint **743 B/doc** on disk (CSR + delta + zlib), ~2 KB/doc dense in-RAM; `save`/`load` lossless and appendable.

### Scaling (measured, `scripts/bench_scaling.py`, `bench_sharded.py`)

- Full-dense latency grows ~linearly with N (0.36 ms @ 1.5 K → 1.78 ms @ 30 K). Champion (M=500) is flat (0.30 → 0.38 ms).
- ShardedIndex: K=4/16 give recall *identical* to a single index, top-10 overlap 98.6–99.4% (float16 tie-ordering only). Validated on real 171 K / 382 K corpora — which surfaced and fixed a real bug (the prime pool was fixed at 200 K but a 171 K-doc multi-view vocabulary exceeds it; now grows dynamically).

## 4. Negative results (the honest core)

These are as important as the wins; each is a measured dead end.

1. **The letter-prime ψ-encoder is surface, not semantic** (`bench_psi_encoder.py`). A 24-D
   vector from letter-prime chains scored **0.10** nDCG reranking alone — documents with
   similar letter distributions land close regardless of meaning. There is no free semantic
   encoder hiding in the formula's geometry.

2. **Unsupervised semantic signals add drift, not lift.** Window-cooccurrence PPMI query
   expansion (`bench_ppmi_retrieval.py`) was neutral-to-negative on real corpora (first-order
   PPMI is topical *association*, not synonymy → query drift). A meet-overlap "manifold"
   rerank added +0.002 (redundant with BM25). The 4 KB/99 ms budget study added +0.0006.
   On lexically-clean corpora, well-tuned lexical BM25+positional is hard to beat — the gap
   is *signal*-limited, not budget- or geometry-limited.

3. **The lift came only when the signal was supervised.** Every unsupervised signal failed
   because it had no idea what was *relevant*. Qrels are the missing ingredient: counting
   relevance pairs into bridges lifted held-out scifact 0.700 → 0.759 — the first thing to
   beat the lexical ceiling, and it stays deterministic/append-only because it is counting.

4. **Champion lists are *not* lossless** (`bench_champion_lists.py`). They give 2–3× more
   speed but top-10 overlap is ~90% at M=200 (−0.008 nDCG on scifact); they converge to
   lossless only as M→2000. A genuine speed/accuracy knob, not a free lunch — labelled as such.

5. **Bridge "consensus" (a meet) did not help** (`bench_bridge_consensus.py`). The
   hypothesis — trust a bridge-reached doc only when ≥2 query terms agree — was a no-op
   (single weak bridges weren't in the top-10 anyway) and the weighted form actively hurt.
   The real lever was just `min_pairs` as a per-corpus knob.

6. **A pure-lattice scorer only *ties* BM25** (`bench_pure_lattice.py`). Replacing BM25's
   denominator with geometric TF-saturation / κ-cardinality length norm is competitive
   (within 1.4% on scifact) but does not leap past it. BM25's idf·tf-sat·length-norm is
   near-optimal for lexical matching; the lattice's edge is *not* the scoring formula.

## 5. The structural moat

The lattice's advantage over BM25 is not the score — it is what the score is built on:

- **Append-only, O(1), no reindex** — `add()` only appends to posting lists; a vector store
  must re-embed/re-index. Adding 100 docs to a warm index was 121× faster than a rebuild.
- **No forgetting, continual supervision** — bridges only accrue counts; more relevance data
  monotonically improves held-out accuracy (measured supervision curve), with no retrain.
- **Deterministic and verifiable** — identical regardless of ingest order; every score traces
  to named primes, every bridge to named training queries.
- **Multi-view recall and typo-robustness** — char-grams find terms through any view.
- **Distributed-exact sharding** — grow horizontally with no accuracy loss.
- **Self-repairing** — detect a knowledge gap, author a doc, append it, recover the answer.

## 6. When to use it

Use lattice-rag where lexical signal plus relevance feedback is strong, where the corpus
grows continuously (no reindex), where determinism/auditability matters, or as the fast,
transparent lexical arm of a hybrid system. Do **not** expect it to replace a trained dense
retriever on large question→answer semantic gaps — fuse the two instead (RRF), which is
where the prior work found the lattice's signals most complementary.

---

*Every number here is reproduced by a script in `scripts/`. The method studies are isolated:
`bench_supervised_bridges.py`, `bench_active_learning.py`, `bench_fast_query.py`,
`bench_numpy_scorer.py`, `bench_compress_dense.py`, `bench_scaling.py`, `bench_sharded.py`,
`bench_author_doc_loop.py`, and the negatives `bench_psi_encoder.py`, `bench_ppmi_retrieval.py`,
`bench_champion_lists.py`, `bench_bridge_consensus.py`, `bench_pure_lattice.py`.*
