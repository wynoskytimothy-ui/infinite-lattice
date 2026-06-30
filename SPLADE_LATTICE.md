# SPLADE into the lattice — and distilled to serve encoder-free

*Goal (Timothy): add SPLADE into the lattice "fully reverse-engineered for it" to keep speed + footprint and
improve accuracy; then — since the lattice can ingest + learn — distill what SPLADE knows INTO the lattice so it
serves smart WITHOUT the query encoder. All numbers measured on nfcorpus (the semantic-mismatch corpus where
learned-sparse helps most). `_o1_splade_lattice.py`, `_o1_splade_roles.py`, `_o1_distill.py`.*

## 1. SPLADE on the lattice — same serve, more accuracy

SPLADE's output IS sparse term-weights = lattice postings. Store the LEARNED weights in the same CSR; score
with the same scatter-add (the meet). The lattice is unchanged.

| index | B/doc | serve | nDCG@10 |
|---|---|---|---|
| lexical lattice | ~198 | 0.10 ms | 0.3065 |
| **SPLADE on lattice** | 763 (top-128; ~287 w/ uint8+FOR) | **0.11 ms** | **0.3357 (+0.029)** |

Serve speed unchanged (scatter-add); footprint stays *sparse* (prunable to ~287 B/doc, vs dense's 1,536). The
encoder is the only addition — and on CPU it's the bottleneck (**4 docs/s**; a GPU is ~50–100×). One-time at
ingest, plus the query encoder at serve — which §3 removes.

## 2. SPLADE's four roles (nfcorpus, 323 queries)

| role | nDCG@10 | Recall@100 |
|---|---|---|
| lexical | 0.307 | 0.238 |
| **SPLADE (retrieval)** | **0.336** | **0.282 (+0.044, +18%)** |
| RRF (lex+SPLADE) | 0.327 | 0.270 |
| SPLADE-rerank (of lex pool) | 0.326 | 0.238 |
| union ceiling | — | 0.310 |

- **Recall ↑ is the headline** — SPLADE's expansion reaches docs lexical misses (0.238 → 0.282), lifting the
  ceiling on everything downstream.
- **The recall win needs SPLADE in the INDEX, not just as a reranker** — reranking the lexical pool lifts nDCG
  (0.307 → 0.326) but recall stays pinned at the lexical pool's 0.238. You can't rerank to docs you never
  retrieved.
- **Fusion must be weighted per corpus** — on a SPLADE-favorable corpus, equal-weight RRF dilutes the strong
  signal (0.270 < 0.282). The union ceiling (0.310) shows the docs are reachable; weighted fusion is the lever.

## 3. Distill SPLADE INTO the lattice → serve ENCODER-FREE (the breakthrough)

Save SPLADE's per-word expansion as a static table at ingest ("teach it what SPLADE knows, save it"). At serve,
a raw word looks up its learned correlations from the table — **no neural net at query time**.

| mode | nDCG@10 | Recall@100 | encoder at serve? |
|---|---|---|---|
| lexical | 0.307 | 0.238 | no |
| SPLADE full | 0.336 | 0.282 | **YES** |
| **distilled** | 0.321 | **0.275** | **NO** |

**Distilled (max-pool) recovers 92% of SPLADE's recall gain (0.238 → 0.279 vs full 0.282) — with no query
encoder.** The semantic graph is built at ingest, before any question; a word (or subword) triggers its saved
correlations. Pure-lattice serve speed, most of SPLADE's recall, no GPU at query time.

**The combination method is the lever, not more context.** Switching per-word distillation from sum → **max-pool**
(matching SPLADE's own pooling) jumped recovery **84% → 92%** (`_o1_distill_pairs.py`).

**Word-pair distillation was tested and REFUTED** (the cross-term-context hypothesis): encoding co-occurring
word-pairs "w1 w2" jointly and composing query vectors from them *hurts* — pairs-only recall recovery is **−84%**
(worse than lexical) and word+pair (88%) is below word-alone (92%). A 2-word fragment isn't full-query context,
and max-pooling many pair expansions adds more noise than signal. Honest conclusion: **~92% is the encoder-free
ceiling** for this approach; the last 8% is genuine full-query context that needs the actual encoder.

## The payoff — one substrate, a dial of accuracy/cost

Everything rides the same lattice CSR + scatter-add serve:
- **lexical** (no encoder, fastest) → BM25-class
- **+ counting-bridges** (feedback, no GPU) → +0.04
- **distilled SPLADE** (encoder-free serve, max-pool, distilled once at ingest) → **92% of SPLADE's recall, lattice speed**
- **SPLADE weights** (encoder at serve) → full SPLADE-class
- **+ WAND pools** → flat serve at any scale

Pick the accuracy/cost point per deployment; the lattice is the universal sparse-serving engine. Timothy's
distillation makes the *encoder-free* tier genuinely semantic — the key to keeping speed while gaining recall.

## Productized: `aethos_splade_lattice.DistilledSpladeIndex`

The validated tier, packaged as a first-class, persistent module:
- `build_docs(corpus, encoder)` — SPLADE-encode docs once → CSR (needs GPU at ingest)
- `distill(vocab, encoder)` — per-word expansion table once
- `search(query)` — **encoder-free** (max-pool distilled expansion → scatter-add); numpy only, no torch
- `save(path)` / `load(path, mmap=True)` — the model is **not needed to serve**

Self-test (nfcorpus, reusing cache): Recall@100 lexical 0.234 → **distilled 0.279 (93% of SPLADE's gain)** →
SPLADE-full 0.282, **0.18 ms/q, no query encoder**; save/load serves identically, 863 B/doc on disk.

## Honest caveats
- Measured on nfcorpus only (the SPLADE-favorable corpus); scifact/fiqa not yet run (CPU encode is slow).
- The distilled table here covers the query vocabulary (one-time, offline); production distills the full/corpus
  vocab — same mechanism, more one-time encoding.
- CPU encode is 4 docs/s — ingest needs a GPU at scale; serve (distilled) needs none.
- Footprint 763 B/doc is top-128 float16; the MARCO-native path (uint8 + FOR) reaches ~287 B/doc.
