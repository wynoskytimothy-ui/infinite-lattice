# Edge RAG — the best small / fast / accurate version (no GPU required)

*Goal (Timothy, 2026-06-29): the best RAG — scalable, fast, accurate, runnable on phones / edge devices,
small RAM, GPU optional. Every number below traces to a captured run; the headline is measured on held-out
test queries, not projected.*

## The headline (measured, scifact 5,183 docs, held-out test nDCG@10)

| config | B/doc | total | nDCG@10 | Recall@10 | speed | model? |
|---|---|---|---|---|---|---|
| full multiview (word+trigram+prefix) | 743 | 3.9 MB | 0.7023 | 0.827 | 0.56 ms | none |
| full + bridges (server champion) | 743 | 3.9 MB | **0.7269** | 0.840 | 10.1 ms | none |
| word-only (drop trigram gear) | 198 | 1.0 MB | 0.6712 | 0.784 | 0.08 ms | none |
| **word-only + bridges (EDGE CHAMPION)** | **198** | **1.0 MB** | **0.7112** | **0.812** | **0.80 ms** | **none** |

Run: `_o1_edge_champion.py`. BM25 reference on scifact ≈ 0.665.

**The edge champion (word-only + bridges) BEATS the fat full-multiview index (0.7112 > 0.7023) at 3.8× smaller
footprint and sub-millisecond latency — with zero neural model.** It crushes BM25 (+0.046).

## Why it works — the two levers

1. **Drop the trigram gear (footprint).** The char-trigram view (`^word$` → trigrams) is ~80% of all postings
   (3.24M → 0.48M). Dropping it shrinks the index 3.8× (743 → 198 B/doc) and speeds the scan 7× (0.56 → 0.08 ms).
   Cost: −0.031 nDCG (typo-robustness lost). *(`_o1_edge_prune.py`: trigram gear is the dominant cost.)*
2. **Add counting-bridges (accuracy, neural-free).** Query-term → doc-term bridges, learned by *counting*
   relevant (query, gold-doc) pairs from train qrels (deterministic, append-only, verifiable). On the word-only
   index they add **+0.040 nDCG** (0.6712 → 0.7112) — *more than recovering* the trigram-drop loss.
   *(Honesty note: the old bench cached `br.score()`, a method that no longer exists; this +0.040 is a fresh
   re-measurement reconstructing the score from the learned `bridge` dict — `bridge_score()` in the script.)*

Net: the footprint we spent on fuzzy trigram matching is better spent on *supervised* query→doc links. Smaller
**and** more accurate.

## The architecture

```
INGEST   : word-gear lattice postings only (append-only; ~198 B/doc on disk via save():
           delta-coded doc-ids + float16 weights + zlib). Trigram/prefix gears OFF.
LEARN    : RelevanceBridges.learn(train qrels) — counts query→gold-doc term co-occurrence.
           Pure counting: no backprop, no GPU, fully traceable to named train queries.
SERVE    : (1) word-gear BM25 over the candidate pool (binary-reader merge, touches only
               the query's term postings — RAM = working set, not the whole index)
           (2) rerank top-100 by the learned bridges.  Total < 1 ms (CPU).
```

The save() format is already CSR (delta + float16, per-prime segments) — **mmap-ready**, so at scale the index
stays on storage and RAM = the per-query working set, never the whole index. That is what makes it phone-native.

## Scaling (B/doc = 198, the edge champion; corpus-linear footprint)

| corpus | index size | fits phone RAM? | serve (est.) |
|---|---|---|---|
| 10k docs (an app's KB) | 2 MB | trivially | < 1 ms |
| 100k docs (personal corpus) | 20 MB | yes | a few ms |
| 1M docs | 198 MB | yes (4–8 GB phones) | ~10–30 ms |
| 8.8M MARCO | 1.74 GB | mmap on storage (RAM = working set) | bounded by query, not corpus |

The serve cost is **working-set-bound** (only the query's terms are touched), so it scales by query, not by
corpus — one index format from a 2 MB app KB to an 8.8M-passage MARCO, edge to cloud.

## Accuracy tiers — pick by device

- **No GPU (the edge champion):** word-only + counting-bridges = **0.711** nDCG, 198 B/doc, < 1 ms. Beats
  BM25 and the fat lexical index. This is the default — runs on any phone CPU.
- **NPU / small GPU (optional):** add a few-MB INT8 distilled query encoder (SPLADE-tiny) to reach SPLADE-class
  accuracy. Keeps RAM small — the model is single-digit MB and the doc index stays sparse postings (no GB-scale
  dense vectors). The lattice serve and bridges are unchanged; the encoder only enriches the query.

## What's honest about this

- The headline is on scifact (an edge-sized corpus). The *mechanism* (trigram-drop + counting-bridges)
  is corpus-general, but the exact +0.040 is scifact-specific; other corpora need their own qrels to learn
  bridges. The no-bridge word-only floor (0.671, still > BM25) needs no training.
- Bridges need *some* train qrels. Cold-start (no qrels) → the word-only floor (still beats BM25), then bridges
  improve it online as judgments arrive (the `learn` step is append-only).
- The mmap "RAM = working set" claim follows from the CSR save format + the binary-reader serve (which provably
  touches only the query's term postings); the end-to-end mmap deployment is the next build, not yet benchmarked.

## Files
- `_o1_edge_champion.py` — the head-to-head table above (A/B/C/D, real save() bytes, held-out nDCG).
- `_o1_edge_prune.py` — the footprint↔accuracy knee (trigram-drop + per-doc top-k curve).
- `_o1_edge_rag.py` — the per-query working-set / RAM profile + scaling projection.
