# MEASUREMENTS — raw evidence for every number in RAG_FOR_ANDREA.md

Every quantitative claim must trace to a line here. Captured 2026-06-25. The lesson from the
self-audit: a number that lives only in a summary doc is an *unsupported* number. These are the runs.

---

## 1. MARCO native SPLADE-on-lattice — FOOTPRINT (VERIFIED)

`stat C:/Users/wynos/trng/marco_data/splade_native_full/splade_index_for.npz` = **2,536,480,416 bytes**
Docs = **8,841,823** (`collection.offsets.npy`).  → **2,536,480,416 / 8,841,823 = 286.9 B/doc**, 2.54 GB.

From `_splade_index_serve.log`:
```
pass1: 1,059,501,065 postings over 8,841,823 docs, 27,715 active terms
postings=1,059,501,065  on-disk=2536.5 MB  (286.9 B/doc, di-gap 11.15 bits/posting)
FOR round-trip on 20 sampled terms: MATCH        <- lossless
[chamber probe] 4.0MB gap-blob -> 2.66MB (1.50x) <- experimental chamber codec, ~1.5x on gaps
```
Note: the served index is the single `splade_index_for.npz` (2.54 GB). The 45 `chunk_*.npz`
files (3.28 GB) are a resumable *encode cache* that gets inverted into the index; the serve loads
only the index. 286.9 B/doc is the served-index figure.

## 2. MARCO native SPLADE-on-lattice — ACCURACY + SPEED (`_serve_sample.log`, 200-q sample)

```
===== native SPLADE-on-lattice, dev-small (NO pool, NO CE) full =====
  queries scored        : 200  (gold-in-index 200; full 8.8M corpus, gold competes vs ALL docs)
  serve latency (meet)  : median 3234.31 ms  p90 4336.22 ms     <- 3.2 s/query, NOT ms
  recall@100            : 92.00%
  MRR@10                : 0.3989                                  <- SPLADE++/ColBERT band, no CE
```
- **0.3989** is the real full-corpus number on a **200-query sample** (std err ~±0.02–0.03). The
  full 6,980-query run is ~13 h at 3.2 s/q and has NOT been run.
- The hardcoded "OPTIMISTIC vs ~50k" string in the print is a leftover from the 50k calibration
  path; on `--full` every gold competes against all 8.8M docs, so 0.3989 is honest, not optimistic.

## 3. BEIR — LEXICAL multi-view lattice + CE rerank (`_r1_beir.out`) — a DIFFERENT pipeline

This is the "Route 1" lexical lattice (the branch Andrea tested), **not** the SPLADE index above.
```
scifact  : lattice-only 0.7023 (BM25 order, no CE)  -> +CE best nDCG@10 0.6786 @ depth 100
nfcorpus : lattice-only 0.3203                       -> +CE best nDCG@10 0.3489 @ depth 200
fiqa     : lattice-only 0.2392                       -> +CE best nDCG@10 0.3522 @ depth 1000
CE rerank latency: ~113–1275 ms/query (depth-dependent)
```
- The clean, real BEIR win: **scifact lattice-only 0.7023 > BM25 0.665**, with NO cross-encoder.
- nfcorpus/fiqa lattice-only LOSE to BM25; they only reach 0.349/0.352 *with* a cross-encoder.
- The learned (SPLADE/route3) scorer on these is lower: nfcorpus 0.3194 (< BM25 0.3346),
  fiqa 0.1765 (< BM25 0.2307) — `_route3_main.out`.

## 4. Lossless codec (corrected — NOT 6.2×)

- FOR vs dense-float baseline: **2.123 GB → 0.428 GB = 4.97×**, byte-exact round-trip (MATCH).
- Experimental chamber codec on the posting-gap stream: **1.50×** (full MARCO) / 2.26× (50k slice);
  chamber 9.24 vs FOR 13.37 bits/posting = ~1.45× (`marco_chamber_blocks.py`).
- The "6.2×" figure that appeared in earlier drafts is **not measured anywhere** — removed.

## 5. Baselines (published literature, not our measurement)

BM25 scifact 0.665 / nfcorpus 0.325 / fiqa 0.236 — matches the BEIR/Anserini table (fair).
ColBERT fiqa 0.317 — matches BEIR Table 2 (fair). ColBERT nfcorpus "0.344" was **dropped** — the
BEIR zero-shot ColBERT(v1) nfcorpus is ≈0.305, so the old "+0.005 win" rested on a bad baseline.
MARCO dense ~0.34, SPLADE++/ColBERT 0.37–0.40 — accurate bands.
