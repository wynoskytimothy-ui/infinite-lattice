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

## 6. SPEED FIX — rarest-address candidate pooling (`_serve_verify.log`, head-to-head, same 250 q)

The 3.2 s/query came from scatter-adding the FULL posting lists of a few high-DF SPLADE query terms.
Fix (`ServedIndex.search_fast`, query-side only — index UNCHANGED at 286.9 B/doc): the short
(discriminative) query-term posting lists build a small candidate set C; every term then refines via
`searchsorted(C)` against its sorted posting list — O(|C|·log|posting|), never O(|posting|).

Head-to-head on the SAME 250 dev-small queries (shipped `search()` vs `search_fast()`):
```
FAST : MRR@10 0.3909   recall@100 88.40%   median   88 ms   p90  115 ms
FULL : MRR@10 0.3977   recall@100 91.60%   median 3144 ms   p90 4259 ms
  speedup 36x  |  delta MRR -0.0068  |  top-10 overlap 92.0%  |  footprint 286.9 B/doc unchanged
```
Honest: a **small real accuracy cost (-0.0068 MRR, -3.2 pts recall@100)** bought for a **36x speedup**
(3.1 s -> 88 ms). The cost is pool coverage (gold docs matching only common terms); it is a **dial** —
larger `pool_cap` recovers accuracy at higher latency (sweep `_serve_fast.log`: pool 150k → MRR ~0.403
at ~209 ms). Default config topq=30 / pool_cap=80k. Sub-100 ms = production band.

## 7. COMPOSITE / CORRELATION pooling — the lattice meet (`_serve_corr*.log`, `_build_composites2.log`)

Using the lattice's correlation structure for pool selection instead of the generic shortest-list union.
A composite (upper prime) = a term-pair; its doc-list is the MEET (intersection) of the constituents'
postings. Same 250 dev-small queries:

| pool method | MRR@10 | recall@100 | median | footprint |
|---|---|---|---|---|
| full scatter (ceiling) | 0.3977 | 91.6% | 3144 ms | 287 B |
| **composite-meet on-the-fly** (intersect 6 discriminative terms + rarest floor) | **0.3982** | 91.2% | 127 ms | **287 B** |
| rarest-union (the generic heuristic) | 0.3909 | 88.4% | 94 ms | 287 B |
| stored composites + recall floor=2 | 0.3893 | 90.0% | 88 ms | 315 B |
| stored composites, floor=0 (pure) | 0.2931 | 65.6% | **14 ms** | 315 B |

Findings (honest):
- The composite-meet **recovers the full-scatter accuracy** (0.398 = ceiling) that the generic union loses
  (0.391) — correlations select *proper* docs. Confirmed. At 287 B/doc, ~127 ms. = accuracy-optimal serve.
- Pre-stored composites (`build_composites.py` → 31M composites, 106M postings, **+28 B/doc FOR-packed →
  ~315 B/doc**, under the 500 B budget) give a real **speed dial down to 14 ms**, but pure composites are
  too narrow (recall 65.6%); a recall floor restores accuracy but brings latency back to ~the union's.
- Net: the correlation layer **matches/slightly-beats** the simple method and shifts the speed/accuracy
  frontier; it does not simultaneously dominate on both. No breakthrough claimed. DF-gating and
  weight-prefix "curation" variants were measured and underperformed (`_serve_corr2/3.log`).
- Shipped default `SERVE_MODE=corr` (accuracy-optimal 0.398/127ms/287B); `fast`=rarest-union 0.391/88ms;
  `full`=scatter. Stored-composite speed dial available via `composites.npz`.
