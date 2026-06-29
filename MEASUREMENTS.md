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

---

## Speed / Footprint / O(1) deep dive (2026-06-28, `_o1_*.py`, `_dd_wand.py`)

Final pass focused on SPEED + FOOTPRINT + O(1) (NOT accuracy). All CPU, no GPU. Two genuine new wins,
several confirmed dead-ends.

### NEW WIN 1 — sub-4-bit weights cut the footprint ~15% at held accuracy (`_o1_bitplane_weight_quant.py`)
Per-term-max SCALING (1 fp16 scale/term) then uniform quantize. 50k-calib, 6980 cached queries, exact rerank.
| weights | wt B/doc | total B/doc | MRR@10 | ΔMRR | recall@100 |
|---|---|---|---|---|---|
| fp32 (baseline) | 483.0 | 605.0 | 0.9240 | — | 99.77% |
| 5-bit (prior ref) | 75.5 | 197.5 | 0.9201 | −0.0039 | 99.77% |
| **3-bit per-term** | **46.2** | **168.2** | **0.9226** | **−0.0014** | **99.77%** |
| 2-bit per-term | 31.1 | 153.1 | 0.9187 | −0.0053 | 99.77% |
Two-sided: global Lloyd-Max FAILS below 5-bit (3-bit −0.0078) — it starves the rare high-weight postings
(~9.7% are w>30) that dominate the dot. Per-term scaling keeps each list's dynamic range. **New near-lossless
footprint = 168 B/doc (gamma doc-ids 122 + 3-bit weights 46), down from 197.5, MRR −0.0014.** Measured on 50k;
relative shrink is the load-bearing claim.

### NEW WIN 2 — the pooled serve is the fast path, not WAND (`_o1_serve_shootout.py`, full 8.8M, 250 q)
| serve | MRR@10 | recall@100 | med ms | p90 | p99 |
|---|---|---|---|---|---|
| full scatter (exact) | 0.3973 | 91.60% | 3068 | 4065 | 4779 |
| WAND (numba, exact) | 0.3973 | 91.60% | 878 | 2421 | 3664 |
| **search_corr (pooled)** | **0.3986** | 91.20% | **123** | 285 | 480 |
| search_fast (pooled) | 0.3909 | 88.40% | **91** | 115 | 181 |
**search_corr = 123 ms at MRR 0.3986 (≥ exact) — 25× faster than WAND/scatter, matched accuracy.** WAND is
exact + 3.5× over naive but ~7× slower than pooling; its only niche is provably-exact top-k. Confirms the
lattice's rarest-address/meet pooling IS the WAND-class optimization and already beats textbook WAND.

### O(1) content-address — the signature differentiator (`_o1_content_address.py`)
Proven meet `(a+p+q,p+q,p)`, det=−1. O(1) lookup ties a dict (629 vs 582 ns @1M), beats searchsorted's
O(log N) (2114 ns). 10M/10M exact invertible. **0 collisions on 10M keys at 0 bits/key** (vs 32-bit hash 456
collisions on 2M; MPH needs 2–3 bits/key). Coordination-free: 200k/200k independent-node agreement.

### Confirmed (no further chase)
- **Elias-Fano doc-ids** (`_o1_succinct_postings.py`): 119.2 B/doc = 0.974× gamma + O(1) skip (gamma is
  sequential). Modest bytes, unlocks skip — pair with WAND for the exact path.
- **min-plus meet = exact graph algebra** (`_o1_minplus_free_graph.py`): 1.14M pairs, 0 disagreements vs
  scipy AND networkx Floyd-Warshall; exact (max,+) scheduling + Viterbi from one 3-line kernel. Exactness +
  operator-reuse, NOT a speed win (10–16× slower than C Floyd-Warshall in pure numpy).
- **chamber-PQ codebook** (`_o1_pq_3d_chambers.py`): ties k-means; only edges it at 3-bit (under-resourced K);
  niche fast zero-param low-bitrate quantizer, not a quality leap.
- **DEAD: corpus-as-a-number** (`_o1_algebraic_number_coldstore.py`): not compression (∑log₂prime > ∑log₂gap);
  ties FOR, loses to a plain ID list 1.22×; blind decode 5,436× slower.
- **DEAD: chamber routing** (`_o1_chamber_routing.py`): no operating point with both candidate reduction AND
  recall ≥98.5% — value-based chambers scatter relevant docs (random-hash control fails identically →
  intrinsic). Slower than the already-DF-bound scatter in every config.
