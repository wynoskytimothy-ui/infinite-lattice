# AETHOS RAG — PROVEN claims (each reproducible, CPU-only, no GPU)

*Every line below has a passing test you can run live. `cd "C:/Users/wynos/New folder (3)"` then `PYTHONUTF8=1 python <script>`.*

## Proven scorecard (safe to pitch)
| # | Claim | Measured number | Reproduce |
|---|---|---|---|
| 1 | **Beats BM25, CPU-only, index alone** | scifact nDCG@10 **0.7023** vs BM25 0.665 (**+0.037**), no GPU, no supervision | `_prove_retrieval_cpu.py` |
| 2 | **Sub-ms, tiny, fast** | **0.59–0.62 ms/query** (p90 ~1.0), **~2,250 docs/s** ingest, **743 B/doc** | `_prove_retrieval_cpu.py` |
| 3 | **Deterministic / bit-identical** | 50×10 results byte-identical across a fresh subprocess (sha256-stable), 0 collisions/3 runs | `_prove_structural.py` |
| 4 | **Invertible / zero-collision meet** | **20,000/20,000** triples round-trip meet→unmeet exactly, 0 collisions; swap_meet bijection 2000/2000 | `_prove_structural.py` |
| 5 | **Append-only, no rebuild** | 100 docs in **1.88 ms**, **0** existing entries mutated, incremental == full rebuild **50/50** | `_prove_structural.py` |
| 6 | **No GPU** | **0/7** of {torch, transformers, tensorflow, jax, cupy, onnxruntime, sentence_transformers} loaded | `_prove_structural.py` |
| 7 | **Glass-box: the explanation IS the score** | per-term idf×BM25 parts reconstruct the engine score to **3.55e-15** (machine precision) | `_prove_glassbox.py` |
| 8 | **Exact-match completeness** | **100.0000%** candidate recall (4314/4314 + 300/300 phrase), 0 false negatives, 0 leaks after tombstoning 200 docs | `_prove_compliance_dedup.py` |

## New problems the engine solves (new selling points, measured)
- **Multi-hop / bridge retrieval, no stored graph.** One content-meet turns rare-entity co-occurrence into a
  free second hop. On 60 mined hard bridge pairs (single-hop **0/60** by construction) the meet-chain recovers
  the answer doc **45/60 = 0.750** — **+47** docs single-hop lexical scoring structurally cannot reach.
  → `_prove_multihop.py`
- **Dedup / near-duplicate, no embeddings.** Word-prime Jaccard (the meet) with rarest-prime blocking:
  **precision 100%, recall 99.17%, F1 99.58%** at Jaccard≥0.80, 240 candidate pairs vs 352,380 full N² in **2 ms**.
  → `_prove_compliance_dedup.py`
- **Compliance "find every record."** The exact-match 100% candidate recall = no scoring threshold can drop a
  containing record. Regulated/legal e-discovery. → `_prove_compliance_dedup.py`

## The 3 strongest investor bullets (each with a live command)
1. **Beats BM25 on accuracy, entirely on CPU, sub-millisecond.** scifact nDCG@10 **0.7023 vs 0.665**,
   **0.6 ms/query**, **743 B/doc**, **~2,250 docs/s** — no GPU, no model, no supervision.
   `PYTHONUTF8=1 python _prove_retrieval_cpu.py`
2. **Glass-box + deterministic — auditable in a way vector DBs structurally are not.** Bit-identical across
   processes; every score decomposes to its terms to machine precision; exact-term recall **100%**, zero
   false negatives. `python _prove_glassbox.py` · `python _prove_compliance_dedup.py`
3. **Append-only + self-extending; one operator unlocks what BM25 can't.** 100 docs live in **1.88 ms**
   (0 entries mutated), the meet is a verified **zero-collision invertible bijection**, powering free
   multi-hop (**+47** docs) and embedding-free dedup (**F1 99.58%**). `python _prove_structural.py` ·
   `_prove_multihop.py` · `_prove_compliance_dedup.py`

## Do NOT pitch as-is (honest — caught before an investor would)
- **0.7645 is NOT index-alone** — the airtight index-alone number is **0.7023**. 0.7645 needs the supervised
  bridges reranker (qrels); only pitch it with the reranker named + its own passing run.
- **"Rankings never change on add" is FALSE** (BM25 idf is live). Pitch the *stronger* true guarantee:
  0 existing entries mutated + immediately searchable + incremental == rebuild.
- **Multi-hop 0.750 is recovery on mined pseudo-bridges**, not a head-to-head multi-hop nDCG win; claim
  "the meet makes a free second hop," not a benchmark victory. Misses 15/60.
- **"MS MARCO MRR 0.948"** from the old overview — drop entirely; real MARCO is ~0.39. **"4 B/doc + high
  accuracy"** — different versions; can't co-claim. **IMS 1.4% early detection** — demoted (a 1-line RMS
  alarm fires earlier). **CMAPSS beats LSTM** — a RandomForest beats it.
