# AETHOS research inventory — compression / speed / footprint / accuracy

*Cross-repo excavation 2026-06-29 (local + OneDrive). Captured-evidence-first. Repos read: prime_hotel,
formuilas, trng, final-build-aethos-13, aethos_master, aethos13-ultrafast, RAG, prime_lattice.*

## TL;DR — what's genuinely reusable on the 8.8M lattice

1. **V27 lossless posting packing** (int16 doc_idx + int8 tf = 3 B/posting): ~399 B/doc, **6–212× faster
   query, NDCG UNCHANGED**. Free smaller+faster, zero accuracy cost. `trng/bench_v27_compressed.py`, `v27_small.log`.
2. **V19 lazy triple regeneration** (don't materialize tuples; regenerate at query time by 3-way merge):
   3–5× smaller index, accuracy neutral-to-+0.43pp. The storage shape MARCO needs. `trng/bench_prime_triple_v19_compressed.py`.
3. **Markov-path codec** (lossless 5–8× text): FiQA 257→49 B/doc, SciFact 400→219, NFC 421→217.
   `final-build-aethos-13/src/aethos/retrieval/_markov_codec.py`.

## VERIFIED wins (captured result file on disk)

| Win | Measured | vs baseline | File |
|---|---|---|---|
| V27 lossless packing | 380–399 B/doc, 3 B/posting, P50 0.03–0.12 ms | ~2× smaller, 6.6–61.9× faster, **NDCG unchanged** | `trng/bench_v27_compressed.py` |
| V19 on-the-fly triples | window 244×, sentence 42×, index 3–5× smaller; **NDCG +0.43pp nfc** | smaller + neutral/positive | `trng/bench_prime_triple_v19_compressed.py` |
| V21 32-dim lattice vectors | 21–22× smaller (lossy, cell-aggregated) | 21–22× vs BM25 | `trng/bench_prime_triple_v21_bm25compress.py` |
| V21 exact prime-address BM25 | 2.38× smaller, lossless (64-bit prime products) | 2.38× | same |
| Lattice-BM25 (Zipf duality, uint16) | 4.32× faster, NDCG +0.16pp, r=0.9836 log(prime)↔IDF | faster + neutral | `trng/lattice_bm25_findings.md` |
| Markov-path codec | 5–8× lossless text | — | `final-build-aethos-13/STAGE7_FINDINGS.md` |
| AETHOS-13 record compression | 99.8–100% on IoT/financial/medical, 9.47 ms | vs gzip ~78%, LZMA 110 ms (44× faster) | `prime_hotel/compression_benchmark_results_*.json` |
| UltraFast int8 vectors | 24 B/doc; 8.8M→~2 GB | 40× vs 80 GB dense | `aethos13-ultrafast/README.md` |
| Statistical RAG vs ChromaDB | 1378× smaller, 706× ingest, 328× retrieve | vs ChromaDB | `prime_hotel/.../RAG_COMPETITION_COMPARISON.md` |

**Caveat on AETHOS-13 record compression:** it's *lossy pattern-deletion* (learns a min/max/mean rule,
deletes rows that fit, reconstructs approximately). Real on highly-patterned IoT/sensor data; NOT lossless
text. Don't pitch the 99.9% as text compression.

## DOC-ONLY claims — do NOT ship as numbers (no captured run)
- 80 GB → 2 GB = "600–1500×" (arithmetic is 40×; the rest is asymptotic projection)
- "inverse scaling: more data = better compression" (marketing framing of record dedup)
- 24 B/doc → "BEIR-comparable NDCG" (no A/B baseline captured)
- `formuilas/` RAG compression 40–75% (in-code only, synthetic chunks, no captured run)
- "Complete Moser 40 B/doc" (captured but NDCG 0.005–0.20 = non-functional retrieval)

## NEGATIVE / closed (don't reopen)
- V27 top-8 **lossy**: nfc −6.2pp, scifact −19.5pp (67 B/doc but accuracy collapses)
- Polarity-prime non-semantic axis: −0.48pp scifact
- Word-Markov: null; PubMed bulk training: −0.61pp

## THE accuracy-from-compression test — verified, two-sided
**Stage 18 recursive lattice** (`final-build-aethos-13/STAGE18_RECURSIVE_LATTICE_FINDINGS.md`,
`results_recursive_lattice_*.json`). Materializes only OBSERVED prime-tuple addresses; docs cluster by
shared rare-prime subsets (depth = rarity); deeper match = stronger structural correlation = a relevance
signal on top of BM25. Compression-by-sparsity and clustering are the same operation.

| corpus | baseline nDCG@10 | +RL @α0.10 | Δ | storage |
|---|---|---|---|---|
| **scifact** | 0.7884 | **0.7957** | **+0.73pp** ✅ | 1373 MB / 5,183 docs (264 KB/doc) |
| nfcorpus | 0.3488 | 0.3476 | −0.11pp ❌ | 1012 MB |
| fiqa | 0.2979 | 0.2958 | −0.19pp ❌ | 5685 MB |

**Honest read:** real lift on scifact only; slight hurt on the mismatch corpora; and it's an EXPANSION, not
compression (10.3M addresses). This is the unsupervised co-occurrence ceiling (cf.
`unsupervised-correlation-mining-marginal`). The mechanism is sound; the two flaws are storage and noise.
Fix = rare-prime budget (compressive) + a relevance signal (supervision). The supervised version of this
same co-occurrence idea is the proven +3.5pp scifact bridges.

## The lazy mechanism — real in two places, portable to 8.8M
- `trng/living_network.py` (LivingLatticeNetwork): `lazy_correlations`, never materializes the full matrix;
  regenerates triples on demand by 3-way sorted-list merge. O(|p1|+|p2|+|p3|).
- `final-build-aethos-13/src/aethos/retrieval/_recursive_lattice.py`: observed-only address dict;
  coordinates derived on demand from `factor(address)`, never stored.
- SPEC-ONLY (not implemented, don't count on): `formuilas/lazy_compression.py` (stores full text in first
  chunk per coord), prime_hotel `insert()` (eager — all 8 coords at insert).

## Core formula map
| Formula | File |
|---|---|
| k-prime coordinate (Z = S+n identity) | `final-build-aethos-13/src/aethos/core/_lattice_core.py` |
| 8 base vectors + VA1–VB4 (32-cell tables) | `final-build-aethos-13/src/aethos/retrieval/_lattice_formula.py` |
| Markov-path / turtle codec | `final-build-aethos-13/src/aethos/compress/path_codec.py` |
| Recursive lattice | `final-build-aethos-13/src/aethos/retrieval/_recursive_lattice.py` |
| Timothy additive π (right-triangle, ~1/4ⁿ) | `final-build-aethos-13/src/aethos/pi/timothy.py`, `trng/pi_formula.py` |
| AETHOS compress modes (semantic/product/signature) | `trng/aethos_compress.py` |
| Lazy correlation store | `trng/living_network.py` |
| PAT balanced-ternary / HPATE harmonic | `formuilas/pat.py`, `formuilas/hpate.py` |
| recursive π = 2^(n+1)·sin(π/2^(n+1)) | `prime_hotel/recursive_geometry.py` |

*Spec-only (NOT standalone code): complex-plane/drift formulas, Moser-band network (only `moser_spheres.py`
in a subdir), meet/join in prime_hotel core.*

## 3 threads to revive (door NOT closed)
1. **V27 lossless packing** as the default posting format on 8.8M — free smaller+faster, lowest risk.
2. **Recursive lattice + rare-prime budget** — fix the storage, see if the scifact lift survives compressive
   (`_o1_recursive_lattice_revive.py`); then add supervision for generality.
3. **Lazy triple regeneration** (V19 + living_network) as the index-build policy — 3–5× smaller, the MARCO shape.
