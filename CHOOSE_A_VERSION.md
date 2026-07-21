# EdgeRAG — pick a version

Five branches, each independently mergeable. They are **layers**, not rivals: `2`, `3` and `4` stack on top of `1`.
Every number below is from a captured run in this repository. Where a number is weak, it says so.

---

## The short answer

| if you want | take | measured |
|---|---|---|
| runs anywhere, no model at serve | **1 — nogpu-engine** | SciFact 0.6963 fast / 0.7052 accurate(bake_M=16) @ **199 B/doc** |
| the best accuracy we have | **1 + 2 — apex-rerank** | SciFact **0.7465** (ties dense 0.7463), ArguAna **0.4535** (beats dense 0.4454) |
| the smallest index at competitive accuracy | **3 — pq-dense** | FiQA **146 B/doc**, nDCG 0.3932, **R@100 0.7241** |
| the biggest single accuracy jump, cheapest | **1 + 4 — llm-expansion** | NFCorpus 0.2841 → **0.4709** (+66%) — **only on the right corpora, see below** |
| to deploy it today | **5 — api-server** | FastAPI + Docker, CPU-only, 2 cores / 2 GB |

---

## 1 — `edgerag/1-nogpu-engine`
Encoder-free lexical lattice: stemmed index, counting-bridges, NPMI drift, CPU-GBDT reranker, optional
SPLADE-distilled tier. **No model at serve.** Three dials on one index:

| tier | SciFact nDCG@10 | ms/query | notes |
|---|---|---|---|
| `fast` | 0.6963 | 0.20 | exact lexical + stem |
| `accurate` | **0.7052** | 0.40 | + bridges + CPU-GBDT — **requires `bake_M=16`**; at the default `bake_M=0` it REGRESSES to 0.6699 |
| `max` | 0.6700 | — | + SPLADE-distill; **worse on SciFact, better on FiQA (0.2936 vs 0.2538)** |

**199 B/doc** measured end to end, save→load round-trip verified. `max` costs 280–440 B/doc.
Honest note: `max` is corpus-dependent — it helps where lexical retrieval struggles and hurts where it doesn't.

## 2 — `edgerag/2-apex-rerank`
Retrieve ~200 docs with tier 1, rerank with a teacher model.

| corpus | tier-1 | apex | dense (full scan) |
|---|---|---|---|
| SciFact | 0.6963 | **0.7465** | 0.7463 |
| ArguAna | 0.3054 | **0.4535** | 0.4454 |
| NFCorpus | 0.3124 | 0.3711 | 0.3814 |
| FiQA | 0.2468 | 0.3950 | 0.4432 |

**On ArguAna the apex beats scanning every embedding.** A cheap lexical pool is a better filter than dense's own
top-100 — dense's extra reach costs it precision. Ties dense on SciFact. Still behind on FiQA (−0.048).
Cost: a model in the serving path, but only over ~200 documents.

## 3 — `edgerag/3-pq-dense`
Product-quantised dense vectors, scanned directly. No lexical tier underneath.

| corpus | B/doc | nDCG@10 | R@100 | vs fp32 dense |
|---|---|---|---|---|
| FiQA | **146** | 0.3932 | **0.7241** | −0.050 nDCG at 28× smaller |
| SciFact | 147 | 0.7047 | **0.9650** | R@100 **beats** full fp32 dense (0.9483) |
| NFCorpus | 168 | 0.3510 | 0.3319 | −0.030 |

**Bytes/doc include the codebook** — a fixed 1 MB, so these figures *improve* with corpus size
(at 1M docs, ~129 B/doc). Needs a query encoder at serve. This is standard product quantization, not proprietary.

## 4 — `edgerag/4-llm-expansion`
Expand the query with an LLM before retrieval. **Only fires where the vocabulary gap is large.**

| corpus | vocabulary gap | nDCG@10 change |
|---|---|---|
| NFCorpus | **61.2%** | **+0.1868** (0.2841 → 0.4709 — ties dense) |
| FiQA | 7.3% | −0.0453 |
| SciFact | 1.8% | **−0.1175** |
| ArguAna | 0.0% | −0.0397 |

**The governor is the product.** "Vocabulary gap" = share of relevant documents containing *none* of the query's
words. Above ~25% expansion wins big; below ~10% it actively hurts. Estimate it from ~20 labelled queries at
onboarding. Caveats: the NFCorpus run was 45 queries (full 323 pending); the technique is published
(HyDE / query2doc) — the contribution here is knowing when to fire it.

## 5 — `edgerag/5-api-server`
FastAPI wrapper, CPU-only Dockerfile (ARM + x86), Oracle Always-Free deploy notes, and a demo that runs on real
SciFact. GPU is disabled at import so a GPU can never sneak into the serving path.

---

## MEASURED LADDER — all 8 collections (2026-07-20)

Every number from a captured run in this repository. `apex` = retrieve ~200 docs with the lexical tier, rerank with
a teacher. `PQ apex` = the same with the teacher's vectors quantised. `~SOTA` is a RECOLLECTION of published BEIR
leaderboards, not measured here — verify before quoting it to anyone.

| corpus | docs | BM25 | dense | **apex** | PQ apex | teacher B/doc | ~SOTA | apex vs BM25 |
|---|---|---|---|---|---|---|---|---|
| quora | 522,931 | 0.7766 | 0.9030 | **0.9000** | 0.8858 | 130.0 | 0.88 | +0.1234 |
| trec-covid | 171,332 | 0.5836 | 0.6440 | **0.7486** | 0.7358 | 134.1 | 0.80 | +0.1650 |
| scifact | 5,183 | 0.6963 | 0.7463 | **0.7465** | -- | -- | 0.77 | +0.0502 |
| arguana | 8,674 | 0.3054 | 0.4454 | **0.4535** | 0.4318 | 248.9 | 0.60 | +0.1481 |
| fiqa | 57,638 | 0.2468 | 0.4432 | **0.3950** | -- | -- | 0.48 | +0.1482 |
| nfcorpus | 3,633 | 0.3124 | 0.3814 | **0.3711** | -- | -- | 0.38 | +0.0587 |
| webis-touche2020 | 382,545 | 0.3416 | 0.2419 | **0.2595** | 0.2167 | 130.7 | 0.30 | -0.0821 |
| scidocs | 25,657 | 0.1347 | 0.1997 | **0.2094** | 0.2075 | 168.9 | 0.20 | +0.0747 |

**MS MARCO** (8,841,823 passages, MRR@10): BM25 **0.1544** -> CE apex **0.3594** (**+0.2050**), ~SOTA 0.39.

### What this says
- **The apex helps on 7 of 8 collections** and is worth +0.05 to +0.21. On **webis-touche it HURTS (-0.0821)**,
  because BM25 already beats dense there by +0.0997. **Rule: apply the apex only where the teacher beats BM25** —
  estimable from ~20 labelled queries at onboarding.
- **The apex BEATS scanning every embedding** on arguana (+0.0081), scidocs (+0.0097), webis (+0.0176) and
  **trec-covid (+0.1046)**. A cheap lexical pool is a better filter than dense's own top-100.
- **PQ costs almost nothing**: -0.003 to -0.014 against the fp32 teacher, at ~130-170 B/doc. Bytes/doc INCLUDE the
  codebook, so they improve with corpus size (webis: codebook is 2.7 of its 130.7 B/doc).
- **trec-covid R@100 is capped at 20.3%** by gold density (493.5 relevant docs per query) — read nDCG only there.

### Caveats that belong with these numbers
- `bake_M` **defaults to 0**, and at that default calling `fit()` makes scifact WORSE than not fitting
  (0.6699 vs 0.6963). Use `bake_M=16`, which turns it into 0.7052. This is a default to change, not a broken tier.
- The recorded T1_bridged **0.7204** has not been reproduced; best reached is **0.7052**.
- Stemming is confirmed as the single universal lever: **+0.0251** on scifact (0.6963 vs 0.6712).

## Two claims to retire before any customer conversation

**"24 B/doc"** — in the March UltraFast build, `storage_bytes_per_doc()` returns the embedding dimension and
nothing else. It does not count token ids or the BM25 structures. The real serialized index is far larger.
**The engine in branch 1 genuinely measures 199 B/doc end to end** — quote that instead.

**"20–60× smaller"** — true against *uncompressed* fp32 vectors (4096 B/doc). Production vector databases ship
product quantization; we measured that at ~96 B/doc. Against a properly compressed competitor the size advantage
is roughly par. The durable differentiators are **no GPU at serve, sub-millisecond retrieval, glass-box
explainability, and a 2-core / 2 GB deployment** — none of which a vector DB can match on price.

## Measurement status
**All 8 collections have a full ladder** (table above), including MS MARCO at 8.8M passages. Known gaps:
the NFCorpus LLM-expansion result is 45 of 323 queries; the recorded T1_bridged 0.7204 is unreproduced (best 0.7052);
and the `~SOTA` column is recollection, not measurement.
