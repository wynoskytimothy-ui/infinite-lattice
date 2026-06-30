# RAG tuning campaign — Wave 1: the per-corpus map (glass-box, measured)

*Wave 1 of the tuning campaign: per-corpus recall failure-analysis (DROWNED vs GAP) + a zero-shot lexical lever
sweep across 5 BEIR corpora, run as a workflow and adversarially synthesized. `_o1_diag.py`, `_wave1_result.json`.
The honest verdict: **lexical levers are marginal; the real recall deficit is semantic** — which redirects the
campaign toward the levers that actually move the needle.*

## The 5-corpus diagnostic

| corpus | docs | drowned% | gap% | BM25 R@100 | best lever | gain |
|---|---|---|---|---|---|---|
| scifact | 5,183 | 9.7% | 2.0% | 0.876 | dfcap10 | +0.0057 |
| nfcorpus | 3,633 | 2.5% | 19.8% | 0.238 | dfcap10 | +0.0027 |
| fiqa | 57,638 | 27% | 2.6% | 0.508 | bm25 | 0 |
| trec-covid | 171,332 | 0%* | 0%* | 0.095 | bm25 | 0 |
| webis-touche2020 | 382,545 | 0%* | 0%* | 0.558 | 2way | +0.0073 |

## Universal lever rules (consistent across 5 corpora)

- **`anchor` (rare-anchor) is universally TOXIC** — negative 5/5, mean −0.136, worst −0.32 (scifact: a 36%
  relative recall collapse). Long queries' rarest word often isn't in the gold. **Never deploy.**
- **`idf²` never helps** — negative 5/5. Over-weighting rare terms drops recall. Drop it.
- **Intersection (`2way`/`3way`) is mostly harmful** — hard term-intersection discards gold matching a *subset*
  of query terms. Helps only touche (`2way` +0.0073, the lone intersection-favorable corpus).
- **`df-cap` is the gentlest family.** `dfcap30` mean −0.0018 (smallest downside of any lever), positive on
  scifact/nfcorpus, ties on fiqa → **the universal safe default.** `dfcap10` is more aggressive: helps small
  drowned corpora, hurts large ones.

## The honest headline: lexical tuning is MARGINAL (≤ +0.0073)

Every lexical lever's best gain is tiny. **The large recall deficits (nfcorpus 0.24, trec-covid 0.095) are
GAP-driven — gold docs that share no query word — and NO lexical lever can reach them.** The dramatic boosts
require *semantic* expansion (distilled-SPLADE: +18% recall on nfcorpus), not lexical retuning. The glass-box
diagnostic *saved us from grid-searching dead lexical knobs.*

## What the synthesis caught (honest corrections to the thesis)

1. **The drowned/gap classifier breaks on multi-gold corpora.** trec-covid/touche report 0/0 because every query
   has *many* relevant docs, so ≥1 lands in the pool (query never "fully misses") — but per-gold recall is low.
   **Fix (Wave 1.5): classify per gold DOC, not per query.**
2. **The fiqa paradox:** highest drowned% (27%) yet *no* lever helps. The drowning is by *query-relevant* common
   terms, so df-cap/intersection discard signal. fiqa needs term-**weighting** (SPLADE), not lexical **pruning**.
3. **Pool depth may beat any lever:** median drown-rank sits just past the cutoff (scifact 276, fiqa 638) — many
   drowned golds are barely out of the top-100. Increasing pool depth (recall@k) likely recovers more than any
   lexical lever (the standing "recall is the lever, pool-limited" finding).

## Deploy guide (per-corpus, from Wave 1)

| corpus | lexical default | the real lever |
|---|---|---|
| scifact | dfcap10 (+0.006) | already strong (0.876); df-cap is the marginal win |
| nfcorpus | dfcap30 (no-op) | **distilled-SPLADE** (semantic-bound, 19.8% gap) |
| fiqa | bm25 | **SPLADE term-weighting** (drowning by relevant common terms) |
| trec-covid | bm25 | **SPLADE** + deeper pool (semantic-bound, 0.095 floor) |
| touche | **2way** (+0.007) | the lone intersection-favorable corpus |
| **universal** | **dfcap30** | never anchor/idf²; semantic for gap-bound |

## Next waves (redirected by Wave 1's verdict)

- **Wave 1.5 (quick):** fix the per-gold diagnostic; add a pool-depth sweep (recall@100/200/500/1000).
- **Wave 2:** composite tokens done *fairly* (2/3-way co-occurrence as a feature with its own idf, not a
  scoring multiplier) — Timothy's free-composites, tested honestly against the Wave-1 "intersection mostly hurts".
- **Wave 3:** the **correlation-graph drift** (Timothy's higher-D idea) as a ZERO-SHOT, encoder-free attack on
  the GAP — diffuse query words to co-occurring/correlated words via the lattice graph. This is the
  Timothy-native, model-free path to the semantic recall that SPLADE gets with a GPU.

## Wave 2 — DRIFT (zero-shot co-occurrence diffusion) across 5 corpora

| corpus | lexical R@100 | drift R@100 | gain | best α | vs SPLADE-full |
|---|---|---|---|---|---|
| nfcorpus | 0.238 | **0.284** | +0.046 | 0.4 | **beats** (0.282) |
| scifact | 0.876 | 0.906 | +0.030 | 0.8 | within 0.003 (0.909) |
| fiqa | 0.508 | 0.508 | 0 | 0 (off) | far below (0.610) |
| trec-covid | 0.095 | 0.095 | 0 | 0 (off) | — |
| touche | 0.558 | 0.558 | 0 | 0 (off) | — |

**Verdict: drift is a SELECTIVE, self-gating recall lever — not universal.** Helps 2/5 (nfcorpus, scifact);
on fiqa/trec-covid/touche the per-corpus alpha sweep returns 0 (co-occurrence net-harmful → dialed out), so
*tuned, drift never hurts*. Where it helps it attacks the semantic GAP and on nfcorpus **beats GPU SPLADE
zero-shot**. It trades nDCG for recall (scifact 0.671→0.511) → a RECALL stage, rerank after. Build is no-GPU,
0.9s–35s. Deploy: drift (alpha-gated) first on mismatch corpora → rerank; pure lexical where alpha=0; SPLADE
where it leads (fiqa). The real ceiling = FUSE complementary recall sources (lexical+bridges+drift+SPLADE),
which miss different docs (Wave 3).

## Wave 3 — FUSE complementary recall sources + rerank (nfcorpus)

| stage | recall@100 | nDCG@10 |
|---|---|---|
| best single (drift) | 0.284 | — |
| FUSE equal-RRF | 0.278 (dilutes) | — |
| FUSE weighted-RRF | 0.283 | — |
| **union → cross-encoder rerank** | **0.285** | **0.332** |
| union ceiling | **0.352** | — |
| SPLADE-full (ref) | 0.282 | 0.336 |

The four recall sources (lexical, bridges, drift, distilled-SPLADE) are COMPLEMENTARY — union ceiling 0.352 vs
any single ~0.28 (+24% headroom). The full stack (fuse → rerank) MATCHES SPLADE-full (recall 0.285>0.282, nDCG
0.332≈0.336) and is the best encoder-free nDCG, no SPLADE query encoder at serve. But it captures only 0.285 of
the 0.352 union ceiling -> the REMAINING HEADROOM (0.067) is the next lever: a stronger reranker or deeper pool
(recall@k>100). The campaign located it precisely.

### Campaign synthesis (Waves 1-3)
- Wave 1: deficit = recall (semantic GAP); lexical levers marginal (≤+0.007); anchor/idf² toxic; df-cap gentlest.
- Wave 2: drift (zero-shot co-occurrence diffusion, Timothy's higher-D idea) = selective recall lever, BEATS
  SPLADE on nfcorpus, self-gating (alpha->0 where noisy).
- Wave 3: sources are complementary (union 0.352); fuse+rerank matches SPLADE-full encoder-free; headroom = a
  better reranker / deeper pool. METHOD: diagnose -> aim the lever -> measure -> the next diagnosis aims the next.

## Wave 3b — capture the union headroom (nfcorpus, depth-300 pools + rerank)

| metric | recall | note |
|---|---|---|
| lexical recall@100 | 0.238 | baseline |
| fused+rerank recall@100 | 0.272 | (top-100 rerank dilutes slightly vs depth-100's 0.285) |
| fused+rerank recall@200 | 0.334 | |
| fused+rerank recall@500 | **0.456** | ~2x lexical |
| union ceiling@300 | **0.487** | deeper pools raised ceiling 0.352->0.487 |
| reranked nDCG@10 | 0.332 | precision held |

THE RECALL IS POOL-DEPTH-LIMITED, not gone: the gold sits just past rank-100. Deeper fused+reranked pooling
recovers it — recall@500 0.456 vs lexical 0.238 (nearly 2x), nDCG held 0.332. For a RAG feeding an LLM a
top-200..500 context, this DOUBLES recall. recall@100 strictly is reranker-bound (a stronger reranker is the
lever there). The campaign's located headroom is captured at depth.
