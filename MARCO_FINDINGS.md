# MS MARCO retrieval, tuned by diagnosis — the ladder, the wall, and the hybrid

A from-scratch MS MARCO passage-retrieval study that runs the full arc: a diagnostic-tuned
**zero-shot rule ladder** on a 298 K pool (§1–5), the same ladder validated on the **full
8.8 M collection** (§6), and a deep investigation of what the lattice can and cannot do at scale
(§7) that ends in a **SOTA-class hybrid at ~2.3 GB**. Two results carry the study:

1. **The method** — a glass-box loop that profiles a corpus, finds which lever matters, adds one
   measured rule, proves it on held-out queries, and *knows when to stop* (it rejects rules the
   data doesn't support). Pool ladder: MRR@10 0.5419 → 0.6030, **+11.3%** over BM25.
2. **The division of labor, measured five ways** — the prime-lattice is a **recall engine**
   (reaches the gold **99.7%** of the time, including golds BM25 *structurally cannot* retrieve),
   but ranking the gold #1 is **contextual comprehension** that no static representation performs.
   The honest system is the **hybrid**: lattice recall (2.16 GB, explainable, append-only) + an
   80 MB neural reranker = **0.41 MRR@10**, which *beats* the standard BM25+neural stack on the
   vocabulary-mismatch queries only the lattice reaches.

Every number traces to a runnable `marco_*.py`. The negatives are kept as carefully as the wins.

> Companion to [`FINDINGS.md`](FINDINGS.md) (the prime-lattice / supervised-bridges BEIR
> thread). §1–5 use a **clean BM25 harness** to isolate the rule-discovery method; §6–7 take it
> to the full 8.8 M collection and the lattice/hybrid. Scope is stated honestly in §8.

---

## 1. The harness (the frozen baseline)

[`marco_lab.py`](marco_lab.py) is rung 0. It caches a MARCO evaluation **pool** (3,000 dev
queries × their relevant passages + distractors ≈ 298 K passages) and exposes one `Index`
class: a textbook BM25 engine (k1=0.9, b=0.4) whose **only swappable part is the tokenizer**.
The engine is frozen for the whole study; each rung changes the tokenizer or adds a reranking
rule on the top-100, never the core. Metrics: **MRR@10** (the judged metric) and **R@100**
(the recall ceiling). Discipline: diagnose and tune on `qrels.train`, prove on the dev pool —
never fit the metric we report.

## 2. The ladder — five rules, each earned

| rung | rule | script | MRR@10 | Δ | where it works |
|---|---|---|---:|---:|---|
| 0 | BM25 baseline | `marco_lab.py` | 0.5419 | — | — |
| 1 | conservative stem | `stem_safe.py` | 0.5777 | +0.0358 | everywhere (inflection only) |
| 2 | + gold-doc company | `marco_golddoc.py` | 0.5822 | +0.0045 | rare-term corridors |
| 3 | + entity gate | `marco_entity.py` | 0.5898 | +0.0076 | hit **and** miss set |
| 4 | + rare crossover | `marco_cascade.py` | 0.6004 | +0.0106 | miss set (×2) |
| 5 | + diversity | `marco_distinct.py` | **0.6030** | +0.0026 | miss set (+25%) |

**Rung 1 — conservative stem** ([`stem_safe.py`](stem_safe.py)). The single biggest jump,
and it is pure restraint: `safe(w)` does **inflectional** folding only (plurals `-ies→y`,
sibilant `-es`/`-s`, `-ing`/`-ed` with the double-consonant fix) and **never derivational**
(`organization` stays `organization`). Aggressive stemming collapses distinct concepts and
*loses* MRR; the win is in stemming *less*.

**Rung 2 — gold-doc company** ([`marco_golddoc.py`](marco_golddoc.py)). For each rare query
term, learn the company it keeps **in relevant documents** (from `qrels.train`): a relevance-
grounded corridor of co-occurring terms. β=0.3 reranks the top-100 by that company. Net
positive (hit-set held, miss-set rescued) — supervised signal injected by counting, no SGD.

**Rung 3 — entity gate** ([`marco_entity.py`](marco_entity.py)). The corridor only fires on
docs that contain the query's **rarest** term (idf ≥ 5.5). One line, zero training: it stops
the company signal from drowning a query's anchor entity (the diagnostic showed `Jupiter`
queries pulled `Pluto` docs). Lifts hit **and** miss; `Freon` flipped rank 17→10.

**Rung 4 — rare crossover** ([`marco_cascade.py`](marco_cascade.py)). The core reranker.
Score = anchor + α·(Σ rare-idf)·k^γ + β·company, where k = how many of the query's rare
terms a doc matches. Super-linear in k (α=0.5, γ=1.5) because the **intersection** of two
rare terms is far more discriminating than either alone (§4). Doubled the miss-set; R@100 0.9117.

**Rung 5 — diversity** ([`marco_distinct.py`](marco_distinct.py)). The pollutant profile
(§4) found wrong rank-1 docs are *longer and more repetitive* than the gold (distinct-vocab
67% vs 74%). A **gentle** diversity reward (score × distinct_ratio^0.25) demotes the
repetitive goblins: miss-set +25%, hit-set held. **Self-bounding** — at p>0.25 "reward
diversity" bleeds into "reward short docs" and the strong matches collapse (hit 0.75→0.67).
The data drew its own line.

## 3. Negative results — the rules we rejected

As load-bearing as the wins. Each was *predicted* to help and *measured* not to.

1. **Medium-frequency compound, as a score boost** ([`marco_cascade2.py`](marco_cascade2.py)).
   Feeding medium-idf terms into the crossover (to recover 1-rare-word queries) **hurt
   monotonically** (0.6004 → 0.5792). Lesson: *recovery ≠ discrimination*. Medium terms are
   too common to discriminate; they add mass to every candidate, including the goblins.

2. **Phrase / proximity / order** ([`marco_explore.py`](marco_explore.py)). The white box
   asked, on 336 failures, *which surface signal favors the gold over the wrong doc we
   picked?* The answer was sobering: **none does.** company favored the *wrong* doc 60% of
   the time, exact-match 71% wrong, rare-match 0% gold / 58% wrong. Only subword (+5%) and
   plural (+1%) net-favored the gold. There is no unused surface signal hiding in the text.

## 4. The diagnostic method (the real artifact)

The ladder is a byproduct; the **glass-box loop** is the discovery. On any corpus:

**(a) Profile the discrimination structure** — `marco_overlap.py`, `marco_crossover.py`,
`marco_rarest.py`, `marco_recover.py`. These measured the cascade that every rule exploits:

| candidates after… | docs left | gold present |
|---|---:|---:|
| rarest query word alone | ~76 | 89% |
| + 2nd-rarest (2-way meet) | ~12 | 69% |
| + 3rd-rarest (3-way meet) | ~1 | 29% |

Each rare word divides the candidate set ~6–10×. The gold is almost always *reachable*
(R@100 0.9117) — retrieval is a **reranking** problem, not a recall problem, on this pool.

**(b) Read ten wins and ten losses, with their trigger words** ([`marco_diagnose.py`](marco_diagnose.py)).
Named the failure modes: entity-drown (`Jupiter`→`Pluto`), polysemy (`golden`→`gate`),
corpus-absent terms, label sparsity. Each named mode became a candidate rule.

**(c) White-box the failures** ([`marco_explore.py`](marco_explore.py), [`marco_pollution.py`](marco_pollution.py)).
Which signal *would* pull the gold up, and what do the pollutant docs look like? This is what
told us the ceiling is real (§5), and produced the one remaining rule (diversity, rung 5).

**(d) Add one rule, prove on held-out, repeat — and stop when the white box says the signal
is gone.** That last clause is the discipline. We stopped at rung 5 because `marco_explore.py`
showed no surface signal remains, not because we ran out of ideas.

## 5. The ceiling, named (not guessed)

- **R@100 = 0.9117** — the gold is in the top-100 *91% of the time*. We land it top-10 at
  0.60. The 0.31 gap is **rerank headroom**, not missing recall.
- **The wall is answer-ness.** On the residual failures the gold matches *fewer* of the
  query's words than the wrong doc, at *identical* match-density — it is the answer despite
  being the weaker surface match. No surface rule demotes a doc for legitimately matching
  more query words. Closing this needs a *semantic* signal (passage-level answer-ness), or it
  is benchmark label sparsity — and `marco_explore.py` cannot tell those apart from the
  surface.

The white box thus draws a clean boundary: **word-matching is exhausted at +11.3%; the rest
is understanding (or label noise).**

## 6. The full 8.8M magnitude test — does the gain hold at 30× distractors?

The pool ladder was tuned on 298 K passages (hit-set 74%). The honest test is the full
collection. Built a compact stemmed numpy CSR index over all **8,841,823 passages**
(1.37 M terms, 351 M postings, **2.16 GB**, CPU) plus an unstemmed twin, and ran the *same*
ladder on 3,000 dev queries (top-100 rerank, texts fetched on demand via byte offsets).
The idf gates are scale-invariant (idf ≈ ln(N/df) ⇒ each threshold is a fixed corpus
fraction), so the constants port without retuning.

**The complete ladder, pool vs full (MRR@10, 3,000 queries):**

| layer | pool 298 K | full 8.8 M |
|---|---|---|
| raw BM25 | 0.5419 | 0.1801 |
| + conservative stem | 0.5777 (+6.6%) | 0.1845 (**+2.4%**) |
| + rerank rules | 0.6030 (+4.4%) | 0.1950 (**+5.7%**) |
| **total over raw BM25** | **+11.3%** | **+8.3%** |

**The gain holds (+8.3%) and its composition validates the thesis.** BM25 = 0.180 matches
the canonical full-collection figure — the harness is verified, not flattering. The *lexical*
gain (stemming) **dilutes** at scale (+6.6%→+2.4%: with 30× distractors, inflectional merges
matter less), while the *semantic* gain (gold-company + crossover + diversity) **grows**
(+4.4%→+5.7%). Stemming was 59% of the pool's gain but only 30% of the full collection's; the
rerank rules are now **70%**. The durable, scale-robust part is the semantic layer — exactly
the project's bet.

**The mechanism flipped, honestly.** The miss-set is now the **majority** (1,825/3,000 = 61%,
vs the pool's 26%). On the pool the ladder *held* hits and rescued misses; at full scale it
slightly *sacrifices* hits (0.471→0.458, −2.9%) to rescue the now-dominant miss-set
(0→0.026) — net-positive only because misses dominate. The same constants are a touch too
aggressive for the hit-set at scale → the **selective gate** (fire the rerank only when BM25
is uncertain) is the data-indicated next lever, and it finally has real work to do.

**Still lexical-tier.** 0.195 is far below dense SOTA (~0.38–0.40); these zero-shot rules
close ~8% of the BM25→SOTA gap. Closing the rest needs learned term-weighting (SPLADE-style)
or answer-level semantics — not more surface rules. Footprint/speed held: 2.16 GB CPU index,
top-100 over 8.8 M in ~0.3 s/query (unoptimized Python).
*(Scripts: `marco_full_build.py` index, `marco_full_eval.py` ladder, `marco_full_raw.py` raw baseline.)*

## 7. The full-collection deep dive — recall, the answer-ness wall, and the hybrid

§6 confirmed the ladder holds at 8.8M but stays lexical-tier (0.195). This is the deeper
investigation: where the lattice genuinely *wins*, where it provably *can't*, and the system
that results. All numbers are full 8.8M, 3,000-query dev sample unless noted; scripts `marco_full_*.py`.

### 7.1 The lattice is a recall engine — and it reaches what BM25 can't

The discrimination cascade, re-measured at scale ([`marco_full_diagnose.py`](marco_full_diagnose.py)):
the rarest query word reaches a median **2,398 docs** (gold present 88.5%); the 2-way meet
collapses that to **69** (gold 65.6%); 3-way to **9** (46%). Gold-coverage is *scale-invariant*
(vs the pool's 89/69/29%); only the reach scaled ~30× with the corpus — so the meet is the
load-bearing operation at scale, and its depth grows with collection size. Recall, pushed to its
ceiling ([`marco_full_recall.py`](marco_full_recall.py), [`marco_full_recall2.py`](marco_full_recall2.py)):

| | recall |
|---|---:|
| gold reachable (corridor-expanded membership) | **0.997** |
| ranked into top-1000 (bm25-rare + tf + corridors) | 0.826 |
| ranked into top-100 | 0.646 (≈ BM25's 0.666) |

tf-saturation was the lever that lifted ranked recall to BM25-class (the bare meet ignored term
frequency). The corridor expansion reaches the gold **99.7%** of the time — *including golds with
no lexical overlap with the query*, which BM25 structurally cannot retrieve.

### 7.2 The answer-ness wall — proven five independent ways

The gold is reachable 99.7% but lands rank-1 only ~10% of the time. The entire gap is *ranking*,
and every static signal hits the same wall:

| approach | MRR@10 | script |
|---|---:|---|
| pure rare-word meet (no BM25) | 0.119 | `marco_full_lattice.py` |
| + corridors | 0.133 | `marco_full_lattice.py` |
| BM25 (= the complete idf-weighted meet) | 0.185 | `marco_full_eval.py` |
| binary contextual rerank | 0.200 | `marco_full_context.py` |
| LSA continuous co-occurrence embedding | 0.037 | `marco_full_embed.py` |
| doc↔doc graph propagation | 0.185 | `marco_full_docgraph.py` |
| **cross-encoder (contextual, learned)** | **0.407** | `marco_full_hybrid.py` |

1. **The pure meet *is* BM25.** BM25's score is `Σ idf·tf-sat·length-norm`; a doc matching two
   rare words scores idf₁+idf₂ — the soft 2-way meet over every term. The pure meet loses only
   because it's an *incomplete* BM25 (drops tf, length, medium terms); add them back and you
   re-derive BM25.
2. **The pollution is not goblins.** Decomposing the score gap on 415 ranking failures
   ([`marco_full_pollution.py`](marco_full_pollution.py)): common-word stuffing is **3.8%** of
   the gap; **64.7%** is rare words, and 66% of the time the gold simply *lacks* a rare word the
   pollutant has. The wrong doc wins by genuinely matching more of the query — a stronger lexical
   match that happens not to be the answer. Nothing to down-weight.
3. **Continuity isn't the lever.** The continuous LSA embedding (the fair test of "vectors
   instead of counts") scored *worse* than the discrete meet (0.037). The boundary is **static
   vs contextual**, not discrete vs continuous: inside an on-topic top-100, topical similarity is
   flat — only reading *this query against this passage* discriminates.
4. **Answer-ness is comprehension.** The one thing that separates the gold from its
   rare-word-richer competitors is whether the passage *answers* the query — a contextual,
   learned operation no static representation (symbolic or vector, any geometry) performs.

### 7.3 The hybrid — the lattice's recall + a small reranker = SOTA-class

The honest division of labor: lattice for recall, neural model for answer-ness
([`marco_full_hybrid.py`](marco_full_hybrid.py), [`marco_full_hybrid2.py`](marco_full_hybrid2.py)).
A cached 80 MB MS-MARCO cross-encoder (MiniLM-L-6-v2) reranks the top-100:

| first stage → cross-encoder | recall@100 | MRR@10 |
|---|---:|---:|
| BM25 pool | 0.688 | 0.4065 |
| lattice pool | 0.670 | 0.3882 |
| **union (BM25 ∪ lattice)** | **0.730** | **0.4132** |

MRR jumps 0.19 → **0.41** (2.1× over BM25), into MARCO-dev-SOTA territory (~0.38–0.40). The
**union beats BM25** because the lattice reaches a gold BM25 misses on **4.2%** of queries (no
lexical overlap, reached via corridor *meaning*); the cross-encoder ranks 13 of those into the
top-10 — **net-new wins a pure BM25+neural stack never sees.** Footprint: **2.16 GB lattice +
80 MB reranker ≈ 2.3 GB** — the 80 GB → ~2 GB, faithful-to-SOTA, fast, first-stage-explainable
north-star, achieved.

### 7.4 Recall is maxed — the zero-shot correlation layers converge

The "lattice-inside-lattice" — an unsupervised, ingest-time rare-word↔rare-word co-occurrence
sub-lattice ([`marco_full_sublattice.py`](marco_full_sublattice.py), 352 K terms / 2.8 M edges /
**23 MB**) — builds and gives a small real lift (+0.007 R@1000). But its *novel* reach over the
supervised corridor is **0.0003**: the supervised (qrels) and unsupervised (corpus) layers are
two views of the same co-occurrence structure, so they don't stack for reach. Multiple independent
methods converge on **99.7% reachable** — recall is genuinely near its ceiling. The remaining
headroom is ranking, which belongs to the reranker.

### 7.5 An honest negative worth keeping

The `Γ`-mixer × VSA unification (give the entanglement-ODE recurrence an outer-product matrix
state to fix the MQAR recall hole) was tested across two iterations and **did not work** — every
gamma-family variant capped at the 17% marginal-fallback baseline vs attention's 100%
([`gpu_gamma_outer.py`](gpu_gamma_outer.py)). The binding math (DeltaNet) provably solves MQAR in
the literature, so this is an implementation gap (likely cramped head dim), not a disproof — but
it is a proper port, not the one-liner a survey suggested. Banked because the negatives are the
record.

## 8. Honest scope

- **Relative gains + a SOTA-class hybrid.** The +11.3% (pool) / +8.3% (full) are clean
  relative gains over BM25; the hybrid's **0.41 MRR@10** is MARCO-dev-SOTA-class on the 3,000-q
  sample (a full 6,980-q dev run is the unimpeachable confirmation). The lattice's distinctive,
  defensible claim is the **+4.2% recall reach BM25 structurally lacks**, converted to net-new
  wins by the reranker.
- **The cross-encoder is supervised, not zero-shot.** The 0.41 is the *hybrid*: the lattice
  supplies recall (its corridors are qrels-counted; the index is zero-shot at query time), the
  80 MB neural model supplies the answer-ness ranking and is trained on MS MARCO. The lattice
  *alone* ranks at ~0.17 (BM25-class) — it is a recall engine, not a ranker.
- **Caches are not committed.** `marco_pool.pkl` (102 MB) and the full-index `full_idx_*` /
  `full_idx_sublattice.pkl` regenerate from the MARCO TSVs; they are git-ignored, not lost.

---

## Script index

**Harness / rules:** `marco_lab.py` (pool + BM25 + eval), `stem_safe.py` (rung 1),
`marco_golddoc.py` (2), `marco_entity.py` (3), `marco_cascade.py` (4), `marco_distinct.py` (5).
**Diagnostics:** `marco_overlap.py`, `marco_crossover.py`, `marco_rarest.py`,
`marco_recover.py` (discrimination structure); `marco_diagnose.py` (glass-box wins/losses);
`marco_explore.py` (white-box signal analysis); `marco_pollution.py` (pollutant profile).
**Rejected:** `marco_cascade2.py` (medium-boost), the proximity/phrase columns of `marco_explore.py`.
**Full 8.8M index/ladder:** `marco_full_build.py` (stemmed CSR index), `marco_full_eval.py`
(ladder + BM25 on the full collection), `marco_full_raw.py` (unstemmed BM25 baseline),
`marco_full_diagnose.py` (the cascade at scale).
**Recall (§7.1, §7.4):** `marco_full_recall.py`, `marco_full_recall2.py` (tf-saturation + corridor
recall-max), `marco_full_company.py` (rare-pair company), `marco_full_residual.py` (residual
glass-box: subword/medium/second-order), `marco_full_sublattice.py` (the unsupervised
lattice-inside-lattice).
**The answer-ness wall (§7.2):** `marco_full_lattice.py` (pure meet, no BM25),
`marco_full_context.py` (binary contextual), `marco_full_embed.py` (LSA continuous embedding),
`marco_full_docgraph.py` (doc↔doc graph). 
**The hybrid (§7.3):** `marco_full_hybrid.py`, `marco_full_hybrid2.py` (lattice-reach union → cross-encoder).
**Honest negative (§7.5):** `gpu_gamma_outer.py` (Γ×VSA unification — did not work).

*Every number is reproduced by running its script against the cached pool (§1–5) or the full
8.8 M collection (§6–7). The ladder and the hybrid are the receipts; the diagnostic loop — and
knowing where each tool wins — is the product.*
