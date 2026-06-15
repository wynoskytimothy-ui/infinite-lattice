# MS MARCO retrieval, tuned by diagnosis — the ladder and the method

A from-scratch MS MARCO passage-retrieval lab where **every accuracy rung is a zero-shot
rule a diagnostic pointed at**, not a guess. The headline is not the final number
(MRR@10 0.5419 → 0.6030, **+11.3%** over BM25); it is the *method* — a glass-box loop that
profiles a corpus, finds which lever matters, adds one measured rule, proves it on held-out
queries, and **knows when to stop**. Every number here traces to a runnable `marco_*.py`.

> Companion to [`FINDINGS.md`](FINDINGS.md) (the prime-lattice / supervised-bridges BEIR
> thread). This study is built on a **clean BM25 harness**, not the lattice engine — it
> isolates the *rule-discovery method* so it can later be ported to the lattice and the full
> collection. Scope is stated honestly in §6.

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

## 6. Honest scope

- **Relative, not absolute SOTA.** +11.3% over a clean BM25 on a 298 K **pool** is solid and
  reproducible. Full-collection (8.8 M) absolute MRR is **unmeasured here**; pool numbers run
  higher than full-collection because the distractor set is smaller. MARCO dev SOTA is ~0.38–
  0.40 (dense); SPLADE++ ~0.38 is the sparse/CPU comparator. This study establishes a *method
  and a relative gain*, not a leaderboard entry.
- **BM25 harness, not the lattice.** Built on `marco_lab.py`'s plain BM25 to isolate the
  rule-discovery loop. Porting the ladder onto the prime-lattice engine (`FINDINGS.md`) and
  running the full collection are the two named next steps.
- **Caches are not committed.** `marco_pool.pkl` (102 MB) regenerates from the MARCO TSVs via
  `marco_lab.load_pool()`; it is git-ignored, not lost.

---

## Script index

**Harness / rules:** `marco_lab.py` (pool + BM25 + eval), `stem_safe.py` (rung 1),
`marco_golddoc.py` (2), `marco_entity.py` (3), `marco_cascade.py` (4), `marco_distinct.py` (5).
**Diagnostics:** `marco_overlap.py`, `marco_crossover.py`, `marco_rarest.py`,
`marco_recover.py` (discrimination structure); `marco_diagnose.py` (glass-box wins/losses);
`marco_explore.py` (white-box signal analysis); `marco_pollution.py` (pollutant profile).
**Rejected:** `marco_cascade2.py` (medium-boost), the proximity/phrase columns of `marco_explore.py`.

*Every rung is reproduced by running its script against the cached pool. The ladder is the
receipt; the diagnostic loop is the product.*
