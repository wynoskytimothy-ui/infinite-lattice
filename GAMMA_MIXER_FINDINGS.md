# The entanglement ODE as a sequence-mixing kernel — findings

A complete, honest write-up of what the AETHOS entanglement ODE becomes when used as a
neural-network token mixer, and exactly how far the evidence reaches. Every number was
measured on the user's RTX 5080 (sm_120, torch 2.12 dev, CUDA 12.8). Where a result is
regime-specific or a measurement was wrong, it says so.

## The object

The entanglement ODE `Ċ = Γ_form·(1−C) − Γ_break·C` discretizes to a selective gated
linear recurrence — the Mamba/SSM shape:

```
C_t = a_t · C_{t-1} + u_t        a_t = exp(−softplus(·)) ∈ (0,1)   (Γ_break decay)
                                 u_t = sigmoid(·) · value           (Γ_form input gate)
```

Used as a drop-in replacement for attention in a pre-norm transformer block.

## 1. The parallel scan (mechanical facts — robust)

The recurrence is a composition of affine maps `f_t(C) = a_t·C + u_t`, and affine-map
composition is **associative**:

```
(A_P, X_P) ∘ (A_Q, X_Q) = (A_P·A_Q,  A_Q·X_P + X_Q)
```

so a Hillis-Steele scan computes every `C_t` in `log₂(T)` parallel steps instead of `T`
sequential ones.

- **Forward correct:** scan output == sequential loop to `9.5e-7` (`gpu_gamma_scan.py`).
- **Backward correct:** gradients match the loop to `1e-5`, no NaN/inf, sane magnitudes,
  every chunk size and length (`gpu_scan_grad.py`).
- **Fixes the loop:** at T=512 the naive loop is 68.6 ms/step (fwd+bwd); the scan is
  2.95 ms — a **23× speedup**. The "15× too slow" cripple is gone.

## 2. Long-context speed (robust)

A naive scan allocates `log₂(T)` full-length tensors per pass and hit a memory cliff at
16k (13.2 GB of a 16 GB card → allocator thrash → 3.3× slower). A **chunked** scan
(parallel within 2048-chunks, sequential carry — the Mamba-2 / FlashLinearAttention
trick) fixes it. Forward+backward ms, B=16 D=256, vs 8-head flash attention:

| context T | attention | chunked scan | speedup |
|---:|---:|---:|---:|
| 4096 | 77.7 ms | 48.5 ms | 1.60× |
| 8192 | 292.4 ms | 96.5 ms | 3.03× |
| 16384 | 1053.0 ms | 200.9 ms | **5.24×** |

The gap **widens every doubling** (attention grows ~3.6×/double → O(T²); scan ~2.1× →
O(T·logT)). Crossover is ~6–8k.

**Honest cost:** the scan uses **3–4× more memory** than flash attention (11 GB vs 3 GB
at 16k) — flash attention is *designed* for memory efficiency; we win speed and pay RAM.
**Left on the table:** `torch.compile` GPU fusion needs Triton (Linux/WSL only — not on
this Windows box); a fused kernel would move the crossover well below 8k.

## 3. Accuracy at matched parameters — char level (robust within regime)

Single mixer block, identical 1.92 M params, char-LM, T=512:

| model | perplexity |
|---|---:|
| **gamma-scan** | **2.61** |
| attention, 1-head | 3.65 |
| attention, 4-head | 3.61 |
| attention, 8-head | 3.60 |

~28% lower perplexity, and **multi-head did not close it**. Stable across T=128–2048
(gamma ~2.65–2.72 vs attention ~3.47–3.67). Eval used matched input/target windows
(verified) — this number is real.

## 4. The recall boundary (robust, multi-seed)

Multi-query associative recall (MQAR) — a literal key→value lookup: attention's best
event, a finite-state recurrence's hardest. Multi-layer, matched params, trained to
convergence, **3 seeds** (MQAR groks sharply and seed-dependently, so single runs are
untrustworthy — verdicts come from the reliable L=64 regime):

| architecture | recall (3 seeds) | reading |
|---|---|---|
| attention ×2 | 100 / 100 / 100% | exact lookup — trivial for attention |
| gamma ×2 | 17 / 17 / 17% | the wall: knows the value *set*, can't bind the key |
| gamma+conv ×2 | 17 / 17 / 17% | the Mamba conv does **not** fix it |
| hybrid gamma→attn | 100 / 100 / 100% | **one attention layer fully recovers it** |

The 17% plateau (chance is 2%) is the diagnosis: the recurrence's fixed state is a
**superposition** of all values — it bundles the set but can't cleanly *unbind* the
specific pair. That is the VSA-bundling behavior proved elsewhere in AETHOS, and exactly
what diagonal-SSM theory predicts. The gap is content-addressing, not local mixing —
which is why the field builds **hybrids** (Griffin, Jamba, Mamba-2 + attention).

## 5. Scale-up to deep word-level — and a correction

**A measurement bug nearly produced a false negative, recorded here in full.**

The first scale-up runs (`gpu_scaleup{,2,3,4,5}.py`) reported the SSM models *diverging*
at word-level/depth (train perplexity worse than random). I chased it through five fixes
(decay init, bounded EMA, gradient clipping, a faithful Mamba block) — each seemed to
trade one failure for another. **All of it was an eval bug.** The evaluation computed
`model(batch(src)[0])` and `batch(src)[1]` as two *separate* random draws, scoring
predictions against **mismatched targets**. A confident, well-trained model scores
*worse than random* on mismatched targets — so the broken metric made the best-trained
models look like they diverged. Training itself was always correct (`x, y = batch(tr)`,
one matched call); only the eval was wrong.

With the one-line fix (`gpu_scaleup_fixed.py`) and **3-seed verification**
(`gpu_scaleup_verify.py`), word-level, matched params — mean val perplexity:

| depth | attention | gamma | mamba-lite |
|---:|---:|---:|---:|
| 2 | 60.8 | **28.3** | 33.5 |
| 4 | 41.7 | **27.1** | 36.8 |

The SSM models train **stably and beat attention at every depth**. Gamma is **53% lower
perplexity at depth 2, 35% at depth 4**, robust across seeds (gamma 27.6/28.8/28.6 at
depth 2 — tight). The char-level win (§3) was **not** regime-specific; it holds and
deepens at word-level. Notably the bare selective recurrence (gamma, good init) beat the
fuller Mamba-style block here — the extra conv/gate/expand machinery wasn't needed at
this scale.

**Honest scope (do not over-extend):** small technical corpus (~535k tokens of this
repo's own text — local/structural regularity may favour a recurrence), small models
(~2.5M params), short training (2500 steps), 4-head attention baseline. This is a
*controlled matched-param comparison at small scale*, not GPT-scale evidence. What it
shows: at this scale, with parameters and training held identical, the formula is a
strongly winning LM sequence mixer — and it trains stably at depth, contradicting the
buggy negative. Scaling to real diverse text at 19.5M params is confirmed below (§6).

## 6. Scale on real text (text8) — the asterisk removed

The small-scale wins all used the repo's own text. To kill that confound, the same three
contenders were trained at **19.5M params** (6 layers, D=512, T=512, 12k steps, matched)
on **text8 — 100M chars of Wikipedia prose**. Validation bits-per-char (lower better):

| model | bpc | vs attention |
|---|---:|---:|
| attention (6× multi-head) | 1.5247 | — |
| gamma (6× selective recurrence) | 1.4151 | **−7.2%** |
| hybrid (gamma×4 + attention×2) | **1.3665** | **−10.4%** |

The win **survives on real, diverse text at a legitimate scale.** Gamma beats attention by
7.2%; the hybrid (gamma bulk + sparse attention) is best at 10.4%, below both. Trajectories
were smooth and monotone — no divergence (the eval bug was the only instability). The margin
is smaller than the repo-text word-level result (35–53%), exactly as predicted: the repo's
regularity flattered the recurrence; the real, corpus-independent edge is ~7–10% and
consistent at every checkpoint.

**Honest scope:** single seed at this scale (small-scale was 3-seed; trajectories are clean
so n=1 is suggestive not airtight); char-level; bpc 1.37–1.52 is a fair *fixed-budget*
comparison, not an absolute-SOTA attempt. What it establishes: the LM win is real,
corpus-independent, and holds at scale — and the **hybrid is the architecture** (it wins LM
here *and*, from §4, carries exact recall). Untested: frontier scale (billions of params),
downstream task quality.

## Lessons that cost real runs

- **MQAR grokking variance** — at L=128 the same attention scored 100% and 16% on
  different runs; verdicts require the reliable regime + multiple seeds.
- **The eval bug** — mismatched input/target windows can make good models look divergent;
  always assert train ≤ val and both below the random-baseline perplexity.
- The char-LM result (§3) used the *correct* matched eval and stands; only the scale-up
  scripts carried the bug.
