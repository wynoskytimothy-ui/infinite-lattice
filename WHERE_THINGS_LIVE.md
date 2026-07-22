# Where things live

> **This push** — branch `aethos13/shadow-rescue` · code commit `a0638fc` · 2026-07-21
> Repo: https://github.com/wynoskytimothy-ui/infinite-lattice
> Branch tree: https://github.com/wynoskytimothy-ui/infinite-lattice/tree/aethos13/shadow-rescue
> **Update this file on every push. Never leave session work only in chat.**

## This push (`aethos13/shadow-rescue`)

The best zero-shot RAG serve path, moved off local disk:
**bake EdgeRAG bridges offline → serve lexical fast → Auto-Arm Shadow only on pool miss → append-only rescue.**

| Face | Path | Status |
|------|------|--------|
| Serve + Auto-Arm gate (pool-fill ARM) | [`edgerag_serve.py`](https://github.com/wynoskytimothy-ui/infinite-lattice/blob/aethos13/shadow-rescue/edgerag_serve.py) | this push |
| Shadow Rescue (append-only recall net) | [`aethos13_shadow_rescue.py`](https://github.com/wynoskytimothy-ui/infinite-lattice/blob/aethos13/shadow-rescue/aethos13_shadow_rescue.py) | this push |
| Phase 5 (shadow + anchor-gravity) | [`aethos13_phase5.py`](https://github.com/wynoskytimothy-ui/infinite-lattice/blob/aethos13/shadow-rescue/aethos13_phase5.py) | this push |
| AETHOS-13 substrate | `aethos13_{lattice,engine,phase3,phase4,glassbox}.py` | this push |
| Gate probe + ablation | `_gate_probe.py`, `_gate_ablation.py` | this push |
| Evidence JSON | `_edgerag_serve_*.json`, `_shadow_rescue_*.json`, `_phase5_real_*.json` + run logs | this push |

**Run:**
```
python edgerag_serve.py all 200 100        # auto-arm serve, invariants asserted (real BEIR)
python aethos13_shadow_rescue.py scifact    # append-only shadow rescue, oracle + deployable triggers
python aethos13_phase5.py                    # constructed corpus: ablation ladder + controls
python aethos13_phase5.py real nfcorpus      # Phase 5 generalization (the honest net-negative)
```

**Measured (real BEIR, invariants asserted):** gate dormant on aligned corpora (scifact/fiqa, 0 % fire),
arms on nfcorpus (~45 %) to bridge zero-overlap gold BM25 cannot reach at any depth — **+0.026 recall,
P@1 delta 0.0000**.

## Product branches (do NOT touch)

| Face | Branch | Status |
|------|--------|--------|
| EdgeRAG product 1–5 | `edgerag/1-nogpu-engine` … `edgerag/5-api-server` | on origin — **untouched** |

## Sister repo — geometry / scale / partner (`aethos_master`)

Local Cursor repo at `C:\Users\wynos\aethos_master` (no cloud remote; local git only).

| Face | Path |
|------|------|
| **Champion Serve** (unified governed path, measured) | `docs/corpus/CHAMPION_SERVE.md` · `src/aethos_master/aethos/champion_serve.py` |
| Zero-shot strategy | `docs/corpus/ZERO_SHOT_BAKE_SERVE.md` |
| 1M tip RAM / escalate-shard | `docs/corpus/ESCALATE_SHARD.md` |
| Glass Box (30 tests) | `docs/corpus/GLASS_BOX.md` |
| Partner brief (Andrea) | `docs/corpus/ANDREA_PARTNER_BRIEF.md` |
| Auto-Arm (smoke, Phase-5 Top-1) | `src/aethos_master/aethos/auto_arming_gate.py` |

## How they connect

**Bake Offline (EdgeRAG-2) → Serve fast → Auto-Arm on pool miss → Shadow Rescue append-only.**
Engine laws (Door ≠ Overpass, tip ≠ Door O(1)) live in `aethos_master`; the BEIR customer serve lives here.
The pool-fill ARM gate signal is shared; the Phase-5 **Top-1 smoke** (constructed corpus) and the BEIR
**append-only Shadow** are kept as *separate faces* — the smoke shows the mechanism, the append-only path
is the honest, safe production form.
