# AETHOS `core/` foundation

Micro-step stack: **L1 characters → L2 subwords → bridge → production registry**.

## Prime namespace

| Band | Source | Aligns with |
|------|--------|-------------|
| L1 letters a–z | `core.primes.LETTER_PRIMES` (first 26 odd primes) | `aethos_words.LETTER_PRIMES` |
| L2 subwords | `PrimePool` tier `L2` (~41% of `PROMOTION_POOL`) | `aethos_pool_tiers` / `PromotionRegistry._alloc_prime` |
| L3 words | `PrimePool` tier `L3` | Same pool bands |

**Rule:** Production ingest uses `aethos_words.letter_to_prime` and `PromotionRegistry._promote`. The core package defines the same L1 table and PMI rules; `bridge_registry.sync_l2_to_registry` applies core **eligibility** and lets the registry **allocate** pool primes.

## Modules

- `primes.py` — odd-prime chains, `PrimePool`, FTA helpers
- `l1_characters.py` — `char_prime`, `intersection_prime` (letter sum)
- `l2_subwords.py` — `SubwordStats`, PMI/z, `SubwordPromoter`, `decompose`, `shared_l2_factors`
- `phi_lattice.py` — VA1–VA4, 32 wings, Z plateau, `swap_meet`, `prime_factor_similarity`
- `bridge_registry.py` — sync L2 into `PromotionRegistry`
- `bridge_library.py` — adapter for `aethos_library.PrimeAssigner` (when pasted)
- `learning_engine.py` — bad-correlation queue, factor analogy, distilled registry brain, BEIR FP hooks

## Tests

```bash
python -m pytest tests/ -q
```

## BEIR / retrieval

Use `fast_ingest=False` or multi-pass Pass 1 so subword stats populate before L2 promotion. See `eval_beir.py` comments and `aethos_iterative.run_pass1(use_core_l2=True)`.

## Verification (SMALL_CORPUS)

| Path | L2 promoted |
|------|-------------|
| `promote_l2_subwords` (legacy) | 38 |
| `run_core_l2_pass` (core bridge) | 38 |

Core eligibility matches production on the smoke corpus. SciFact NDCG A/B: run `eval_beir.py` with `build_multi_pass(..., use_core_l2=True)` when ready (full eval is slow).
