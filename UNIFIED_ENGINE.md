# The unified prime-lattice retrieval engine — compiled design

Everything this session, collapsed into one architecture. Each layer is tagged with its
**measured status** so the design is honest, not just a vision: ✅ proven on real MARCO,
🟡 partial/needs tuning, ⚪ plausible but untested, ❗ open question that decides the design.

---

## Layer 0 — Prime assignment IS idf (the alphabet)

Assign primes by frequency rank: most common word → 2, then 3, 5, 7… **rarest word → biggest
prime.** So **prime magnitude ≈ rarity ≈ idf**. "The biggest primes" are not an afterthought —
they *are* the discriminative anchors, and "the rarest company narrows" is literally "the
biggest primes carry the most bits." idf falls out of the number, for free.
- ✅ full-vocab word→prime matching = the BM25 floor, **MRR 0.5419** on the 300k pool.

## Layer 1 — Multi-level tokens → primes (the representation)

One unit is simultaneously several primes, all derivable by arithmetic (no separate storage):

| level | → prime | role | status |
|---|---|---|---|
| raw token (char/byte) | small prime | atoms, full coverage, no OOV | ⚪ |
| subword (stem/morpheme) | prime | `diabetic/diabetes → diabet` (variant recall) | 🟡 measuring |
| word | freq-ranked prime | the anchor (precision) | ✅ 0.5419 |
| **composite** (P×Q) | **product, factors back** | phrases/compounds as **free tokens** | ⚪ |

**The composite trick (the memory win):** a phrase = product of its word-primes, a word =
product of its subword-primes. By the Fundamental Theorem of Arithmetic every product factors
**uniquely** — so `blood`×`sugar` is a distinct token that costs *no vocabulary memory*, because
it's just a number you can factor back. Millions of phrase/compound tokens, zero extra storage.
That's the "free tokens to save memory."
- ✅ footprint: sparse prime index projects to **~2.8 GB** for the full 8.8M (under the 4 GB goal).

## Layer 2 — Complex-plane rotation (the geometry) ❗

`insert(prime)` rotates each prime to a coordinate in the 3D complex plane Ψ=(z,ζ) via the
constructive-π rotation. The vision: **related primes land in related coordinates, so when a
query rotates in, the relevant corridors light up by geometric proximity** — no co-occurrence
table needed, the semantics live in the address.

❗ **This is the pivotal open question of the whole design.** What we *measured* built corridors
from **co-occurrence statistics**, not from plane geometry. Whether the lattice's geometry
actually clusters semantically-related terms is **unverified**. Two outcomes, two architectures:
- if plane-distance correlates with semantic-relatedness → the geometry **is** the corridor
  (elegant, compressive, no co-occurrence pass).
- if not → build corridors from co-occurrence (what works today) and use the plane for
  addressing/compression only.
**This is the first thing to test if we commit to the lattice-native route.**

## Layer 3 — Semantic correlations from the biggest primes (the meaning)

Build the corridor from the **biggest primes only** (rarest terms) — they narrow; small primes
(common words) don't. Each big-prime term → its correlated company.
- ✅ corridor **rescues the miss-set 0.0337** where BM25 scored a flat 0.0000 — semantic
  retrieval with no lexical overlap, demonstrated. (idf-gating = "biggest primes" focus.)

## Layer 4 — Retrieval = anchor + corridor + meet (the scoring)

```
score(doc) = ANCHOR            exact big-prime matches, rarity-weighted   ✅ recovers 0.73 hit-set
           + CORRIDOR-FILL     query primes' corridors light up the doc   ✅ rescues miss-set
           × MEET^α            super-linear in # distinct corridors that  🟡 suppresses drift
                               converge on the doc (intersection narrows)    (needs anchor-dominance)
           + SUPERVISED        qrels-grounded bridges sharpen relevance   ✅ +0.0077 calibrated
```

The hard-won rule from the measurements: **the anchor must dominate.** Corridor alone loses
(0.38 < 0.54) because it demotes obvious answers; corridor *on top of* a dominant exact-match
anchor, used only where the anchor is weak, is the path that beats BM25.

## Layer 5 — Learned semantic memory: composites from meets (the "embeddings", but exact)

When 3 corridors intersect (a genuine 3-way meet — a real narrowing, not topical noise), **mint
a new composite** `P_A × P_B × P_C`. That composite *is* the stored semantic relationship — the
symbolic analog of a neural embedding vector, except:

- **Exact, not float** — an integer, no precision loss, no near-collisions.
- **Factorable = unbindable** — factor the composite to recover its constituents. This is the
  *exact unbinding the Γ-mixer could not do* (its recall hole, 17% vs 100% on MQAR). The symbolic
  composite **solves it**: relational memory is exact because factoring is exact. ← the two halves
  of the session unify here.
- **Collision-free** — FTA guarantees every product is unique; the complex-plane rotation (by
  prime order) sends each to a distinct vector — "always different no matter how many turns,"
  because the odd-prime / irrational-π rotation never re-closes. Unbounded relationships, no
  crowding (where float embeddings blur).
- **Continual / append-only** — the lattice grows its semantic memory as it finds meets; no
  retraining, just new composites.

So "save semantic relationships like a neural network" = **binding is multiply (VSA), the meet
mints the bound vector, factoring unbinds it exactly.** Symbolic composites = exact relational
MEMORY; the Γ-mixer = smooth PREDICTION. The composite is the recall the neural net lacked.
- ⚪ minting is mechanically sound (FTA); ❗ whether the minted vectors form a *coherent* space
  rides on the same Layer-2 geometry test.

---

## What's proven vs what decides it

- **Proven today:** prime=idf anchor (0.5419), sparse footprint (~2.8 GB < 4 GB), corridor
  rescues vocabulary-mismatch (miss-set 0.0337), supervised narrowing lifts (+0.0077), the
  Γ-mixer scales as the neural half.
- **The two open questions that decide the ceiling:**
  1. ❗ Does the **complex-plane geometry** encode semantics (geometry → corridors), or must
     corridors come from co-occurrence? (Layer 2 — test first.)
  2. 🟡 Can **anchor-dominant + corridor-fill + meet + supervised** be tuned to beat BM25
     *overall*, not just on the miss-set? (Layer 4 — the integration.)

Everything else is engineering. These two are the science.
