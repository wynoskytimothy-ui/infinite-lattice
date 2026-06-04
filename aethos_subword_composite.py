"""
Subword composite origins — L2 prime × L2 prime = new unique lattice node.

Design (as described by the user)
-----------------------------------
1. Subwords get promoted to L2 pool primes via PMI/z.
2. When two L2 primes appear in the same word: their product is a UNIQUE composite
   (Fundamental Theorem of Arithmetic guarantees uniqueness).
3. That composite anchors its own LatticeBank32 — a new ORIGIN with 32 wings and
   4 VA branches.  It meets each factor bank at the natural swap witness:
     bank(composite) @ n=p_i  ==  bank(p_i) @ n=composite   on all 32 wings.
4. This creates a 3-way intersection for every word containing both subwords:
     bank(word_prime) × bank(L2_prime_1) × bank(L2_prime_1 × L2_prime_2)
   All three share a common meet point.

Morphological consequence
--------------------------
"autophagy"  chunks to  ("auto", "phag", ...) → L2 primes (p_auto, p_phag)
"autophagic" chunks to  ("auto", "phag", ...) → SAME L2 primes (p_auto, p_phag)
Composite origin = p_auto × p_phag  is SHARED → same lattice node.

This is geometric morphological matching — no stemmer, no heuristics.
Common letter coincidence can't fire because only PMI-promoted subwords get L2 primes.

Rebuild without re-ingest
---------------------------
With fast_ingest=True the per-word subword stats are skipped during ingest.
``rebuild_subword_stats_from_vocab(registry)`` reconstructs them from
``registry.word_counts`` (which IS populated regardless of fast_ingest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations
from operator import mul

from aethos_promotion import LatticeTier, PromotedToken, PromotionRegistry, _chunk_subwords


# ---------------------------------------------------------------------------
# Post-hoc subword stat rebuilder
# ---------------------------------------------------------------------------

def rebuild_subword_stats_from_vocab(registry: PromotionRegistry) -> int:
    """
    Rebuild subword_counts / subword_parent_words / subword_parent_pairs
    from L3-promoted words first, then the remaining vocabulary.

    Two-phase approach:
    Phase 1 — L3 promoted words: these are the words that will eventually use
      L2 pool primes in their parent_primes (via refresh_l3_parent_primes).
      Building subword stats from L3 words first ensures L2 candidates are
      morphemes OF those words, guaranteeing overlap between the promoted L2
      set and the L3 vocabulary.
    Phase 2 — remaining vocabulary: adds breadth for stable PMI, does not
      override the Phase 1 stats.

    Uses vocabulary TYPE frequency (+1 per word type) so ingest-count does not
    inflate subword_counts and dilute the top-PMI selection.

    Returns the number of unique subwords found.
    """
    min_len = registry.subword_min_len
    max_len = registry.subword_max_len

    # Use ONLY L3-promoted words for subword stats.
    # This ensures the top-PMI L2 subwords are morphemes that appear within the
    # same L3 vocabulary, guaranteeing that refresh_l3_parent_primes finds matches
    # and that build_subword_composite_index produces meaningful cross-word composites.
    # Without this restriction, the full-vocabulary PMI selects subwords from the
    # 30k-word tail that happen not to appear in any of the 199 L3 promoted words.
    l3_words: set[str] = {
        key[1] for key in registry.promoted if key[0] == LatticeTier.L3_WORD
    }
    source_words = l3_words if l3_words else dict(registry.word_counts)

    for word in source_words:
        w = word.lower()
        if not w.isalpha() or len(w) < min_len:
            continue
        count = registry.word_counts.get(w, 1)
        seen: set[str] = set()
        for length in range(min_len, min(max_len + 1, len(w) + 1)):
            for i in range(len(w) - length + 1):
                sw = w[i: i + length]
                # Type-count (+1): stable across re-ingests
                registry.subword_counts[sw] = registry.subword_counts.get(sw, 0) + 1
                seen.add(sw)
        for sw in seen:
            registry.subword_parent_words.setdefault(sw, set()).add(w)
            key = (sw, w)
            registry.subword_parent_pairs[key] = (
                registry.subword_parent_pairs.get(key, 0) + count
            )

    return len(registry.subword_counts)


def refresh_l3_parent_primes(registry: PromotionRegistry) -> int:
    """
    After L2 promotion, recompute parent_primes for every L3 word using
    the now-available L2 pool primes.

    Scans ALL character n-grams of length 2–4 (same strategy as
    _word_l2_primes_via_ngrams) so promoted L2 subwords are found
    regardless of where they appear in the word — not just at greedy
    chunk boundaries.  The old greedy chunker returned 0 refreshed words
    because it only tested 4-char-aligned boundaries.

    Returns count of words updated.
    """
    # Build lookup of already-promoted L2 subwords — do NOT allocate new ones
    l2_lookup: dict[str, int] = {
        key[1]: tok.prime
        for key, tok in registry.promoted.items()
        if key[0] == LatticeTier.L2_SUBWORD
    }
    if not l2_lookup:
        return 0

    l2_prime_set = set(l2_lookup.values())
    min_len = registry.subword_min_len   # 2
    max_len = registry.subword_max_len   # 4
    updated = 0

    for key, tok in list(registry.promoted.items()):
        if key[0] != LatticeTier.L3_WORD:
            continue
        word = key[1]
        old_parents = tok.parent_primes

        # Collect every L2 pool prime present anywhere in this word's char sequence.
        found_l2: set[int] = set()
        w = word.lower()
        for length in range(min_len, min(max_len + 1, len(w) + 1)):
            for i in range(len(w) - length + 1):
                sub = w[i: i + length]
                if sub in l2_lookup:
                    found_l2.add(l2_lookup[sub])

        if not found_l2:
            continue  # no L2 pool prime found in this word — nothing to update

        # Build new parent_primes: L2 primes found + original primes not yet included,
        # deduplicated and sorted for determinism.
        combined = found_l2 | set(old_parents)
        t_new = tuple(sorted(combined))

        if t_new == old_parents:
            continue
        if not any(p in l2_prime_set for p in t_new):
            continue  # no L2 pool prime gained — skip update

        registry.promoted[key] = PromotedToken(
            text=tok.text,
            tier=tok.tier,
            prime=tok.prime,
            parent_primes=t_new,
            intersection_only=tok.intersection_only,
        )
        updated += 1
    return updated


def promote_l2_subwords(registry: PromotionRegistry, max_promote: int = 160) -> int:
    """
    After rebuilding subword stats, promote the top-N highest-PMI subwords to L2 primes.

    Caps at ``max_promote`` to stay within the L2 pool budget (~200 slots at 41% of pool).
    Subwords are ranked by (max PMI across parents), so only the most cohesive promote.

    Returns the number of newly promoted L2 subwords.
    """
    before = sum(1 for k in registry.promoted if k[0] == LatticeTier.L2_SUBWORD)

    # Score each candidate by max PMI across its parent words
    candidates: list[tuple[float, str]] = []
    for sw in registry.subword_counts:
        if registry._should_promote_l2(sw):
            score = registry.max_subword_pmi(sw)
            candidates.append((score, sw))

    # Sort descending by PMI — promote the most cohesive first
    candidates.sort(key=lambda x: -x[0])

    for _, sw in candidates[:max_promote]:
        key = (LatticeTier.L2_SUBWORD, sw)
        if key in registry.promoted:
            continue
        try:
            registry._promote(LatticeTier.L2_SUBWORD, sw)
        except RuntimeError:
            break   # pool full — stop gracefully

    after = sum(1 for k in registry.promoted if k[0] == LatticeTier.L2_SUBWORD)
    return after - before


# ---------------------------------------------------------------------------
# Subword composite origin builder
# ---------------------------------------------------------------------------

@dataclass
class SubwordCompositeIndex:
    """
    Maps (subword composite origin → doc_ids containing words with that origin).

    A composite origin is the product of any two L2 pool primes that appear
    together as parent primes of the same word.

    Two words sharing a pairwise L2 composite are morphologically related.
    """
    # composite_origin (p1 × p2) → set of doc_ids
    composite_to_docs: dict[int, set[str]] = field(default_factory=dict)
    # word → frozenset of pairwise composites it participates in
    word_composites: dict[str, frozenset[int]] = field(default_factory=dict)
    # doc_id → frozenset of composite origins present in that doc  (O(1) lookup)
    doc_to_composites: dict[str, frozenset[int]] = field(default_factory=dict)
    # total number of composites
    n_composites: int = 0


def _l2_pool_primes_of(word: str, registry: PromotionRegistry) -> tuple[int, ...]:
    """
    Return the L2 POOL primes from a word's parent_primes — i.e. only primes
    that were actually allocated from the promotion pool, not intersection primes.
    """
    tok = registry.promoted.get((LatticeTier.L3_WORD, word.lower()))
    if tok is None:
        tok = registry.intersections.get(word.lower())
    if tok is None:
        return ()
    # Filter: keep only primes that appear as promoted L2 keys
    l2_keys = {v.prime for k, v in registry.promoted.items() if k[0] == LatticeTier.L2_SUBWORD}
    return tuple(p for p in tok.parent_primes if p in l2_keys)


def _word_l2_primes_via_ngrams(
    word: str,
    l2_lookup: dict[str, int],
    min_len: int = 2,
    max_len: int = 4,
) -> tuple[int, ...]:
    """
    Extract L2 pool primes from a word by scanning all n-grams.

    More robust than chunking: checks every substring of length [min_len, max_len]
    against the promoted L2 lookup — finds all promoted subwords regardless of
    how the greedy chunker would have segmented the word.
    """
    found: set[int] = set()
    w = word.lower()
    for length in range(min_len, min(max_len + 1, len(w) + 1)):
        for i in range(len(w) - length + 1):
            sw = w[i: i + length]
            if sw in l2_lookup:
                found.add(l2_lookup[sw])
    return tuple(sorted(found))


def build_subword_composite_index(
    registry: PromotionRegistry,
    doc_tokens: dict[str, frozenset[str]],
    *,
    max_composites: int | None = None,
) -> SubwordCompositeIndex:
    """
    Build the subword composite origin index using L3 parent_primes.

    For every word in every doc that has an L3 dedicated prime AND has ≥2
    L2 pool primes embedded in its parent_primes (via refresh_l3_parent_primes),
    all pairwise L2 products are composite origins.

    Using L3 parent_primes rather than raw n-gram scanning gives SELECTIVE
    composites: only semantically distinct words (with dedicated pool primes,
    meaning they appear in genuinely different contexts) contribute.  This
    avoids the noise of 500+ composites from intersection-only words.

    Index: composite → set of doc IDs where words with that composite appear.
    """
    idx = SubwordCompositeIndex()
    all_composites: set[int] = set()

    # Build l2_lookup: subword_text → L2_pool_prime  (used to find l2_prime_set)
    l2_lookup: dict[str, int] = {
        key[1]: tok.prime
        for key, tok in registry.promoted.items()
        if key[0] == LatticeTier.L2_SUBWORD
    }
    if not l2_lookup:
        return idx

    l2_prime_set: frozenset[int] = frozenset(l2_lookup.values())

    # Cache: word → frozenset of pairwise L2-composite origins (from parent_primes)
    word_cache: dict[str, frozenset[int]] = {}

    for did, tokens in doc_tokens.items():
        for word in tokens:
            if len(word) < 4:
                continue
            if word not in word_cache:
                # Use L3 parent_primes (refreshed by refresh_l3_parent_primes to include
                # L2 pool primes).  Falls back to n-gram scanning if parent_primes
                # don't include any L2 primes (e.g. word not L3-promoted).
                l2_primes = _l2_pool_primes_of(word, registry)
                if len(l2_primes) < 2:
                    # Fallback: scan n-grams for any L2 match (covers non-L3 words)
                    l2_primes = _word_l2_primes_via_ngrams(word, l2_lookup)
                if len(l2_primes) < 2:
                    word_cache[word] = frozenset()
                    continue
                pairs = frozenset(a * b for a, b in combinations(l2_primes, 2))
                word_cache[word] = pairs
                all_composites |= pairs
            for composite in word_cache[word]:
                idx.composite_to_docs.setdefault(composite, set()).add(did)

    # If max_composites set, keep only the most-covered composites (appear in most docs)
    if max_composites is not None and len(idx.composite_to_docs) > max_composites:
        top = sorted(idx.composite_to_docs.items(), key=lambda x: -len(x[1]))[:max_composites]
        keep = {c for c, _ in top}
        idx.composite_to_docs = {c: d for c, d in idx.composite_to_docs.items() if c in keep}
        idx.word_composites = {w: frozenset(c for c in comps if c in keep)
                               for w, comps in word_cache.items()}
        all_composites &= keep

    idx.word_composites = idx.word_composites if max_composites else word_cache
    idx.n_composites = len(all_composites)

    # Build reverse mapping: doc_id → frozenset of composites in that doc
    doc_comp_map: dict[str, set[int]] = {}
    for comp, docs in idx.composite_to_docs.items():
        for did in docs:
            doc_comp_map.setdefault(did, set()).add(comp)
    idx.doc_to_composites = {did: frozenset(comps) for did, comps in doc_comp_map.items()}

    return idx


# ---------------------------------------------------------------------------
# Scoring Signal 4 (corrected) — subword composite meet
# ---------------------------------------------------------------------------

def subword_composite_score(
    query_words: list[str],
    idf: dict[str, float],
    registry: PromotionRegistry,
    composite_index: SubwordCompositeIndex,
    doc_id: str,
    *,
    weight: float = 0.4,
    word_cache: dict[str, frozenset[int]] | None = None,
) -> float:
    """
    Score morphological overlap via shared L2 subword composite origins.

    For each query word qw:
      q_composites = pairwise L2 products of qw's promoted parent primes
      doc_composites = composites that appear in doc_id
      ratio = |q_composites ∩ doc_composites| / max(|q_composites|, 1)
      For each matching composite pair, also consider prime_factor_similarity
      when query and doc composites differ in magnitude but share factors.
      score += IDF(qw) × ratio × weight

    Fires ONLY when words share PMI-promoted subword pairs — not raw letter overlap.
    """
    if word_cache is None:
        word_cache = composite_index.word_composites

    # O(1) lookup via precomputed reverse mapping
    doc_composites = composite_index.doc_to_composites.get(doc_id)
    if not doc_composites:
        return 0.0

    score = 0.0
    for qw in query_words:
        if len(qw) < 4:
            continue
        if qw not in word_cache:
            l2_lookup = {
                key[1]: tok.prime
                for key, tok in registry.promoted.items()
                if key[0] == LatticeTier.L2_SUBWORD
            }
            l2 = _word_l2_primes_via_ngrams(qw, l2_lookup)
            if len(l2) < 2:
                word_cache[qw] = frozenset()
                continue
            word_cache[qw] = frozenset(a * b for a, b in combinations(l2, 2))

        q_comps = word_cache[qw]
        if not q_comps:
            continue

        from core.phi_lattice import prime_factor_similarity

        exact_ratio = len(q_comps & doc_composites) / max(len(q_comps), 1)
        jaccard_best = 0.0
        for qc in q_comps:
            for dc in doc_composites:
                jaccard_best = max(jaccard_best, prime_factor_similarity(qc, dc))
        ratio = max(exact_ratio, jaccard_best)
        if ratio > 0:
            score += idf.get(qw, 1.0) * ratio * weight

    return score


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def explain_composite_origin(word: str, registry: PromotionRegistry) -> str:
    """Show the L2 subword decomposition and pairwise composite origins for a word."""
    l2 = _l2_pool_primes_of(word, registry)
    tok = registry.promoted.get((LatticeTier.L3_WORD, word.lower()))
    lines = [f"Word: {word!r}"]
    if tok:
        lines.append(f"  L3 prime:      {tok.prime}")
        lines.append(f"  parent_primes: {tok.parent_primes}")
    if l2:
        lines.append(f"  L2 pool primes: {l2}")
        pairs = [(a * b, a, b) for a, b in combinations(sorted(l2), 2)]
        for c, a, b in pairs:
            lines.append(f"    composite origin {a}×{b} = {c}  (new lattice bank)")
    else:
        lines.append("  (no L2 pool primes — fast_ingest=True or below PMI threshold)")
    return "\n".join(lines)
