"""
L4 Phrase Composite Nodes — prime product addressing for co-occurring word pairs.

Architecture (from design doc)
--------------------------------
  L1  letters  → letter primes (a=3, b=5, ...)
  L2  subwords → L2 pool primes via PMI (promotes "auto", "phag", "tion", ...)
  L3  words    → product of L2 subword primes (apple = p_ap × p_ple)
  L4  phrases  → product of L3 word primes (apple×phone=67367, apple×pie=68701)
  L5  bridges  → when two phrase composites share enough context → new bridge prime
  L6  meta     → bridges between bridges

FTA guarantee:  p_apple × p_phone  ≠  p_apple × p_pie  (different composites)
                 → different lattice nodes
                 → correct disambiguation without embeddings

Swap meet holds for composites:
  solo(67367) @ n=p_apple  ==  solo(p_apple) @ n=67367  on all 32 wings
  → the phrase node deterministically connects back to each component word

Retrieval (Signal 5)
---------------------
  Query "apple phone" → compute composite 67367
  Docs containing "apple" near "phone" → also computed 67367 during ingest
  Score: IDF("apple") × IDF("phone") × pair_weight for matching composites

  This is O(|query_pairs|) per query (not per doc): precompute query composites,
  look up docs in the index, score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from operator import mul
from functools import reduce
from collections import defaultdict

from aethos_promotion import LatticeTier, PromotionRegistry


# Weight for phrase composite matches relative to single-word BM25
PHRASE_WEIGHT = 0.0   # disabled until more L3 pool primes are promoted (need quality mode ingest)


# ---------------------------------------------------------------------------
# Word prime resolution
# ---------------------------------------------------------------------------

def word_prime(word: str, registry: PromotionRegistry) -> int | None:
    """
    Get the L3 dedicated prime for a word.
    Returns None if the word is intersection-only (no dedicated pool prime).
    """
    key = (LatticeTier.L3_WORD, word.lower())
    tok = registry.promoted.get(key)
    if tok is not None and not tok.intersection_only:
        return tok.prime
    return None


def word_prime_or_intersection(word: str, registry: PromotionRegistry) -> int:
    """
    Get the L3 prime (pool or intersection).
    Always returns a value — pool prime if promoted, intersection prime otherwise.
    """
    from aethos_promotion import intersection_prime
    key = (LatticeTier.L3_WORD, word.lower())
    tok = registry.promoted.get(key)
    if tok is not None:
        return tok.prime
    itok = registry.intersections.get(word.lower())
    if itok is not None:
        return itok.prime
    return intersection_prime(word)


# ---------------------------------------------------------------------------
# Phrase composite computation
# ---------------------------------------------------------------------------

def phrase_composite(prime_a: int, prime_b: int) -> int:
    """
    Product of two word primes = unique L4 phrase composite (FTA).

    apple(667) × phone(101) = 67,367  [apple phone]
    apple(667) × pie(103)   = 68,701  [apple pie]

    These are DIFFERENT composites → different lattice nodes → correct disambiguation.
    The swap meet guarantees: solo(67367)@n=101 == solo(101)@n=67367 on all 32 wings.
    """
    return prime_a * prime_b


def phrase_composite_triple(p1: int, p2: int, p3: int) -> int:
    """Three-way phrase composite: p1 × p2 × p3."""
    return p1 * p2 * p3


# ---------------------------------------------------------------------------
# Phrase composite index
# ---------------------------------------------------------------------------

@dataclass
class PhraseCompositeIndex:
    """
    L4 phrase composite index for retrieval.

    composite → set of doc IDs where this word pair co-occurred.
    word_pair_composites → per-doc word pairs that were indexed.

    Size: O(|unique_composites|) — much smaller than full cross-product because
    only co-occurring word pairs in the corpus are indexed.
    """
    # composite → set of doc_ids
    composite_to_docs: dict[int, set[str]] = field(default_factory=dict)
    # doc_id → frozenset of composites present in that doc
    doc_composites: dict[str, frozenset[int]] = field(default_factory=dict)
    # prime → set of composites containing it (for fast query lookup)
    prime_to_composites: dict[int, set[int]] = field(default_factory=dict)
    # composite → (prime_a, prime_b) — for diagnostics / factorization
    composite_factors: dict[int, tuple[int, int]] = field(default_factory=dict)
    n_composites: int = 0
    n_pairs_indexed: int = 0

    def composites_for_doc(self, doc_id: str) -> frozenset[int]:
        return self.doc_composites.get(doc_id, frozenset())

    def docs_for_composite(self, composite: int) -> set[str]:
        return self.composite_to_docs.get(composite, set())

    def composites_for_prime(self, prime: int) -> set[int]:
        return self.prime_to_composites.get(prime, set())


def build_phrase_composite_index(
    registry: PromotionRegistry,
    doc_tokens: dict[str, frozenset[str]],
    *,
    min_word_len: int = 4,
    min_pair_count: int = 2,
    max_pairs_per_doc: int = 64,
    use_pool_primes_only: bool = False,
) -> PhraseCompositeIndex:
    """
    Build the L4 phrase composite index.

    For each document, for each pair of co-occurring content words that are
    both in the registry and have meaningful primes, compute their phrase
    composite and index it.

    Parameters
    ----------
    min_word_len      : minimum word length to include in pairs
    min_pair_count    : minimum co-occurrence count across corpus to index a pair
    max_pairs_per_doc : cap pairs per document to control index size
    use_pool_primes_only : if True, only index words with dedicated L3 pool primes
                           (more precise but misses intersection-only words)
    """
    idx = PhraseCompositeIndex()

    # Pre-resolve word primes for all words in the vocabulary
    prime_cache: dict[str, int | None] = {}
    def get_prime(w: str) -> int | None:
        if w not in prime_cache:
            if use_pool_primes_only:
                prime_cache[w] = word_prime(w, registry)
            else:
                p = word_prime(w, registry)
                if p is None:
                    p = word_prime_or_intersection(w, registry)
                prime_cache[w] = p
        return prime_cache[w]

    # Count how many docs each word pair co-occurs in (for min_pair_count filter)
    pair_doc_count: dict[tuple[int, int], int] = defaultdict(int)
    doc_pair_map: dict[str, list[tuple[int, int, int]]] = {}  # doc → [(pa, pb, comp)]

    for did, tokens in doc_tokens.items():
        content = [w for w in tokens if len(w) >= min_word_len and not _is_short_common(w)]
        pairs_this_doc: list[tuple[int, int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()

        for w1, w2 in combinations(sorted(content), 2):
            if len(pairs_this_doc) >= max_pairs_per_doc:
                break
            p1 = get_prime(w1)
            p2 = get_prime(w2)
            if p1 is None or p2 is None or p1 == p2:
                continue
            pair = (min(p1, p2), max(p1, p2))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            comp = pair[0] * pair[1]
            pairs_this_doc.append((pair[0], pair[1], comp))
            pair_doc_count[pair] += 1

        doc_pair_map[did] = pairs_this_doc

    # Build the index, filtering by min_pair_count
    total_pairs = 0
    for did, pairs in doc_pair_map.items():
        doc_comp_set: set[int] = set()
        for pa, pb, comp in pairs:
            pair = (pa, pb)
            if pair_doc_count[pair] < min_pair_count:
                continue
            idx.composite_to_docs.setdefault(comp, set()).add(did)
            idx.composite_factors[comp] = pair
            idx.prime_to_composites.setdefault(pa, set()).add(comp)
            idx.prime_to_composites.setdefault(pb, set()).add(comp)
            doc_comp_set.add(comp)
            total_pairs += 1
        if doc_comp_set:
            idx.doc_composites[did] = frozenset(doc_comp_set)

    idx.n_composites = len(idx.composite_to_docs)
    idx.n_pairs_indexed = total_pairs
    return idx


def _is_short_common(w: str) -> bool:
    """Quick filter for very short or common words that make noisy pairs."""
    return len(w) <= 2


# ---------------------------------------------------------------------------
# Query phrase composites
# ---------------------------------------------------------------------------

def query_phrase_composites(
    query_words: list[str],
    phrase_idx: PhraseCompositeIndex,
    registry: PromotionRegistry,
    idf: dict[str, float],
    *,
    min_word_len: int = 4,
    weight: float = PHRASE_WEIGHT,
) -> dict[int, float]:
    """
    Precompute all phrase composite scores for this query — run once per query.
    Returns {composite: effective_score} for composites that exist in the index.
    O(Q²) prime lookups, typically Q=5–10 so ≤45 pairs.
    """
    if weight <= 0:
        return {}
    content = [w for w in query_words if len(w) >= min_word_len and not _is_short_common(w)]
    result: dict[int, float] = {}
    for w1, w2 in combinations(sorted(content), 2):
        p1 = word_prime(w1, registry) or word_prime_or_intersection(w1, registry)
        p2 = word_prime(w2, registry) or word_prime_or_intersection(w2, registry)
        if not p1 or not p2 or p1 == p2:
            continue
        comp = min(p1, p2) * max(p1, p2)
        if comp in phrase_idx.composite_to_docs:
            pair_idf = (idf.get(w1, 1.0) + idf.get(w2, 1.0)) / 2
            result[comp] = pair_idf * weight
    return result


# ---------------------------------------------------------------------------
# Signal 5: phrase composite scoring
# ---------------------------------------------------------------------------

def phrase_composite_score(
    query_words: list[str],
    idf: dict[str, float],
    registry: PromotionRegistry,
    phrase_idx: PhraseCompositeIndex,
    doc_id: str,
    *,
    weight: float = PHRASE_WEIGHT,
    min_word_len: int = 4,
) -> float:
    """
    Score a document based on shared phrase composites with the query.

    For each query word pair (w1, w2) with composite C:
      if C is indexed for this doc:
        score += (IDF(w1) + IDF(w2)) / 2 × weight

    Two documents containing "apple phone" both get composite 67367.
    A document containing "apple pie" gets 68701 — different node, won't match.
    This is the FTA disambiguation: exact co-occurrence identity.

    O(|query_pairs|) per doc — query composites precomputed once.
    """
    doc_comps = phrase_idx.composites_for_doc(doc_id)
    if not doc_comps:
        return 0.0

    content = [w for w in query_words if len(w) >= min_word_len and not _is_short_common(w)]
    if not content:
        return 0.0

    score = 0.0
    for w1, w2 in combinations(sorted(content), 2):
        p1 = word_prime(w1, registry) or word_prime_or_intersection(w1, registry)
        p2 = word_prime(w2, registry) or word_prime_or_intersection(w2, registry)
        if p1 is None or p2 is None or p1 == p2:
            continue
        comp = min(p1, p2) * max(p1, p2)
        if comp in doc_comps:
            pair_idf = (idf.get(w1, 1.0) + idf.get(w2, 1.0)) / 2
            score += pair_idf * weight

    return score


def phrase_composite_score_fast(
    query_phrase_comps: dict[int, float],
    doc_id: str,
    phrase_idx: PhraseCompositeIndex,
) -> float:
    """O(|query_phrase_comps|) per doc via frozenset lookup."""
    doc_comps = phrase_idx.doc_composites.get(doc_id, frozenset())
    return sum(w for c, w in query_phrase_comps.items() if c in doc_comps)


# ---------------------------------------------------------------------------
# Bridge detection (L5) — when two phrase composites share enough context
# ---------------------------------------------------------------------------

@dataclass
class BridgeNode:
    """
    L5 bridge: emerges when two phrase composites share enough context.

    The bridge represents the shared meaning between different phrases
    (e.g., "android phone" ↔ "apple phone" both bridge through "smartphone").
    """
    composite_a: int
    composite_b: int
    bridge_prime: int     # allocated from pool when promoted
    jaccard: float
    shared_docs: frozenset[str]
    is_promoted: bool = False

    def bridge_composite(self) -> int:
        """Product of the two phrase composites = unique bridge identifier."""
        return self.composite_a * self.composite_b


def detect_bridges(
    phrase_idx: PhraseCompositeIndex,
    *,
    jaccard_threshold: float = 0.40,
    min_docs: int = 3,
    max_bridges: int = 200,
) -> list[BridgeNode]:
    """
    Detect L5 bridge nodes: pairs of phrase composites with high doc overlap.

    Two composites with Jaccard(docs_A ∩ docs_B) / Jaccard(docs_A ∪ docs_B) ≥ threshold
    share enough context to warrant a bridge.

    The bridge represents an emergent concept straddling both composites.
    """
    composites = list(phrase_idx.composite_to_docs.keys())
    bridges: list[BridgeNode] = []

    for ca, cb in combinations(composites, 2):
        docs_a = phrase_idx.composite_to_docs[ca]
        docs_b = phrase_idx.composite_to_docs[cb]
        if len(docs_a) < min_docs or len(docs_b) < min_docs:
            continue
        inter = docs_a & docs_b
        if not inter:
            continue
        union = docs_a | docs_b
        jaccard = len(inter) / len(union)
        if jaccard >= jaccard_threshold:
            bridges.append(BridgeNode(
                composite_a=ca,
                composite_b=cb,
                bridge_prime=0,   # assigned if promoted
                jaccard=jaccard,
                shared_docs=frozenset(inter),
            ))
        if len(bridges) >= max_bridges:
            break

    bridges.sort(key=lambda b: -b.jaccard)
    return bridges


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def explain_phrase_composite(
    w1: str, w2: str, registry: PromotionRegistry
) -> str:
    p1 = word_prime(w1, registry)
    p2 = word_prime(w2, registry)
    p1_type = "pool" if p1 else "intersection"
    if not p1:
        p1 = word_prime_or_intersection(w1, registry)
    if not p2:
        p2 = word_prime_or_intersection(w2, registry)
        p2_type = "intersection"
    else:
        p2_type = "pool"
    comp = min(p1, p2) * max(p1, p2) if p1 and p2 else 0
    return (
        f"phrase_composite({w1!r} [{p1_type}={p1}], {w2!r} [{p2_type}={p2}])\n"
        f"  = {p1} × {p2} = {comp}\n"
        f"  Unique by FTA. Swap meet: solo({comp})@n={p1} == solo({p1})@n={comp} (all 32 wings)"
    )
