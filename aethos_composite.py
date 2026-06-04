"""
Composite anchor nodes — prime products as unique lattice addresses.

Core insight (Fundamental Theorem of Arithmetic):
  Every finite set of distinct primes has a unique product.
  → Any word set {w₁,…,wₙ} with primes {p₁,…,pₙ} maps to one unique composite C.
  → C anchors its own LatticeBank32 with the same 32-wing, 4-branch structure.
  → The swap meet (a@n=b == b@n=a) works for ANY two distinct integers,
    so C meets each factor pᵢ at the natural witness (n_C=pᵢ, n_pᵢ=C).

Morphological similarity via letter composites
----------------------------------------------
Two words w₁, w₂ sharing letter-prime factors have gcd(letter_composite(w₁), letter_composite(w₂)) > 1.
The GCD ratio  gcd(c₁, c₂) / c₁  ∈ [0,1]  measures the fraction of w₁'s
letter-prime structure present in w₂ — a deterministic morphological proximity
that requires no stemmer, no learned model, and no heuristic suffix rules.

Example:
  "autophagy"  letter set {a,u,t,o,p,h,g,y}  → product C₁
  "autophagic" letter set {a,u,t,o,p,h,g,i,c} → product C₂
  gcd(C₁,C₂) = product of {a,u,t,o,p,h,g}  (shared letters)
  ratio = gcd/C₁ = 7/8 = 0.875  → very high morphological similarity

Retrieval signal (Signal 4 — Morphological meet)
-------------------------------------------------
For each query word qw and each doc hub word hw:
  ratio = gcd(letter_composite(qw), letter_composite(hw)) / letter_composite(qw)
  if ratio >= MORPH_THRESHOLD:
      score += IDF(qw) × ratio × MORPH_WEIGHT

This is O(Q × K) per doc — same as Signals 2–3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from operator import mul

from aethos_words import word_sorted_chain


# Minimum shared-letter fraction to count as a morphological match.
# 0.75 keeps "autophagy"↔"autophagic" (0.875) while dropping noise.
MORPH_THRESHOLD = 0.75
# Minimum word length for both query word and hub word to apply Signal 4.
# Short words (≤5 chars) share letters with too many unrelated words.
MORPH_MIN_LEN = 6
# Weight relative to IDF for a morphological (non-exact) match
MORPH_WEIGHT = 0.0   # disabled — letter GCD too coarse; replaced by L2 subword meet


# ---------------------------------------------------------------------------
# Letter composite — product of distinct letter primes
# ---------------------------------------------------------------------------

def letter_prime_set(word: str) -> frozenset[int]:
    """Set of distinct letter primes for ``word``."""
    return frozenset(word_sorted_chain(word))


def letter_composite(word: str) -> int:
    """
    Product of the word's DISTINCT letter primes (sorted chain).

    By FTA, this integer is unique to the letter multiset of ``word``.
    Stored in HubEntry for composite-based routing (Step 2c MeetIndex).
    """
    chain = word_sorted_chain(word)
    if not chain:
        return 1
    return reduce(mul, chain, 1)


def prime_factor_jaccard(c1: int, c2: int) -> float:
    """Scale-invariant Jaccard on prime factors of composites (φ-lattice)."""
    from core.phi_lattice import prime_factor_similarity

    return prime_factor_similarity(c1, c2)


def letter_composite_gcd_ratio(w1: str, w2: str) -> float:
    """
    Fraction of w1's distinct letter primes present in w2.

    Returns |primes(w1) ∩ primes(w2)| / |primes(w1)|  ∈ [0, 1].

    Uses prime COUNTING (not product ratio) so a single non-shared letter
    doesn't dominate:  "autophagy"/"autophagic" → 7/8 = 0.875 ✓
    """
    s1 = frozenset(word_sorted_chain(w1))
    s2 = frozenset(word_sorted_chain(w2))
    if not s1:
        return 0.0
    return len(s1 & s2) / len(s1)


# ---------------------------------------------------------------------------
# Composite cluster node — product of word primes for a co-occurring group
# ---------------------------------------------------------------------------

def cluster_composite(word_primes: tuple[int, ...]) -> int:
    """
    Product of all word primes in a co-occurring cluster.

    By FTA, this composite is globally unique to this exact set of primes.
    The composite anchors its own LatticeBank32 — same 32 wings, 4 branches.
    Each word prime pᵢ is a factor, so the bank meets bank(pᵢ) at the
    swap witness (n_composite=pᵢ, n_pᵢ=composite) on all 32 wings.
    """
    if not word_primes:
        return 1
    return reduce(mul, word_primes, 1)


def composite_factors(composite: int, prime_vocab: set[int]) -> list[int]:
    """
    Which word primes from prime_vocab divide this composite?

    Uses trial division against the known vocab — fast when |prime_vocab| is small.
    Returns the factors in ascending order.
    """
    return sorted(p for p in prime_vocab if composite % p == 0)


# ---------------------------------------------------------------------------
# Precomputed letter composite cache for fast scoring
# ---------------------------------------------------------------------------

@dataclass
class CompositeIndex:
    """
    Per-document letter prime SET table — built once after hub signatures.

    Stores prime sets (not products) for fast counting-based ratio computation.
    word_cache maps word → frozenset[int] of its letter primes.
    """
    # doc_id → list of (hub_word, prime_set) pairs
    doc_hub_composites: dict[str, list[tuple[str, frozenset[int]]]] = field(default_factory=dict)
    # word → frozenset of letter primes (shared cache across all queries)
    word_cache: dict[str, frozenset[int]] = field(default_factory=dict)

    def get_prime_set(self, word: str) -> frozenset[int]:
        if word not in self.word_cache:
            self.word_cache[word] = frozenset(word_sorted_chain(word))
        return self.word_cache[word]


def build_composite_index(hub_sigs: dict) -> CompositeIndex:
    """
    Build letter prime sets for all hub words.  O(docs × K × word_len).
    Called once after hub signature build.
    """
    idx = CompositeIndex()
    for did, sig in hub_sigs.items():
        pairs: list[tuple[str, frozenset[int]]] = []
        for word in sig.hub_words():
            if len(word) < MORPH_MIN_LEN:
                continue
            ps = idx.get_prime_set(word)
            if ps:
                pairs.append((word, ps))
        idx.doc_hub_composites[did] = pairs
    return idx


# ---------------------------------------------------------------------------
# Morphological meet score — Signal 4
# ---------------------------------------------------------------------------

def morph_meet_score(
    query_words: list[str],
    idf: dict[str, float],
    doc_composites: list[tuple[str, frozenset[int]]],
    word_cache: dict[str, frozenset[int]],
    *,
    threshold: float = MORPH_THRESHOLD,
    weight: float = MORPH_WEIGHT,
) -> float:
    """
    Score morphological overlap between query words and doc hub prime sets.

    For each query word qw and hub (hw, ps_hw):
      ratio = |ps_qw ∩ ps_hw| / |ps_qw|   (shared letter prime fraction)
      if ratio >= threshold and qw != hw:
          score += IDF(qw) × ratio × weight

    "autophagy" vs "autophagic" → ratio = 7/8 = 0.875 ✓
    O(|query_words| × |doc_composites|) — typically 10 × 12 = 120 ops per doc.
    """
    if not doc_composites:
        return 0.0

    score = 0.0
    for qw in query_words:
        if len(qw) < MORPH_MIN_LEN:
            continue
        if qw not in word_cache:
            word_cache[qw] = frozenset(word_sorted_chain(qw))
        ps_qw = word_cache[qw]
        if not ps_qw:
            continue
        n_qw = len(ps_qw)
        qw_idf = idf.get(qw, 1.0)

        for hw, ps_hw in doc_composites:
            if hw == qw or len(hw) < MORPH_MIN_LEN:
                continue   # exact match or too short
            shared = len(ps_qw & ps_hw)
            ratio = shared / n_qw
            if ratio >= threshold:
                score += qw_idf * ratio * weight

    return score


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def explain_composite_similarity(w1: str, w2: str) -> str:
    """Human-readable explanation of the composite similarity between two words."""
    from aethos_words import word_sorted_chain
    c1 = word_sorted_chain(w1)
    c2 = set(word_sorted_chain(w2))
    shared = [p for p in c1 if p in c2]
    ratio = len(shared) / max(len(c1), 1)
    return (
        f"composite_similarity({w1!r}, {w2!r})\n"
        f"  {w1} primes:  {c1}\n"
        f"  {w2} primes:  {tuple(sorted(c2))}\n"
        f"  shared:       {tuple(shared)}\n"
        f"  GCD ratio:    {ratio:.3f}  ({'MATCH' if ratio >= MORPH_THRESHOLD else 'below threshold'})"
    )
