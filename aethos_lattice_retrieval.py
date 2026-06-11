"""
aethos_lattice_retrieval.py - retrieval via the RecursiveLattice.

The first real wire-up of the recursive-lattice machinery into a retriever.

Ingest:
  tokens get base primes (rarer token -> higher prime, so |z|^2 acts as IDF)
  each doc becomes a chain of its token primes
  the chain is promoted to a doc-prime; doc-prime is a parent of every token

Query:
  tokens -> their primes -> lattice.walk_up(prime) returns containing docs
  intersection across query tokens gives candidate docs (without inverted index)
  score = |z|^2 of wing_transform(VA1, doc_chain, n=shared_anchor, wing=1)

Footprint:
  per doc: prime ID + chain pointer + parent links
  no stored coordinates - Psi is computed at query time via wing_transform
  natural cascade-free property: adding a doc doesn't move existing entries
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aethos_complex_plane import wing_transform
from aethos_lattice import BranchKind, LatticeId, lattice_id_parts
from aethos_recursive_lattice import RecursiveLattice, RecursiveNode
from aethos_symbol_subjects import (
    MASTER_CHAMBER,
    DATASET_SUBJECTS,
    infer_doc_subjects,
    vote_query_chambers,
    subjects_for_dataset,
)
from aethos_words import letter_to_prime, LETTER_PRIMES
from core.primes import chain_primes


# 4-wing consensus: one wing per VA branch (matches aethos_hub_signature.CONSENSUS_WINGS).
# A genuine meet fires on ALL chambers; coincidental single-wing matches fire on fewer.
CONSENSUS_LATTICE_IDS: tuple[int, ...] = (1, 9, 17, 25)


_TOKEN_RE = re.compile(r"[a-z]+")

# Conservative stopword list; tune for domain
STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "by", "with", "as", "be", "been",
    "has", "have", "had", "do", "does", "did", "this", "that", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they",
    "from", "into", "about", "if", "then", "than", "so", "not", "no",
})


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in STOPWORDS and len(t) > 1]


# =====================================================================
# Lattice retriever
# =====================================================================

class LatticeRetriever:
    """RAG retrieval keyed entirely on the RecursiveLattice."""

    # Cap each doc's lattice chain to its top-K rarest token primes.
    # Common tokens (low primes) contribute little IDF; dropping them shrinks
    # both the doc node footprint AND the inverted index (token parents).
    MAX_CHAIN_PER_DOC: int = 1024     # effectively unlimited (avg ~95)
    # Tokens appearing in more than MAX_DF_RATIO of docs are skipped at ingest.
    MAX_DF_RATIO: float = 1.0          # no filter on common tokens
    # Drop tokens appearing in fewer than MIN_DF docs (singletons, typos)
    MIN_DF: int = 1

    # L2 subword config (PMI-style morpheme mining)
    L2_MIN_LEN: int = 3
    L2_MAX_LEN: int = 4
    L2_TOP_K: int = 800        # number of L2 subwords to promote per corpus
    L2_MIN_PARENT_WORDS: int = 5  # subword must appear in >= N distinct words

    # Plan B: compound (bigram) promotion.
    # When two adjacent words co-occur with high PMI, the pair becomes its own
    # composite "compound prime" in the lattice. Effects:
    #   - High-IDF compound primes give heavy weight to phrase matches like
    #     "stem cell" or "breast cancer" or "neural network"
    #   - Common flood words like "cell" or "gene" get retrieved only when
    #     they appear as parts of meaningful compounds in queries
    #   - Query bigrams expand the routing without storing per-pair tables
    # Tuning history (SciFact 5183/300):
    #   (MIN_COOC=3, MIN_PMI=1.5, TOP_K=3000) -> 0.6713  -0.9 vs baseline
    #   (MIN_COOC=5, MIN_PMI=3.0, TOP_K=1500) -> 0.6736  -0.7 vs baseline
    # Compounds add candidate noise without enough precision gain. Kept the
    # mechanism in place; set TOP_K=0 disables it cleanly. Plan C (PMI graph
    # bridges) attacks the audit's actual leak (vocabulary mismatch).
    COMPOUND_MIN_COOC: int = 5
    COMPOUND_MIN_PMI: float = 3.0
    COMPOUND_TOP_K: int = 0           # disabled until per-query gating wired

    # Plan C: PMI semantic bridge graph.
    # Word-pair PMI across the WHOLE doc (not just adjacent) finds semantic
    # neighbors. Then at query time, for each query word, look up top-K
    # high-PMI partners. Boost docs containing those partners by a small
    # multiplicative factor only when MULTIPLE query words agree on the same
    # partner (triangulation - reduces noise).
    PMI_MIN_COOC: int = 4       # word pair must co-occur in >= N docs
    PMI_MIN_VAL: float = 3.0    # log PMI threshold (rarer = stronger)
    PMI_TOP_PER_WORD: int = 10  # keep top-K partners per word
    BRIDGE_LAMBDA: float = 0.04 # smaller boost - noise control
    BRIDGE_MIN_TRIANGULATION: int = 1  # any strong PMI partner counts

    # Subject chamber semantic boost (the 32-chamber parallel classifier system).
    # Each doc and query gets routed to 1-3 of 31 semantic subjects via
    # aethos_symbol_subjects (physics, biology, medicine, linguistics, etc.).
    # Docs whose subjects overlap with the query's subjects get a multiplicative
    # boost. This is the lattice's chamber structure functioning as a parallel
    # semantic feature space, not just an indexing structure.
    LAMBDA_SUBJECT: float = 0.15   # multiplier per matched chamber
    SUBJECT_MAX_PER_DOC: int = 3
    SUBJECT_MAX_PER_QUERY: int = 3

    # Learned per-chamber CategoryVector (Plan A from earlier menu).
    # For each of the 31 subjects, learn from corpus which word primes
    # have high log P(word|chamber) / P(word). At query time, both query
    # and doc get a "soft chamber distribution" via dot product against
    # the learned vocabularies. Cosine similarity of the two distributions
    # adds a multiplicative boost. This captures within-subject topic
    # structure that the hand-picked keyword vote alone misses.
    LAMBDA_CHAMBER_SOFT: float = 0.0   # disabled: chamber dist sim was redundant with existing signals on SciFact
    CHAMBER_VOCAB_MIN_AFFINITY: float = 0.5   # only keep word -> chamber affinities above this
    CHAMBER_VOCAB_MIN_DF: int = 3             # word must be in >= N docs to learn
    CHAMBER_VOCAB_TOP_PER_CHAMBER: int = 400  # keep top-K most diagnostic words per chamber

    # Plan A position encoding via EXPONENTS (not multipliers).
    # Mathematically: by FTA, two prime factorizations differ when their
    # prime exponents differ. So we encode chunk position as the EXPONENT
    # raised on that chunk's prime:
    #
    #   "dog" = d^1 * o^2 * g^3        # d at pos 0 raised to 1, etc
    #   "god" = g^1 * o^2 * d^3        # different exponents -> different composite
    #
    # Morph variants: "dogs" = d^1 * o^2 * g^3 * s^4, and dogs / dog = s^4
    # is a clean integer, so dog | dogs by divisibility.
    # gcd(runs, runner) = run, all by integer factorization.
    #
    # Exponents capped at MAX_POS_EXP so composites stay tractable. Past that
    # position, exponent stops growing (suffix order beyond first ~6 chars
    # doesn't help morphology).
    POS_ANCHOR_COUNT: int = 0      # no longer used; kept for back-compat
    MAX_POS_EXP: int = 6           # cap exponent to keep composites small

    def __init__(self, token_pool_size: int = 20000, doc_pool_size: int = 100000,
                 l2_pool_size: int = 2000, compound_pool_size: int = 5000):
        # Doc / L2 subword / compound bigram primes all from distinct slices.
        total = (
            doc_pool_size + l2_pool_size + self.POS_ANCHOR_COUNT
            + compound_pool_size + 1000
        )
        all_primes = chain_primes(total)
        self._doc_primes = all_primes[:doc_pool_size]
        self._l2_primes = all_primes[doc_pool_size:doc_pool_size + l2_pool_size]
        pos_start = doc_pool_size + l2_pool_size
        self._pos_anchors = all_primes[pos_start:pos_start + self.POS_ANCHOR_COUNT]
        comp_start = pos_start + self.POS_ANCHOR_COUNT
        self._compound_primes = all_primes[comp_start:comp_start + compound_pool_size]
        self._compound_to_prime: dict[tuple[str, str], int] = {}
        self._prime_to_compound: dict[int, tuple[str, str]] = {}
        self._next_compound_idx = 0
        # Plan C: per-token top-K PMI partners (semantic bridge graph)
        # word_prime -> tuple of (partner_prime, pmi_weight)
        self._pmi_partners: dict[int, tuple[tuple[int, float], ...]] = {}
        # Learned chamber vocabularies (Plan A from menu).
        # chamber_id -> dict[word_prime, affinity_log_ratio]
        self._chamber_vocab: dict[int, dict[int, float]] = {}
        # Per-doc soft chamber vector (sparse: only nonzero chambers stored)
        self.doc_chamber_dist: dict[str, dict[int, float]] = {}

        self.lattice = RecursiveLattice()

        # word -> ICN composite (unique anagram-class address; encodes morphology
        # via shared letter prime factors).
        self.token_to_prime: dict[str, int] = {}
        self.prime_to_token: dict[int, str] = {}
        # Cache of letter-prime factor sets per word for morph bonus scoring
        self._word_factors: dict[int, frozenset[int]] = {}
        self.token_doc_count: Counter = Counter()
        # L2 subword promotion: subword string -> pool prime
        self._l2_to_prime: dict[str, int] = {}
        self._prime_to_l2: dict[int, str] = {}
        # Cache: word -> set of L2 subword primes present in it
        self._word_l2_primes: dict[str, frozenset[int]] = {}
        self._next_l2_idx = 0

        self.doc_id_to_prime: dict[str, int] = {}
        self.prime_to_doc_id: dict[int, str] = {}
        # Per-doc token counts for BM25-style tf saturation in scoring
        self.doc_token_counts: dict[str, Counter] = {}
        self.doc_lengths: dict[str, int] = {}
        self._avg_doc_len: float = 1.0
        # 32-chamber subject classification per doc (semantic dim)
        self.doc_subjects: dict[str, frozenset[int]] = {}
        self._dataset_subjects: frozenset[int] = frozenset()

        self._next_doc_idx = 0

    def _chunk_word(self, word: str) -> list[tuple[str, int]]:
        """Greedy left-to-right chunking.

        At each position, pick the longest L2 subword match (4 chars > 3 chars).
        If no L2 match, emit the letter prime for that single character.
        Returns ordered list of (chunk_kind, prime) where kind is 'l2' or 'letter'.
        Position information is implicit in list order.
        """
        w = word.lower()
        chunks: list[tuple[str, int]] = []
        i = 0
        max_l2 = self.L2_MAX_LEN
        min_l2 = self.L2_MIN_LEN
        while i < len(w):
            ch = w[i]
            if not ("a" <= ch <= "z"):
                i += 1
                continue
            matched = False
            # Try L2 lengths from longest to shortest
            for length in range(min(max_l2, len(w) - i), min_l2 - 1, -1):
                sw = w[i:i + length]
                p = self._l2_to_prime.get(sw)
                if p is not None:
                    chunks.append(("l2", p))
                    i += length
                    matched = True
                    break
            if not matched:
                chunks.append(("letter", letter_to_prime(ch)))
                i += 1
        return chunks

    def _allocate_token_prime(self, token: str) -> int:
        """Word -> FTA composite of letter primes WITH MULTIPLICITY.

        Plan A (position-anchored exponents) was tested and HURT nDCG by 1.3
        points because it broke morphological routing: "weight" and "underweight"
        no longer share divisibility. The audit's anagram issues only fire for
        3-letter all-unique-letter words (god/dog, pge/gep) which are rare in
        real BEIR vocabulary. Multiplicity already handles morph correctly:
            weight     = w * e * i * g * h * t
            underweight = u * n * d * e^2 * r * w * i * g * h * t
            underweight / weight = u * n * d * e * r   (INTEGER, perfect prefix)
        gcd(runs, runner) = run by integer factorization.
        """
        if token in self.token_to_prime:
            return self.token_to_prime[token]
        composite = 1
        unique: set[int] = set()
        for ch in token.lower():
            if "a" <= ch <= "z":
                lp = letter_to_prime(ch)
                composite *= lp
                unique.add(lp)
        if composite == 1:
            return 0
        self.token_to_prime[token] = composite
        self.prime_to_token[composite] = token
        self._word_factors[composite] = frozenset(unique)
        self.lattice.register_base(composite, label=token)
        return composite

    def _allocate_doc_prime(self, doc_id: str) -> int:
        if self._next_doc_idx >= len(self._doc_primes):
            raise RuntimeError("doc prime pool exhausted")
        p = self._doc_primes[self._next_doc_idx]
        self._next_doc_idx += 1
        return p

    def set_dataset_subjects(self, dataset: str):
        """Set the dataset-level subject prior (used as fallback when a doc's
        keywords don't match any subject chamber). E.g. 'scifact' -> {1,9,10}."""
        self._dataset_subjects = subjects_for_dataset(dataset)

    def build_from_corpus(self, corpus: dict[str, str]):
        """Multi-pass:
          1. Mine document frequencies + L2 subword stats from kept vocabulary
          2. Promote top-PMI L2 subwords to pool primes
          3. Allocate word composites and cache their L2 subwords
        """
        # Pass 1: document frequency
        df_counts: Counter = Counter()
        for text in corpus.values():
            df_counts.update(set(tokenize(text)))

        n_docs = max(len(corpus), 1)
        max_df = max(int(n_docs * self.MAX_DF_RATIO), 2)
        self._kept_tokens: set[str] = {
            t for t, df in df_counts.items()
            if self.MIN_DF <= df <= max_df
        }

        # Pass 2: mine L2 subword candidates from kept vocabulary
        # PMI-like scoring: prefer subwords that appear in many distinct words
        # AND are not just single-character coincidences.
        l2_sw_doc_count: Counter = Counter()    # number of docs containing subword
        l2_sw_word_count: Counter = Counter()   # number of distinct words containing subword
        for text in corpus.values():
            sw_in_doc: set[str] = set()
            for word in tokenize(text):
                w = word.lower()
                if len(w) < self.L2_MIN_LEN:
                    continue
                word_subs: set[str] = set()
                for length in range(self.L2_MIN_LEN, min(self.L2_MAX_LEN + 1, len(w) + 1)):
                    for i in range(len(w) - length + 1):
                        word_subs.add(w[i:i + length])
                sw_in_doc.update(word_subs)
                for sw in word_subs:
                    l2_sw_word_count[sw] += 1
            for sw in sw_in_doc:
                l2_sw_doc_count[sw] += 1

        # Score subwords: prefer those appearing in many distinct words
        # but not in TOO many docs (would be a stopword-like cross-corpus noise).
        l2_scored: list[tuple[str, float]] = []
        for sw, word_count in l2_sw_word_count.items():
            if word_count < self.L2_MIN_PARENT_WORDS:
                continue
            doc_count = l2_sw_doc_count.get(sw, 0)
            if doc_count > n_docs * 0.5:  # in over half corpus -> noise
                continue
            # Score: log(word_count) * log(n_docs / doc_count)
            score = math.log1p(word_count) * math.log((n_docs + 1.0) / (doc_count + 1.0))
            l2_scored.append((sw, score))
        l2_scored.sort(key=lambda x: -x[1])
        # Promote top-K L2 subwords
        for sw, _ in l2_scored[:self.L2_TOP_K]:
            self._promote_l2(sw)

        # Pass 2.5: mine bigram compounds via PMI.
        bigram_cooc: Counter = Counter()      # raw co-occurrence count
        bigram_doc_count: Counter = Counter() # distinct docs containing the bigram
        for text in corpus.values():
            tokens = [t for t in tokenize(text) if t in self._kept_tokens]
            seen_in_doc: set[tuple[str, str]] = set()
            for i in range(len(tokens) - 1):
                bg = (tokens[i], tokens[i + 1])
                bigram_cooc[bg] += 1
                seen_in_doc.add(bg)
            for bg in seen_in_doc:
                bigram_doc_count[bg] += 1

        # Score bigrams by PMI; promote top-K above threshold
        bigram_scored: list[tuple[tuple[str, str], float]] = []
        for bg, cooc in bigram_cooc.items():
            doc_count = bigram_doc_count.get(bg, 0)
            if doc_count < self.COMPOUND_MIN_COOC:
                continue
            w1, w2 = bg
            df1 = df_counts.get(w1, 1)
            df2 = df_counts.get(w2, 1)
            # PMI = log( N * cooc / (df1 * df2) )
            pmi = math.log((n_docs * cooc + 1.0) / (df1 * df2 + 1.0))
            if pmi < self.COMPOUND_MIN_PMI:
                continue
            bigram_scored.append((bg, pmi))
        bigram_scored.sort(key=lambda x: -x[1])
        for bg, _ in bigram_scored[:self.COMPOUND_TOP_K]:
            self._promote_compound(bg)

        # Pass 3: assign primes to kept tokens (with their L2 subwords cached)
        kept_sorted = sorted(
            (t for t in self._kept_tokens),
            key=lambda t: -df_counts[t],
        )
        for token in kept_sorted:
            self._allocate_token_prime(token)
            # Cache the L2 subword primes for this token
            self._word_l2_primes[token] = self._l2_subword_primes_in(token)

        # Pass 4: ingest each doc (was missing - bug)
        for doc_id, text in corpus.items():
            self._ingest_doc(doc_id, text)

        # Compute avg doc length AND avg chain length (|K|) for normalization.
        if self.doc_lengths:
            self._avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        chain_lens = [
            len(self.lattice.resolve(p).sub_chain or ())
            for p in self.doc_id_to_prime.values()
        ]
        if chain_lens:
            self._avg_doc_len = sum(chain_lens) / len(chain_lens)

        # Pass 5: build PMI semantic bridge graph (Plan C).
        # For each word, find its top-K high-PMI partners across the whole corpus.
        # At query time these become bridge candidates when triangulated.
        if self.BRIDGE_LAMBDA > 0:
            self._build_pmi_graph(corpus)

        # Pass 6: learn per-chamber vocabularies from corpus (Plan A from menu).
        # Each of the 31 chambers gets a learned dict of word_prime -> affinity.
        # Then each doc gets a soft chamber distribution stored in
        # doc_chamber_dist for query-time cosine similarity.
        if self.LAMBDA_CHAMBER_SOFT > 0:
            self._learn_chamber_vocabularies(corpus)

    def _learn_chamber_vocabularies(self, corpus: dict[str, str]) -> None:
        """Learn per-chamber word affinities from corpus co-occurrence.

        For each chamber k with seed keywords S_k, the words that fire
        for chamber k get an affinity = log( P(word | chamber=k) / P(word) ).
        Words much more common in chamber k's docs than overall have high
        affinity and become part of chamber k's learned vocabulary.

        Then each doc's chamber distribution is computed as a sum over its
        words of (affinity per chamber), normalized.
        """
        from aethos_symbol_subjects import _SUBJECT_KEYWORDS

        n_total = max(len(corpus), 1)

        # Step 1: which docs fire for which chamber (binary, seed-based)
        chamber_docs: dict[int, list[str]] = {}
        for doc_id, text in corpus.items():
            tokens = set(tokenize(text))
            for k, seeds in _SUBJECT_KEYWORDS.items():
                if tokens & seeds:
                    chamber_docs.setdefault(k, []).append(doc_id)

        # Step 2: per-chamber word affinity via log probability ratio
        for k, docs_in_k in chamber_docs.items():
            n_k = len(docs_in_k)
            if n_k < 5:
                continue
            word_count_in_chamber: Counter = Counter()
            for doc_id in docs_in_k:
                text = corpus.get(doc_id, "")
                for t in set(tokenize(text)):
                    if t in self._kept_tokens:
                        word_count_in_chamber[t] += 1

            vocab: dict[int, float] = {}
            for word, count_k in word_count_in_chamber.items():
                p_word = self.token_doc_count.get(word, count_k)
                if p_word < self.CHAMBER_VOCAB_MIN_DF:
                    continue
                # log ratio: P(w | k) / P(w) = (count_k / n_k) / (df / N)
                ratio = (count_k / n_k) / (p_word / n_total)
                if ratio <= 1.0:
                    continue
                log_ratio = math.log(ratio)
                if log_ratio < self.CHAMBER_VOCAB_MIN_AFFINITY:
                    continue
                word_prime = self.token_to_prime.get(word)
                if word_prime is not None:
                    vocab[word_prime] = log_ratio

            # Keep top-K most diagnostic words per chamber
            if len(vocab) > self.CHAMBER_VOCAB_TOP_PER_CHAMBER:
                top = sorted(vocab.items(), key=lambda x: -x[1])
                vocab = dict(top[:self.CHAMBER_VOCAB_TOP_PER_CHAMBER])
            self._chamber_vocab[k] = vocab

        # Step 3: precompute soft chamber distribution per doc
        # doc_dist[k] = sum over doc's word primes of chamber_vocab[k][prime]
        for doc_id in self.doc_id_to_prime:
            doc_prime = self.doc_id_to_prime[doc_id]
            doc_chain = self.lattice.resolve(doc_prime).sub_chain or ()
            doc_set = set(doc_chain)
            dist: dict[int, float] = {}
            for k, vocab in self._chamber_vocab.items():
                s = 0.0
                for p in doc_set:
                    aff = vocab.get(p)
                    if aff is not None:
                        s += aff
                if s > 0:
                    dist[k] = s
            # Normalize L2 so cosine sim is bounded in [0, 1]
            norm = math.sqrt(sum(v * v for v in dist.values()))
            if norm > 0:
                self.doc_chamber_dist[doc_id] = {
                    k: v / norm for k, v in dist.items()
                }

    def _query_chamber_dist(self, query_primes_set: set[int]) -> dict[int, float]:
        """Soft chamber distribution for a query."""
        dist: dict[int, float] = {}
        for k, vocab in self._chamber_vocab.items():
            s = 0.0
            for p in query_primes_set:
                aff = vocab.get(p)
                if aff is not None:
                    s += aff
            if s > 0:
                dist[k] = s
        norm = math.sqrt(sum(v * v for v in dist.values()))
        if norm > 0:
            return {k: v / norm for k, v in dist.items()}
        return {}

    def _build_pmi_graph(self, corpus: dict[str, str]) -> None:
        # Step 1: word pair co-occurrence at the WHOLE-DOC level.
        # Only count pairs where both words are in our vocab AND
        # at least one is not a flood word.
        kept = self._kept_tokens
        pair_doc_count: dict[tuple[int, int], int] = {}
        for text in corpus.values():
            unique_word_primes: set[int] = set()
            for t in set(tokenize(text)):
                if t in kept:
                    p = self.token_to_prime.get(t)
                    if p is not None:
                        unique_word_primes.add(p)
            # Quadratic in unique tokens per doc; cap to avoid blow-up
            sorted_primes = sorted(unique_word_primes)
            if len(sorted_primes) > 80:
                # Keep the rarer ones (highest primes = assigned last = rarer)
                sorted_primes = sorted_primes[-80:]
            for i, a in enumerate(sorted_primes):
                for b in sorted_primes[i + 1:]:
                    key = (a, b) if a < b else (b, a)
                    pair_doc_count[key] = pair_doc_count.get(key, 0) + 1

        # Step 2: compute PMI for each pair seen >= PMI_MIN_COOC times.
        n_docs = max(len(corpus), 1)
        word_df: dict[int, int] = {
            p: len(self.lattice.resolve(p).parents)
            for p in self.prime_to_token
        }
        per_word_candidates: dict[int, list[tuple[float, int]]] = {}
        for (a, b), c in pair_doc_count.items():
            if c < self.PMI_MIN_COOC:
                continue
            df_a = word_df.get(a, 1)
            df_b = word_df.get(b, 1)
            pmi = math.log((n_docs * c + 1.0) / (df_a * df_b + 1.0))
            if pmi < self.PMI_MIN_VAL:
                continue
            per_word_candidates.setdefault(a, []).append((pmi, b))
            per_word_candidates.setdefault(b, []).append((pmi, a))

        # Step 3: keep top-K partners per word
        for w, cands in per_word_candidates.items():
            cands.sort(key=lambda x: -x[0])
            top = tuple((p, pmi) for pmi, p in cands[:self.PMI_TOP_PER_WORD])
            self._pmi_partners[w] = top

    def _promote_compound(self, bigram: tuple[str, str]) -> int:
        if bigram in self._compound_to_prime:
            return self._compound_to_prime[bigram]
        if self._next_compound_idx >= len(self._compound_primes):
            return 0
        p = self._compound_primes[self._next_compound_idx]
        self._next_compound_idx += 1
        self._compound_to_prime[bigram] = p
        self._prime_to_compound[p] = bigram
        self.lattice.register_base(p, label=f"C:{bigram[0]}_{bigram[1]}")
        return p

    def _doc_compound_primes(self, text: str) -> set[int]:
        """Return the set of compound primes whose bigrams appear in text."""
        tokens = [t for t in tokenize(text) if t in self._kept_tokens]
        out: set[int] = set()
        for i in range(len(tokens) - 1):
            bg = (tokens[i], tokens[i + 1])
            cp = self._compound_to_prime.get(bg)
            if cp:
                out.add(cp)
        return out

    def _promote_l2(self, sw: str) -> int:
        if sw in self._l2_to_prime:
            return self._l2_to_prime[sw]
        if self._next_l2_idx >= len(self._l2_primes):
            return 0  # pool exhausted
        p = self._l2_primes[self._next_l2_idx]
        self._next_l2_idx += 1
        self._l2_to_prime[sw] = p
        self._prime_to_l2[p] = sw
        self.lattice.register_base(p, label=f"L2:{sw}")
        return p

    def _l2_subword_primes_in(self, word: str) -> frozenset[int]:
        w = word.lower()
        if len(w) < self.L2_MIN_LEN:
            return frozenset()
        found: set[int] = set()
        for length in range(self.L2_MIN_LEN, min(self.L2_MAX_LEN + 1, len(w) + 1)):
            for i in range(len(w) - length + 1):
                sw = w[i:i + length]
                p = self._l2_to_prime.get(sw)
                if p:
                    found.add(p)
        return frozenset(found)

        # Pass 2: ingest docs in given order
        for doc_id, text in corpus.items():
            self._ingest_doc(doc_id, text)

        # Compute avg doc length AND avg chain length (|K|) for normalization.
        if self.doc_lengths:
            self._avg_doc_len = sum(self.doc_lengths.values()) / len(self.doc_lengths)
        # avg_chain_len is the geometric "doc length" — number of unique kept anchors
        chain_lens = [
            len(self.lattice.resolve(p).sub_chain or ())
            for p in self.doc_id_to_prime.values()
        ]
        if chain_lens:
            self._avg_doc_len = sum(chain_lens) / len(chain_lens)

    def _ingest_doc(self, doc_id: str, text: str) -> int | None:
        tokens = tokenize(text)
        if not tokens:
            return None

        kept_vocab = getattr(self, "_kept_tokens", None)
        # Per-doc term frequencies (only for tokens in our kept vocabulary)
        tf_counts: Counter = Counter(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        # Classify doc into subject chambers (parallel semantic dim).
        # Each chamber acts as a specialized classifier; doc routes to top-K subjects.
        self.doc_subjects[doc_id] = infer_doc_subjects(
            text,
            fallback=self._dataset_subjects,
            max_subjects=self.SUBJECT_MAX_PER_DOC,
        )

        # Build (prime, tf) pairs for unique tokens in the kept vocabulary.
        prime_tf_pairs: list[tuple[int, int]] = []
        l2_primes_in_doc: set[int] = set()
        for t, c in tf_counts.items():
            if kept_vocab is not None and t not in kept_vocab:
                continue
            p = self.token_to_prime.get(t)
            if p is None:
                continue
            prime_tf_pairs.append((p, c))
            self.token_doc_count[t] += 1
            # Collect L2 subword primes present in this token
            l2_primes_in_doc.update(self._word_l2_primes.get(t, frozenset()))
        # Cap to top-K rarest if needed
        if len(prime_tf_pairs) > self.MAX_CHAIN_PER_DOC:
            prime_tf_pairs.sort(key=lambda pt: -pt[0])
            prime_tf_pairs = prime_tf_pairs[:self.MAX_CHAIN_PER_DOC]
        prime_tf_pairs.sort(key=lambda pt: pt[0])
        # Doc chain = word composites UNION L2 subword primes UNION compound primes
        compound_primes_in_doc = self._doc_compound_primes(text)
        chain_set = {p for p, _ in prime_tf_pairs} | l2_primes_in_doc | compound_primes_in_doc
        chain = tuple(sorted(chain_set))
        if not chain:
            return None
        # Store tf only for word composites (L2 subwords get default tf=1)
        self.doc_token_counts[doc_id] = Counter({
            self.prime_to_token[p]: tf for p, tf in prime_tf_pairs
        })

        # Allocate the doc's prime and create the lattice node directly
        doc_prime = self._allocate_doc_prime(doc_id)
        max_level = 0
        for p in chain:
            node = self.lattice.resolve(p)
            max_level = max(max_level, node.level)
            if doc_prime not in node.parents:
                node.parents.append(doc_prime)
        doc_node = RecursiveNode(
            prime=doc_prime,
            level=max_level + 1,
            sub_chain=chain,
            label=f"doc:{doc_id}",
        )
        self.lattice.nodes[doc_prime] = doc_node

        self.doc_id_to_prime[doc_id] = doc_prime
        self.prime_to_doc_id[doc_prime] = doc_id
        return doc_prime

    # BM25 backbone + lattice morphology bonus.
    # Tokens are now letter-prime composites (text_icn semantics) so
    # morphologically related words share most of their letter prime factors.
    # We score factor-Jaccard between query word and doc word as a "soft match",
    # giving free stemming-like behavior without any stemmer.
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    # Containment-based morphology (no Jaccard noise).
    # Tried set-subset: weight ⊂ underweight works, but "act" ⊂ "fact" is
    # a false positive (letter set match without true morph root). Disabled
    # until we have L2 PMI subwords (genuine morph units, not raw letters).
    LAMBDA_MORPH: float = 0.0
    MORPH_MIN_Q_LETTERS: int = 4
    MORPH_MAX_DOC_OVER_Q: float = 2.0
    LAMBDA_PAIR: float = 0.0
    PI_DEPTH_ALPHA: float = 0.0
    # PRF / 3-way meet expansion: take rare-IDF terms recurring in top-K
    # candidates and let them bring related docs (incl gold) to the top.
    # 3-way meet expansion: for each pair of query composites, find docs
    # containing both, then take the rarest recurring third composite as
    # the "meet branch" -- a query expansion term that's not in the query
    # but geometrically lives at the same lattice node as the query pair.
    PRF_TOP_K: int = 0          # disabled: classical PRF blind to relevance
    TRIPLE_MAX_PAIRS: int = 6   # cap pairs of query terms to consider
    TRIPLE_PER_PAIR: int = 3    # expansion terms per pair (rarest recurring)
    TRIPLE_LAMBDA: float = 0.0  # disabled: top-K precision too low for branches
    TRIPLE_MIN_PAIR_DOCS: int = 2
    TRIPLE_MIN_RECUR: int = 2

    def query(self, text: str, k: int = 10, wing: int = 1) -> list[tuple[str, float]]:
        """Hybrid BM25 + lattice pair-meet score on lattice-routed candidates.

        Score(q, d) = BM25(q, d, IDF*) + LAMBDA_PAIR * S_pair(q, d)

        - BM25 with IDF* = IDF_lex * (1 + alpha * depth_frac) -- pi-depth boost
        - S_pair: sum over all query pairs (a,b) both in doc of sqrt(IDF*(a)*IDF*(b))
          (the compositional signal BM25 misses; query co-occurrence in doc).
        """
        tokens = tokenize(text)
        if not tokens:
            return []
        query_primes_set: set[int] = set()
        # Track L2 subword primes separately so we can weight them lighter
        query_l2_primes: set[int] = set()
        for t in tokens:
            p = self.token_to_prime.get(t)
            if p is not None:
                query_primes_set.add(p)
            # Get L2 subword primes for this query token (whether or not
            # the word itself is in the vocab - this lets unseen query words
            # still route through their morphemes).
            l2_primes = self._l2_subword_primes_in(t)
            query_l2_primes.update(l2_primes)
        # Compound bigram primes for adjacent token pairs in the query
        query_compound_primes: set[int] = set()
        for i in range(len(tokens) - 1):
            bg = (tokens[i], tokens[i + 1])
            cp = self._compound_to_prime.get(bg)
            if cp:
                query_compound_primes.add(cp)

        if not query_primes_set and not query_l2_primes and not query_compound_primes:
            return []
        # Include L2 subword primes AND compound bigram primes in the routing set
        query_primes_set = query_primes_set | query_l2_primes | query_compound_primes

        # Classify the QUERY into subject chambers (the parallel semantic
        # classifier system; same machinery as doc classification).
        query_subjects: frozenset[int] = vote_query_chambers(
            tokens, max_chambers=self.SUBJECT_MAX_PER_QUERY,
        )
        # Soft chamber distribution from learned vocabularies (Plan A)
        q_chamber_dist: dict[int, float] = {}
        if self.LAMBDA_CHAMBER_SOFT > 0 and self._chamber_vocab:
            q_chamber_dist = self._query_chamber_dist(
                query_primes_set | {self.token_to_prime[t] for t in tokens
                                    if t in self.token_to_prime}
            )
        query_primes = sorted(query_primes_set)
        query_chain = tuple(query_primes)
        n_q = len(query_primes)

        # walk_up returns containing parents -> these include doc primes
        candidate_hits: Counter[str] = Counter()
        for qp in query_primes_set:
            for parent_p in self.lattice.walk_up(qp):
                if parent_p in self.prime_to_doc_id:
                    candidate_hits[self.prime_to_doc_id[parent_p]] += 1
        if not candidate_hits:
            return []

        # IDF*(p): lexical IDF augmented by pi-depth (length of query chain as proxy).
        # Longer query chains => higher depth boost (rare-term queries score higher).
        n_docs = max(len(self.doc_id_to_prime), 1)
        depth_frac = math.log1p(n_q) / math.log1p(16.0)
        depth_boost = 1.0 + self.PI_DEPTH_ALPHA * depth_frac
        idf_for: dict[int, float] = {}
        for qp in query_primes_set:
            df = max(len(self.lattice.resolve(qp).parents), 1)
            idf_lex = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            idf_for[qp] = idf_lex * depth_boost

        # Precompute query Psi at each query anchor (one wing_transform per query prime).
        q_psi: dict[int, "ComplexPlane3D"] = {}
        for qp in query_primes:
            q_psi[qp] = wing_transform(BranchKind.VA1, query_chain, qp, wing=wing)

        scored: list[tuple[str, float]] = []
        for doc_id, hit_count in candidate_hits.items():
            doc_prime = self.doc_id_to_prime[doc_id]
            doc_chain = self.lattice.resolve(doc_prime).sub_chain
            if not doc_chain:
                continue
            doc_set = set(doc_chain)
            shared = query_primes_set & doc_set
            if not shared:
                continue

            # 1) BM25 backbone with IDF on shared word composites + L2 subwords.
            #    Word composites use stored tf; L2 subword anchors have tf=1
            #    (each unique subword counts once per doc).
            doc_len = len(doc_chain)
            doc_counts = self.doc_token_counts.get(doc_id)
            length_norm = 1.0 - self.BM25_B + self.BM25_B * (doc_len / max(self._avg_doc_len, 1.0))
            bm25_term = 0.0
            for anchor in shared:
                tf = 1
                # Only word composites have stored tf; L2 primes do not.
                word = self.prime_to_token.get(anchor)
                if word is not None and doc_counts is not None:
                    tf = doc_counts.get(word, 1)
                tf_sat = (tf * (self.BM25_K1 + 1)) / (tf + self.BM25_K1 * length_norm)
                # L2 subwords: 0.6x weight (morph hint, less direct)
                # Compound bigrams: 1.0x (high IDF already gives them weight,
                # over-boosting them over-scores docs that just happen to have
                # the phrase even when topical relevance is weak)
                # Exact word match: 1.0x baseline
                if anchor in self._prime_to_l2:
                    anchor_weight = 0.6
                else:
                    anchor_weight = 1.0
                bm25_term += idf_for[anchor] * tf_sat * anchor_weight

            # 2) Containment morphology: q_letters subset of d_letters.
            #    Only fires when the doc word strictly EXTENDS the query word's
            #    letter set (weight -> underweight; rate -> rates). Cheap O(1)
            #    set comparison; no jaccard noise.
            morph_score = 0.0
            if self.LAMBDA_MORPH > 0:
                unmatched_doc = doc_set - query_primes_set
                for q_comp in query_primes:
                    q_factors = self._word_factors.get(q_comp)
                    if q_factors is None or len(q_factors) < self.MORPH_MIN_Q_LETTERS:
                        continue
                    q_size = len(q_factors)
                    max_d_size = int(q_size * self.MORPH_MAX_DOC_OVER_Q)
                    best_credit = 0.0
                    for d_comp in unmatched_doc:
                        d_factors = self._word_factors.get(d_comp)
                        if d_factors is None: continue
                        d_size = len(d_factors)
                        if d_size > max_d_size: continue
                        if d_size < q_size: continue  # must be at least as big
                        if not q_factors.issubset(d_factors): continue
                        # Credit by how tightly the doc word matches:
                        # exact set equality = 1.0, length 1.5x = 0.67
                        credit = q_size / d_size
                        if credit > best_credit:
                            best_credit = credit
                    if best_credit > 0:
                        morph_score += best_credit * idf_for[q_comp]

            score = bm25_term + self.LAMBDA_MORPH * morph_score

            # Hard subject-chamber boost via keyword vote
            if query_subjects and self.LAMBDA_SUBJECT > 0:
                d_subjects = self.doc_subjects.get(doc_id, frozenset())
                if d_subjects:
                    overlap = len(query_subjects & d_subjects)
                    if overlap > 0:
                        score *= (1.0 + self.LAMBDA_SUBJECT * overlap)

            # Soft chamber-distribution similarity via learned vocabularies
            if q_chamber_dist:
                d_chamber_dist = self.doc_chamber_dist.get(doc_id)
                if d_chamber_dist:
                    # Cosine: both distributions are L2-normalized
                    cos_sim = 0.0
                    smaller = q_chamber_dist if len(q_chamber_dist) < len(d_chamber_dist) else d_chamber_dist
                    other = d_chamber_dist if smaller is q_chamber_dist else q_chamber_dist
                    for k, v in smaller.items():
                        ov = other.get(k)
                        if ov is not None:
                            cos_sim += v * ov
                    if cos_sim > 0:
                        score *= (1.0 + self.LAMBDA_CHAMBER_SOFT * cos_sim)

            scored.append((doc_id, score))

        scored.sort(key=lambda x: -x[1])

        # ============================================================
        # Plan C: PMI semantic bridge boost.
        # For each query word, look up its top PMI partners. If MULTIPLE
        # query words agree on the same partner (triangulation), that
        # partner is a bridge candidate. Boost docs containing it.
        # Triangulation is the safety net: a single word's PMI partner
        # could be noise; agreement across 2+ query words means it's a
        # genuine topic-context word, not just co-occurrence.
        # ============================================================
        if (self.BRIDGE_LAMBDA > 0 and self._pmi_partners
                and len(query_primes) >= self.BRIDGE_MIN_TRIANGULATION):
            # Tally bridge candidates by triangulation count
            bridge_votes: dict[int, float] = {}
            for q_word in query_primes:
                partners = self._pmi_partners.get(q_word)
                if not partners:
                    continue
                for partner_p, pmi in partners:
                    if partner_p in query_primes_set:
                        continue  # already in query
                    bridge_votes[partner_p] = bridge_votes.get(partner_p, 0.0) + pmi
            # Keep only partners voted by >= MIN_TRIANGULATION query words
            # (approximation: high cumulative PMI implies multiple votes)
            min_cum_pmi = self.PMI_MIN_VAL * self.BRIDGE_MIN_TRIANGULATION
            bridge_set = {p: v for p, v in bridge_votes.items() if v >= min_cum_pmi}

            if bridge_set:
                # Boost scored docs that contain bridge primes
                rescored: list[tuple[str, float]] = []
                for d_id, s in scored:
                    d_prime = self.doc_id_to_prime[d_id]
                    d_chain_set = set(self.lattice.resolve(d_prime).sub_chain or ())
                    overlap = bridge_set.keys() & d_chain_set
                    if overlap:
                        boost = sum(bridge_set[p] for p in overlap)
                        rescored.append((d_id, s + self.BRIDGE_LAMBDA * boost))
                    else:
                        rescored.append((d_id, s))
                rescored.sort(key=lambda x: -x[1])
                scored = rescored

        # ============================================================
        # 3-way meet expansion (lattice-native, not PRF):
        # For each pair of query word composites (q_i, q_j), walk_up
        # both and intersect -- those are docs containing BOTH terms.
        # Among those docs, the rarest recurring THIRD term is the
        # 3-way meet branch -- a word that geometrically lives at the
        # same lattice node as the query pair.
        # ============================================================
        n_docs_total = max(len(self.doc_id_to_prime), 1)
        if (self.TRIPLE_LAMBDA > 0 and n_q >= 2
                and len(query_primes) <= 12):  # limit pair combinations
            # Cache walk_up sets per query prime (intersection-friendly)
            qp_parents: dict[int, set[int]] = {}
            for qp in query_primes:
                qp_parents[qp] = set(self.lattice.walk_up(qp))

            # Score each pair: their co-occurring docs' rare term recurrence
            pair_count = 0
            branch_score: dict[int, float] = {}
            for i, p1 in enumerate(query_primes):
                if pair_count >= self.TRIPLE_MAX_PAIRS: break
                for p2 in query_primes[i + 1:]:
                    if pair_count >= self.TRIPLE_MAX_PAIRS: break
                    pair_co_docs = qp_parents[p1] & qp_parents[p2]
                    pair_co_doc_ids = [
                        self.prime_to_doc_id[p] for p in pair_co_docs
                        if p in self.prime_to_doc_id
                    ]
                    if len(pair_co_doc_ids) < self.TRIPLE_MIN_PAIR_DOCS:
                        continue
                    pair_count += 1
                    # Tally third terms across the pair's co-occurring docs
                    third_counts: Counter = Counter()
                    for d_id in pair_co_doc_ids:
                        d_prime = self.doc_id_to_prime[d_id]
                        d_chain = self.lattice.resolve(d_prime).sub_chain or ()
                        for t in d_chain:
                            if t in query_primes_set: continue
                            third_counts[t] += 1
                    # Keep rare ones (high IDF, recur >= TRIPLE_MIN_RECUR)
                    candidates = []
                    for t, cnt in third_counts.items():
                        if cnt < self.TRIPLE_MIN_RECUR: continue
                        df = max(len(self.lattice.resolve(t).parents), 1)
                        if df > n_docs_total * 0.05:  # too common
                            continue
                        idf = math.log((n_docs_total + 1.0) / (df + 1.0)) + 1.0
                        candidates.append((t, idf * math.sqrt(cnt)))
                    candidates.sort(key=lambda x: -x[1])
                    for t, s in candidates[:self.TRIPLE_PER_PAIR]:
                        branch_score[t] = branch_score.get(t, 0.0) + s

            # Apply branch boost only to docs we already scored
            if branch_score:
                rescored: list[tuple[str, float]] = []
                for d_id, s in scored:
                    d_prime = self.doc_id_to_prime[d_id]
                    d_chain_set = set(self.lattice.resolve(d_prime).sub_chain or ())
                    overlap_score = sum(
                        v for t, v in branch_score.items() if t in d_chain_set
                    )
                    rescored.append((d_id, s + self.TRIPLE_LAMBDA * overlap_score))
                rescored.sort(key=lambda x: -x[1])
                return rescored[:k]

        # ============================================================
        # Classical PRF fallback (disabled by default; PRF_TOP_K=0)
        # ============================================================
        if self.PRF_TOP_K > 0 and self.PRF_LAMBDA > 0 and len(scored) >= self.PRF_TOP_K:
            top_docs = scored[:self.PRF_TOP_K]
            # Tally rare-IDF terms across top-K (terms not in query)
            n_docs_total = max(len(self.doc_id_to_prime), 1)
            expansion_score: dict[int, float] = {}
            expansion_recur: dict[int, int] = {}
            for top_doc_id, _ in top_docs:
                top_doc_prime = self.doc_id_to_prime[top_doc_id]
                top_chain = self.lattice.resolve(top_doc_prime).sub_chain
                if not top_chain:
                    continue
                top_doc_counts = self.doc_token_counts.get(top_doc_id)
                for term_comp in top_chain:
                    if term_comp in query_primes_set:
                        continue
                    df = max(len(self.lattice.resolve(term_comp).parents), 1)
                    # Skip terms that occur in too many docs (low IDF, noise)
                    if df > n_docs_total * 0.1:
                        continue
                    idf = math.log((n_docs_total + 1.0) / (df + 1.0)) + 1.0
                    tf = 1
                    if top_doc_counts is not None:
                        tf = top_doc_counts.get(self.prime_to_token.get(term_comp, ""), 1)
                    expansion_score[term_comp] = expansion_score.get(term_comp, 0.0) + idf * math.sqrt(tf)
                    expansion_recur[term_comp] = expansion_recur.get(term_comp, 0) + 1

            # Keep only terms that recur in PRF_MIN_RECUR+ docs
            expansion_candidates = [
                (t, s) for t, s in expansion_score.items()
                if expansion_recur.get(t, 0) >= self.PRF_MIN_RECUR
            ]
            expansion_candidates.sort(key=lambda x: -x[1])
            expansion_terms = expansion_candidates[:self.PRF_EXPANSION_N]

            if expansion_terms:
                # Only boost docs that are ALREADY candidates we scored
                # (not every doc in the corpus containing some expansion term).
                # Boost = count of expansion terms doc contains, IDF-weighted.
                exp_set = {t for t, _ in expansion_terms}
                exp_idf: dict[int, float] = {}
                for term, _ in expansion_terms:
                    df = max(len(self.lattice.resolve(term).parents), 1)
                    exp_idf[term] = math.log((n_docs_total + 1.0) / (df + 1.0)) + 1.0

                rescored: list[tuple[str, float]] = []
                for d_id, s in scored:
                    doc_prime = self.doc_id_to_prime[d_id]
                    doc_chain_set = set(self.lattice.resolve(doc_prime).sub_chain or ())
                    overlap = doc_chain_set & exp_set
                    if not overlap:
                        rescored.append((d_id, s))
                        continue
                    boost = sum(exp_idf[t] for t in overlap)
                    rescored.append((d_id, s + self.PRF_LAMBDA * boost))
                rescored.sort(key=lambda x: -x[1])
                return rescored[:k]

        return scored[:k]

    def estimated_footprint(self) -> dict:
        """Rough bytes-on-disk estimate (excluding Python object overhead)."""
        n_docs = len(self.doc_id_to_prime)
        n_tokens = len(self.token_to_prime)
        # per doc node: prime(4) + level(1) + chain pointer(8) + parents pointer(8) = ~24 B
        # plus actual chain ints (8 B each)
        avg_chain = (sum(len(self.lattice.resolve(p).sub_chain or ())
                         for p in self.doc_id_to_prime.values())
                     / max(n_docs, 1))
        bytes_per_doc = 24 + avg_chain * 8
        avg_parents = (sum(len(self.lattice.resolve(p).parents)
                           for p in self.token_to_prime.values())
                       / max(n_tokens, 1))
        bytes_per_token = 24 + avg_parents * 8
        total = n_docs * bytes_per_doc + n_tokens * bytes_per_token
        return {
            "n_docs": n_docs,
            "n_tokens": n_tokens,
            "avg_chain_len": avg_chain,
            "avg_parents_per_token": avg_parents,
            "bytes_per_doc": bytes_per_doc,
            "bytes_per_token": bytes_per_token,
            "total_bytes": total,
        }


# =====================================================================
# TF-IDF baseline (minimal, just for comparison)
# =====================================================================

class TfIdfBaseline:
    def __init__(self):
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_token_counts: dict[str, Counter] = {}
        self.df: Counter = Counter()
        self.n_docs = 0

    def ingest(self, doc_id: str, text: str):
        tokens = tokenize(text)
        self.doc_tokens[doc_id] = tokens
        self.doc_token_counts[doc_id] = Counter(tokens)
        for t in set(tokens):
            self.df[t] += 1
        self.n_docs += 1

    def query(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        q_tokens = tokenize(text)
        scored: list[tuple[str, float]] = []
        for doc_id, counts in self.doc_token_counts.items():
            score = 0.0
            for qt in q_tokens:
                tf = counts.get(qt, 0)
                df = self.df.get(qt, 0)
                if tf > 0 and df > 0:
                    idf = math.log((self.n_docs + 1) / (df + 1)) + 1
                    score += tf * idf
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def estimated_footprint(self) -> dict:
        total_tokens = sum(len(t) for t in self.doc_tokens.values())
        # Inverted-index-equivalent: each token gets (str + count) entries per doc
        # Realistic: ~16 B/posting (varint doc_id + count)
        bytes_postings = total_tokens * 16
        # Vocabulary: 6 B/token avg
        bytes_vocab = len(self.df) * 12
        total = bytes_postings + bytes_vocab
        return {
            "n_docs": self.n_docs,
            "n_postings": total_tokens,
            "bytes_per_doc": total / max(self.n_docs, 1),
            "total_bytes": total,
        }


# =====================================================================
# Demo
# =====================================================================

DEMO_CORPUS = {
    "d1":  "the quick brown fox jumps over the lazy dog",
    "d2":  "machine learning algorithms classify large data sets",
    "d3":  "the lazy dog sleeps under the brown tree",
    "d4":  "neural networks learn patterns in data",
    "d5":  "the quick fox hunts at dawn near the river",
    "d6":  "deep learning models process images efficiently",
    "d7":  "lazy summer afternoon under the oak tree",
    "d8":  "supervised learning requires labeled data examples",
    "d9":  "the fox and the dog became unlikely friends",
    "d10": "convolutional neural networks process images effectively",
    "d11": "transformer models revolutionize natural language processing",
    "d12": "the river runs through ancient mountain valleys",
    "d13": "data scientists train neural networks on large datasets",
    "d14": "wolves and foxes share ancestry in canine evolution",
    "d15": "image classification benefits from deep convolutional architectures",
}

DEMO_QUERIES = [
    ("quick fox",                  ["d1", "d5", "d9", "d14"]),
    ("neural networks images",     ["d10", "d15", "d4", "d6", "d13"]),
    ("lazy dog tree",              ["d3", "d7", "d1", "d12"]),
    ("learning data",              ["d2", "d4", "d6", "d8", "d13"]),
    ("river dawn",                 ["d5", "d12"]),
    ("convolutional networks",     ["d10", "d15", "d4"]),
]


def hit_at_k(retrieved: list[str], gold: list[str], k: int) -> int:
    return sum(1 for d in retrieved[:k] if d in gold)


def main():
    print("=" * 78)
    print("LATTICE RETRIEVAL - first wire-up of the recursive lattice as a retriever")
    print("=" * 78)
    print()

    # ----- build both indexes -----
    t0 = time.time()
    lat = LatticeRetriever()
    lat.build_from_corpus(DEMO_CORPUS)
    lat_build_ms = (time.time() - t0) * 1000

    t0 = time.time()
    bm = TfIdfBaseline()
    for d, t in DEMO_CORPUS.items():
        bm.ingest(d, t)
    bm_build_ms = (time.time() - t0) * 1000

    print(f"Corpus: {len(DEMO_CORPUS)} docs")
    print(f"\nBUILD time:")
    print(f"  Lattice:  {lat_build_ms:.1f} ms")
    print(f"  TF-IDF :  {bm_build_ms:.1f} ms")

    lf = lat.estimated_footprint()
    bf = bm.estimated_footprint()
    print(f"\nFOOTPRINT estimate:")
    print(f"  Lattice:  {lf['bytes_per_doc']:.0f} B/doc (chain len {lf['avg_chain_len']:.1f}, "
          f"avg {lf['avg_parents_per_token']:.1f} parents/token)")
    print(f"  TF-IDF :  {bf['bytes_per_doc']:.0f} B/doc ({bf['n_postings']} postings total)")
    if bf['bytes_per_doc'] > 0:
        ratio = bf['bytes_per_doc'] / lf['bytes_per_doc']
        print(f"  Lattice is {ratio:.2f}x {'smaller' if ratio > 1 else 'larger'}")

    # ----- queries -----
    print()
    print("=" * 78)
    print("QUERIES (lattice  vs  TF-IDF baseline)")
    print("=" * 78)

    total_lat_hit3 = total_lat_hit5 = total_bm_hit3 = total_bm_hit5 = total_gold_3 = total_gold_5 = 0
    lat_latency_total = bm_latency_total = 0.0

    for query_text, gold in DEMO_QUERIES:
        t0 = time.time()
        lat_results = lat.query(query_text, k=5)
        lat_ms = (time.time() - t0) * 1000
        lat_latency_total += lat_ms

        t0 = time.time()
        bm_results = bm.query(query_text, k=5)
        bm_ms = (time.time() - t0) * 1000
        bm_latency_total += bm_ms

        lat_ids = [d for d, _ in lat_results]
        bm_ids = [d for d, _ in bm_results]

        lat_h3 = hit_at_k(lat_ids, gold, 3)
        lat_h5 = hit_at_k(lat_ids, gold, 5)
        bm_h3 = hit_at_k(bm_ids, gold, 3)
        bm_h5 = hit_at_k(bm_ids, gold, 5)
        gold_at_3 = min(3, len(gold))
        gold_at_5 = min(5, len(gold))

        total_lat_hit3 += lat_h3
        total_lat_hit5 += lat_h5
        total_bm_hit3 += bm_h3
        total_bm_hit5 += bm_h5
        total_gold_3 += gold_at_3
        total_gold_5 += gold_at_5

        print(f"\nQuery: '{query_text}'   gold: {gold}")
        print(f"  Lattice  ({lat_ms:.2f} ms)  Hit@3={lat_h3}/{gold_at_3}  Hit@5={lat_h5}/{gold_at_5}")
        for d, s in lat_results:
            mark = "+" if d in gold else "."
            print(f"    [{mark}] {d}  score={s:>7.1f}   {DEMO_CORPUS[d]}")
        print(f"  TF-IDF   ({bm_ms:.2f} ms)  Hit@3={bm_h3}/{gold_at_3}  Hit@5={bm_h5}/{gold_at_5}")
        for d, s in bm_results:
            mark = "+" if d in gold else "."
            print(f"    [{mark}] {d}  score={s:>7.3f}   {DEMO_CORPUS[d]}")

    # ----- summary -----
    print()
    print("=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)
    print(f"{'metric':<25} {'Lattice':>12} {'TF-IDF':>12}")
    print(f"{'-' * 25} {'-' * 12} {'-' * 12}")
    lat_acc3 = total_lat_hit3 / max(total_gold_3, 1)
    bm_acc3 = total_bm_hit3 / max(total_gold_3, 1)
    lat_acc5 = total_lat_hit5 / max(total_gold_5, 1)
    bm_acc5 = total_bm_hit5 / max(total_gold_5, 1)
    print(f"{'Hit@3 recall':<25} {lat_acc3:>11.1%} {bm_acc3:>11.1%}")
    print(f"{'Hit@5 recall':<25} {lat_acc5:>11.1%} {bm_acc5:>11.1%}")
    print(f"{'avg query latency (ms)':<25} {lat_latency_total / len(DEMO_QUERIES):>12.2f} "
          f"{bm_latency_total / len(DEMO_QUERIES):>12.2f}")
    print(f"{'bytes/doc':<25} {lf['bytes_per_doc']:>12.0f} {bf['bytes_per_doc']:>12.0f}")

    # ----- show one lattice sample -----
    print()
    print("=" * 78)
    print("LATTICE STATE")
    print("=" * 78)
    s = lat.lattice.stats()
    print(f"  levels: {s['level_counts']}    total nodes: {s['total_nodes']}")

    # Walk up from a common token
    fox_prime = lat.token_to_prime.get("fox")
    if fox_prime:
        print(f"\n  walk_up('fox' prime={fox_prime}):")
        for parent in lat.lattice.walk_up(fox_prime):
            node = lat.lattice.resolve(parent)
            print(f"    -> {parent} '{node.label}' [L{node.level}]")


if __name__ == "__main__":
    main()
