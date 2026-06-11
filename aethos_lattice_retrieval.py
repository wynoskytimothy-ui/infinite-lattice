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

    def __init__(self, token_pool_size: int = 20000, doc_pool_size: int = 100000):
        # Doc primes still come from chain_primes; word "primes" are composites of
        # letter primes (text_icn semantics), so no separate token pool needed.
        all_primes = chain_primes(doc_pool_size + 1000)
        self._doc_primes = all_primes[:doc_pool_size]

        self.lattice = RecursiveLattice()

        # word -> ICN composite (unique anagram-class address; encodes morphology
        # via shared letter prime factors).
        self.token_to_prime: dict[str, int] = {}
        self.prime_to_token: dict[int, str] = {}
        # Cache of letter-prime factor sets per word for morph bonus scoring
        self._word_factors: dict[int, frozenset[int]] = {}
        self.token_doc_count: Counter = Counter()

        self.doc_id_to_prime: dict[str, int] = {}
        self.prime_to_doc_id: dict[int, str] = {}
        # Per-doc token counts for BM25-style tf saturation in scoring
        self.doc_token_counts: dict[str, Counter] = {}
        self.doc_lengths: dict[str, int] = {}
        self._avg_doc_len: float = 1.0

        self._next_doc_idx = 0

    def _allocate_token_prime(self, token: str) -> int:
        """Word -> letter-prime composite WITH MULTIPLICITY (proper text_icn).

        Audit of failing SciFact queries showed unique-set composite collapses
        anagrams catastrophically:
            properties = prosite    (same unique letter set)
            mortality  = immortality (same unique letter set)
            perinatal  = intraparietal
            pge        = gep
        Multiplying with multiplicity (m^1, t^2, etc) gives distinct composites
        per FTA. Morphological factor set (for future morph scoring) still uses
        the unique-prime set, kept in _word_factors.
        """
        if token in self.token_to_prime:
            return self.token_to_prime[token]
        # FTA composite: multiply once PER occurrence (not per unique letter)
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

    def build_from_corpus(self, corpus: dict[str, str]):
        """Two-pass: filter by document frequency, then assign primes by rarity."""
        # Pass 1: document frequency (how many docs each token appears in)
        df_counts: Counter = Counter()
        for text in corpus.values():
            df_counts.update(set(tokenize(text)))

        n_docs = max(len(corpus), 1)
        max_df = max(int(n_docs * self.MAX_DF_RATIO), 2)
        # Build the kept-vocabulary set: not too common, not too rare
        self._kept_tokens: set[str] = {
            t for t, df in df_counts.items()
            if self.MIN_DF <= df <= max_df
        }

        # Assign primes to kept tokens only - common-but-kept first, rare last
        kept_sorted = sorted(
            (t for t in self._kept_tokens),
            key=lambda t: -df_counts[t],
        )
        for token in kept_sorted:
            self._allocate_token_prime(token)

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

        # Build (prime, tf) pairs for unique tokens in the kept vocabulary.
        prime_tf_pairs: list[tuple[int, int]] = []
        for t, c in tf_counts.items():
            if kept_vocab is not None and t not in kept_vocab:
                continue
            p = self.token_to_prime.get(t)
            if p is None:
                continue
            prime_tf_pairs.append((p, c))
            self.token_doc_count[t] += 1
        # Cap to top-K rarest if needed
        if len(prime_tf_pairs) > self.MAX_CHAIN_PER_DOC:
            prime_tf_pairs.sort(key=lambda pt: -pt[0])
            prime_tf_pairs = prime_tf_pairs[:self.MAX_CHAIN_PER_DOC]
        prime_tf_pairs.sort(key=lambda pt: pt[0])
        chain = tuple(p for p, _ in prime_tf_pairs)
        if not chain:
            return None
        # Store tf only for kept primes
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
    LAMBDA_MORPH: float = 0.0       # disabled: raw 26-letter morph is noisy
    MORPH_MIN_SHARED: int = 4
    MORPH_MIN_JACCARD: float = 0.7
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
        for t in tokens:
            p = self.token_to_prime.get(t)
            if p is not None:
                query_primes_set.add(p)
        if not query_primes_set:
            return []
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

            # 1) BM25 backbone with IDF on shared word composites
            doc_len = len(doc_chain)
            doc_counts = self.doc_token_counts.get(doc_id)
            length_norm = 1.0 - self.BM25_B + self.BM25_B * (doc_len / max(self._avg_doc_len, 1.0))
            bm25_term = 0.0
            for anchor in shared:
                tf = 1
                if doc_counts is not None:
                    tf = doc_counts.get(self.prime_to_token[anchor], 1)
                tf_sat = (tf * (self.BM25_K1 + 1)) / (tf + self.BM25_K1 * length_norm)
                bm25_term += idf_for[anchor] * tf_sat

            # 2) Lattice morphology bonus: words that DON'T exact-match but
            #    share enough letter prime factors with a query word are scored
            #    as soft matches. This is stemming-free morphology.
            morph_score = 0.0
            if self.LAMBDA_MORPH > 0:
                unmatched_doc = doc_set - query_primes_set
                for q_comp in query_primes:
                    q_factors = self._word_factors.get(q_comp)
                    if q_factors is None:
                        continue
                    best_jaccard = 0.0
                    for d_comp in unmatched_doc:
                        d_factors = self._word_factors.get(d_comp)
                        if d_factors is None:
                            continue
                        inter = q_factors & d_factors
                        if len(inter) < self.MORPH_MIN_SHARED:
                            continue
                        union = q_factors | d_factors
                        j = len(inter) / len(union)
                        if j >= self.MORPH_MIN_JACCARD and j > best_jaccard:
                            best_jaccard = j
                    if best_jaccard > 0:
                        # Only credit each query word once at its best morph kin
                        morph_score += best_jaccard * idf_for[q_comp]

            score = bm25_term + self.LAMBDA_MORPH * morph_score
            scored.append((doc_id, score))

        scored.sort(key=lambda x: -x[1])

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
