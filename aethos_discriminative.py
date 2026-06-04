"""
Discriminative heavy anchor training for AETHOS retrieval.

Heavy anchors are rare word-pair composites (p1 × p2) weighted by their
actual discrimination power — how few docs contain them and how often they
lead to correct retrievals.

Core insight
------------
Phrase composites (p1 × p2) should be weighted by discrimination power:
  - How few docs contain them (rarity → high discrimination)
  - How often they lead to correct retrievals (accuracy via training)

The current system uses uniform weights, causing regressions on rare
technical queries. This module learns per-composite weights from qrels.

Prime strategy: pool primes preferred; intersection primes used as fallback.
Intersection prime collisions are tolerated — the learned_weight=0.0 guard
means untrained anchors emit no signal, and false-positive composites
accumulate wrong_count → learned_weight stays near 0.

Usage
-----
    anchor_idx = build_heavy_anchor_index(registry, doc_tokens, doc_freq)
    n_trained = train_on_qrels(anchor_idx, queries, qrels_train, ...)
    # Once per query:
    qac = query_anchor_composites(profile.word_set, anchor_idx, registry, idf=profile.idf)
    # Once per candidate doc:
    score += score_with_heavy_anchors(qac, doc_id, anchor_idx)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from aethos_phrase_composite import word_prime, word_prime_or_intersection
from aethos_promotion import is_stopword
from aethos_tokenize import tokenize_words


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class HeavyAnchor:
    """
    A rare word-pair composite with learned discrimination weight.

    ``composite``    : p1 × p2 (pool primes preferred; intersection primes as fallback)
    ``discrimination``: 1.0 / len(doc_ids) — high means rare = discriminative
    ``correct_count`` : times this composite led to a correct top-K retrieval
    ``wrong_count``   : times this composite led to a wrong top-K retrieval
    """

    composite: int
    prime_a: int
    prime_b: int
    word_a: str
    word_b: str
    doc_ids: frozenset[str]
    discrimination: float
    correct_count: int = 0
    wrong_count: int = 0

    @property
    def learned_weight(self) -> float:
        """Blend of rarity and observed accuracy.

        Before any training: returns 0.0 (no evidence → no signal).
        After training: discrimination × Laplace-smoothed accuracy × 1.0.
        Anchors with more wrong than correct observations get low/zero weight.
        """
        total = self.correct_count + self.wrong_count
        if total == 0:
            return 0.0  # untrained — no evidence, emit no signal
        # Laplace-smoothed accuracy: avoids 0/1 extremes after single observation
        accuracy = (self.correct_count + 1) / (total + 2)
        return self.discrimination * accuracy * 1.0


@dataclass
class HeavyAnchorIndex:
    """
    Pre-built lookup structures for O(1)-per-doc heavy anchor scoring.

    ``anchors``        : composite → HeavyAnchor
    ``doc_to_anchors`` : doc_id → frozenset of composites in that doc
    ``prime_to_anchors``: prime → set of composites containing it
    """

    anchors: dict[int, HeavyAnchor] = field(default_factory=dict)
    doc_to_anchors: dict[str, frozenset[int]] = field(default_factory=dict)
    prime_to_anchors: dict[int, set[int]] = field(default_factory=dict)

    @property
    def n_anchors(self) -> int:
        return len(self.anchors)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def build_heavy_anchor_index(
    registry,
    doc_tokens: dict[str, frozenset[str]],
    doc_freq: dict[str, int],
    *,
    max_doc_count: int = 10,
    rarity_threshold: float = 0.02,
) -> HeavyAnchorIndex:
    """
    Scan every doc's word pairs; keep only rare×rare composites.

    A composite is "heavy" (discriminative) if:
      1. Both words appear in < rarity_threshold fraction of docs
         (rare words — common words yield noisy composites)
      2. The composite itself appears in ≤ max_doc_count docs
         (rare composite — otherwise not discriminative)
      3. Both words have a prime (pool prime preferred; intersection prime as fallback)

    Parameters
    ----------
    max_doc_count     : composites appearing in more docs are discarded
    rarity_threshold  : word must appear in < this fraction of docs

    Intersection prime collisions (two words mapping to the same prime) are
    handled by the p1==p2 skip guard.  Untrained anchors contribute 0.0 via
    learned_weight, so collision artifacts don't affect ranking until training
    confirms the composite is genuinely discriminative.
    """
    n_docs = len(doc_tokens)
    if n_docs == 0:
        return HeavyAnchorIndex()

    max_df = max(1, int(rarity_threshold * n_docs))

    # Cache prime lookups (word → int); use pool prime when available, else intersection
    prime_cache: dict[str, int] = {}

    def get_prime(w: str) -> int:
        if w not in prime_cache:
            prime_cache[w] = word_prime_or_intersection(w, registry)
        return prime_cache[w]

    # Pass 1: accumulate composite → docs, composite → (pa, pb, wa, wb)
    composite_docs: dict[int, set[str]] = defaultdict(set)
    composite_info: dict[int, tuple[int, int, str, str]] = {}

    for did, tokens in doc_tokens.items():
        # Filter to rare content words (pool prime preferred; intersection prime as fallback)
        rare_words: list[tuple[str, int]] = []
        for w in sorted(tokens):       # sorted → deterministic pair ordering
            if len(w) < 4 or not w.isalpha():
                continue
            if doc_freq.get(w, 0) >= max_df:
                continue               # too common — skip
            rare_words.append((w, get_prime(w)))

        seen_this_doc: set[int] = set()
        for (w1, p1), (w2, p2) in combinations(rare_words, 2):
            if p1 == p2:
                continue
            # Canonicalize: smaller prime first
            pa, pb = (p1, p2) if p1 < p2 else (p2, p1)
            wa, wb = (w1, w2) if p1 < p2 else (w2, w1)
            comp = pa * pb
            if comp in seen_this_doc:
                continue
            seen_this_doc.add(comp)
            composite_docs[comp].add(did)
            if comp not in composite_info:
                composite_info[comp] = (pa, pb, wa, wb)

    # Pass 2: filter to ≤ max_doc_count, build index structures
    anchors: dict[int, HeavyAnchor] = {}
    doc_to_mut: dict[str, set[int]] = defaultdict(set)
    prime_to_mut: dict[int, set[int]] = defaultdict(set)

    for comp, docs in composite_docs.items():
        if len(docs) > max_doc_count:
            continue
        info = composite_info.get(comp)
        if info is None:
            continue
        pa, pb, wa, wb = info
        anchor = HeavyAnchor(
            composite=comp,
            prime_a=pa,
            prime_b=pb,
            word_a=wa,
            word_b=wb,
            doc_ids=frozenset(docs),
            discrimination=1.0 / len(docs),
        )
        anchors[comp] = anchor
        for did in docs:
            doc_to_mut[did].add(comp)
        prime_to_mut[pa].add(comp)
        prime_to_mut[pb].add(comp)

    return HeavyAnchorIndex(
        anchors=anchors,
        doc_to_anchors={did: frozenset(cs) for did, cs in doc_to_mut.items()},
        prime_to_anchors=dict(prime_to_mut),
    )


# ---------------------------------------------------------------------------
# Discriminative rarity score (IDF-like)
# ---------------------------------------------------------------------------

def discriminative_score(
    composite: int,
    anchor_idx: HeavyAnchorIndex,
    n_docs: int,
) -> float:
    """
    IDF-like rarity score for a composite.

    discriminative_score = log(N / df) / log(N)

    Range [0, 1].  Composites in 1 doc → score near 1.0.
    """
    anchor = anchor_idx.anchors.get(composite)
    if anchor is None or n_docs <= 1:
        return 0.0
    df = len(anchor.doc_ids)
    if df <= 0:
        return 0.0
    return math.log(n_docs / df) / math.log(n_docs)


# ---------------------------------------------------------------------------
# Per-query anchor precomputation (call once per query, not once per doc)
# ---------------------------------------------------------------------------

def query_anchor_composites(
    query_words,
    anchor_idx: HeavyAnchorIndex,
    registry,
    *,
    idf: dict[int, float] | None = None,
) -> dict[int, float]:
    """
    O(Q²) once per query — returns {composite: effective_weight} for every
    trained heavy-anchor composite that could fire on this query.

    effective_weight = anchor.learned_weight × mean_idf_of_pair
    (If ``idf`` is not supplied, idf_scale defaults to 1.0.)

    A typical 8-word query has C(8,2)=28 word pairs → at most 28 composites.
    The per-doc path then checks only these composites instead of iterating
    the full anchor list, dropping per-doc cost from O(anchor_count) to
    O(|composites|) ≈ O(0–28).
    """
    content = [w for w in query_words if len(w) >= 4 and w.isalpha()]
    if len(content) < 2:
        return {}

    prime_cache: dict[str, int] = {}

    def get_prime(w: str) -> int:
        if w not in prime_cache:
            prime_cache[w] = word_prime_or_intersection(w, registry)
        return prime_cache[w]

    q_primes: list[tuple[str, int]] = [(w, get_prime(w)) for w in sorted(content)]
    result: dict[int, float] = {}

    for (w1, p1), (w2, p2) in combinations(q_primes, 2):
        if p1 == p2:
            continue
        comp = min(p1, p2) * max(p1, p2)
        anchor = anchor_idx.anchors.get(comp)
        if anchor is None or anchor.learned_weight < 0.05:
            continue
        if idf is not None:
            idf_scale = (idf.get(w1, 1.0) + idf.get(w2, 1.0)) / 2.0
        else:
            idf_scale = 1.0
        result[comp] = anchor.learned_weight * idf_scale

    return result


# ---------------------------------------------------------------------------
# Per-document scoring (Signal 6)
# ---------------------------------------------------------------------------

def score_with_heavy_anchors(
    query_anchor_comps: dict[int, float],
    doc_id: str,
    anchor_idx: HeavyAnchorIndex,
) -> float:
    """
    Score a document using precomputed query anchor composites.

    O(|query_anchor_comps| × 1) per doc — both the outer iteration and the
    frozenset membership test are O(1).  Call ``query_anchor_composites``
    once per query to build ``query_anchor_comps`` (O(Q²)), then call this
    function once per candidate doc.

    Parameters
    ----------
    query_anchor_comps : precomputed {composite: effective_weight} dict —
        produced by ``query_anchor_composites()``.  Typically 0–28 entries.
    doc_id            : document to score.
    anchor_idx        : index with doc_to_anchors frozensets.
    """
    if not query_anchor_comps:
        return 0.0
    doc_comps = anchor_idx.doc_to_anchors.get(doc_id)
    if not doc_comps:
        return 0.0
    score = 0.0
    for comp, eff_weight in query_anchor_comps.items():
        if comp in doc_comps:
            score += eff_weight
    return score


# ---------------------------------------------------------------------------
# Training on qrels
# ---------------------------------------------------------------------------

def train_on_qrels(
    anchor_idx: HeavyAnchorIndex,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    registry,
    doc_freq: dict[str, int],
    n_docs: int,
    *,
    top_k: int = 100,
    rarity_threshold: float = 0.02,
    max_doc_count: int = 10,
) -> int:
    """
    Train anchor weights using relevance judgments (qrels supervision).

    For each training query:
      1. Resolve pool primes for query words
      2. Compute which indexed heavy anchors match the query
      3. Score all docs by sum of matching anchor.learned_weight
      4. Take top_k docs by this score
      5. For anchors matching this query:
           - In top_k AND gold → correct_count += 1
           - In top_k AND NOT gold → wrong_count += 1
      6. Scan gold docs for rare pairs not yet indexed → add them

    Returns the number of queries used for training.
    """
    if not anchor_idx.anchors:
        return 0

    max_df = max(1, int(rarity_threshold * n_docs))
    prime_cache: dict[str, int] = {}

    def get_prime(w: str) -> int:
        if w not in prime_cache:
            prime_cache[w] = word_prime_or_intersection(w, registry)
        return prime_cache[w]

    trained = 0

    for qid, gold_docs in qrels.items():
        if qid not in queries or not gold_docs:
            continue

        # --- step 1-3: query composites → score all docs ---
        q_words_raw = tokenize_words(queries[qid])
        q_primes: list[tuple[str, int]] = []
        for w in sorted(set(q_words_raw)):
            if len(w) < 4 or not w.isalpha():
                continue
            q_primes.append((w, get_prime(w)))

        if len(q_primes) >= 2:
            # Composites that are both query-relevant AND indexed
            query_comps: list[int] = []
            for (w1, p1), (w2, p2) in combinations(q_primes, 2):
                if p1 == p2:
                    continue
                comp = min(p1, p2) * max(p1, p2)
                if comp in anchor_idx.anchors:
                    query_comps.append(comp)

            if query_comps:
                # Score every doc by sum of matching anchor weights
                doc_scores: list[tuple[float, str]] = []
                for did in doc_ids:
                    doc_comps = anchor_idx.doc_to_anchors.get(did)
                    if not doc_comps:
                        continue
                    s = 0.0
                    for c in query_comps:
                        if c in doc_comps:
                            s += anchor_idx.anchors[c].learned_weight
                    if s > 0.0:
                        doc_scores.append((s, did))

                doc_scores.sort(key=lambda x: -x[0])
                top_set = {did for _, did in doc_scores[:top_k]}

                # Update correct / wrong counts for each matching anchor
                for comp in query_comps:
                    anchor = anchor_idx.anchors[comp]
                    for did in top_set:
                        if did in anchor.doc_ids:
                            if did in gold_docs:
                                anchor.correct_count += 1
                            else:
                                anchor.wrong_count += 1

        # --- step 6: discover new anchors from gold docs ---
        for gold_did in gold_docs:
            tokens = doc_tokens.get(gold_did)
            if tokens is None:
                continue

            rare_words: list[tuple[str, int]] = []
            for w in sorted(tokens):
                if len(w) < 4 or not w.isalpha():
                    continue
                if doc_freq.get(w, 0) >= max_df:
                    continue
                rare_words.append((w, get_prime(w)))

            seen_new: set[int] = set()
            for (w1, p1), (w2, p2) in combinations(rare_words, 2):
                if p1 == p2:
                    continue
                pa, pb = (p1, p2) if p1 < p2 else (p2, p1)
                wa, wb = (w1, w2) if p1 < p2 else (w2, w1)
                comp = pa * pb
                if comp in seen_new or comp in anchor_idx.anchors:
                    continue
                seen_new.add(comp)

                # Compute true doc set: docs where BOTH words appear
                # Efficient because rare words have small inv lists
                new_docs = frozenset(
                    did for did, toks in doc_tokens.items()
                    if wa in toks and wb in toks
                )
                if len(new_docs) == 0 or len(new_docs) > max_doc_count:
                    continue

                new_anchor = HeavyAnchor(
                    composite=comp,
                    prime_a=pa,
                    prime_b=pb,
                    word_a=wa,
                    word_b=wb,
                    doc_ids=new_docs,
                    discrimination=1.0 / len(new_docs),
                    correct_count=1,   # gold doc counts as one correct
                    wrong_count=0,
                )
                anchor_idx.anchors[comp] = new_anchor

                # Update lookup maps
                for did in new_docs:
                    existing = anchor_idx.doc_to_anchors.get(did, frozenset())
                    anchor_idx.doc_to_anchors[did] = existing | {comp}
                anchor_idx.prime_to_anchors.setdefault(pa, set()).add(comp)
                anchor_idx.prime_to_anchors.setdefault(pb, set()).add(comp)

        trained += 1

    return trained


# ---------------------------------------------------------------------------
# Multi-round convergence training loop
# ---------------------------------------------------------------------------

def _ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    """Local NDCG@K — avoids circular import with eval_beir."""
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    if not ideal:
        return 0.0
    return sum(1.0 / math.log2(r + 2) for r, d in enumerate(ranked[:k]) if d in rel) / ideal


def train_convergence_loop(
    anchor_idx: HeavyAnchorIndex,
    registry,
    phrase_idx,
    doc_tokens: dict[str, frozenset[str]],
    doc_tf: dict[str, dict[str, int]],
    doc_len: dict[str, int],
    avg_dl: float,
    doc_freq: dict[str, int],
    queries: dict[str, str],
    qrels_train: dict[str, dict[str, int]],
    qrels_eval: dict[str, dict[str, int]],
    doc_ids: list[str],
    hub_sigs,
    neighbor_map,
    sub_comp_idx,
    *,
    max_rounds: int = 8,
    convergence_threshold: float = 0.002,
    verbose: bool = True,
) -> list[float]:
    """
    Repeatedly train on qrels_train and evaluate on qrels_eval.

    Each round: call train_on_qrels (accumulates correct/wrong counts) then
    score all eval queries with the updated anchor weights.  Stop when the
    NDCG@10 improvement between consecutive rounds falls below
    convergence_threshold for one round.

    Optimisations:
    - Inverted index built once (not per-round) for candidate generation.
    - QueryProfiles precomputed once (IDF is static across rounds).
    - Only query_anchor_composites is recomputed per round (learned_weight changes).

    Returns NDCG@10 history (one entry per completed round).
    """
    from collections import defaultdict as _dd
    from aethos_hub_signature import rank_with_hub_signatures, build_query_profile
    from aethos_phrase_composite import query_phrase_composites as _qpc

    n_docs = len(doc_ids)

    # Build inverted index once for fast candidate generation
    _inv: dict[str, set[str]] = _dd(set)
    for did, tokens in doc_tokens.items():
        for w in tokens:
            _inv[w].add(did)
    inv: dict[str, set[str]] = dict(_inv)

    # Precompute query profiles — IDF depends only on doc_freq/n_docs (static)
    eval_qids = [q for q in qrels_eval if q in queries]
    profiles = {
        qid: build_query_profile(
            queries[qid], registry,
            neighbor_map=neighbor_map,
            doc_freq=doc_freq,
            n_docs=n_docs,
        )
        for qid in eval_qids
    }

    # Precompute phrase composites — phrase_idx is static across rounds
    q_phrase_map: dict[str, dict[int, float]] = {}
    if phrase_idx is not None:
        for qid in eval_qids:
            prof = profiles[qid]
            q_phrase_map[qid] = _qpc(prof.words, phrase_idx, registry, prof.idf)

    history: list[float] = []

    for round_num in range(1, max_rounds + 1):
        # --- Train: accumulate correct/wrong counts ---
        # Compound improvement works through brain loading: Run 2 starts with
        # non-zero anchor weights (loaded from Run 1 brain), so retrieval changes
        # → new correct/wrong signals → anchors refine further each run.
        n_train = train_on_qrels(
            anchor_idx,
            queries,
            qrels_train,
            doc_ids,
            doc_tokens,
            registry,
            doc_freq,
            n_docs,
        )

        # --- Eval: score all eval queries with updated anchor weights ---
        ndcgs: list[float] = []
        for qid in eval_qids:
            profile = profiles[qid]

            # Candidate generation via inverted index + neighbor expansion
            cand: set[str] = set()
            for w in profile.words:
                cand |= inv.get(w, set())
                for nb in neighbor_map.get(w, {}):
                    cand |= inv.get(nb, set())
            cands = list(cand) if cand else doc_ids

            # Recompute anchor composites each round (learned_weight changes)
            q_anchor = query_anchor_composites(
                list(profile.word_set), anchor_idx, registry, idf=profile.idf
            )

            ranked = rank_with_hub_signatures(
                profile, cands, hub_sigs, doc_ids,
                doc_tokens=doc_tokens,
                doc_tf=doc_tf,
                doc_len=doc_len,
                avg_dl=avg_dl,
                sub_comp_idx=sub_comp_idx,
                registry=registry,
                phrase_idx=phrase_idx,
                query_anchor_comps=q_anchor,
                query_phrase_comps=q_phrase_map.get(qid),
                anchor_idx=anchor_idx,
                top_k=100,
            )
            ndcgs.append(_ndcg_at_k(ranked, qrels_eval[qid], 10))

        ndcg = sum(ndcgs) / max(len(ndcgs), 1)
        history.append(ndcg)

        if verbose:
            print(
                f"  Round {round_num}: NDCG@10={ndcg:.4f} "
                f"(trained on {n_train} queries)",
                flush=True,
            )

        # Convergence: stop when last improvement < threshold
        if len(history) >= 2 and (history[-1] - history[-2]) < convergence_threshold:
            if verbose:
                print(f"  Converged after {round_num} rounds.", flush=True)
            break

    return history


# ---------------------------------------------------------------------------
# Discriminating intersection discovery
# ---------------------------------------------------------------------------

def discover_discriminating_intersections(
    anchor_idx: HeavyAnchorIndex,
    registry,
    queries: dict[str, str],
    qrels_train: dict[str, dict[str, int]],
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    doc_freq: dict[str, int],
    n_docs: int,
    hub_sigs,
    neighbor_map: dict,
    doc_tf: dict,
    doc_len: dict,
    avg_dl: float,
    sub_comp_idx,
    phrase_idx,
    *,
    max_doc_count: int = 5,
    rarity_threshold: float = 0.018,
    min_word_len: int = 4,
    max_new_anchors: int = 2000,
    verbose: bool = True,
) -> int:
    """
    Discover new discriminating anchors by analyzing training failures.

    For each training query where the correct doc is not ranked #1:
      1. Find words in gold doc NOT in query AND NOT in wrong docs → discriminating
      2. For each discriminating word paired with each query word:
         composite = word_prime(q_word) × word_prime(disc_word)
         If composite appears in ≤ max_doc_count docs → add as new anchor
      3. New anchors start with correct_count=1, wrong_count=0
         → learned_weight > 0 immediately on next training pass

    This creates composite anchors that UNIQUELY connect query vocabulary to
    gold doc vocabulary — the core of the "3-way intersection" insight.

    Returns the number of new anchors added.
    """
    from aethos_hub_signature import build_query_profile, rank_with_hub_signatures
    from aethos_phrase_composite import query_phrase_composites as _qpc

    train_qids = [q for q in qrels_train if q in queries]
    if not train_qids:
        return 0

    # Build inverted index
    inv: dict[str, set[str]] = {}
    from collections import defaultdict as _dd
    _inv = _dd(set)
    for did, tokens in doc_tokens.items():
        for w in tokens:
            _inv[w].add(did)
    inv = dict(_inv)

    # --- D4 word importance: count (query, gold_doc) pairs each word appears in ---
    # Words with high D4 are more discriminative: they appear in gold docs
    # across many different queries. Prioritizing them produces better anchors.
    d4_word_count: dict[str, int] = {}
    for qid in train_qids:
        for did in qrels_train.get(qid, {}):
            for w in doc_tokens.get(did, frozenset()):
                d4_word_count[w] = d4_word_count.get(w, 0) + 1
    # Normalize by IDF: rare words get boosted (more discriminative)
    d4_score: dict[str, float] = {
        w: cnt * math.log((n_docs + 1.0) / (doc_freq.get(w, 1) + 1.0))
        for w, cnt in d4_word_count.items()
    }

    # Build composite → doc_ids index for rarity checking
    comp_to_docs: dict[int, set[str]] = {}
    for comp, anchor in anchor_idx.anchors.items():
        comp_to_docs[comp] = set(anchor.doc_ids)

    new_count = 0

    for qid in train_qids:
        if new_count >= max_new_anchors:
            break

        gold_docs = set(qrels_train[qid].keys())
        if not gold_docs:
            continue

        # Build query profile
        profile = build_query_profile(
            queries[qid], registry,
            neighbor_map=neighbor_map,
            doc_freq=doc_freq, n_docs=n_docs,
        )
        if not profile.word_set:
            continue

        # Quick rank to find wrong docs
        cands: set[str] = set()
        for w in profile.words:
            cands |= inv.get(w, set())
        for w in profile.word_set:
            for nb in neighbor_map.get(w, {}):
                cands |= inv.get(nb, set())
        if not cands:
            cands = set(doc_ids)

        q_anchor = query_anchor_composites(
            list(profile.word_set), anchor_idx, registry, idf=profile.idf
        )
        q_phrase = _qpc(profile.words, phrase_idx, registry, profile.idf) if phrase_idx else {}

        ranked = rank_with_hub_signatures(
            profile, list(cands), hub_sigs, doc_ids,
            doc_tokens=doc_tokens, doc_tf=doc_tf, doc_len=doc_len, avg_dl=avg_dl,
            sub_comp_idx=sub_comp_idx, registry=registry, phrase_idx=phrase_idx,
            query_anchor_comps=q_anchor, query_phrase_comps=q_phrase,
            anchor_idx=anchor_idx, top_k=20,
        )

        ranked_set = set(ranked[:20])
        wrong_docs = ranked_set - gold_docs

        # For each gold doc, find discriminating words
        for gold_id in gold_docs:
            gold_tokens = doc_tokens.get(gold_id, frozenset())
            if not gold_tokens:
                continue

            # Gold rank — skip if already at top
            gold_rank = next((i + 1 for i, d in enumerate(ranked) if d == gold_id), 999)
            if gold_rank == 1:
                continue  # already optimal

            # Collect wrong doc tokens
            wrong_tokens: set[str] = set()
            for wd in wrong_docs:
                wrong_tokens |= doc_tokens.get(wd, frozenset())

            # Discriminating words: in gold, not in query, not in wrong docs
            # Also not too common (must be rare)
            disc_words = []
            for w in gold_tokens:
                if (len(w) >= min_word_len
                        and not w in profile.word_set
                        and not w in wrong_tokens
                        and doc_freq.get(w, n_docs) / n_docs < rarity_threshold):
                    disc_words.append(w)

            if not disc_words:
                continue

            # Sort discriminating words by D4 score (high = appears in more gold docs
            # across training queries = more likely to produce useful anchors).
            disc_words.sort(key=lambda w: -d4_score.get(w, 0.0))

            # Build new composites: query_word × discriminating_word
            for q_word in profile.word_set:
                if len(q_word) < min_word_len:
                    continue
                p_q = word_prime(q_word, registry) or word_prime_or_intersection(q_word, registry)
                if not p_q:
                    continue

                for d_word in disc_words[:8]:  # cap per gold doc
                    p_d = word_prime(d_word, registry) or word_prime_or_intersection(d_word, registry)
                    if not p_d or p_d == p_q:
                        continue

                    comp = min(p_q, p_d) * max(p_q, p_d)
                    if comp in anchor_idx.anchors:
                        # Already exists — just mark as correct
                        anchor_idx.anchors[comp].correct_count += 1
                        continue

                    # Count docs with this composite
                    docs_with_comp: set[str] = set()
                    for did in inv.get(q_word, set()) & inv.get(d_word, set()):
                        docs_with_comp.add(did)

                    if not docs_with_comp or len(docs_with_comp) > max_doc_count:
                        continue

                    # --- Coherence gate ---
                    # The docs sharing this composite should be semantically related,
                    # not coincidental co-occurrences. Check that they share at least
                    # 30% of their non-stopword vocabulary (vocabulary overlap coherence).
                    if len(docs_with_comp) >= 2:
                        doc_list = list(docs_with_comp)
                        tok_a = {w for w in doc_tokens.get(doc_list[0], frozenset())
                                 if len(w) >= 3 and not is_stopword(w)}
                        tok_b = {w for w in doc_tokens.get(doc_list[1], frozenset())
                                 if len(w) >= 3 and not is_stopword(w)}
                        if tok_a and tok_b:
                            overlap = len(tok_a & tok_b) / min(len(tok_a), len(tok_b))
                            if overlap < 0.15:
                                continue  # incoherent — likely noise

                    # Rare and useful — add as new anchor with initial correct evidence
                    disc = 1.0 - len(docs_with_comp) / n_docs
                    new_anchor = HeavyAnchor(
                        composite=comp,
                        prime_a=min(p_q, p_d),
                        prime_b=max(p_q, p_d),
                        word_a=q_word if p_q <= p_d else d_word,
                        word_b=d_word if p_q <= p_d else q_word,
                        doc_ids=frozenset(docs_with_comp),
                        discrimination=disc,
                        correct_count=1,   # starts pre-trained
                        wrong_count=0,
                    )
                    anchor_idx.anchors[comp] = new_anchor

                    # Update reverse maps
                    for did in docs_with_comp:
                        existing = set(anchor_idx.doc_to_anchors.get(did, frozenset()))
                        existing.add(comp)
                        anchor_idx.doc_to_anchors[did] = frozenset(existing)
                    anchor_idx.prime_to_anchors.setdefault(min(p_q, p_d), set()).add(comp)
                    anchor_idx.prime_to_anchors.setdefault(max(p_q, p_d), set()).add(comp)

                    new_count += 1
                    if new_count >= max_new_anchors:
                        break

                if new_count >= max_new_anchors:
                    break

    if verbose:
        print(f"  discriminating intersections: {new_count} new anchors added", flush=True)
    return new_count


def discover_meta_intersections(
    anchor_idx: HeavyAnchorIndex,
    doc_tokens: dict[str, frozenset[str]],
    doc_freq: dict[str, int],
    n_docs: int,
    *,
    max_doc_count: int = 3,
    coherence_threshold: float = 0.20,
    max_new: int = 500,
    verbose: bool = True,
) -> int:
    """
    Meta-intersections: composites of TWO existing anchor composites.

    When two separate anchors (A and B) both fire on the same small set of docs,
    their product A×B is an ultra-rare composite that represents a deeper concept —
    the intersection of both discriminating pairs. This is exactly Steps 25-28
    from the architecture spec: promoted primes at Depth 3+.

    Only anchors with learned_weight > 0 (trained positive anchors) are used.
    Coherence gate: docs sharing both composites must share ≥ coherence_threshold
    of their vocabulary (same topic check).

    Returns the number of new meta-anchors added.
    """
    from aethos_promotion import is_stopword

    # Only use strongly trained positive anchors as seeds
    positive_anchors = [
        a for a in anchor_idx.anchors.values()
        if a.learned_weight >= 0.5 and len(a.doc_ids) <= max_doc_count
    ]
    if len(positive_anchors) < 2:
        return 0

    # Index by doc coverage for fast intersection
    doc_to_anchors_list: dict[str, list] = {}
    for anc in positive_anchors:
        for did in anc.doc_ids:
            doc_to_anchors_list.setdefault(did, []).append(anc)

    new_count = 0
    seen_meta: set[int] = set()

    # Find pairs of anchors that share the same doc(s)
    for did, anchors_in_doc in doc_to_anchors_list.items():
        if len(anchors_in_doc) < 2:
            continue

        from itertools import combinations as _comb
        for a1, a2 in _comb(anchors_in_doc[:10], 2):  # cap to avoid O(n²) explosion
            if new_count >= max_new:
                break

            # Meta composite = product of the two component composites
            meta_comp = a1.composite * a2.composite
            if meta_comp in anchor_idx.anchors or meta_comp in seen_meta:
                continue
            seen_meta.add(meta_comp)

            # Docs that have BOTH anchors
            docs_with_both = a1.doc_ids & a2.doc_ids
            if not docs_with_both or len(docs_with_both) > max_doc_count:
                continue

            # Coherence gate: docs must share vocabulary
            if len(docs_with_both) >= 2:
                doc_list = list(docs_with_both)
                tok_a = {w for w in doc_tokens.get(doc_list[0], frozenset())
                         if len(w) >= 3 and not is_stopword(w)}
                tok_b = {w for w in doc_tokens.get(doc_list[1], frozenset())
                         if len(w) >= 3 and not is_stopword(w)}
                if tok_a and tok_b:
                    overlap = len(tok_a & tok_b) / min(len(tok_a), len(tok_b))
                    if overlap < coherence_threshold:
                        continue

            disc = 1.0 - len(docs_with_both) / n_docs
            meta_anchor = HeavyAnchor(
                composite=meta_comp,
                prime_a=a1.composite,
                prime_b=a2.composite,
                word_a=f"{a1.word_a}×{a1.word_b}",
                word_b=f"{a2.word_a}×{a2.word_b}",
                doc_ids=docs_with_both,
                discrimination=disc,
                correct_count=max(a1.correct_count, a2.correct_count),
                wrong_count=max(a1.wrong_count, a2.wrong_count),
            )
            anchor_idx.anchors[meta_comp] = meta_anchor
            for d in docs_with_both:
                existing = set(anchor_idx.doc_to_anchors.get(d, frozenset()))
                existing.add(meta_comp)
                anchor_idx.doc_to_anchors[d] = frozenset(existing)
            new_count += 1

        if new_count >= max_new:
            break

    if verbose:
        print(f"  meta-intersections: {new_count} new depth-3 anchors", flush=True)
    return new_count


def train_negative_anchors(
    anchor_idx: HeavyAnchorIndex,
    registry,
    queries: dict[str, str],
    qrels_train: dict[str, dict[str, int]],
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    doc_freq: dict[str, int],
    n_docs: int,
    hub_sigs,
    neighbor_map: dict,
    doc_tf: dict,
    doc_len: dict,
    avg_dl: float,
    sub_comp_idx,
    phrase_idx,
    *,
    max_doc_count: int = 5,
    rarity_threshold: float = 0.018,
    min_word_len: int = 4,
    min_wrong_count: int = 3,
    max_new_negatives: int = 500,
    verbose: bool = True,
) -> int:
    """
    Build NEGATIVE anchors — composites that reliably predict irrelevant docs.

    For each training query where a wrong doc is in top-10:
      1. Find words in wrong doc NOT in gold docs AND in multiple wrong results
      2. Build composite: query_word × wrong_discriminating_word
      3. If this composite fires consistently on wrong docs, add it with
         correct_count=0, wrong_count=min_wrong_count
      → learned_weight ≈ 0 immediately, won't pollute scoring

    These anchors suppress composites associated with wrong doc vocabulary.
    """
    from aethos_hub_signature import build_query_profile, rank_with_hub_signatures
    from aethos_phrase_composite import query_phrase_composites as _qpc
    from collections import defaultdict as _dd

    train_qids = [q for q in qrels_train if q in queries]
    if not train_qids:
        return 0

    inv: dict[str, set[str]] = {}
    _inv = _dd(set)
    for did, tokens in doc_tokens.items():
        for w in tokens:
            _inv[w].add(did)
    inv = dict(_inv)

    # Track which composites appear in wrong docs across multiple queries
    composite_wrong_count: dict[int, int] = {}
    composite_info: dict[int, tuple[int, int, str, str, frozenset]] = {}

    new_count = 0

    for qid in train_qids:
        if new_count >= max_new_negatives:
            break

        gold_docs = set(qrels_train[qid].keys())
        if not gold_docs:
            continue

        profile = build_query_profile(
            queries[qid], registry,
            neighbor_map=neighbor_map,
            doc_freq=doc_freq, n_docs=n_docs,
        )
        if not profile.word_set:
            continue

        cands: set[str] = set()
        for w in profile.words:
            cands |= inv.get(w, set())
        if not cands:
            continue

        q_anchor = query_anchor_composites(
            list(profile.word_set), anchor_idx, registry, idf=profile.idf
        )
        q_phrase = _qpc(profile.words, phrase_idx, registry, profile.idf) if phrase_idx else {}

        ranked = rank_with_hub_signatures(
            profile, list(cands), hub_sigs, doc_ids,
            doc_tokens=doc_tokens, doc_tf=doc_tf, doc_len=doc_len, avg_dl=avg_dl,
            sub_comp_idx=sub_comp_idx, registry=registry, phrase_idx=phrase_idx,
            query_anchor_comps=q_anchor, query_phrase_comps=q_phrase,
            anchor_idx=anchor_idx, top_k=10,
        )

        wrong_top10 = [d for d in ranked[:10] if d not in gold_docs]
        if not wrong_top10:
            continue

        gold_tokens: set[str] = set()
        for gd in gold_docs:
            gold_tokens |= doc_tokens.get(gd, frozenset())

        for wrong_id in wrong_top10[:3]:
            wrong_tokens = doc_tokens.get(wrong_id, frozenset())

            # Words unique to wrong docs: in wrong, NOT in gold, rare
            bad_words = [
                w for w in wrong_tokens
                if (len(w) >= min_word_len
                    and w not in gold_tokens
                    and w not in profile.word_set
                    and doc_freq.get(w, n_docs) / n_docs < rarity_threshold)
            ]
            if not bad_words:
                continue

            for q_word in profile.word_set:
                if len(q_word) < min_word_len:
                    continue
                p_q = word_prime(q_word, registry) or word_prime_or_intersection(q_word, registry)
                if not p_q:
                    continue

                for b_word in bad_words[:4]:
                    p_b = word_prime(b_word, registry) or word_prime_or_intersection(b_word, registry)
                    if not p_b or p_b == p_q:
                        continue

                    comp = min(p_q, p_b) * max(p_q, p_b)

                    # Skip if it already exists and is positive
                    if comp in anchor_idx.anchors:
                        existing = anchor_idx.anchors[comp]
                        if existing.correct_count > existing.wrong_count:
                            continue  # already a positive — don't demote
                        existing.wrong_count += 1
                        continue

                    composite_wrong_count[comp] = composite_wrong_count.get(comp, 0) + 1
                    if comp not in composite_info:
                        docs_with_comp = frozenset(
                            inv.get(q_word, set()) & inv.get(b_word, set())
                        )
                        if docs_with_comp and len(docs_with_comp) <= max_doc_count:
                            disc = 1.0 - len(docs_with_comp) / n_docs
                            composite_info[comp] = (
                                min(p_q, p_b), max(p_q, p_b),
                                q_word if p_q <= p_b else b_word,
                                b_word if p_q <= p_b else q_word,
                                docs_with_comp,
                            )

    # Promote consistently wrong composites to negative anchors
    for comp, wrong_cnt in composite_wrong_count.items():
        if wrong_cnt < min_wrong_count or comp in anchor_idx.anchors:
            continue
        info = composite_info.get(comp)
        if not info:
            continue
        pa, pb, wa, wb, docs = info
        neg_anchor = HeavyAnchor(
            composite=comp, prime_a=pa, prime_b=pb, word_a=wa, word_b=wb,
            doc_ids=docs,
            discrimination=1.0 - len(docs) / n_docs,
            correct_count=0,
            wrong_count=wrong_cnt,  # pre-trained negative
        )
        anchor_idx.anchors[comp] = neg_anchor
        for did in docs:
            existing = set(anchor_idx.doc_to_anchors.get(did, frozenset()))
            existing.add(comp)
            anchor_idx.doc_to_anchors[did] = frozenset(existing)
        anchor_idx.prime_to_anchors.setdefault(pa, set()).add(comp)
        anchor_idx.prime_to_anchors.setdefault(pb, set()).add(comp)
        new_count += 1
        if new_count >= max_new_negatives:
            break

    if verbose:
        print(f"  negative anchors: {new_count} added (wrong_count pre-trained)", flush=True)
    return new_count


# ---------------------------------------------------------------------------
# λ calibration — grid-search LAMBDA_COORD × LAMBDA_NEIGHBOR on train qrels
# ---------------------------------------------------------------------------

def calibrate_signal_weights(
    hub_sigs: dict,
    doc_tokens: dict,
    doc_tf: dict,
    doc_len: dict,
    avg_dl: float,
    sub_comp_idx,
    anchor_idx: "HeavyAnchorIndex | None",
    phrase_idx,
    registry,
    queries: dict[str, str],
    qrels_train: dict[str, dict[str, int]],
    doc_ids: list[str],
    neighbor_map: dict,
    doc_freq: dict[str, int],
    n_docs: int,
    *,
    lambda_coord_grid: tuple[float, ...] = (0.2, 0.5, 1.0, 1.5),
    lambda_neighbor_grid: tuple[float, ...] = (0.1, 0.3, 0.5),
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Grid-search LAMBDA_COORD × LAMBDA_NEIGHBOR on training qrels.

    Returns (best_lambda_coord, best_lambda_neighbor) — the values that
    maximise NDCG@10 on the training set.  Modifies the module-level
    constants in ``aethos_hub_signature`` for the duration of the calling
    process.

    Complexity: len(lambda_coord_grid) × len(lambda_neighbor_grid) eval passes,
    each O(|train_qids| × |doc_ids|).  For SciFact (809 train queries,
    4 × 3 = 12 grid points) this is fast.
    """
    import aethos_hub_signature as _hs
    from aethos_hub_signature import build_query_profile, rank_with_hub_signatures

    train_qids = [q for q in qrels_train if q in queries]
    if not train_qids:
        if verbose:
            print("  calibrate_signal_weights: no train queries, skipping.", flush=True)
        return _hs.LAMBDA_COORD, _hs.LAMBDA_NEIGHBOR

    # Pre-build query profiles (IDF static across grid)
    profiles: dict[str, object] = {}
    q_phrase_map: dict[str, dict] = {}
    q_anchor_map: dict[str, dict] = {}
    inv: dict[str, set] = {}
    from collections import defaultdict
    for did, toks in doc_tokens.items():
        for t in toks:
            inv.setdefault(t, set()).add(did)

    for qid in train_qids:
        p = build_query_profile(
            queries[qid], registry,
            neighbor_map=neighbor_map,
            doc_freq=doc_freq, n_docs=n_docs,
        )
        profiles[qid] = p
        if phrase_idx is not None:
            from aethos_phrase_composite import query_phrase_composites, PHRASE_WEIGHT
            if PHRASE_WEIGHT > 0:
                q_phrase_map[qid] = query_phrase_composites(
                    p.words, phrase_idx, registry, p.idf
                )
        if anchor_idx is not None:
            q_anchor_map[qid] = query_anchor_composites(
                list(p.word_set), anchor_idx, registry, idf=p.idf
            )

    best_ndcg = -1.0
    best_lc = _hs.LAMBDA_COORD
    best_ln = _hs.LAMBDA_NEIGHBOR

    for lc in lambda_coord_grid:
        for ln in lambda_neighbor_grid:
            _hs.LAMBDA_COORD = lc
            _hs.LAMBDA_NEIGHBOR = ln

            ndcgs: list[float] = []
            for qid in train_qids:
                p = profiles[qid]
                cand: set[str] = set()
                for w in p.words:
                    cand |= inv.get(w, set())
                    for nb in neighbor_map.get(w, {}):
                        cand |= inv.get(nb, set())
                cands = list(cand) if cand else doc_ids

                ranked = rank_with_hub_signatures(
                    p, cands, hub_sigs, doc_ids,
                    doc_tokens=doc_tokens,
                    doc_tf=doc_tf, doc_len=doc_len, avg_dl=avg_dl,
                    sub_comp_idx=sub_comp_idx,
                    registry=registry,
                    phrase_idx=phrase_idx,
                    query_anchor_comps=q_anchor_map.get(qid),
                    query_phrase_comps=q_phrase_map.get(qid),
                    anchor_idx=anchor_idx,
                    top_k=100,
                )
                ndcgs.append(_ndcg_at_k(ranked, qrels_train[qid], 10))

            ndcg = sum(ndcgs) / max(len(ndcgs), 1)
            if verbose:
                print(
                    f"  grid λ_coord={lc}  λ_neighbor={ln}  NDCG@10={ndcg:.4f}",
                    flush=True,
                )
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_lc = lc
                best_ln = ln

    # Lock in the best values
    _hs.LAMBDA_COORD = best_lc
    _hs.LAMBDA_NEIGHBOR = best_ln
    if verbose:
        print(
            f"  calibrated: λ_coord={best_lc}  λ_neighbor={best_ln}  "
            f"best_NDCG@10={best_ndcg:.4f}",
            flush=True,
        )
    return best_lc, best_ln
