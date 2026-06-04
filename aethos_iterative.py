"""
Iterative corpus passes — each pass builds deeper correlations.

Pass 1 (Foundation):
    Direct co-occurrences → L2 subword candidates → L3 word promotions
    Result: keyword-level retrieval, most words intersection-only

Pass 2 (Structure):
    Same corpus, but now L2 subwords are promoted.
    Words get updated parent_primes using L2 pool primes.
    Phrase composites (L4) are now genuinely unique by FTA.
    Bridge detection: pairs of phrase composites with Jaccard ≥ threshold.
    Result: phrase-level retrieval, semantic disambiguation

Pass 3 (Semantics):
    Meta-bridges: bridges between bridges.
    Cross-domain connections emerge.
    Cluster boundaries stabilize.
    Result: domain-level retrieval

Each pass discovers fewer new things but at greater depth.
Convergence criteria: < 1% new discoveries relative to previous pass.

Why this matters for quality vs fast_ingest
-------------------------------------------
fast_ingest=True skips subword observation → most L3 words stay intersection-only
→ phrase composites use intersection primes → collisions → signal noise.

Multi-pass fixes this WITHOUT re-ingesting all documents:
  Pass 1: fast_ingest=True (build word/co-occurrence stats)
  Rebuild subword stats from vocabulary (aethos_subword_composite.rebuild_subword_stats_from_vocab)
  Promote L2 subwords top-K by PMI
  Pass 2: re-observe vocabulary to update L3 parent_primes with new L2 primes
    → phrase composites now genuinely unique
    → bridge detection becomes meaningful
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations


@dataclass
class PassDiscoveries:
    """What was discovered in one pass."""
    pass_number: int
    elapsed_ms: float
    new_l2_subwords: int = 0
    new_l3_words: int = 0
    new_l4_phrase_composites: int = 0
    new_bridges: int = 0
    new_meta_bridges: int = 0
    refreshed_l3_parents: int = 0
    total_l2: int = 0
    total_l3_promoted: int = 0
    total_correlations: int = 0

    @property
    def total_new(self) -> int:
        return (self.new_l2_subwords + self.new_l3_words +
                self.new_l4_phrase_composites + self.new_bridges +
                self.new_meta_bridges)

    def summary(self) -> str:
        return (
            f"Pass {self.pass_number} ({self.elapsed_ms:.0f} ms)\n"
            f"  L2 subwords promoted:  {self.total_l2} total (+{self.new_l2_subwords})\n"
            f"  L3 words promoted:     {self.total_l3_promoted} total (+{self.new_l3_words})\n"
            f"  L3 parents refreshed:  {self.refreshed_l3_parents}\n"
            f"  L4 phrase composites:  {self.new_l4_phrase_composites}\n"
            f"  Bridges detected:      {self.new_bridges}\n"
            f"  Meta-bridges detected: {self.new_meta_bridges}\n"
            f"  Correlations total:    {self.total_correlations}"
        )


def has_converged(history: list[PassDiscoveries], min_improvement: float = 0.01) -> bool:
    if len(history) < 2:
        return False
    current = history[-1].total_new
    previous = history[-2].total_new
    if previous == 0:
        return current == 0
    return abs(current - previous) / previous < min_improvement


# ---------------------------------------------------------------------------
# Pass 1: Foundation
# ---------------------------------------------------------------------------

def run_pass1(
    pipe,
    corpus_texts: list[str],
    *,
    max_l2_promote: int = 160,
    use_core_l2: bool = False,
) -> PassDiscoveries:
    """
    Pass 1: post-hoc rebuild subword stats and promote top L2 subwords, then
    refresh L3 parent_primes to incorporate the new L2 pool primes.

    Does NOT re-ingest the corpus.  ingest_corpus + flush already ran.  Re-ingesting
    here would give 2× word_counts, contaminating rebuild_subword_stats_from_vocab
    with hapax-word subwords that dilute the top-PMI L2 selection and reduce
    subword composite quality.

    After Pass 1 the registry has:
      - word_counts for all tokens (from ingest_corpus, 1×)
      - co-occurrence correlations (from ingest_corpus)
      - L2 subword pool primes (post-hoc rebuild from vocabulary)
      - L3 promoted words with UPDATED parent_primes (L2 primes via n-gram refresh)
    """
    from aethos_promotion import LatticeTier
    from aethos_subword_composite import (
        promote_l2_subwords,
        rebuild_subword_stats_from_vocab,
        refresh_l3_parent_primes,
    )

    if use_core_l2:
        from core.bridge_registry import run_core_l2_pass

    t0 = time.perf_counter()
    d = PassDiscoveries(pass_number=1, elapsed_ms=0)

    reg = pipe.registry

    # Count L3 promotions already established by ingest_corpus
    l3_before = sum(1 for k in reg.promoted if k[0] == LatticeTier.L3_WORD)

    # Post-hoc subword rebuild + L2 promotion
    n_sw = rebuild_subword_stats_from_vocab(reg)
    l2_before = sum(1 for k in reg.promoted if k[0] == LatticeTier.L2_SUBWORD)
    if use_core_l2:
        n_prom, n_refreshed = run_core_l2_pass(reg, max_promote=max_l2_promote)
    else:
        n_prom = promote_l2_subwords(reg, max_promote=max_l2_promote)
        n_refreshed = refresh_l3_parent_primes(reg)
    l2_after = sum(1 for k in reg.promoted if k[0] == LatticeTier.L2_SUBWORD)

    d.new_l2_subwords = l2_after - l2_before
    d.new_l3_words = l3_before  # all L3 from this ingest are "new"
    d.refreshed_l3_parents = n_refreshed
    d.total_l2 = l2_after
    d.total_l3_promoted = l3_before
    d.total_correlations = len(reg.correlations)
    d.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return d


# ---------------------------------------------------------------------------
# Pass 2: Structure — phrase composites + bridges
# ---------------------------------------------------------------------------

def run_pass2(
    pipe,
    corpus_texts: list[str],
    doc_tokens: dict[str, frozenset[str]],
    *,
    min_pair_count: int = 2,
    max_pairs_per_doc: int = 32,
    bridge_jaccard: float = 0.35,
    max_bridges: int = 500,
) -> tuple[PassDiscoveries, object, list]:
    """
    Pass 2: build L4 phrase composites and detect L5 bridges.

    Requires Pass 1 to have been run (L2 subwords promoted, L3 refreshed).
    Returns (discoveries, phrase_idx, bridges).
    """
    from aethos_phrase_composite import (
        PhraseCompositeIndex,
        build_phrase_composite_index,
        detect_bridges,
    )
    from aethos_promotion import LatticeTier

    t0 = time.perf_counter()
    d = PassDiscoveries(pass_number=2, elapsed_ms=0)
    reg = pipe.registry

    l3_before = sum(1 for k in reg.promoted if k[0] == LatticeTier.L3_WORD)

    # Re-observe corpus with focus on L3 context enrichment
    # (Updates context sets which enables better polysemy splits)
    for text in corpus_texts:
        pipe.ingest_one(text, finalize=False)
    try:
        pipe.flush()
    except (IndexError, RuntimeError):
        pass

    l3_after = sum(1 for k in reg.promoted if k[0] == LatticeTier.L3_WORD)

    # Build L4 phrase composite index (pool primes only → FTA unique)
    phrase_idx = build_phrase_composite_index(
        reg, doc_tokens,
        min_word_len=4,
        min_pair_count=min_pair_count,
        max_pairs_per_doc=max_pairs_per_doc,
        use_pool_primes_only=True,
    )

    # Detect L5 bridges between phrase composites
    bridges = detect_bridges(
        phrase_idx,
        jaccard_threshold=bridge_jaccard,
        max_bridges=max_bridges,
    )

    d.new_l3_words = l3_after - l3_before
    d.total_l3_promoted = l3_after
    d.new_l4_phrase_composites = phrase_idx.n_composites
    d.new_bridges = len(bridges)
    d.total_correlations = len(reg.correlations)
    d.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return d, phrase_idx, bridges


# ---------------------------------------------------------------------------
# Pass 3: Semantics — meta-bridges + cross-domain
# ---------------------------------------------------------------------------

def run_pass3(
    pipe,
    phrase_idx,
    bridges: list,
    doc_tokens: dict[str, frozenset[str]],
    *,
    meta_jaccard: float = 0.30,
    max_meta: int = 200,
) -> tuple[PassDiscoveries, list]:
    """
    Pass 3: detect meta-bridges (bridges between bridges).

    Meta-bridges represent cross-domain connections:
      "quantum-entanglement" ↔ "quantum-information" → "quantum-science" meta-bridge
    """
    from aethos_phrase_composite import detect_bridges, BridgeNode
    from dataclasses import asdict

    t0 = time.perf_counter()
    d = PassDiscoveries(pass_number=3, elapsed_ms=0)

    # Build a sub-index from bridge shared docs to find bridge-bridge overlaps
    bridge_doc_map: dict[int, set[str]] = {}
    for b in bridges:
        key = b.composite_a * b.composite_b  # unique bridge identifier
        bridge_doc_map[key] = b.shared_docs

    # Find meta-bridges: pairs of bridges sharing enough docs
    meta_bridges = []
    bridge_keys = list(bridge_doc_map.keys())
    for i, ka in enumerate(bridge_keys):
        for kb in bridge_keys[i + 1:]:
            docs_a = bridge_doc_map[ka]
            docs_b = bridge_doc_map[kb]
            inter = docs_a & docs_b
            if not inter:
                continue
            union = docs_a | docs_b
            j = len(inter) / len(union)
            if j >= meta_jaccard:
                meta_bridges.append((ka, kb, j, frozenset(inter)))
            if len(meta_bridges) >= max_meta:
                break
        if len(meta_bridges) >= max_meta:
            break

    meta_bridges.sort(key=lambda x: -x[2])

    d.pass_number = 3
    d.new_meta_bridges = len(meta_bridges)
    d.new_bridges = len(bridges)
    d.new_l4_phrase_composites = phrase_idx.n_composites
    d.total_correlations = len(pipe.registry.correlations)
    d.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return d, meta_bridges


# ---------------------------------------------------------------------------
# Full multi-pass runner
# ---------------------------------------------------------------------------

@dataclass
class MultiPassResult:
    """Results from a complete multi-pass build."""
    passes: list[PassDiscoveries] = field(default_factory=list)
    phrase_idx: object = None
    bridges: list = field(default_factory=list)
    meta_bridges: list = field(default_factory=list)
    converged: bool = False
    total_ms: float = 0.0

    def summary(self) -> str:
        lines = ["=== Multi-Pass Build Results ==="]
        for d in self.passes:
            lines.append(d.summary())
        lines.append(f"Converged: {self.converged}")
        lines.append(f"Total build time: {self.total_ms:.0f} ms")
        return "\n".join(lines)


def build_multi_pass(
    pipe,
    corpus_texts: list[str],
    doc_tokens: dict[str, frozenset[str]],
    *,
    n_passes: int = 3,
    max_l2_promote: int = 160,
    use_core_l2: bool = False,
    verbose: bool = True,
) -> MultiPassResult:
    """
    Run N passes over the corpus to build increasingly deep correlations.

    Pass 1: Foundation (direct co-occurrences, L2/L3 promotions)
    Pass 2: Structure  (L4 phrase composites, L5 bridges)
    Pass 3: Semantics  (meta-bridges, cross-domain)

    Returns MultiPassResult with phrase_idx, bridges, meta_bridges.
    """
    t_total = time.perf_counter()
    result = MultiPassResult()

    # Pass 1
    if verbose:
        print("  Pass 1: foundation (direct co-occurrences + L2/L3 promotions)...", flush=True)
    d1 = run_pass1(
        pipe, corpus_texts, max_l2_promote=max_l2_promote, use_core_l2=use_core_l2
    )
    result.passes.append(d1)
    if verbose:
        print(f"    L2={d1.total_l2} L3={d1.total_l3_promoted} refreshed={d1.refreshed_l3_parents} corr={d1.total_correlations} ({d1.elapsed_ms:.0f}ms)", flush=True)

    if n_passes < 2:
        result.total_ms = (time.perf_counter() - t_total) * 1000.0
        return result

    # Pass 2
    if verbose:
        print("  Pass 2: structure (phrase composites + bridges)...", flush=True)
    d2, phrase_idx, bridges = run_pass2(
        pipe, corpus_texts, doc_tokens
    )
    result.passes.append(d2)
    result.phrase_idx = phrase_idx
    result.bridges = bridges
    if verbose:
        print(f"    L4={d2.new_l4_phrase_composites} bridges={d2.new_bridges} corr={d2.total_correlations} ({d2.elapsed_ms:.0f}ms)", flush=True)

    if n_passes < 3 or has_converged(result.passes):
        result.converged = has_converged(result.passes)
        result.total_ms = (time.perf_counter() - t_total) * 1000.0
        return result

    # Pass 3
    if verbose:
        print("  Pass 3: semantics (meta-bridges)...", flush=True)
    d3, meta_bridges = run_pass3(pipe, phrase_idx, bridges, doc_tokens)
    result.passes.append(d3)
    result.meta_bridges = meta_bridges
    if verbose:
        print(f"    meta-bridges={d3.new_meta_bridges} ({d3.elapsed_ms:.0f}ms)", flush=True)

    result.converged = has_converged(result.passes)
    result.total_ms = (time.perf_counter() - t_total) * 1000.0
    return result
