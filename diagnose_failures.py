#!/usr/bin/env python3
"""
AETHOS Retrieval Failure Diagnostic
====================================
Runs the full AETHOS pipeline on SciFact, finds NDCG=0 queries, and prints
a complete per-signal trace explaining exactly why each query failed.

Every invisible retrieval decision is made visible:
  - What L3 prime did each query word get? (pool vs intersection)
  - What signals fired for the correct doc vs the wrong #1 doc?
  - What composites does the correct doc have that the query didn't check?
  - What fix would close the gap?

Usage:
  python diagnose_failures.py
  python diagnose_failures.py --dataset scifact --n 5
  python diagnose_failures.py --dataset nfcorpus --n 3 --max-docs 1000
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from beir_data_root import resolve_beir_root
from eval_beir import (
    BM25_REF,
    build_corpus_index,
    build_neighbor_weights,
    candidate_ids,
    doc_text,
    ingest_corpus,
    load_corpus,
    load_paths,
    load_qrels,
    load_queries,
    make_pipeline,
    merge_qrels,
    ndcg_at_k,
)
from aethos_composite import build_composite_index
from aethos_discriminative import (
    build_heavy_anchor_index,
    query_anchor_composites,
    score_with_heavy_anchors,
    train_convergence_loop,
    train_on_qrels,
)
from aethos_hub_signature import (
    QueryProfile,
    build_all_hub_signatures,
    build_query_profile,
    rank_with_hub_signatures,
)
from aethos_iterative import build_multi_pass
from aethos_phrase_composite import (
    PHRASE_WEIGHT,
    build_phrase_composite_index,
    phrase_composite_score_fast,
    query_phrase_composites as _query_phrase_composites,
    word_prime,
    word_prime_or_intersection,
)
from aethos_subword_composite import build_subword_composite_index, subword_composite_score
from aethos_tokenize import tokenize_words

# Rarity threshold — must match build_heavy_anchor_index default
RARITY_THRESHOLD = 0.018


# ---------------------------------------------------------------------------
# Per-signal breakdown helpers
# ---------------------------------------------------------------------------

@dataclass
class SignalBreakdown:
    s1_bm25: float = 0.0
    s1_per_word: dict[str, float] = None       # word → BM25 contribution
    s2_coord: float = 0.0
    s2_meets: list[tuple[str, str]] = None      # (query_word, hub_word) coord meets
    s3_neighbors: float = 0.0
    s3_hits: list[tuple[str, float]] = None     # (hub_word, weight) neighbor hits
    s4_subword: float = 0.0
    s5_phrase: float = 0.0
    s6_anchors: float = 0.0
    s6_fired: list[tuple[str, str, int, float]] = None  # (wa, wb, comp, weight)
    total: float = 0.0

    def __post_init__(self):
        if self.s1_per_word is None:
            self.s1_per_word = {}
        if self.s2_meets is None:
            self.s2_meets = []
        if self.s3_hits is None:
            self.s3_hits = []
        if self.s6_fired is None:
            self.s6_fired = []


def compute_signal_breakdown(
    profile: QueryProfile,
    doc_id: str,
    cidx,
    hub_sigs: dict,
    comp_idx,
    sub_comp_idx,
    phrase_idx,
    anchor_idx,
    q_anchor_comps: dict[int, float] | None,
    q_phrase_comps: dict[int, float] | None,
    registry,
) -> SignalBreakdown:
    """Replay scoring for a single doc with per-signal visibility."""
    bd = SignalBreakdown()

    doc_tokens = cidx.doc_tokens.get(doc_id, frozenset())
    sig = hub_sigs.get(doc_id)
    tf = cidx.doc_tf.get(doc_id) if cidx.doc_tf else None
    dl = cidx.doc_len.get(doc_id, 0)
    avg_dl = cidx.avg_dl
    k1, b = 1.5, 0.75

    # Signal 1 — BM25
    for w in profile.word_set:
        if w not in doc_tokens:
            continue
        idf = profile.idf.get(w, 1.0)
        if tf and dl > 0:
            tf_val = tf.get(w, 0)
            norm = tf_val * (k1 + 1.0) / (tf_val + k1 * (1.0 - b + b * dl / avg_dl))
        else:
            norm = 1.0
        contribution = idf * norm
        bd.s1_bm25 += contribution
        bd.s1_per_word[w] = contribution

    # Signal 2 — Coord meet
    if sig:
        for coord, hub_word in sig.hub_coords.items():
            if coord in profile.coords:
                q_word = profile.coords[coord]
                if q_word != hub_word:
                    bd.s2_coord += 1.5
                    bd.s2_meets.append((q_word, hub_word))

    # Signal 3 — Neighbor expansion
    if sig:
        for hub_word in sig.hub_words():
            w_weight = profile.flat_neighbors.get(hub_word, 0.0)
            if w_weight > 0:
                bd.s3_neighbors += w_weight
                bd.s3_hits.append((hub_word, w_weight))
        bd.s3_hits.sort(key=lambda x: -x[1])

    # Signal 4 — Subword composite
    if sub_comp_idx is not None and registry is not None:
        bd.s4_subword = subword_composite_score(
            list(profile.word_set), profile.idf, registry, sub_comp_idx, doc_id,
            word_cache=sub_comp_idx.word_composites,
        )

    # Signal 5 — Phrase composite (PHRASE_WEIGHT=0 so always 0.0, but show for transparency)
    if q_phrase_comps is not None and phrase_idx is not None:
        bd.s5_phrase = phrase_composite_score_fast(q_phrase_comps, doc_id, phrase_idx)

    # Signal 6 — Heavy anchor discriminative
    if q_anchor_comps and anchor_idx is not None:
        doc_comps = anchor_idx.doc_to_anchors.get(doc_id, frozenset())
        for comp, eff_weight in q_anchor_comps.items():
            if comp in doc_comps:
                anchor = anchor_idx.anchors.get(comp)
                if anchor:
                    bd.s6_anchors += eff_weight
                    bd.s6_fired.append((anchor.word_a, anchor.word_b, comp, eff_weight))
        bd.s6_fired.sort(key=lambda x: -x[3])

    bd.total = bd.s1_bm25 + bd.s2_coord + bd.s3_neighbors + bd.s4_subword + bd.s5_phrase + bd.s6_anchors
    return bd


def get_prime_info(word: str, registry) -> tuple[int | None, int, str]:
    """Return (pool_prime_or_None, effective_prime, 'POOL'|'INTERSECTION')."""
    pool_p = word_prime(word, registry)
    actual_p = word_prime_or_intersection(word, registry)
    ptype = "POOL" if pool_p is not None else "INTERSECTION"
    return pool_p, actual_p, ptype


# ---------------------------------------------------------------------------
# Composite gap analysis
# ---------------------------------------------------------------------------

def find_composite_gap(
    rel_id: str,
    profile: QueryProfile,
    cidx,
    anchor_idx,
    q_anchor_comps: dict[int, float] | None,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (active_composites, unchecked_composites) for the correct doc.

    active_composites   — composites the query DID check and the doc has
    unchecked_composites — composites the doc has but the query did NOT fire
                          (words present in doc but not in query)
    """
    if anchor_idx is None:
        return [], []
    doc_comps = anchor_idx.doc_to_anchors.get(rel_id, frozenset())
    active, unchecked = [], []
    for comp in doc_comps:
        anchor = anchor_idx.anchors.get(comp)
        if anchor is None:
            continue
        info = {
            "composite": comp,
            "word_a": anchor.word_a,
            "word_b": anchor.word_b,
            "learned_weight": anchor.learned_weight,
            "discrimination": anchor.discrimination,
            "n_docs": len(anchor.doc_ids),
            "correct_count": anchor.correct_count,
            "wrong_count": anchor.wrong_count,
        }
        if q_anchor_comps and comp in q_anchor_comps:
            info["effective_weight"] = q_anchor_comps[comp]
            active.append(info)
        else:
            # These were NOT fired — either:
            # (a) words not in query, or (b) learned_weight < 0.05 threshold
            info["fired_reason"] = (
                "weight_too_low" if anchor.learned_weight < 0.05
                else "words_not_in_query"
            )
            unchecked.append(info)
    active.sort(key=lambda x: -x["effective_weight"])
    unchecked.sort(key=lambda x: -x["learned_weight"])
    return active, unchecked


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

W = 72  # separator width

def _sep(char: str = "─") -> str:
    return char * W

def _print_breakdown(label: str, bd: SignalBreakdown) -> None:
    print(f"  {label}")
    words_hit = list(bd.s1_per_word.keys())
    print(f"    Sig 1 BM25       {bd.s1_bm25:>8.3f}  words: {words_hit}")
    if bd.s2_meets:
        print(f"    Sig 2 coord      {bd.s2_coord:>8.3f}  meets: {bd.s2_meets}")
    else:
        print(f"    Sig 2 coord      {bd.s2_coord:>8.3f}  (no coord meet)")
    if bd.s3_hits:
        top_nb = [(w, f"{s:.3f}") for w, s in bd.s3_hits[:3]]
        print(f"    Sig 3 neighbors  {bd.s3_neighbors:>8.3f}  top: {top_nb}")
    else:
        print(f"    Sig 3 neighbors  {bd.s3_neighbors:>8.3f}")
    print(f"    Sig 4 subword    {bd.s4_subword:>8.3f}")
    print(f"    Sig 5 phrase     {bd.s5_phrase:>8.3f}  (PHRASE_WEIGHT={PHRASE_WEIGHT}, {'disabled' if PHRASE_WEIGHT == 0 else 'active'})")
    if bd.s6_fired:
        for wa, wb, comp, w in bd.s6_fired:
            print(f"    Sig 6 anchor     {' ':>8}  {wa}×{wb}={comp}  fired w={w:.4f}")
        print(f"    Sig 6 total      {bd.s6_anchors:>8.3f}")
    else:
        print(f"    Sig 6 anchors    {bd.s6_anchors:>8.3f}  (none fired)")
    print(f"    {'─'*40}")
    print(f"    TOTAL            {bd.total:>8.3f}")


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_pipeline(dataset: str, max_docs: int | None, verbose: bool):
    """Build the full AETHOS pipeline, return all index structures."""
    root = Path(resolve_beir_root())
    paths = load_paths(root, dataset)
    if not paths.corpus.exists():
        raise FileNotFoundError(f"Corpus not found: {paths.corpus}")

    print(f"\nLoading {dataset}...", flush=True)
    corpus   = load_corpus(paths.corpus, max_docs=max_docs)
    queries  = load_queries(paths.queries)
    qrels    = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))
    qrels_train = load_qrels(paths.qrels_train)
    qrels_test  = load_qrels(paths.qrels_test)

    print(f"  {len(corpus)} docs  {len(queries)} queries  "
          f"{sum(len(v) for v in qrels.values())} relevance judgments", flush=True)

    # ── Ingest ──────────────────────────────────────────────────────────────
    pipe = make_pipeline("scale")
    t0 = time.perf_counter()
    metrics, cidx = ingest_corpus(pipe, corpus, mode="scale")
    try:
        pipe.flush()
    except (IndexError, RuntimeError) as exc:
        print(f"  cluster flush skipped ({exc})", flush=True)
    print(f"  ingest done: {len(cidx.doc_ids)} docs in {(time.perf_counter()-t0)*1000:.0f} ms",
          flush=True)

    # ── Multi-pass build ─────────────────────────────────────────────────────
    corpus_texts = [doc_text(doc) for doc in corpus.values()]
    mp = build_multi_pass(pipe, corpus_texts, cidx.doc_tokens, n_passes=2, verbose=True)
    phrase_idx = mp.phrase_idx

    # ── Hub signatures ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    hub_sigs = build_all_hub_signatures(cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=12)
    print(f"  hub signatures: {len(hub_sigs)} docs in {(time.perf_counter()-t0)*1000:.0f} ms",
          flush=True)

    # ── Subword composite index ──────────────────────────────────────────────
    sub_comp_idx = build_subword_composite_index(
        pipe.registry, cidx.doc_tokens, max_composites=500
    )
    print(f"  subword composites: {sub_comp_idx.n_composites}", flush=True)

    # ── Legacy composite index (Signal 4 legacy, weight=0 in practice) ───────
    comp_idx = build_composite_index(hub_sigs)

    # ── Phrase composite index ────────────────────────────────────────────────
    if phrase_idx is None:
        phrase_idx = build_phrase_composite_index(
            pipe.registry, cidx.doc_tokens,
            min_word_len=4, min_pair_count=3, max_pairs_per_doc=32,
            use_pool_primes_only=True,
        )
    if phrase_idx is not None:
        print(f"  phrase composites: {phrase_idx.n_composites} nodes "
              f"({phrase_idx.n_pairs_indexed} pairs)", flush=True)

    # ── Heavy anchor index ────────────────────────────────────────────────────
    t0 = time.perf_counter()
    anchor_idx = build_heavy_anchor_index(
        pipe.registry, cidx.doc_tokens, cidx.doc_freq,
        max_doc_count=5, rarity_threshold=RARITY_THRESHOLD,
    )
    print(f"  heavy anchors: {anchor_idx.n_anchors} in {(time.perf_counter()-t0)*1000:.0f} ms",
          flush=True)

    # ── Anchor training + convergence ─────────────────────────────────────────
    neighbor_map = build_neighbor_weights(pipe.registry)

    if qrels_train and anchor_idx.n_anchors > 0:
        n_trained = train_on_qrels(
            anchor_idx, queries, qrels_train, cidx.doc_ids, cidx.doc_tokens,
            pipe.registry, cidx.doc_freq, len(cidx.doc_ids),
        )
        n_pos = sum(1 for a in anchor_idx.anchors.values() if a.correct_count > 0)
        print(f"  anchor training: {n_trained} queries  ({n_pos} net-positive anchors)",
              flush=True)

        if qrels_test:
            print("  running convergence loop...", flush=True)
            history = train_convergence_loop(
                anchor_idx, pipe.registry, phrase_idx,
                cidx.doc_tokens, cidx.doc_tf, cidx.doc_len, cidx.avg_dl, cidx.doc_freq,
                queries, qrels_train, qrels_test, cidx.doc_ids,
                hub_sigs, neighbor_map, sub_comp_idx,
                max_rounds=8, convergence_threshold=0.002, verbose=True,
            )
            print(f"  NDCG@10 history: {[f'{x:.4f}' for x in history]}", flush=True)

    return dict(
        paths=paths,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        qrels_train=qrels_train,
        qrels_test=qrels_test,
        pipe=pipe,
        cidx=cidx,
        hub_sigs=hub_sigs,
        comp_idx=comp_idx,
        sub_comp_idx=sub_comp_idx,
        phrase_idx=phrase_idx,
        anchor_idx=anchor_idx,
        neighbor_map=neighbor_map,
    )


def score_query(qid: str, state: dict) -> tuple[list[str], QueryProfile,
                                                  dict[int, float] | None,
                                                  dict[int, float] | None]:
    """Score a single query; return (ranked_ids, profile, q_anchor_comps, q_phrase_comps)."""
    cidx        = state["cidx"]
    pipe        = state["pipe"]
    hub_sigs    = state["hub_sigs"]
    comp_idx    = state["comp_idx"]
    sub_comp_idx= state["sub_comp_idx"]
    phrase_idx  = state["phrase_idx"]
    anchor_idx  = state["anchor_idx"]
    neighbor_map= state["neighbor_map"]
    n_docs      = len(cidx.doc_ids)

    profile = build_query_profile(
        state["queries"][qid], pipe.registry,
        neighbor_map=neighbor_map,
        doc_freq=cidx.doc_freq,
        n_docs=n_docs,
    )
    cands = candidate_ids(profile.words, cidx.inv, neighbor_map, cidx.doc_ids)

    q_anchor_comps = None
    if anchor_idx.n_anchors > 0:
        q_anchor_comps = query_anchor_composites(
            list(profile.word_set), anchor_idx, pipe.registry, idf=profile.idf,
        )

    q_phrase_comps = None
    if phrase_idx is not None:
        q_phrase_comps = _query_phrase_composites(
            profile.words, phrase_idx, pipe.registry, profile.idf,
        )

    ranked = rank_with_hub_signatures(
        profile, cands, hub_sigs, cidx.doc_ids,
        doc_tokens=cidx.doc_tokens,
        doc_tf=cidx.doc_tf,
        doc_len=cidx.doc_len,
        avg_dl=cidx.avg_dl,
        composite_index=comp_idx,
        sub_comp_idx=sub_comp_idx,
        registry=pipe.registry,
        phrase_idx=phrase_idx,
        anchor_idx=anchor_idx,
        query_anchor_comps=q_anchor_comps,
        query_phrase_comps=q_phrase_comps,
        top_k=100,
    )

    return ranked, profile, q_anchor_comps, q_phrase_comps


def print_query_trace(
    qid: str,
    ranked: list[str],
    profile: QueryProfile,
    q_anchor_comps: dict[int, float] | None,
    q_phrase_comps: dict[int, float] | None,
    state: dict,
) -> None:
    """Print the full diagnostic trace for one query."""
    cidx        = state["cidx"]
    pipe        = state["pipe"]
    hub_sigs    = state["hub_sigs"]
    comp_idx    = state["comp_idx"]
    sub_comp_idx= state["sub_comp_idx"]
    phrase_idx  = state["phrase_idx"]
    anchor_idx  = state["anchor_idx"]
    neighbor_map= state["neighbor_map"]
    queries     = state["queries"]
    qrels       = state["qrels"]
    n_docs      = len(cidx.doc_ids)

    query_text = queries[qid]
    rel_docs   = qrels[qid]
    q_words    = profile.words
    rarity_max = max(1, int(RARITY_THRESHOLD * n_docs))

    print(f"\n{'='*W}")
    print(f"Query {qid}: \"{query_text}\"")

    # Rank of each relevant doc
    for rel_id in rel_docs:
        try:
            rank = ranked.index(rel_id) + 1
            in_top = "✓ IN TOP 10" if rank <= 10 else f"rank #{rank}"
        except ValueError:
            in_top = "rank >100 (not retrieved)"
        print(f"  Relevant doc {rel_id}: {in_top}")

    top1 = ranked[0] if ranked else None
    top1_is_wrong = top1 is not None and top1 not in rel_docs

    # ── 1. Query word analysis ───────────────────────────────────────────────
    print(f"\n  {_sep('─')}")
    print(f"  Query word analysis  (rarity: df < {rarity_max}/{n_docs} = {RARITY_THRESHOLD*100:.1f}%)")
    print(f"  {_sep('─')}")
    for w in sorted(set(q_words)):
        if not w.isalpha():
            continue
        df       = cidx.doc_freq.get(w, 0)
        pct      = df / n_docs * 100
        pool_p, actual_p, ptype = get_prime_info(w, pipe.registry)
        is_rare  = df < rarity_max
        rare_tag = "RARE ✓" if is_rare else "common"
        idf_val  = profile.idf.get(w, 0.0)
        if ptype == "POOL":
            prime_str = f"pool_prime={actual_p}"
        else:
            prime_str = f"intersection_prime={actual_p} (no pool prime)"
        print(f"    {w:<22}  df={df}/{n_docs} ({pct:>5.1f}%)  {prime_str:<38}  {rare_tag}  idf={idf_val:.2f}")

    # ── 2. Relevant doc coverage ─────────────────────────────────────────────
    for rel_id in rel_docs:
        if rel_id not in cidx.doc_tokens:
            print(f"\n  Relevant doc {rel_id}: NOT IN CORPUS (max_docs cutoff?)")
            continue
        doc_toks = cidx.doc_tokens[rel_id]
        print(f"\n  {_sep('─')}")
        print(f"  Relevant doc {rel_id}  word coverage:")
        print(f"  {_sep('─')}")
        for w in sorted(set(q_words)):
            if not w.isalpha():
                continue
            if w in doc_toks:
                print(f"    ✓  {w}")
            else:
                # Look for morphological near-misses (shared prefix ≥ 4 chars)
                stem = w[:5]
                similar = sorted(
                    t for t in doc_toks
                    if len(t) >= 4 and t != w and (
                        t.startswith(stem) or w.startswith(t[:5]) or
                        (len(t) >= 4 and t[:4] == w[:4])
                    )
                )[:4]
                note = f"  ← doc has: {similar}" if similar else ""
                morph_note = ""
                if similar:
                    morph_note = "  ← MORPHOLOGY MISS"
                print(f"    ✗  {w}{note}{morph_note}")

    # ── 3. Signal breakdown — correct doc ───────────────────────────────────
    for rel_id in rel_docs:
        if rel_id not in cidx.doc_tokens:
            continue
        bd_rel = compute_signal_breakdown(
            profile, rel_id, cidx, hub_sigs, comp_idx, sub_comp_idx,
            phrase_idx, anchor_idx, q_anchor_comps, q_phrase_comps, pipe.registry,
        )
        print(f"\n  {_sep('─')}")
        print(f"  Signal breakdown — CORRECT doc {rel_id}:")
        print(f"  {_sep('─')}")
        _print_breakdown("", bd_rel)

    # ── 4. Signal breakdown — top-1 wrong doc ───────────────────────────────
    if top1_is_wrong:
        bd_top1 = compute_signal_breakdown(
            profile, top1, cidx, hub_sigs, comp_idx, sub_comp_idx,
            phrase_idx, anchor_idx, q_anchor_comps, q_phrase_comps, pipe.registry,
        )
        print(f"\n  {_sep('─')}")
        print(f"  Signal breakdown — #1 RANKED (wrong) doc {top1}:")
        print(f"  {_sep('─')}")
        _print_breakdown("", bd_top1)

        # Which query words are in the wrong doc?
        top1_toks = cidx.doc_tokens.get(top1, frozenset())
        hit_words = [w for w in profile.word_set if w in top1_toks]
        miss_words = [w for w in profile.word_set if w not in top1_toks]
        print(f"\n  Wrong doc {top1} query-word coverage:")
        for w in sorted(hit_words):
            print(f"    ✓  {w}")
        for w in sorted(miss_words):
            print(f"    ✗  {w}")

    # ── 5. Composite gap analysis ────────────────────────────────────────────
    for rel_id in rel_docs:
        if rel_id not in cidx.doc_tokens or anchor_idx is None:
            continue
        active, unchecked = find_composite_gap(
            rel_id, profile, cidx, anchor_idx, q_anchor_comps,
        )
        if not active and not unchecked:
            print(f"\n  Composite gap: no heavy anchors in correct doc {rel_id}")
            continue
        print(f"\n  {_sep('─')}")
        print(f"  Composite gap analysis — correct doc {rel_id}:")
        print(f"  {_sep('─')}")
        if active:
            print(f"    ACTIVE anchors (query fired these for this doc):")
            for g in active:
                print(f"      {g['word_a']:16} × {g['word_b']:16}  "
                      f"eff_w={g['effective_weight']:.4f}  "
                      f"n_docs={g['n_docs']}  "
                      f"c={g['correct_count']} w={g['wrong_count']}")
        else:
            print(f"    (no active anchors — query did not fire any anchor for this doc)")

        if unchecked:
            print(f"    UNCHECKED anchors (doc has these, but query didn't fire them):")
            for g in unchecked[:8]:
                reason = g["fired_reason"]
                print(f"      {g['word_a']:16} × {g['word_b']:16}  "
                      f"base_w={g['learned_weight']:.4f}  "
                      f"n_docs={g['n_docs']}  "
                      f"reason={reason}")

    # ── 6. WHY IT FAILED + fix suggestions ──────────────────────────────────
    print(f"\n  {_sep('─')}")
    print(f"  DIAGNOSIS:")
    print(f"  {_sep('─')}")
    _diagnose(qid, ranked, profile, q_anchor_comps, state, rarity_max)
    print()


def _diagnose(
    qid: str,
    ranked: list[str],
    profile: QueryProfile,
    q_anchor_comps: dict[int, float] | None,
    state: dict,
    rarity_max: int,
) -> None:
    cidx      = state["cidx"]
    pipe      = state["pipe"]
    anchor_idx= state["anchor_idx"]
    qrels     = state["qrels"]
    queries   = state["queries"]
    n_docs    = len(cidx.doc_ids)

    rel_docs  = qrels[qid]
    q_words   = [w for w in tokenize_words(queries[qid]) if w.isalpha() and len(w) >= 3]

    issues: list[str] = []

    for rel_id in rel_docs:
        doc_toks = cidx.doc_tokens.get(rel_id)

        # Issue: doc not in corpus
        if doc_toks is None:
            issues.append(f"✗ Doc {rel_id} not loaded (increase --max-docs)")
            continue

        # Issue: doc not retrieved at all
        try:
            rank = ranked.index(rel_id) + 1
        except ValueError:
            rank = None

        if rank is None:
            # Not even in top-100 — candidate expansion miss?
            word_overlap = profile.word_set & doc_toks
            nb_overlap   = frozenset(profile.flat_neighbors.keys()) & doc_toks
            if not word_overlap and not nb_overlap:
                issues.append(
                    f"✗ CANDIDATE MISS: doc {rel_id} shares NO query words and NO neighbor words "
                    f"→ inverted index never returns it. "
                    f"FIX: add subword/morphological expansion to candidate generation."
                )
            else:
                issues.append(
                    f"✗ SCORE MISS: doc {rel_id} was a candidate (via {list(word_overlap)[:3]}) "
                    f"but scored outside top-100."
                )
        elif rank > 10:
            issues.append(f"✗ RANKED TOO LOW: doc {rel_id} at rank #{rank}")

        # Issue: morphological misses
        morph_misses = [w for w in q_words if w not in doc_toks]
        for mw in morph_misses:
            stem = mw[:5]
            similar = [t for t in doc_toks if len(t) >= 4 and t != mw and t[:4] == mw[:4]][:3]
            if similar:
                issues.append(
                    f"  MORPHOLOGY: '{mw}' not in doc {rel_id} but doc has {similar}. "
                    f"FIX: add subword '{mw[:4]}' to bridge variants via L2 prime."
                )

        # Issue: anchor pollution (common words in active anchors)
        if q_anchor_comps:
            polluted = []
            for comp, eff_w in q_anchor_comps.items():
                anch = anchor_idx.anchors.get(comp)
                if anch:
                    df_a = cidx.doc_freq.get(anch.word_a, 0)
                    df_b = cidx.doc_freq.get(anch.word_b, 0)
                    if df_a >= rarity_max or df_b >= rarity_max:
                        polluted.append(
                            f"{anch.word_a}({df_a/n_docs*100:.1f}%)"
                            f"×{anch.word_b}({df_b/n_docs*100:.1f}%)"
                        )
            if polluted:
                issues.append(
                    f"  ANCHOR POLLUTION: active anchors include common-word pairs: {polluted}. "
                    f"FIX: filter anchors to rare×rare only, or add word-df weight penalty."
                )

    # Issue: no trained anchors at all for this query
    if not q_anchor_comps:
        q_long = [w for w in q_words if len(w) >= 4 and w.isalpha()]
        issues.append(
            f"  NO ANCHORS: query has no trained heavy anchors "
            f"(content words: {q_long}). "
            f"This query relies entirely on BM25+coord+neighbors."
        )

    if not issues:
        issues.append("  (no obvious single cause — check signal totals above)")

    for issue in issues:
        print(f"  {issue}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def diagnose(
    dataset: str = "scifact",
    n_failing: int = 5,
    max_docs: int | None = None,
    verbose: bool = False,
) -> None:
    print(f"\n{'#'*W}")
    print(f"# AETHOS Retrieval Failure Diagnostic")
    print(f"# dataset={dataset}  n_failing={n_failing}  max_docs={max_docs}")
    print(f"{'#'*W}", flush=True)

    state = run_pipeline(dataset, max_docs, verbose)

    cidx   = state["cidx"]
    qrels  = state["qrels"]
    queries= state["queries"]
    qids   = [q for q in qrels if q in queries]

    # ── Find NDCG=0 queries ─────────────────────────────────────────────────
    print(f"\nScoring {len(qids)} queries to find failures...", flush=True)
    failing: list[tuple[str, list[str], QueryProfile,
                         dict[int, float] | None, dict[int, float] | None]] = []
    all_ndcgs: list[float] = []

    for qid in qids:
        ranked, profile, q_ac, q_pc = score_query(qid, state)
        ndcg = ndcg_at_k(ranked, qrels[qid], 10)
        all_ndcgs.append(ndcg)
        if ndcg == 0.0:
            failing.append((qid, ranked, profile, q_ac, q_pc))

    mean_ndcg = sum(all_ndcgs) / max(len(all_ndcgs), 1)
    n_zero    = sum(1 for x in all_ndcgs if x == 0.0)
    n_perfect = sum(1 for x in all_ndcgs if x >= 0.999)
    print(f"\n  NDCG@10 summary over {len(qids)} queries:")
    print(f"    mean     = {mean_ndcg:.4f}")
    print(f"    NDCG=0   = {n_zero}  ({n_zero/max(len(qids),1)*100:.1f}%)")
    print(f"    NDCG=1   = {n_perfect}  ({n_perfect/max(len(qids),1)*100:.1f}%)")
    bm25_ref = BM25_REF.get(dataset)
    if bm25_ref:
        delta = mean_ndcg - bm25_ref
        print(f"    BM25 ref = {bm25_ref:.4f}  (delta {delta:+.4f})")

    print(f"\nDiagnosing top {min(n_failing, len(failing))} of {len(failing)} NDCG=0 queries...",
          flush=True)

    for qid, ranked, profile, q_ac, q_pc in failing[:n_failing]:
        print_query_trace(qid, ranked, profile, q_ac, q_pc, state)

    print(f"\n{'#'*W}")
    print(f"# Diagnostic complete — {len(failing)} NDCG=0 queries found")
    print(f"{'#'*W}")


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHOS retrieval failure diagnostic")
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--n", type=int, default=5, help="Number of failing queries to trace")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    diagnose(
        dataset=args.dataset,
        n_failing=args.n,
        max_docs=args.max_docs,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
