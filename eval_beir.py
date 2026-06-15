#!/usr/bin/env python3
"""
BEIR evaluation for AETHOS pipeline (folder 3).

Retrieval uses LatticeHubSignatures — compact per-doc lattice index built
from compression_strength hubs + formula_coord + L4-L6 neighbors.
Scoring is O(Q × K) per document (K = hub count, default 12) instead of
the previous O(Q × D_tokens) cross-product.

Usage:
  python eval_beir.py --datasets scifact
  python eval_beir.py --datasets scifact nfcorpus
  python eval_beir.py --datasets scifact --mode scale --max-docs 2000
  python eval_beir.py --list
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# BIT 10: pre-filter routed pool before rank (κ Jaccard top-N + BM25 fill).
DEFAULT_KAPPA_CANDIDATE_CAP = 350

from aethos_composite import build_composite_index
from aethos_discriminative import (
    HeavyAnchorIndex,
    build_heavy_anchor_index,
    calibrate_signal_weights,
    discover_discriminating_intersections,
    discover_meta_intersections,
    query_anchor_composites,
    train_negative_anchors,
    train_on_qrels as train_heavy_anchors,
    train_convergence_loop,
)
from aethos_persist import brain_path_for_dataset, load_brain, save_brain
from aethos_iterative import build_multi_pass
from itertools import combinations
from aethos_promotion import LatticeTier
from aethos_phrase_composite import (
    PhraseCompositeIndex,
    build_phrase_composite_index,
    phrase_composite_score,
    phrase_composite_score_fast,
    query_phrase_composites as _query_phrase_composites,
    explain_phrase_composite,
)
from aethos_subword_composite import (
    SubwordCompositeIndex,
    build_subword_composite_index,
    subword_composite_score,
)
from aethos_hub_signature import (
    LatticeHubSignature,
    QueryProfile,
    build_all_hub_signatures,
    build_query_profile,
    rank_with_hub_signatures,
    signature_report,
)
from aethos_pipeline import AethosPipeline
from aethos_scale import ScaleConfig, ScaleMetrics, timed_ingest_one
from aethos_tokenize import tokenize_words
from beir_data_root import resolve_beir_root
from core.learning_engine import (
    BadCorrelationStore,
    bad_correlation_path,
    load_distilled_registry,
    record_retrieval_false_positives,
    save_distilled_registry,
)
from eval_checkpoint import (
    EvalBundle,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint as persist_eval_checkpoint,
)

BM25_REF = {
    "scifact": 0.643,
    "nfcorpus": 0.321,
    "fiqa": 0.236,
    "trec-covid": 0.656,
    "webis-touche2020": 0.442,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    dataset: str
    mode: str
    n_docs: int
    n_queries: int
    ndcg10: float
    r10: float
    p50_ingest_ms: float
    p99_ingest_ms: float
    bytes_per_doc: float
    hub_bytes_per_doc: float
    p50_query_ms: float
    bm25_ref: float | None = None

    def summary(self) -> str:
        ref = ""
        if self.bm25_ref is not None:
            delta = self.ndcg10 - self.bm25_ref
            ref = f"  BM25 ref:        {self.bm25_ref:.3f}  (delta {delta:+.3f})\n"
        return (
            f"{self.dataset} [{self.mode}]  docs={self.n_docs}  queries={self.n_queries}\n"
            f"  NDCG@10:         {self.ndcg10:.4f}\n"
            f"  R@10:            {self.r10:.4f}\n"
            f"{ref}"
            f"  ingest p50:      {self.p50_ingest_ms:.2f} ms/doc\n"
            f"  ingest p99:      {self.p99_ingest_ms:.2f} ms/doc\n"
            f"  fingerprint:     {self.bytes_per_doc:.0f} B/doc\n"
            f"  hub index:       {self.hub_bytes_per_doc:.0f} B/doc\n"
            f"  query p50:       {self.p50_query_ms:.2f} ms/query"
        )


# ---------------------------------------------------------------------------
# BEIR file layout
# ---------------------------------------------------------------------------

@dataclass
class BeirPaths:
    name: str
    corpus: Path
    queries: Path
    qrels_test: Path
    qrels_train: Path


def load_paths(root: Path, name: str) -> BeirPaths:
    base = root / name
    return BeirPaths(
        name=name,
        corpus=base / "corpus.jsonl",
        queries=base / "queries.jsonl",
        qrels_test=base / "qrels" / "test.tsv",
        qrels_train=base / "qrels" / "train.tsv",
    )


def load_corpus(path: Path, *, max_docs: int | None) -> dict[str, dict]:
    docs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
            if max_docs is not None and len(docs) >= max_docs:
                break
    return docs


def load_queries(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            out[o["_id"]] = o["text"]
    return out


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                out[row["query-id"]][row["corpus-id"]] = int(row["score"])
    return out


def merge_qrels(*maps: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for m in maps:
        for qid, rel in m.items():
            out[qid].update(rel)
    return dict(out)


def doc_text(doc: dict) -> str:
    title = doc.get("title", "").strip()
    body = doc.get("text", "").strip()
    return f"{title} {body}".strip() if title else body


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    if not ideal:
        return 0.0
    return sum(1.0 / math.log2(r + 2) for r, d in enumerate(ranked[:k]) if d in rel) / ideal


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


# ---------------------------------------------------------------------------
# Inverted index — for candidate generation only (not scoring)
# ---------------------------------------------------------------------------

def build_neighbor_weights(registry) -> dict[str, dict[str, float]]:
    """
    Log-normalized neighbor weights so correlation signal stays in IDF-comparable range.
    Raw strength can reach hundreds on large corpora; log(1+s) keeps it ≤ ~6 per edge.
    """
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (a, b), link in registry.correlations.items():
        # log-normalize to prevent corpus-size blow-up
        w = math.log1p(float(link.strength)) * (link.dim4 + link.dim6 + 1.0) * 0.1
        out[a][b] = out[a].get(b, 0.0) + w
        out[b][a] = out[b].get(a, 0.0) + w
    return dict(out)


MIN_POOL_PRIME = 107  # PROMOTION_POOL_FIRST — letter primes are corpus-wide noise


def _tier1_lexical(
    query_words: list[str],
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
) -> set[str]:
    """Tier 1: inverted index + L4-L6 neighbor postings (BM25 vocabulary path)."""
    cand: set[str] = set()
    for w in query_words:
        cand |= inv.get(w, set())
        for nb in neighbor_map.get(w, {}):
            cand |= inv.get(nb, set())
    return cand


def _tier2_meet_exact(
    query_words: list[str],
    meet_index: dict[int, set[str]],
    registry,
) -> set[str]:
    """Tier 2: exact pool-prime lookup on word prime + L2 parent primes."""
    cand: set[str] = set()
    for w in query_words:
        if len(w) < 3:
            continue
        try:
            tok = registry.resolve_token(w)
            for p in (tok.prime,) + tok.parent_primes:
                if p >= MIN_POOL_PRIME:
                    cand |= meet_index.get(p, set())
        except Exception:
            pass
    return cand


def _tier3a_meet_pool(
    query_words: list[str],
    meet_index: dict[int, set[str]],
    registry,
) -> set[str]:
    """Tier 3a: union meet postings for every query pool factor."""
    from aethos_hub_signature import pool_factors_for_word

    cand: set[str] = set()
    for w in query_words:
        if len(w) < 3:
            continue
        try:
            for p in pool_factors_for_word(registry, w):
                cand |= meet_index.get(p, set())
        except Exception:
            pass
    return cand


def _tier3b_meet_fuzzy(
    query_words: list[str],
    meet_index: dict[int, set[str]],
    registry,
) -> set[str]:
    """Tier 3b: prime_factor_similarity partial overlap on lattice composites."""
    from aethos_hub_signature import lattice_composite_for_word
    from core.phi_lattice import prime_factor_similarity

    cand: set[str] = set()
    for w in query_words:
        if len(w) < 3:
            continue
        try:
            qc = lattice_composite_for_word(registry, w)
            if qc <= 1:
                continue
            for prime_key, docs in meet_index.items():
                if prime_factor_similarity(qc, prime_key) > 0.0:
                    cand |= docs
        except Exception:
            pass
    return cand


def candidate_generation_tier(
    query_words: list[str],
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    all_ids: list[str],
    meet_index: "dict[int, set[str]] | None" = None,
    registry=None,
    attractor_index=None,
    attractor_radius: int = 1,
    min_attractor_candidates: int = 8,
) -> str:
    """Which tier supplied candidates (for P4 diagnostics)."""
    if attractor_index is not None and registry is not None:
        from pipeline.bit_04_candidate_router import route_query_candidates

        route = route_query_candidates(
            query_words,
            registry,
            attractor_index,
            inv,
            neighbor_map,
            all_ids,
            radius=attractor_radius,
            min_candidates=min_attractor_candidates,
            meet_index=meet_index,
        )
        if route.tier.startswith("bit4_") and route.tier != "bit4_fallback":
            return "bit4_router"
    t1 = _tier1_lexical(query_words, inv, neighbor_map)
    if t1:
        return "tier1_lexical"
    if meet_index is not None and registry is not None:
        t2 = _tier2_meet_exact(query_words, meet_index, registry)
        if t2:
            return "tier2_meet_exact"
        t3a = _tier3a_meet_pool(query_words, meet_index, registry)
        if t3a:
            return "tier3_meet_pool"
        t3b = _tier3b_meet_fuzzy(query_words, meet_index, registry)
        if t3b:
            return "tier3_meet_fuzzy"
    return "tier4_full_corpus"


def candidate_ids(
    query_words: list[str],
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    all_ids: list[str],
    meet_index: "dict[int, set[str]] | None" = None,
    registry=None,
    *,
    attractor_index=None,
    attractor_radius: int = 1,
    min_attractor_candidates: int = 8,
) -> list[str]:
    """
    P4 candidate cascade (invert generation for zero-BM25 queries).

    BIT 4 (when attractor_index set): C(q) from κ neighborhoods, ∪ tier1 if
    |C| ≥ min_attractor_candidates; else fall through to tiers below.

    Tier 1 — lexical inverted index + neighbor postings.
    Tier 2 — exact MeetIndex on pool primes (only if tier 1 empty).
    Tier 3 — pool-factor union + prime_factor_similarity fuzzy meet
              (only if tiers 1–2 empty; replaces old all_ids fallback).
    Tier 4 — full corpus only when meet index unavailable or tier 3 empty.
    """
    if attractor_index is not None and registry is not None:
        from pipeline.bit_04_candidate_router import route_query_candidates

        meet_arg = (
            meet_index.legacy_dict()
            if meet_index is not None and hasattr(meet_index, "legacy_dict")
            else meet_index
        )
        route = route_query_candidates(
            query_words,
            registry,
            attractor_index,
            inv,
            neighbor_map,
            all_ids,
            radius=attractor_radius,
            min_candidates=min_attractor_candidates,
            meet_index=meet_index if hasattr(meet_index, "by_factor") else meet_arg,
        )
        if route.tier != "bit4_fallback":
            return route.doc_ids

    t1 = _tier1_lexical(query_words, inv, neighbor_map)
    if t1:
        return list(t1)

    if meet_index is not None and registry is not None:
        t2 = _tier2_meet_exact(query_words, meet_index, registry)
        if t2:
            return list(t2)
        t3a = _tier3a_meet_pool(query_words, meet_index, registry)
        if t3a:
            return list(t3a)
        t3b = _tier3b_meet_fuzzy(query_words, meet_index, registry)
        if t3b:
            return list(t3b)

    return all_ids


def build_meet_index(
    hub_sigs: dict,
    registry,
    *,
    max_docs_per_factor: int = 500,
) -> "dict[int, set[str]]":
    """
    Build a prime-factor inverted index (MeetIndex) from hub signatures.

    BIT 7: capped MeetWitnessIndex with pool-prime postings.
    Returns legacy dict[int, set[str]] for candidate_ids tiers 2–3.
    """
    from pipeline.bit_07_meet_witness import build_meet_witness_index

    idx = build_meet_witness_index(
        hub_sigs,
        registry,
        max_docs_per_factor=max_docs_per_factor,
    )
    return idx.legacy_dict()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

@dataclass
class CorpusIndex:
    """Built once after ingest; used for all retrieval operations."""
    doc_ids: list[str]
    doc_tokens: dict[str, frozenset[str]]            # doc_id → unique words (for candidate gen)
    doc_tf: dict[str, dict[str, int]]                # doc_id → {word: count} (for BM25 TF)
    doc_len: dict[str, int]                          # doc_id → token count
    avg_dl: float                                    # average document length
    doc_freq: dict[str, int]                         # word → # docs containing it
    inv: dict[str, set[str]]                         # word → doc_ids


# BM25 hyperparameters — standard Robertson/Sparck Jones values
BM25_K1 = 1.5
BM25_B  = 0.75


def build_corpus_index(
    doc_ids: list[str],
    doc_tokens: dict[str, frozenset[str]],
    doc_tf: dict[str, dict[str, int]],
    doc_len: dict[str, int],
) -> CorpusIndex:
    doc_freq: dict[str, int] = {}
    inv: dict[str, set[str]] = defaultdict(set)
    for did, toks in doc_tokens.items():
        for w in toks:
            doc_freq[w] = doc_freq.get(w, 0) + 1
            inv[w].add(did)
    avg_dl = sum(doc_len.values()) / max(len(doc_len), 1)
    return CorpusIndex(
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        doc_tf=doc_tf,
        doc_len=doc_len,
        avg_dl=avg_dl,
        doc_freq=doc_freq,
        inv=dict(inv),
    )


def ingest_corpus(
    pipe: AethosPipeline,
    corpus: dict[str, dict],
    *,
    mode: str,
) -> tuple[ScaleMetrics, CorpusIndex]:
    """Returns metrics and corpus index with BM25 TF counts."""
    from collections import Counter
    metrics = ScaleMetrics()
    doc_tokens: dict[str, frozenset[str]] = {}
    doc_tf: dict[str, dict[str, int]] = {}
    doc_len: dict[str, int] = {}
    doc_ids: list[str] = []

    for i, (did, doc) in enumerate(corpus.items()):
        text = doc_text(doc)
        doc_ids.append(did)
        words = tokenize_words(text)
        tf = Counter(words)
        doc_tokens[did] = frozenset(tf.keys())
        doc_tf[did] = dict(tf)
        doc_len[did] = len(words)
        timing, fp = timed_ingest_one(pipe, i, text)
        metrics.record(timing, fp)
        if (i + 1) % 500 == 0:
            print(f"  ingested {i + 1}/{len(corpus)} docs...", flush=True)

    return metrics, build_corpus_index(doc_ids, doc_tokens, doc_tf, doc_len)


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _tune_registry_for_beir(reg) -> None:
    # word_promote_at=2: rare scientific terms appearing 2+ times get L3 pool primes.
    reg.word_promote_at = 2
    # context_jaccard_max=0.7: scientific papers on the same narrow topic share
    # >45% vocabulary, blocking L3 promotion of rare technical terms with the default
    # 0.45 threshold.  0.7 allows domain-specific terms that appear across related
    # documents to promote, dramatically increasing L3 pool prime coverage.
    reg.context_jaccard_max = 0.7


def make_pipeline(mode: str) -> AethosPipeline:
    if mode == "scale":
        cfg = ScaleConfig(
            rebuild_every=128,
            lazy_clusters=True,
            fast_ingest=True,
            defer_l2_promotion=True,
            fast_cluster=True,
        )
    elif mode == "quality":
        # Quality mode: same as scale but with 2× L2 subword promotion cap (320).
        # Post-hoc subword rebuild promotes top-320 highest-PMI subwords to L2 primes.
        # More L2 primes → better L3 parent_primes → more FTA-unique composites.
        # Clusters stay lazy for speed; the quality gain is in the promotion layer.
        cfg = ScaleConfig(
            rebuild_every=128,
            lazy_clusters=True,
            fast_ingest=True,
            defer_l2_promotion=True,
            fast_cluster=True,
        )
    else:
        cfg = ScaleConfig(
            rebuild_every=32,
            lazy_clusters=False,
            fast_ingest=True,
            defer_l2_promotion=True,
            fast_cluster=False,
            max_corr_pairs_per_doc=512,
        )
    pipe = AethosPipeline(rebuild_every=cfg.rebuild_every)
    pipe.apply_scale_config(cfg)
    _tune_registry_for_beir(pipe.registry)
    return pipe


# ---------------------------------------------------------------------------
# Query scoring (from checkpoint — fast A/B)
# ---------------------------------------------------------------------------

@dataclass
class QueryScoreResult:
    """One scored query — ranked ids plus routing/scoring metadata for progress logs."""

    ranked: list[str]
    profile: QueryProfile
    n_candidates: int
    route_tier: str
    n_kappa_keys: int


def _format_eta(elapsed_s: float, done: int, total: int) -> str:
    if done <= 0 or total <= done:
        return "—"
    rem_s = (elapsed_s / done) * (total - done)
    if rem_s < 90:
        return f"{rem_s:.0f}s"
    return f"{rem_s / 60:.1f}m"


def _print_query_progress(
    *,
    arm_label: str,
    qi: int,
    total: int,
    qid: str,
    elapsed_ms: float,
    n_candidates: int,
    route_tier: str,
    n_kappa_keys: int,
    ndcg: float,
    avg_ndcg: float,
    loop_elapsed_s: float,
    enable_kappa_scoring: bool,
) -> None:
    eta = _format_eta(loop_elapsed_s, qi + 1, total)
    k8 = "8a" if enable_kappa_scoring else "—"
    print(
        f"  [{arm_label}] {qi + 1:4d}/{total}  q={qid:<6}  "
        f"{elapsed_ms:6.0f}ms  |C|={n_candidates:5d}  "
        f"k={n_kappa_keys:3d}  tier={route_tier:<22}  "
        f"8a={k8}  ndcg={ndcg:.3f}  avg={avg_ndcg:.4f}  ETA={eta}",
        flush=True,
    )


def _score_one_query(
    query: str,
    *,
    pipe,
    cidx,
    hub_sigs,
    neighbor_map,
    meet_index,
    sub_comp_idx,
    comp_idx,
    phrase_idx,
    anchor_idx,
    attractor_index=None,
    kappa_candidate_cap: int = DEFAULT_KAPPA_CANDIDATE_CAP,
    enable_kappa_scoring: bool = False,
) -> QueryScoreResult:
    """Score one query — shared by score_from_bundle and evaluate_dataset.

    BIT 4 routing uses ``attractor_index`` whenever it is set.
    Signal 8a (κ Jaccard) fires only when ``enable_kappa_scoring`` is True.
    """
    from pipeline.bit_04_candidate_router import route_query_candidates
    from pipeline.bit_09_query_cell_profile import build_query_cell_profile

    n_docs = len(cidx.doc_ids)
    profile = build_query_profile(
        query,
        pipe.registry,
        neighbor_map=neighbor_map,
        doc_freq=cidx.doc_freq,
        n_docs=n_docs,
    )

    cell = None
    route_tier = "legacy"
    cands: list[str]
    n_kappa_keys = 0
    protected_doc_ids: frozenset[str] = frozenset()

    if attractor_index is not None:
        cell = build_query_cell_profile(
            pipe.registry,
            query,
            neighbor_map=neighbor_map,
            doc_freq=cidx.doc_freq,
            n_docs=n_docs,
        )
        n_kappa_keys = len(cell.kappa_neighbor_q)
        meet_arg = (
            meet_index.legacy_dict()
            if meet_index is not None and hasattr(meet_index, "legacy_dict")
            else meet_index
        )
        route = route_query_candidates(
            profile.words,
            pipe.registry,
            attractor_index,
            cidx.inv,
            neighbor_map,
            cidx.doc_ids,
            meet_index=meet_index if hasattr(meet_index, "by_factor") else meet_arg,
            doc_freq=cidx.doc_freq,
            n_docs=n_docs,
        )
        cands = route.doc_ids
        protected_doc_ids = route.protected_doc_ids
        route_tier = (
            "bit4_router"
            if route.tier.startswith("bit4_") and route.tier != "bit4_fallback"
            else route.tier
        )
    else:
        cands = candidate_ids(
            profile.words,
            cidx.inv,
            neighbor_map,
            cidx.doc_ids,
            meet_index=meet_index,
            registry=pipe.registry,
        )
        route_tier = candidate_generation_tier(
            profile.words,
            cidx.inv,
            neighbor_map,
            cidx.doc_ids,
            meet_index=meet_index,
            registry=pipe.registry,
        )

    q_anchor_comps = None
    if anchor_idx is not None and anchor_idx.n_anchors > 0:
        q_anchor_comps = query_anchor_composites(
            list(profile.word_set),
            anchor_idx,
            pipe.registry,
            idf=profile.idf,
        )

    if pipe.reader.word_to_cluster:
        query_clusters: dict[str, float] = {}
        for w in profile.word_set:
            cid = pipe.reader.word_to_cluster.get(w)
            if cid:
                query_clusters[cid] = max(
                    query_clusters.get(cid, 0.0), profile.idf.get(w, 0.0)
                )
        if query_clusters:
            qc_ids: dict[str, float] = {}
            for w2, cid2 in pipe.reader.word_to_cluster.items():
                if cid2 in query_clusters and w2 not in profile.word_set:
                    if qc_ids.get(w2, 0.0) < query_clusters[cid2]:
                        qc_ids[w2] = query_clusters[cid2]
            profile.query_cluster_ids = qc_ids

    q_phrase_comps = None
    if phrase_idx is not None:
        q_phrase_comps = _query_phrase_composites(
            profile.words, phrase_idx, pipe.registry, profile.idf,
        )

    ranked = rank_with_hub_signatures(
        profile,
        cands,
        hub_sigs,
        cidx.doc_ids,
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
        attractor_index=attractor_index,
        query_kappa_keys=(
            cell.kappa_neighbor_q if cell and enable_kappa_scoring else None
        ),
        kappa_candidate_cap=kappa_candidate_cap,
        protect_doc_ids=protected_doc_ids,
        top_k=100,
    )
    return QueryScoreResult(
        ranked=ranked,
        profile=profile,
        n_candidates=len(cands),
        route_tier=route_tier,
        n_kappa_keys=n_kappa_keys,
    )


def score_from_bundle(
    bundle: EvalBundle,
    *,
    lambda_prime_factor: float | None = None,
    lambda_kappa: float | None = None,
    kappa_candidate_cap: int = DEFAULT_KAPPA_CANDIDATE_CAP,
    bad_store: BadCorrelationStore | None = None,
    verbose: bool = False,
    progress_every: int = 1,
    arm_label: str = "score",
) -> EvalResult:
    """Run the query loop only — reuse a saved EvalBundle from stage 1."""
    import aethos_hub_signature as _hs
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures

    prev_kappa = _hs.LAMBDA_KAPPA
    if lambda_kappa is not None:
        _hs.LAMBDA_KAPPA = float(lambda_kappa)
    if lambda_prime_factor is not None:
        _hs.LAMBDA_PRIME_FACTOR = float(lambda_prime_factor)
    print(
        f"  LAMBDA_PRIME_FACTOR={_hs.LAMBDA_PRIME_FACTOR}  "
        f"LAMBDA_KAPPA={_hs.LAMBDA_KAPPA}",
        flush=True,
    )

    pipe = bundle.pipe
    cidx = bundle.cidx
    hub_sigs = bundle.hub_sigs
    neighbor_map = bundle.neighbor_map
    meet_index = bundle.meet_index
    sub_comp_idx = bundle.sub_comp_idx
    comp_idx = bundle.comp_idx
    phrase_idx = bundle.phrase_idx
    anchor_idx = bundle.anchor_idx
    queries = bundle.queries
    qrels = bundle.qrels
    qids = bundle.qids

    attractor_index = bundle.attractor_index
    if attractor_index is None:
        attractor_index = build_attractor_index_from_hub_signatures(
            pipe.registry, hub_sigs,
        )

    enable_kappa_scoring = lambda_kappa is not None and lambda_kappa > 0

    if bad_store is None:
        bad_store = BadCorrelationStore.load(bad_correlation_path(bundle.dataset, bundle.mode))

    ndcgs: list[float] = []
    r10s: list[float] = []
    q_times: list[float] = []
    try:
        print(
            f"  scoring {len(qids)} queries  arm={arm_label}  "
            f"docs={bundle.n_docs}  routing=BIT4  "
            f"signal_8a={'on' if enable_kappa_scoring else 'off'}",
            flush=True,
        )
        loop_t0 = time.perf_counter()
        for qi, qid in enumerate(qids):
            t0 = time.perf_counter()

            result = _score_one_query(
                queries[qid],
                pipe=pipe,
                cidx=cidx,
                hub_sigs=hub_sigs,
                neighbor_map=neighbor_map,
                meet_index=meet_index,
                sub_comp_idx=sub_comp_idx,
                comp_idx=comp_idx,
                phrase_idx=phrase_idx,
                anchor_idx=anchor_idx,
                attractor_index=attractor_index,
                kappa_candidate_cap=kappa_candidate_cap,
                enable_kappa_scoring=enable_kappa_scoring,
            )

            q_ms = (time.perf_counter() - t0) * 1000.0
            q_times.append(q_ms)
            rel = qrels[qid]
            ndcg = ndcg_at_k(result.ranked, rel, 10)
            ndcgs.append(ndcg)
            r10s.append(recall_at_k(result.ranked, rel, 10))

            record_retrieval_false_positives(
                bad_store,
                result.ranked,
                rel,
                result.profile,
                hub_sigs,
                pipe.registry,
                top_k=10,
            )
            if progress_every > 0 and ((qi + 1) % progress_every == 0 or qi == 0):
                _print_query_progress(
                    arm_label=arm_label,
                    qi=qi,
                    total=len(qids),
                    qid=qid,
                    elapsed_ms=q_ms,
                    n_candidates=result.n_candidates,
                    route_tier=result.route_tier,
                    n_kappa_keys=result.n_kappa_keys,
                    ndcg=ndcg,
                    avg_ndcg=sum(ndcgs) / len(ndcgs),
                    loop_elapsed_s=time.perf_counter() - loop_t0,
                    enable_kappa_scoring=enable_kappa_scoring,
                )
            elif verbose and (qi + 1) % 50 == 0:
                print(
                    f"  query {qi+1}/{len(qids)}  NDCG@10 avg: {sum(ndcgs)/len(ndcgs):.4f}",
                    flush=True,
                )
        if ndcgs:
            print(
                f"  [{arm_label}] done  NDCG@10={sum(ndcgs)/len(ndcgs):.4f}  "
                f"R@10={sum(r10s)/len(r10s):.4f}  "
                f"p50={sorted(q_times)[len(q_times)//2]:.0f}ms/query",
                flush=True,
            )
    finally:
        if lambda_kappa is not None:
            _hs.LAMBDA_KAPPA = prev_kappa

    n_q = max(len(qids), 1)
    p50_q = sorted(q_times)[len(q_times) // 2] if q_times else 0.0

    bad_path = bad_correlation_path(bundle.dataset, bundle.mode)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.save(bad_path)

    return EvalResult(
        dataset=bundle.dataset,
        mode=bundle.mode,
        n_docs=bundle.n_docs,
        n_queries=len(qids),
        ndcg10=sum(ndcgs) / n_q,
        r10=sum(r10s) / n_q,
        p50_ingest_ms=bundle.p50_ingest_ms,
        p99_ingest_ms=bundle.p99_ingest_ms,
        bytes_per_doc=bundle.bytes_per_doc,
        hub_bytes_per_doc=bundle.hub_bytes_per_doc,
        p50_query_ms=p50_q,
        bm25_ref=BM25_REF.get(bundle.dataset),
    )


# ---------------------------------------------------------------------------
# Evaluate one dataset
# ---------------------------------------------------------------------------

def evaluate_dataset(
    paths: BeirPaths,
    *,
    mode: str = "scale",
    max_docs: int | None = None,
    max_queries: int | None = None,
    hub_top_k: int = 12,
    n_passes: int = 2,
    verbose: bool = False,
    lambda_prime_factor: float | None = None,
    lambda_kappa: float | None = None,
    kappa_candidate_cap: int = DEFAULT_KAPPA_CANDIDATE_CAP,
    from_checkpoint: str | Path | None = None,
    save_checkpoint: bool | str | Path | None = None,
    build_only: bool = False,
    skip_training: bool = False,
    max_convergence_rounds: int = 8,
    train_mode: str = "full",
    max_composite_anchors: int = 2000,
    max_composite_meta: int = 500,
    max_composite_negatives: int = 500,
    clear_bad_correlation: bool = False,
) -> EvalResult:
    if not paths.corpus.exists():
        raise FileNotFoundError(f"missing corpus: {paths.corpus}")

    bad_path = bad_correlation_path(paths.name, mode)
    if clear_bad_correlation:
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_store = BadCorrelationStore()
        bad_store.save(bad_path)
        print("  bad-correlation queue: cleared", flush=True)
    else:
        bad_store = BadCorrelationStore.load(bad_path)

    if from_checkpoint is not None:
        ckpt = Path(from_checkpoint)
        print(f"  loading checkpoint: {ckpt.name}", flush=True)
        bundle = load_checkpoint(ckpt)
        result = score_from_bundle(
            bundle,
            lambda_prime_factor=lambda_prime_factor,
            lambda_kappa=lambda_kappa,
            kappa_candidate_cap=kappa_candidate_cap,
            bad_store=bad_store,
            verbose=verbose,
        )
        print(result.summary())
        return result

    import aethos_hub_signature as _hs

    if lambda_prime_factor is not None:
        _hs.LAMBDA_PRIME_FACTOR = float(lambda_prime_factor)
        print(f"  LAMBDA_PRIME_FACTOR={_hs.LAMBDA_PRIME_FACTOR}", flush=True)
    prev_kappa = _hs.LAMBDA_KAPPA
    if lambda_kappa is not None:
        _hs.LAMBDA_KAPPA = float(lambda_kappa)
        print(f"  LAMBDA_KAPPA={_hs.LAMBDA_KAPPA}", flush=True)

    if bad_store.entries:
        print(f"  bad-correlation queue: {len(bad_store.entries)} entries (loaded)", flush=True)

    corpus = load_corpus(paths.corpus, max_docs=max_docs)
    queries = load_queries(paths.queries)
    qrels = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))

    qids = [q for q in qrels if q in queries]
    if max_queries is not None:
        qids = qids[:max_queries]

    # --- multi-pass build ---
    pipe = make_pipeline(mode)
    distilled_path = brain_path_for_dataset(paths.name, mode).with_suffix(".distilled.json")
    if load_distilled_registry(pipe.registry, distilled_path):
        print(f"  distilled registry loaded (pre-ingest): {distilled_path.name}", flush=True)

    t_ingest = time.perf_counter()
    metrics, cidx = ingest_corpus(pipe, corpus, mode=mode)
    # Pass 1 flush (cluster discovery)
    try:
        pipe.flush()
    except (IndexError, RuntimeError) as exc:
        print(f"  cluster flush skipped ({exc})", flush=True)
    ingest_total_ms = (time.perf_counter() - t_ingest) * 1000.0
    print(f"  ingest done: {len(cidx.doc_ids)} docs in {ingest_total_ms:.0f} ms", flush=True)

    # --- multi-pass iterative build (L2 → L3 refresh → L4 phrases → L5 bridges) ---
    corpus_texts = [doc_text(doc) for doc in corpus.values()]
    # Quality mode doubles L2 subword promotion cap for more pool primes.
    # More L2 primes → L3 words get proper non-colliding parent_primes →
    # better FTA-unique composites for all geometric signals.
    max_l2 = 320 if mode == "quality" else 160
    mp = build_multi_pass(
        pipe, corpus_texts, cidx.doc_tokens,
        n_passes=n_passes, verbose=True,
        max_l2_promote=max_l2,
    )
    phrase_idx = mp.phrase_idx

    # --- build hub signatures (after multi-pass so hubs use refreshed parent_primes) ---
    t_hub = time.perf_counter()
    hub_sigs = build_all_hub_signatures(
        cidx.doc_ids, cidx.doc_tokens, pipe.registry, top_k=hub_top_k
    )
    hub_ms = (time.perf_counter() - t_hub) * 1000.0
    hub_bytes = sum(s.encoded_size() for s in hub_sigs.values()) / max(len(hub_sigs), 1)
    print(f"  hub signatures: {len(hub_sigs)} docs in {hub_ms:.0f} ms  ~{hub_bytes:.0f} B/doc", flush=True)

    if verbose and hub_sigs:
        print(signature_report(hub_sigs[cidx.doc_ids[0]]))

    # --- MeetIndex: prime-factor inverted index for geometric candidate routing ---
    # Uses the swap-meet guarantee: solo(p_q)@n=p_h == solo(p_h)@n=p_q on all 32 wings.
    # Finds docs sharing prime factors with query words regardless of surface vocabulary.
    t_meet = time.perf_counter()
    meet_index = build_meet_index(hub_sigs, pipe.registry)
    meet_ms = (time.perf_counter() - t_meet) * 1000.0
    print(f"  meet index: {len(meet_index)} prime factors in {meet_ms:.0f} ms", flush=True)

    # --- BIT 3/4 attractor index (κ buckets — routing always on in eval) ---
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_from_hub_signatures

    t_attr = time.perf_counter()
    attractor_index = build_attractor_index_from_hub_signatures(
        pipe.registry, hub_sigs,
    )
    attr_summary = attractor_index.summary()
    attr_ms = (time.perf_counter() - t_attr) * 1000.0
    print(
        f"  attractor index: {attr_summary['buckets']} buckets, "
        f"{attr_summary['docs']} docs in {attr_ms:.0f} ms  "
        f"(avg {attr_summary['avg_keys_per_doc']:.1f} keys/doc)",
        flush=True,
    )

    # --- subword composite origin index (Signal 4) ---
    # Cap at 500 composites to keep per-query O(1) lookup fast.
    # 32-lattice consensus will replace this with higher-quality signals.
    t_comp = time.perf_counter()
    sub_comp_idx = build_subword_composite_index(
        pipe.registry, cidx.doc_tokens, max_composites=500
    )
    comp_ms = (time.perf_counter() - t_comp) * 1000.0
    print(f"  composite origins: {sub_comp_idx.n_composites} unique in {comp_ms:.0f} ms", flush=True)

    # legacy letter-GCD index (Signal 4 weight=0)
    comp_idx = build_composite_index(hub_sigs)

    # phrase_idx already built by multi-pass Pass 2 (or None if n_passes<2)
    if phrase_idx is None:
        phrase_idx = build_phrase_composite_index(
            pipe.registry, cidx.doc_tokens,
            min_word_len=4, min_pair_count=3, max_pairs_per_doc=32,
            use_pool_primes_only=True,
        )
    if phrase_idx is not None:
        print(f"  phrase composites: {phrase_idx.n_composites} unique nodes "
              f"({phrase_idx.n_pairs_indexed} pairs)", flush=True)

    composite_only = train_mode == "composite_only"
    import aethos_hub_signature as _hs

    # --- heavy anchor discriminative index (Signal 6) ---
    t_anchor = time.perf_counter()
    if composite_only:
        from aethos_discriminative import HeavyAnchorIndex

        anchor_idx = HeavyAnchorIndex()
        _hs.LAMBDA_COORD = 0.5
        _hs.LAMBDA_NEIGHBOR = 0.15
        print("  heavy anchors: composite_only (empty index, skip bulk scan)", flush=True)
    else:
        anchor_idx = build_heavy_anchor_index(
            pipe.registry, cidx.doc_tokens, cidx.doc_freq,
            max_doc_count=5, rarity_threshold=0.018,
        )
        anchor_ms = (time.perf_counter() - t_anchor) * 1000.0
        print(
            f"  heavy anchors: {anchor_idx.n_anchors} anchors in {anchor_ms:.0f} ms",
            flush=True,
        )

    # --- load saved brain (compound learning across runs) ---
    b_path = brain_path_for_dataset(paths.name, mode)
    brain_exists = b_path.exists()
    qrels_train_data = load_qrels(paths.qrels_train)
    if composite_only:
        if brain_exists and verbose:
            print("  composite_only: ignoring saved brain (fresh composite index)", flush=True)
    else:
        init_lc, init_ln = load_brain(b_path, anchor_idx, verbose=True)
        _hs.LAMBDA_COORD = init_lc
        _hs.LAMBDA_NEIGHBOR = init_ln

    do_train = bool(qrels_train_data) and (
        composite_only or anchor_idx.n_anchors > 0
    )
    if skip_training:
        do_train = False
        if brain_exists:
            print("  skip_training: using saved brain (no convergence/calibration)", flush=True)
        else:
            print("  skip_training: skipping anchor training and convergence", flush=True)

    neighbor_map = build_neighbor_weights(pipe.registry)

    # --- train anchors on train qrels if available ---
    if do_train and not composite_only:
        t_train = time.perf_counter()
        n_trained = train_heavy_anchors(
            anchor_idx,
            queries,
            qrels_train_data,
            cidx.doc_ids,
            cidx.doc_tokens,
            pipe.registry,
            cidx.doc_freq,
            len(cidx.doc_ids),
        )
        train_ms = (time.perf_counter() - t_train) * 1000.0
        n_activated = sum(
            1 for a in anchor_idx.anchors.values()
            if a.correct_count + a.wrong_count > 0
        )
        n_positive = sum(
            1 for a in anchor_idx.anchors.values()
            if a.correct_count > 0
        )
        n_above_threshold = sum(
            1 for a in anchor_idx.anchors.values()
            if a.learned_weight >= 0.05
        )
        print(
            f"  anchor training: {n_trained} queries in {train_ms:.0f} ms  "
            f"({n_activated} activated, {n_positive} net-positive, "
            f"{n_above_threshold} w>=0.05 / {anchor_idx.n_anchors} total)",
            flush=True,
        )

    # --- discriminating intersection discovery ---
    if do_train:
        n_disc = discover_discriminating_intersections(
            anchor_idx,
            pipe.registry,
            queries,
            qrels_train_data,
            cidx.doc_ids,
            cidx.doc_tokens,
            cidx.doc_freq,
            len(cidx.doc_ids),
            hub_sigs,
            neighbor_map,
            cidx.doc_tf,
            cidx.doc_len,
            cidx.avg_dl,
            sub_comp_idx,
            phrase_idx,
            max_new_anchors=max_composite_anchors,
            verbose=True,
        )
        n_neg = train_negative_anchors(
            anchor_idx, pipe.registry, queries, qrels_train_data,
            cidx.doc_ids, cidx.doc_tokens, cidx.doc_freq, len(cidx.doc_ids),
            hub_sigs, neighbor_map, cidx.doc_tf, cidx.doc_len, cidx.avg_dl,
            sub_comp_idx, phrase_idx,
            max_new_negatives=max_composite_negatives,
            verbose=True,
        )
        n_meta = discover_meta_intersections(
            anchor_idx, cidx.doc_tokens, cidx.doc_freq, len(cidx.doc_ids),
            max_new=max_composite_meta,
            verbose=True,
        )
        if composite_only:
            print(
                f"  composite_only summary: +{n_disc} discriminators  "
                f"+{n_neg} negatives  +{n_meta} meta  "
                f"total={anchor_idx.n_anchors} anchors",
                flush=True,
            )
        elif do_train:
            n_retrain = train_heavy_anchors(
                anchor_idx,
                queries,
                qrels_train_data,
                cidx.doc_ids,
                cidx.doc_tokens,
                pipe.registry,
                cidx.doc_freq,
                len(cidx.doc_ids),
            )
            print(f"  anchor re-training: {n_retrain} queries (post-discovery)", flush=True)

    # --- multi-round convergence training ---
    if do_train and not composite_only:
        qrels_test_data = load_qrels(paths.qrels_test)
        if qrels_test_data:
            print("  running convergence loop...", flush=True)
            conv_history = train_convergence_loop(
                anchor_idx,
                pipe.registry,
                phrase_idx,
                cidx.doc_tokens,
                cidx.doc_tf,
                cidx.doc_len,
                cidx.avg_dl,
                cidx.doc_freq,
                queries,
                qrels_train_data,
                qrels_test_data,
                cidx.doc_ids,
                hub_sigs,
                neighbor_map,
                sub_comp_idx,
                max_rounds=max_convergence_rounds,
                convergence_threshold=0.002,
                verbose=True,
            )
            print(
                f"  convergence NDCG@10 history: {[f'{x:.4f}' for x in conv_history]}",
                flush=True,
            )

    # --- λ calibration (grid-search LAMBDA_COORD × LAMBDA_NEIGHBOR on train qrels) ---
    if do_train and anchor_idx is not None and not composite_only:
        print("  calibrating signal weights on train qrels...", flush=True)
        calibrate_signal_weights(
            hub_sigs,
            cidx.doc_tokens,
            cidx.doc_tf,
            cidx.doc_len,
            cidx.avg_dl,
            sub_comp_idx,
            anchor_idx,
            phrase_idx,
            pipe.registry,
            queries,
            qrels_train_data,
            cidx.doc_ids,
            neighbor_map,
            cidx.doc_freq,
            len(cidx.doc_ids),
            verbose=True,
        )

    # --- save brain (compound learning for next run) ---
    if do_train and anchor_idx is not None:
        b_path.parent.mkdir(parents=True, exist_ok=True)
        save_brain(anchor_idx, _hs.LAMBDA_COORD, _hs.LAMBDA_NEIGHBOR, b_path)
        trained_count = sum(
            1 for a in anchor_idx.anchors.values()
            if a.correct_count + a.wrong_count > 0
        )
        print(
            f"  brain saved: {trained_count} trained anchors -> {b_path.name}  "
            f"(λ_coord={_hs.LAMBDA_COORD}, λ_neighbor={_hs.LAMBDA_NEIGHBOR}, "
            f"λ_pf={_hs.LAMBDA_PRIME_FACTOR})",
            flush=True,
        )

    save_distilled_registry(pipe.registry, distilled_path)
    print(f"  distilled registry saved: {distilled_path.name}", flush=True)

    bundle = EvalBundle(
        dataset=paths.name,
        mode=mode,
        qids=qids,
        queries=queries,
        qrels=qrels,
        cidx=cidx,
        hub_sigs=hub_sigs,
        neighbor_map=neighbor_map,
        meet_index=meet_index,
        sub_comp_idx=sub_comp_idx,
        comp_idx=comp_idx,
        phrase_idx=phrase_idx,
        anchor_idx=anchor_idx,
        pipe=pipe,
        hub_bytes_per_doc=hub_bytes,
        p50_ingest_ms=metrics.p50_ms,
        p99_ingest_ms=metrics.p99_ms,
        bytes_per_doc=metrics.mean_bytes_per_doc,
        n_docs=len(cidx.doc_ids),
        attractor_index=attractor_index,
    )
    if save_checkpoint:
        ckpt_out = (
            Path(save_checkpoint)
            if isinstance(save_checkpoint, (str, Path))
            else checkpoint_path(paths.name, mode)
        )
        persist_eval_checkpoint(bundle, ckpt_out)
        print(f"  eval checkpoint saved: {ckpt_out.name}", flush=True)

    if build_only:
        print("  build_only: skipping query loop (use run_ab.py --stage ab)", flush=True)
        return EvalResult(
            dataset=paths.name,
            mode=mode,
            n_docs=len(cidx.doc_ids),
            n_queries=len(qids),
            ndcg10=0.0,
            r10=0.0,
            p50_ingest_ms=metrics.p50_ms,
            p99_ingest_ms=metrics.p99_ms,
            bytes_per_doc=metrics.mean_bytes_per_doc,
            hub_bytes_per_doc=hub_bytes,
            p50_query_ms=0.0,
            bm25_ref=BM25_REF.get(paths.name),
        )

    # --- query loop (BIT 4 routing via attractor_index; 8a optional via lambda_kappa) ---
    enable_kappa_scoring = lambda_kappa is not None and lambda_kappa > 0

    ndcgs: list[float] = []
    r10s: list[float] = []
    q_times: list[float] = []

    print(f"  scoring {len(qids)} queries...", flush=True)
    try:
        loop_t0 = time.perf_counter()
        for qi, qid in enumerate(qids):
            t0 = time.perf_counter()

            result = _score_one_query(
                queries[qid],
                pipe=pipe,
                cidx=cidx,
                hub_sigs=hub_sigs,
                neighbor_map=neighbor_map,
                meet_index=meet_index,
                sub_comp_idx=sub_comp_idx,
                comp_idx=comp_idx,
                phrase_idx=phrase_idx,
                anchor_idx=anchor_idx,
                attractor_index=attractor_index,
                kappa_candidate_cap=kappa_candidate_cap,
                enable_kappa_scoring=enable_kappa_scoring,
            )

            q_ms = (time.perf_counter() - t0) * 1000.0
            q_times.append(q_ms)
            rel = qrels[qid]
            ndcg = ndcg_at_k(result.ranked, rel, 10)
            ndcgs.append(ndcg)
            r10s.append(recall_at_k(result.ranked, rel, 10))

            n_bad = record_retrieval_false_positives(
                bad_store,
                result.ranked,
                rel,
                result.profile,
                hub_sigs,
                pipe.registry,
                top_k=10,
            )
            if verbose and n_bad:
                print(f"    q={qid}: recorded {n_bad} bad-correlation signals", flush=True)

            progress_every = 1 if verbose else 50
            if (qi + 1) % progress_every == 0 or qi == 0:
                _print_query_progress(
                    arm_label="eval",
                    qi=qi,
                    total=len(qids),
                    qid=qid,
                    elapsed_ms=q_ms,
                    n_candidates=result.n_candidates,
                    route_tier=result.route_tier,
                    n_kappa_keys=result.n_kappa_keys,
                    ndcg=ndcg,
                    avg_ndcg=sum(ndcgs) / len(ndcgs),
                    loop_elapsed_s=time.perf_counter() - loop_t0,
                    enable_kappa_scoring=enable_kappa_scoring,
                )
    finally:
        if lambda_kappa is not None:
            _hs.LAMBDA_KAPPA = prev_kappa

    n_q = max(len(qids), 1)
    p50_q = sorted(q_times)[len(q_times) // 2] if q_times else 0.0

    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.save(bad_path)
    unresolved = sum(1 for e in bad_store.entries.values() if not e.resolved)
    print(
        f"  bad-correlation queue saved: {len(bad_store.entries)} pairs, "
        f"{unresolved} unresolved -> {bad_path.name}",
        flush=True,
    )
    return EvalResult(
        dataset=paths.name,
        mode=mode,
        n_docs=len(cidx.doc_ids),
        n_queries=len(qids),
        ndcg10=sum(ndcgs) / n_q,
        r10=sum(r10s) / n_q,
        p50_ingest_ms=metrics.p50_ms,
        p99_ingest_ms=metrics.p99_ms,
        bytes_per_doc=metrics.mean_bytes_per_doc,
        hub_bytes_per_doc=hub_bytes,
        p50_query_ms=p50_q,
        bm25_ref=BM25_REF.get(paths.name),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def list_datasets(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [d.name for d in sorted(root.iterdir()) if (d / "corpus.jsonl").exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHOS BEIR evaluation")
    parser.add_argument("--datasets", nargs="+", default=["scifact"])
    parser.add_argument("--mode", choices=("quality", "scale"), default="scale")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--hub-top-k", type=int, default=12)
    parser.add_argument("--passes", type=int, default=2, help="Number of corpus passes (1=BM25 only, 2=+phrases, 3=+meta-bridges)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--lambda-pf",
        type=float,
        default=None,
        help="Signal 5b weight (default 0.35); use 0 for A/B baseline",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build indices and checkpoint; skip query scoring",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Write brains/{dataset}_{mode}.eval.pkl after build",
    )
    parser.add_argument(
        "--from-checkpoint",
        type=str,
        default=None,
        help="Load eval pickle and run query loop only",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip anchor train/convergence if brain file exists",
    )
    parser.add_argument(
        "--max-convergence-rounds",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    root = Path(resolve_beir_root())
    print(f"BEIR root: {root}")

    if args.list:
        names = list_datasets(root)
        for n in names:
            print(f"  {n}")
        return 0 if names else 1

    if not root.is_dir():
        print("BEIR root not found. Set BEIR_DATA_DIR or check trng worktree.")
        return 1

    ok = True
    for name in args.datasets:
        paths = load_paths(root, name)
        print(f"\n=== {name} ===")
        try:
            result = evaluate_dataset(
                paths,
                mode=args.mode,
                max_docs=args.max_docs,
                max_queries=args.max_queries,
                hub_top_k=args.hub_top_k,
                n_passes=args.passes,
                verbose=args.verbose,
                lambda_prime_factor=args.lambda_pf,
                from_checkpoint=args.from_checkpoint,
                save_checkpoint=args.save_checkpoint or None,
                build_only=args.build_only,
                skip_training=args.skip_training,
                max_convergence_rounds=args.max_convergence_rounds,
            )
        except FileNotFoundError as exc:
            print(f"SKIP {name}: {exc}")
            ok = False
            continue
        print()
        print(result.summary())

    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
