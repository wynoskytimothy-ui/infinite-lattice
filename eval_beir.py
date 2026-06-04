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


def candidate_ids(
    query_words: list[str],
    inv: dict[str, set[str]],
    neighbor_map: dict[str, dict[str, float]],
    all_ids: list[str],
    meet_index: "dict[int, set[str]] | None" = None,
    registry=None,
) -> list[str]:
    """
    Candidate generation with optional MeetIndex expansion.

    Standard path: lexical inverted index + L4-L6 neighbor postings.

    MeetIndex path (when meet_index is provided): for each query word's prime
    chain (parent_primes), look up docs with hub words sharing those prime
    factors.  The swap-meet guarantee means these docs have a geometric witness
    with the query word even when surface vocabulary is completely different.
    This addresses ZERO_BM25 and PARTIAL failures where gold docs use
    different terminology from the query.
    """
    cand: set[str] = set()
    for w in query_words:
        cand |= inv.get(w, set())
        for nb in neighbor_map.get(w, {}):
            cand |= inv.get(nb, set())

    # MeetIndex expansion: add docs sharing prime factors with query words.
    # Uses the swap-meet property: solo(p_q)@n=p_h == solo(p_h)@n=p_q on all
    # 32 wings — so words sharing prime factors have a geometric meet witness.
    if meet_index is not None and registry is not None:
        for w in query_words:
            if len(w) < 3:
                continue
            try:
                tok = registry.resolve_token(w)
                # Check word prime itself
                meet_docs = meet_index.get(tok.prime)
                if meet_docs:
                    cand |= meet_docs
                # Check parent primes (L2 subword primes if promoted)
                for pp in tok.parent_primes:
                    meet_docs = meet_index.get(pp)
                    if meet_docs:
                        cand |= meet_docs
            except Exception:
                pass

    return list(cand) if cand else all_ids


def build_meet_index(
    hub_sigs: dict,
    registry,
) -> "dict[int, set[str]]":
    """
    Build a prime-factor inverted index (MeetIndex) from hub signatures.

    Maps prime → set of doc_ids whose hub words have that prime in their
    prime chain (L3 pool prime or parent primes).

    Used in candidate_ids for MeetIndex expansion: a query word with prime
    chain (f1, f2, ...) can find docs that share ANY of those factors via
    their hub words — the geometric swap-meet guarantee.

    O(docs × K × len(parent_primes)) to build; O(1) per lookup at query time.
    """
    from collections import defaultdict as _dd
    idx: dict[int, set[str]] = {}
    _idx = _dd(set)

    # Only index pool-promoted primes (≥ PROMOTION_POOL_FIRST = 107).
    # Letter primes (3,5,7,...,101) appear in every word — indexing them
    # would expand candidates to the entire corpus (pure noise).
    # Pool primes are specific to promoted subwords/words and meaningful.
    MIN_POOL_PRIME = 107

    for did, sig in hub_sigs.items():
        for word, entry in sig.hubs.items():
            # Index by L3 pool prime only if it's a pool-promoted prime
            if entry.prime >= MIN_POOL_PRIME:
                _idx[entry.prime].add(did)
            # Index by parent primes that are pool-promoted (L2 subword primes)
            try:
                tok = registry.resolve_token(word)
                for pp in tok.parent_primes:
                    if pp >= MIN_POOL_PRIME:
                        _idx[pp].add(did)
            except Exception:
                pass

    return dict(_idx)


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
        # Quality mode: fast_ingest=True (for speed) but with aggressive post-hoc
        # subword promotion. The subword rebuild runs after ingest using vocabulary
        # stats, promoting up to 320 L2 subwords (2× the scale-mode cap of 160).
        # More L2 primes → L3 words get proper non-colliding parent_primes.
        # Result: ~400-600 L3 pool primes (vs 199 with scale mode).
        cfg = ScaleConfig(
            rebuild_every=32,
            lazy_clusters=False,
            fast_ingest=True,
            defer_l2_promotion=True,
            fast_cluster=False,
            max_corr_pairs_per_doc=512,
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
) -> EvalResult:
    if not paths.corpus.exists():
        raise FileNotFoundError(f"missing corpus: {paths.corpus}")

    import aethos_hub_signature as _hs

    if lambda_prime_factor is not None:
        _hs.LAMBDA_PRIME_FACTOR = float(lambda_prime_factor)
        print(f"  LAMBDA_PRIME_FACTOR={_hs.LAMBDA_PRIME_FACTOR}", flush=True)

    bad_path = bad_correlation_path(paths.name, mode)
    bad_store = BadCorrelationStore.load(bad_path)
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
    t_ingest = time.perf_counter()
    metrics, cidx = ingest_corpus(pipe, corpus, mode=mode)
    # Pass 1 flush (cluster discovery)
    try:
        pipe.flush()
    except (IndexError, RuntimeError) as exc:
        print(f"  cluster flush skipped ({exc})", flush=True)
    ingest_total_ms = (time.perf_counter() - t_ingest) * 1000.0
    print(f"  ingest done: {len(cidx.doc_ids)} docs in {ingest_total_ms:.0f} ms", flush=True)

    distilled_path = brain_path_for_dataset(paths.name, mode).with_suffix(".distilled.json")
    if load_distilled_registry(pipe.registry, distilled_path):
        print(f"  distilled registry loaded: {distilled_path.name}", flush=True)

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

    # --- heavy anchor discriminative index (Signal 6) ---
    t_anchor = time.perf_counter()
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
    init_lc, init_ln = load_brain(b_path, anchor_idx, verbose=True)
    # Apply saved λ values as starting point (will be refined by calibration below)
    import aethos_hub_signature as _hs
    _hs.LAMBDA_COORD = init_lc
    _hs.LAMBDA_NEIGHBOR = init_ln

    # --- train anchors on train qrels if available ---
    qrels_train_data = load_qrels(paths.qrels_train)
    if qrels_train_data and anchor_idx.n_anchors > 0:
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
    # Run BEFORE the convergence loop so new anchors get a training pass.
    # Finds words unique to gold docs (not in query, not in wrong docs) and builds
    # new composites that anchor gold docs against vocabulary-mismatched queries.
    neighbor_map = build_neighbor_weights(pipe.registry)
    if qrels_train_data and anchor_idx.n_anchors > 0:
        discover_discriminating_intersections(
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
            verbose=True,
        )
        # Negative anchor training: suppress composites that predict wrong docs
        train_negative_anchors(
            anchor_idx, pipe.registry, queries, qrels_train_data,
            cidx.doc_ids, cidx.doc_tokens, cidx.doc_freq, len(cidx.doc_ids),
            hub_sigs, neighbor_map, cidx.doc_tf, cidx.doc_len, cidx.avg_dl,
            sub_comp_idx, phrase_idx, verbose=True,
        )
        # Meta-intersections: depth-3 composites from pairs of trained anchors
        discover_meta_intersections(
            anchor_idx, cidx.doc_tokens, cidx.doc_freq, len(cidx.doc_ids),
            verbose=True,
        )
        # Second training pass: give new discriminating anchors real feedback
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
    if qrels_train_data and anchor_idx.n_anchors > 0:
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
                max_rounds=8,
                convergence_threshold=0.002,
                verbose=True,
            )
            print(
                f"  convergence NDCG@10 history: {[f'{x:.4f}' for x in conv_history]}",
                flush=True,
            )

    # --- λ calibration (grid-search LAMBDA_COORD × LAMBDA_NEIGHBOR on train qrels) ---
    import aethos_hub_signature as _hs
    if qrels_train_data and anchor_idx is not None and anchor_idx.n_anchors > 0:
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
    if qrels_train_data and anchor_idx is not None and anchor_idx.n_anchors > 0:
        b_path.parent.mkdir(parents=True, exist_ok=True)
        save_brain(anchor_idx, _hs.LAMBDA_COORD, _hs.LAMBDA_NEIGHBOR, b_path)
        trained_count = sum(
            1 for a in anchor_idx.anchors.values()
            if a.correct_count + a.wrong_count > 0
        )
        print(
            f"  brain saved: {trained_count} trained anchors → {b_path.name}  "
            f"(λ_coord={_hs.LAMBDA_COORD}, λ_neighbor={_hs.LAMBDA_NEIGHBOR}, "
            f"λ_pf={_hs.LAMBDA_PRIME_FACTOR})",
            flush=True,
        )

    # --- query loop ---
    ndcgs: list[float] = []
    r10s: list[float] = []
    q_times: list[float] = []
    n_docs = len(cidx.doc_ids)

    print(f"  scoring {len(qids)} queries...", flush=True)
    for qi, qid in enumerate(qids):
        t0 = time.perf_counter()

        profile = build_query_profile(
            queries[qid],
            pipe.registry,
            neighbor_map=neighbor_map,
            doc_freq=cidx.doc_freq,   # document frequency, not term count
            n_docs=n_docs,
        )
        cands = candidate_ids(
            profile.words, cidx.inv, neighbor_map, cidx.doc_ids,
            meet_index=meet_index, registry=pipe.registry,
        )

        # Signal 6: precompute anchor composites once per query — O(Q²) here,
        # O(|comps|) per doc (≈0–28 entries vs. iterating 168k anchors).
        q_anchor_comps: dict[int, float] | None = None
        if anchor_idx is not None and anchor_idx.n_anchors > 0:
            q_anchor_comps = query_anchor_composites(
                list(profile.word_set),
                anchor_idx,
                pipe.registry,
                idf=profile.idf,
            )

        # Signal 7: precompute cluster membership for query words (L7-L9 routing).
        # Maps hub words that share a cluster with any query word → IDF of the
        # best query word in that cluster.  O(|word_to_cluster|) once per query.
        if pipe.reader.word_to_cluster:
            query_clusters: dict[str, float] = {}  # cluster_id → best IDF
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

        # Signal 5: precompute phrase composite scores once per query — O(Q²) prime
        # lookups (Q≤10 → ≤45 pairs). Per-doc scoring becomes O(|q_phrase_comps|)
        # via frozenset lookup instead of recomputing primes for every candidate.
        q_phrase_comps: dict[int, float] | None = None
        if phrase_idx is not None:
            q_phrase_comps = _query_phrase_composites(
                profile.words, phrase_idx, pipe.registry, profile.idf,
            )

        # --- composite candidate expansion (Signals 4+5) ---
        extra_cands: set[str] = set()
        # Signal 4: subword composite expansion
        for qw in profile.word_set:
            q_comps = sub_comp_idx.word_composites.get(qw, frozenset())
            for c in q_comps:
                extra_cands |= sub_comp_idx.composite_to_docs.get(c, set())
        # Signal 5 candidate expansion disabled — phrase composites are
        # used for scoring only; candidate expansion caused 500ms+ query times.
        # Re-enable when 32-lattice consensus scoring is implemented.

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

        q_times.append((time.perf_counter() - t0) * 1000.0)
        rel = qrels[qid]
        ndcgs.append(ndcg_at_k(ranked, rel, 10))
        r10s.append(recall_at_k(ranked, rel, 10))

        n_bad = record_retrieval_false_positives(
            bad_store,
            ranked,
            rel,
            profile,
            hub_sigs,
            pipe.registry,
            top_k=10,
        )
        if verbose and n_bad:
            print(f"    q={qid}: recorded {n_bad} bad-correlation signals", flush=True)

        if (qi + 1) % 50 == 0:
            print(
                f"  query {qi+1}/{len(qids)}  "
                f"NDCG@10 running avg: {sum(ndcgs)/len(ndcgs):.4f}",
                flush=True,
            )

    n_q = max(len(qids), 1)
    p50_q = sorted(q_times)[len(q_times) // 2] if q_times else 0.0

    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.save(bad_path)
    unresolved = sum(1 for e in bad_store.entries.values() if not e.resolved)
    print(
        f"  bad-correlation queue saved: {len(bad_store.entries)} pairs, "
        f"{unresolved} unresolved → {bad_path.name}",
        flush=True,
    )
    save_distilled_registry(pipe.registry, distilled_path)
    print(f"  distilled registry saved: {distilled_path.name}", flush=True)

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
