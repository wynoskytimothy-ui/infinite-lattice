"""
Geometry-native fast ingest + train — single-doc κ indexing, incremental plane patch.

Avoids full plane rebuilds; pairs with TrinaryTrainer for one-query supervision.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path
from aethos_symbol_trinary_train import TrinaryTrainer
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_02_attractor_key import kappa_branch_fan
from pipeline.bit_12_symbol_plane_index import (
    SymbolPlaneIndex,
    _rail_from_imag,
    correlation_meet_keys,
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
    symbol_word_chain,
    symbol_word_imag,
)

if TYPE_CHECKING:
    pass

_TOKEN_RE = re.compile(r"[a-z]+")
_ROOT = Path(__file__).resolve().parent


class FastTrinaryTrainer(TrinaryTrainer):
    """TrinaryTrainer with lazy doc-frequency cache (query/gold tokens only)."""

    def __init__(self, knowledge: SymbolKnowledgeIndex) -> None:
        super().__init__(knowledge, max_triples_per_query=8)
        self._doc_freq: dict[str, int] = {}

    def warm_doc_freq(self, *texts: str) -> None:
        """Pre-cache df for supervised tokens in a single corpus pass."""
        pending: set[str] = set()
        for text in texts:
            pending.update(self._tokens(text))
        pending -= set(self._doc_freq)
        if not pending:
            return
        counts = dict.fromkeys(pending, 0)
        for doc_text in self.knowledge.corpus.values():
            doc_toks = set(self._tokens(doc_text))
            for w in pending:
                if w in doc_toks:
                    counts[w] += 1
        self._doc_freq.update(counts)

    def _word_doc_freq(self, word: str) -> int:
        w = word.lower()
        if w not in self._doc_freq:
            self.warm_doc_freq(w)
        return self._doc_freq.get(w, 0)


def _unique_tokens(text: str, *, min_len: int = 2) -> list[str]:
    return list(dict.fromkeys(
        t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= min_len
    ))


def ingest_doc_fast(
    plane: SymbolPlaneIndex,
    knowledge: SymbolKnowledgeIndex,
    doc_id: str,
    text: str,
    *,
    update_knowledge: bool = True,
) -> dict[str, int | float]:
    """
    Index one document on the κ plane (per-token branch fan).

    Optionally ingests text into knowledge first (single-doc fast path).
    """
    t0 = time.perf_counter()
    if update_knowledge:
        knowledge.ingest_corpus({doc_id: text}, subjects=None, lazy_chambers=True)

    keys_added = 0
    tokens = _unique_tokens(text)
    for w in tokens:
        chain = symbol_word_chain(knowledge, w)
        if not chain:
            continue
        rail = _rail_from_imag(symbol_word_imag(knowledge, w))
        for key in kappa_branch_fan(chain, rail, quantize=plane.quantize):
            before = len(plane.doc_keys.get(doc_id, set()))
            plane.add(doc_id, key, word=w)
            after = len(plane.doc_keys.get(doc_id, set()))
            if after > before:
                keys_added += 1

    return {
        "doc_id": doc_id,
        "tokens": len(tokens),
        "keys_added": keys_added,
        "ingest_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def patch_plane_for_words(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    touch_words: set[str],
    *,
    max_new_pairs: int = 64,
    max_neighbors: int = 12,
) -> dict[str, int]:
    """Incremental adjacency + pair_meet patch for query/triple words only."""
    if not plane.word_adjacency:
        plane.word_adjacency = {}

    touched_links = 0
    added_pairs = 0
    for w in sorted(touch_words):
        neighbors = knowledge.neighbors(w)[:max_neighbors]
        for lk in neighbors:
            other = lk.right if lk.left == w else lk.left
            plane.word_adjacency.setdefault(w, [])
            plane.word_adjacency.setdefault(other, [])
            entry = (other, lk.strength, lk.kind)
            rev = (w, lk.strength, lk.kind)
            if entry not in plane.word_adjacency[w]:
                plane.word_adjacency[w].append(entry)
                touched_links += 1
            if rev not in plane.word_adjacency[other]:
                plane.word_adjacency[other].append(rev)
            key_pair = tuple(sorted((w, other)))
            if not plane.pair_keys.get(key_pair):
                meet = correlation_meet_keys(knowledge, w, other, link=lk)
                if meet:
                    plane.pair_keys[key_pair] = meet
                    added_pairs += 1
                    if added_pairs >= max_new_pairs:
                        break
        if added_pairs >= max_new_pairs:
            break
        if w in plane.word_adjacency:
            plane.word_adjacency[w].sort(key=lambda x: (-x[1], x[0]))

    return {"touched_links": touched_links, "pair_meets_added": added_pairs}


def train_query_fast(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    query_id: str,
    query_text: str,
    gold_ids: Sequence[str],
    *,
    trainer: FastTrinaryTrainer | None = None,
) -> dict[str, object]:
    """Trinary train one query + incremental plane patch (no full rebuild)."""
    t0 = time.perf_counter()
    tr = trainer or FastTrinaryTrainer(knowledge)
    gold_texts = [knowledge.corpus.get(gid, "") for gid in gold_ids]
    tr.warm_doc_freq(query_text, *gold_texts)
    report = tr.train_query(query_id, query_text, gold_ids)

    touch: set[str] = set(query_words(query_text))
    for prom in report.promoted:
        touch.update(prom.words)

    patch = patch_plane_for_words(knowledge, plane, touch)
    train_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "query_id": query_id,
        "train_ms": round(train_ms, 2),
        "triples_promoted": report.triples_promoted,
        "plane_patch": patch,
        "touch_words": len(touch),
        "report": report,
        "trainer": tr,
    }


def bench_one_query(
    query_id: str,
    *,
    dataset: str = "scifact",
    brain_name: str | None = None,
    pretrain_q1: bool = False,
) -> dict[str, object]:
    """
    Benchmark one query: load, route-before, train, route-after, rank.

    Returns timing dict with load_ms, route_before_ms, train_ms, route_after_ms,
    rank_after_ms, gold_in_route, ndcg@10, top5.
    """
    from beir_data_root import resolve_beir_root

    brain = brain_name or dataset
    paths = load_paths(Path(resolve_beir_root()), dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    qid = str(query_id)
    if qid not in queries or qid not in qrels:
        raise KeyError(f"query {qid!r} not in {dataset} test split")

    t_load = time.perf_counter()
    knowledge, plane = load_brain_and_plane(brain)
    load_ms = (time.perf_counter() - t_load) * 1000.0

    if pretrain_q1 and qid == "1":
        from aethos_symbol_knowledge import PRETRAIN_QUANTUM_GOLD
        knowledge.compound_learn(PRETRAIN_QUANTUM_GOLD, subjects={1, 9, 10})

    query_text = queries[qid]
    words = query_words(query_text)
    rel = qrels[qid]
    gold_ids = list(rel.keys())

    t0 = time.perf_counter()
    route_before = route_symbol_plane_candidates(knowledge, plane, words)
    route_before_ms = (time.perf_counter() - t0) * 1000.0

    train_out = train_query_fast(knowledge, plane, qid, query_text, gold_ids)
    train_ms = float(train_out["train_ms"])

    t0 = time.perf_counter()
    route_after = route_symbol_plane_candidates(knowledge, plane, words)
    route_after_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    ranked = [doc_id for doc_id, _ in rank_symbol_plane_docs(
        knowledge, plane, words, limit=100,
    )]
    rank_after_ms = (time.perf_counter() - t0) * 1000.0

    gold_in_route = any(g in route_after.doc_ids for g in rel)
    ndcg = ndcg_at_k(ranked, rel, 10)

    return {
        "query_id": qid,
        "query": query_text[:160],
        "dataset": dataset,
        "brain": brain,
        "gold": gold_ids,
        "load_ms": round(load_ms, 2),
        "route_before_ms": round(route_before_ms, 2),
        "train_ms": round(train_ms, 2),
        "route_after_ms": round(route_after_ms, 2),
        "rank_after_ms": round(rank_after_ms, 2),
        "gold_in_route": gold_in_route,
        "gold_in_route_before": any(g in route_before.doc_ids for g in rel),
        "ndcg_at_10": round(ndcg, 6),
        "top5": ranked[:5],
        "triples_promoted": train_out["triples_promoted"],
        "plane_patch": train_out["plane_patch"],
        "route_pool_before": len(route_before.doc_ids),
        "route_pool_after": len(route_after.doc_ids),
    }


def save_benchmark(result: dict[str, object], path: Path | None = None) -> Path:
    import json

    out = path or (_ROOT / "logs" / "geometry_speed_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out
