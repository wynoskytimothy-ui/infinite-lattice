#!/usr/bin/env python3
"""
BEIR evaluation — symbol knowledge brain + BIT 12 κ plane router.

Uses saved SymbolKnowledgeIndex + SymbolPlaneIndex (no hub-signature pipeline).

  python eval_beir_symbol.py --dataset scifact
  python eval_beir_symbol.py --dataset scifact --max-queries 50
  python eval_beir_symbol.py --dataset scifact --split test --out logs/eval_symbol_scifact.json

Metrics: nDCG@10, Recall@10, MRR@10, gold-in-candidate-pool (routing recall).
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path
from aethos_tokenize import tokenize_words
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, merge_qrels, ndcg_at_k, recall_at_k
from pipeline.bit_04_candidate_router import query_words_for_routing
from aethos_cascade_retrieval import search_docs_cascade
from aethos_rare_rank import _DocFreqCache, search_docs_rare_correlations
from pipeline.bit_12_symbol_plane_index import (
    SymbolPlaneIndex,
    rank_symbol_plane_docs,
    rank_symbol_plane_witness,
    route_symbol_plane_candidates,
)


def plane_index_path(dataset: str) -> Path:
    return knowledge_path(dataset).parent / f"{dataset}_plane.pkl"


def load_brain_and_plane(
    brain_name: str,
    *,
    brain_path: Path | None = None,
    plane_path: Path | None = None,
) -> tuple[SymbolKnowledgeIndex, SymbolPlaneIndex]:
    kpath = Path(brain_path) if brain_path else knowledge_path(brain_name)
    ppath = Path(plane_path) if plane_path else plane_index_path(brain_name)

    if not kpath.is_file():
        raise FileNotFoundError(f"symbol knowledge not found: {kpath}")
    if not ppath.is_file():
        raise FileNotFoundError(
            f"plane index not found: {ppath}\n"
            f"  run: python scripts/build_symbol_plane_index.py --dataset {brain_name}",
        )

    t0 = time.perf_counter()
    knowledge = SymbolKnowledgeIndex.load(brain_name, path=kpath)
    with open(ppath, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, tuple) and len(payload) == 2:
        _kb, plane = payload
    elif isinstance(payload, SymbolPlaneIndex):
        plane = payload
    else:
        raise TypeError(f"unexpected plane pickle type: {type(payload)}")
    from pipeline.bit_12_symbol_plane_index import (
        _canonical_pair_link_cache,
        _canonical_surface_map,
    )

    _canonical_surface_map(knowledge)
    _canonical_pair_link_cache(knowledge)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"  loaded brain + plane in {load_ms:.0f} ms  "
        f"(docs={len(knowledge.corpus)}  kappa_buckets={len(plane.by_key)})",
        flush=True,
    )
    return knowledge, plane


def mrr_at_k(ranked: list[str], rel: dict[str, int], k: int = 10) -> float:
    for i, doc_id in enumerate(ranked[:k]):
        if doc_id in rel:
            return 1.0 / (i + 1)
    return 0.0


def query_words(text: str) -> list[str]:
    return query_words_for_routing(tokenize_words(text))


@dataclass
class SymbolEvalResult:
    dataset: str
    split: str
    n_queries: int
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    route_recall: float
    mean_query_ms: float
    p95_query_ms: float
    failures: list[dict[str, object]] = field(default_factory=list)
    n_skipped: int = 0
    skipped_qids: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Symbol BEIR — {self.dataset} ({self.split})",
            f"  queries     : {self.n_queries}"
            + (f"  (skipped {self.n_skipped})" if self.n_skipped else ""),
            f"  nDCG@10     : {self.ndcg_at_10:.4f}",
            f"  Recall@10   : {self.recall_at_10:.4f}",
            f"  MRR@10      : {self.mrr_at_10:.4f}",
            f"  route recall: {self.route_recall:.4f}  (gold in kappa candidate pool)",
            f"  query ms    : mean={self.mean_query_ms:.2f}  p95={self.p95_query_ms:.2f}",
        ]
        if self.failures:
            lines.append(f"  zero-ndcg   : {len(self.failures)} queries")
        return "\n".join(lines)


def evaluate_symbol_beir(
    knowledge: SymbolKnowledgeIndex,
    plane: SymbolPlaneIndex,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    max_queries: int | None = None,
    rank_limit: int = 100,
    route_max: int = 1200,
    max_keys: int = 768,
    max_corr_neighbors: int = 4,
    expand_correlations: bool = True,
    save_failures: int = 25,
    mode: str = "kappa",
    use_idf_weighting: bool = False,
    word_attributed_pool: bool = False,
    rare_boost_hits: bool = False,
    rare_only_hits: bool = False,
    df_cache: _DocFreqCache | None = None,
) -> SymbolEvalResult:
    qids = [q for q in qrels if q in queries]
    if max_queries is not None:
        qids = qids[:max_queries]

    cache = df_cache
    if (word_attributed_pool or use_idf_weighting or rare_boost_hits or rare_only_hits) and cache is None:
        cache = _DocFreqCache(knowledge)
        cache.warm_corpus()

    route_kw = {
        "use_idf_weighting": use_idf_weighting,
        "word_attributed_pool": word_attributed_pool,
        "rare_boost_hits": rare_boost_hits,
        "rare_only_hits": rare_only_hits,
        "df_cache": cache,
    }

    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    route_hits: list[float] = []
    query_ms: list[float] = []
    failures: list[dict[str, object]] = []
    skipped: list[str] = []

    for qid in qids:
        words = query_words(queries[qid])
        rel = qrels[qid]
        t0 = time.perf_counter()
        try:
            if mode == "cascade":
                route, scored = search_docs_cascade(
                    knowledge,
                    plane,
                    words,
                    max_candidates=route_max,
                    limit=rank_limit,
                    max_keys=max_keys,
                    max_corr_neighbors=max_corr_neighbors,
                    expand_correlations=expand_correlations,
                )
                ranked = [doc_id for doc_id, _ in scored]
            elif mode == "rare":
                route, scored = search_docs_rare_correlations(
                    knowledge,
                    plane,
                    words,
                    max_candidates=route_max,
                    limit=rank_limit,
                )
                ranked = [doc_id for doc_id, _ in scored]
            elif mode == "witness":
                route = route_symbol_plane_candidates(
                    knowledge,
                    plane,
                    words,
                    max_candidates=route_max,
                    max_keys=max_keys,
                    max_corr_neighbors=max_corr_neighbors,
                    expand_correlations=expand_correlations,
                    **route_kw,
                )
                ranked = [doc_id for doc_id, _ in rank_symbol_plane_witness(
                    knowledge,
                    plane,
                    words,
                    limit=rank_limit,
                    query_keys=set(route.query_keys),
                    candidate_doc_ids=route.doc_ids,
                    expand_correlations=expand_correlations,
                )]
            else:
                route = route_symbol_plane_candidates(
                    knowledge,
                    plane,
                    words,
                    max_candidates=route_max,
                    max_keys=max_keys,
                    max_corr_neighbors=max_corr_neighbors,
                    expand_correlations=expand_correlations,
                    **route_kw,
                )
                ranked = [doc_id for doc_id, _ in rank_symbol_plane_docs(
                    knowledge,
                    plane,
                    words,
                    limit=rank_limit,
                    query_keys=set(route.query_keys),
                    candidate_doc_ids=route.doc_ids,
                    expand_correlations=expand_correlations,
                )]
        except Exception:
            skipped.append(qid)
            continue
        elapsed = (time.perf_counter() - t0) * 1000.0
        query_ms.append(elapsed)

        ndcg = ndcg_at_k(ranked, rel, 10)
        rec = recall_at_k(ranked, rel, 10)
        mrr = mrr_at_k(ranked, rel, 10)
        in_route = 1.0 if any(g in route.doc_ids for g in rel) else 0.0

        ndcgs.append(ndcg)
        recalls.append(rec)
        mrrs.append(mrr)
        route_hits.append(in_route)

        if ndcg == 0.0 and len(failures) < save_failures:
            failures.append({
                "query_id": qid,
                "query": queries[qid][:120],
                "gold": list(rel.keys()),
                "top5": ranked[:5],
                "route_pool": len(route.doc_ids),
                "gold_in_route": bool(in_route),
                "query_ms": round(elapsed, 2),
            })

    n = max(len(ndcgs), 1)
    query_ms_sorted = sorted(query_ms)
    p95_idx = min(int(math.ceil(0.95 * len(query_ms_sorted))) - 1, len(query_ms_sorted) - 1)

    return SymbolEvalResult(
        dataset="",
        split="",
        n_queries=len(ndcgs),
        ndcg_at_10=sum(ndcgs) / n,
        recall_at_10=sum(recalls) / n,
        mrr_at_10=sum(mrrs) / n,
        route_recall=sum(route_hits) / n,
        mean_query_ms=sum(query_ms) / n if query_ms else 0.0,
        p95_query_ms=query_ms_sorted[p95_idx] if query_ms_sorted else 0.0,
        failures=failures,
        n_skipped=len(skipped),
        skipped_qids=skipped,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="BEIR eval on symbol knowledge + κ plane")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--brain", default=None, help="knowledge dataset name (default: --dataset)")
    p.add_argument("--brain-path", default=None)
    p.add_argument("--plane-path", default=None)
    p.add_argument("--split", choices=("test", "train", "all"), default="test")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--rank-limit", type=int, default=100)
    p.add_argument("--route-max", type=int, default=1200)
    p.add_argument(
        "--word-attributed-pool",
        action="store_true",
        help="word-attributed IDF pool (default: uniform per-κ-key hit count)",
    )
    p.add_argument(
        "--uniform-pool",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias; uniform is now default
    )
    p.add_argument(
        "--mode",
        choices=("cascade", "rare", "kappa", "witness"),
        default="kappa",
        help="kappa = asymmetric κ overlap; witness = intersection rank (meet+rare); cascade/rare = legacy",
    )
    p.add_argument("--max-keys", type=int, default=768)
    p.add_argument("--max-corr-neighbors", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    brain_name = args.brain or args.dataset
    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)

    print(f"=== Symbol BEIR eval: {args.dataset} ===", flush=True)
    print(f"Loading brain={brain_name!r} ...", flush=True)
    knowledge, plane = load_brain_and_plane(
        brain_name,
        brain_path=Path(args.brain_path) if args.brain_path else None,
        plane_path=Path(args.plane_path) if args.plane_path else None,
    )

    queries = load_queries(paths.queries)
    if args.split == "test":
        qrels = load_qrels(paths.qrels_test)
    elif args.split == "train":
        qrels = load_qrels(paths.qrels_train)
    else:
        qrels = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))

    n_eval = len([q for q in qrels if q in queries])
    if args.max_queries is not None:
        n_eval = min(n_eval, args.max_queries)
    print(f"Evaluating {n_eval} queries (split={args.split}) ...", flush=True)
    t0 = time.perf_counter()
    result = evaluate_symbol_beir(
        knowledge,
        plane,
        queries,
        qrels,
        max_queries=args.max_queries,
        rank_limit=args.rank_limit,
        route_max=args.route_max,
        mode=args.mode,
        max_keys=args.max_keys,
        max_corr_neighbors=args.max_corr_neighbors,
        word_attributed_pool=args.word_attributed_pool and not args.uniform_pool,
    )
    result.dataset = args.dataset
    result.split = args.split
    eval_ms = (time.perf_counter() - t0) * 1000.0

    print()
    print(result.summary())
    print(f"  eval wall   : {eval_ms:.0f} ms")

    out = Path(args.out or _ROOT / "logs" / f"eval_beir_symbol_{args.dataset}_{args.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "brain": brain_name,
        "split": args.split,
        "mode": args.mode,
        "route_max": args.route_max,
        "word_attributed_pool": args.word_attributed_pool and not args.uniform_pool,
        "n_queries": result.n_queries,
        "n_skipped": result.n_skipped,
        "skipped_qids": result.skipped_qids,
        "ndcg_at_10": round(result.ndcg_at_10, 6),
        "recall_at_10": round(result.recall_at_10, 6),
        "mrr_at_10": round(result.mrr_at_10, 6),
        "route_recall": round(result.route_recall, 6),
        "mean_query_ms": round(result.mean_query_ms, 3),
        "p95_query_ms": round(result.p95_query_ms, 3),
        "eval_wall_ms": round(eval_ms, 1),
        "failures": result.failures,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
