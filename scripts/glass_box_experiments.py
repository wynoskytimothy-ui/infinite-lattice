#!/usr/bin/env python3
"""
Glass-box experiment matrix for SciFact symbol-plane retrieval.

Runs interpretable configs (no BM25) and logs per-query + aggregate metrics
for rule extraction. Output: logs/glass_box_experiment_matrix.json

  PYTHONPATH=. python scripts/glass_box_experiments.py
  PYTHONPATH=. python scripts/glass_box_experiments.py --max-queries 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_rare_rank import (
    _DocFreqCache,
    _rare_word_cached,
    degree_map_from_plane,
    is_hub_word,
)
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, merge_qrels, ndcg_at_k, recall_at_k
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    _query_pair_meets,
    rank_symbol_plane_docs,
    rank_symbol_plane_witness,
    route_symbol_plane_candidates,
    score_doc_meet_witness,
)

ROUTE_MAX = 1200
NARROW_CAP = 100
RANK_LIMIT = 100


@dataclass
class QueryCtx:
    qid: str
    words: list[str]
    rarest_word: str
    rarest_docs: set[str]
    rare_query: list[str]
    canon_rare: list[str]
    pair_meets_all: list
    pair_meets_rare: list
    query_keys: frozenset


def _docs_for_word(plane, word: str) -> set[str]:
    out: set[str] = set()
    for k in plane.keys_for_word(word):
        out.update(plane.by_key.get(k, ()))
    return out


def _rarest_query_word(
    knowledge,
    words: Sequence[str],
    cache: _DocFreqCache,
) -> str:
    if not words:
        return ""
    return min(
        words,
        key=lambda w: (cache.get(w), len(w), w),
    )


def _hub_penalty(
    knowledge,
    words: Sequence[str],
    doc_id: str,
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> float:
    text = knowledge.corpus.get(doc_id, "")
    if not text:
        return -0.5
    toks = {t for t in text.lower().split() if len(t) >= 3}
    rare_hit = any(
        _rare_word_cached(knowledge, w, df_cache=cache, rare_cache={}, degrees=degrees)
        and w.lower() in toks
        for w in words
    )
    if rare_hit:
        return 0.0
    hub_hits = sum(
        1 for w in words
        if is_hub_word(knowledge, w, df_cache=cache, degrees=degrees) and w.lower() in toks
    )
    if hub_hits >= 2:
        return -2.0
    if hub_hits >= 1:
        return -1.0
    return -0.25


def build_query_ctx(
    knowledge,
    plane,
    qid: str,
    query_text: str,
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
    max_keys: int,
    max_corr_neighbors: int,
) -> QueryCtx:
    from pipeline.bit_12_symbol_plane_index import query_symbol_plane_keys

    words = query_words(query_text)
    rare_cache: dict[str, bool] = {}
    rare_query = [
        w.lower() for w in words
        if _rare_word_cached(
            knowledge, w, df_cache=cache, rare_cache=rare_cache, degrees=degrees,
        )
    ]
    rarest = _rarest_query_word(knowledge, words, cache)
    ca = knowledge.morph_canonical_surface(rarest) if rarest else ""
    rarest_docs = _docs_for_word(plane, ca) if ca else set()
    canon_rare = list(dict.fromkeys(
        knowledge.morph_canonical_surface(w) for w in rare_query
    ))
    routed = list(dict.fromkeys(
        knowledge.morph_canonical_surface(w) for w in words
    ))
    keys = query_symbol_plane_keys(
        knowledge, plane, words,
        max_keys=max_keys, max_corr_neighbors=max_corr_neighbors,
    )
    return QueryCtx(
        qid=qid,
        words=words,
        rarest_word=ca,
        rarest_docs=rarest_docs,
        rare_query=rare_query,
        canon_rare=canon_rare,
        pair_meets_all=_query_pair_meets(knowledge, plane, routed),
        pair_meets_rare=_query_pair_meets(knowledge, plane, canon_rare),
        query_keys=frozenset(keys),
    )


def _meet_score(
    knowledge,
    plane,
    ctx: QueryCtx,
    doc_id: str,
    *,
    rare_only: bool,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> float:
    pair_meets = ctx.pair_meets_rare if rare_only else ctx.pair_meets_all
    return score_doc_meet_witness(
        knowledge, plane, ctx.words, doc_id,
        query_keys=set(ctx.query_keys),
        df_cache=cache,
        degrees=degrees,
        rare_q=ctx.rare_query,
        pair_meets=pair_meets,
    )


def _kappa_score(plane, ctx: QueryCtx, doc_id: str) -> float:
    if not ctx.query_keys:
        return 0.0
    return plane.score_overlap_asymmetric(ctx.query_keys, doc_id)


def narrow_cascade2(
    knowledge,
    plane,
    pool: Sequence[str],
    ctx: QueryCtx,
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
    rare_only_meets: bool,
    cap: int = NARROW_CAP,
) -> list[str]:
    compound: list[tuple[str, float]] = []
    rarest: list[tuple[str, float]] = []
    kappa: list[tuple[str, float]] = []

    for did in pool:
        meet = _meet_score(
            knowledge, plane, ctx, did,
            rare_only=rare_only_meets, cache=cache, degrees=degrees,
        )
        kap = _kappa_score(plane, ctx, did)
        hub = _hub_penalty(
            knowledge, ctx.words, did, cache=cache, degrees=degrees,
        )
        if meet > 0:
            compound.append((did, meet * 3.0 + kap * 0.15 + hub))
        elif did in ctx.rarest_docs:
            rarest.append((did, kap + hub + 0.5))
        elif kap > 0:
            kappa.append((did, kap * 0.5 + hub))

    out: list[str] = []
    for tier in (
        sorted(compound, key=lambda x: (-x[1], x[0])),
        sorted(rarest, key=lambda x: (-x[1], x[0])),
        sorted(kappa, key=lambda x: (-x[1], x[0])),
    ):
        for did, _ in tier:
            if did not in out:
                out.append(did)
            if len(out) >= cap:
                return out[:cap]

    if len(out) < cap:
        for did in pool:
            if did not in out:
                out.append(did)
            if len(out) >= cap:
                break
    return out[:cap]


def narrow_pair_meet(
    knowledge,
    plane,
    pool: Sequence[str],
    ctx: QueryCtx,
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
    rare_only_meets: bool = False,
    cap: int = NARROW_CAP,
) -> list[str]:
    scored: list[tuple[str, float]] = []
    for did in pool:
        meet = _meet_score(
            knowledge, plane, ctx, did,
            rare_only=rare_only_meets, cache=cache, degrees=degrees,
        )
        if meet > 0:
            scored.append((did, meet))
    scored.sort(key=lambda x: (-x[1], x[0]))
    out = [did for did, _ in scored[:cap]]
    if len(out) < cap:
        rest = sorted(
            (d for d in pool if d not in out),
            key=lambda d: (-_kappa_score(plane, ctx, d), d),
        )
        for did in rest:
            out.append(did)
            if len(out) >= cap:
                break
    return out[:cap]


def narrow_rarest_only(
    plane,
    pool: Sequence[str],
    ctx: QueryCtx,
    *,
    cap: int = NARROW_CAP,
) -> list[str]:
    hits = [did for did in pool if did in ctx.rarest_docs]
    hits.sort(key=lambda d: (-_kappa_score(plane, ctx, d), d))
    return hits[:cap]


def gold_bucket(
    knowledge,
    plane,
    ctx: QueryCtx,
    gold_ids: set[str],
    route_pool: set[str],
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> str:
    gold_in_route = [g for g in gold_ids if g in route_pool]
    if not gold_in_route:
        return "missed"
    gid = gold_in_route[0]
    meet = _meet_score(
        knowledge, plane, ctx, gid,
        rare_only=False, cache=cache, degrees=degrees,
    )
    if meet > 0:
        return "compound"
    if gid in ctx.rarest_docs:
        return "rarest-only"
    return "kappa-only"


def gold_rank(ranked: list[str], gold_ids: set[str]) -> int | None:
    for i, did in enumerate(ranked):
        if did in gold_ids:
            return i + 1
    return None


@dataclass
class Config:
    name: str
    label: str
    narrow: str  # none | cascade2 | pair_meet | rarest
    rerank: str  # kappa | witness
    rare_only_meets: bool = False
    witness_all_pairs: bool = True


CONFIGS: list[Config] = [
    Config("baseline_kappa", "uniform route 1200 + asymmetric κ rank", "none", "kappa"),
    Config("witness_1200", "witness rerank on full 1200 pool", "none", "witness"),
    Config(
        "cascade2_witness",
        "route 1200 → cascade2 narrow 100 → witness rerank",
        "cascade2", "witness", rare_only_meets=False, witness_all_pairs=True,
    ),
    Config(
        "narrow_pair_meet",
        "route 1200 → pair-meet narrow 100 → κ rank",
        "pair_meet", "kappa", rare_only_meets=False,
    ),
    Config(
        "narrow_rarest",
        "route 1200 → rarest-word pool cap 100 → κ rank",
        "rarest", "kappa",
    ),
    Config(
        "cascade2_rare_meets",
        "cascade2 narrow + witness (rare-only pair meets)",
        "cascade2", "witness", rare_only_meets=True, witness_all_pairs=False,
    ),
    Config(
        "cascade2_all_meets",
        "cascade2 narrow + witness (all-pair meets)",
        "cascade2", "witness", rare_only_meets=False, witness_all_pairs=True,
    ),
]


@dataclass
class QueryResult:
    qid: str
    route_recall: bool
    recall_10: float
    recall_25: float
    recall_100: float
    ndcg_10: float
    gold_rank: int | None
    bucket: str
    query_ms: float
    narrow_pool: int = 0


@dataclass
class ConfigResult:
    config: Config
    per_query: list[QueryResult] = field(default_factory=list)
    route_recall: float = 0.0
    recall_at_10: float = 0.0
    recall_at_25: float = 0.0
    recall_at_100: float = 0.0
    ndcg_at_10: float = 0.0
    mean_query_ms: float = 0.0
    bucket_counts: dict[str, int] = field(default_factory=dict)
    gold_rank_median: float | None = None


def run_one_query(
    knowledge,
    plane,
    cfg: Config,
    ctx: QueryCtx,
    rel: dict[str, int],
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
    max_keys: int,
    max_corr_neighbors: int,
) -> QueryResult:
    gold_ids = set(rel)
    t0 = time.perf_counter()

    route = route_symbol_plane_candidates(
        knowledge, plane, ctx.words,
        max_candidates=ROUTE_MAX,
        max_keys=max_keys,
        max_corr_neighbors=max_corr_neighbors,
    )
    pool = list(route.doc_ids)
    route_set = set(pool)

    if cfg.narrow == "cascade2":
        candidates = narrow_cascade2(
            knowledge, plane, pool, ctx,
            cache=cache, degrees=degrees,
            rare_only_meets=cfg.rare_only_meets,
        )
    elif cfg.narrow == "pair_meet":
        candidates = narrow_pair_meet(
            knowledge, plane, pool, ctx,
            cache=cache, degrees=degrees,
            rare_only_meets=cfg.rare_only_meets,
        )
    elif cfg.narrow == "rarest":
        candidates = narrow_rarest_only(plane, pool, ctx)
        if not candidates:
            candidates = pool[:NARROW_CAP]
    else:
        candidates = pool

    if cfg.rerank == "witness":
        scored = rank_symbol_plane_witness(
            knowledge, plane, ctx.words,
            limit=RANK_LIMIT,
            query_keys=set(ctx.query_keys),
            candidate_doc_ids=candidates,
            all_pair_meets=cfg.witness_all_pairs,
        )
    else:
        scored = rank_symbol_plane_docs(
            knowledge, plane, ctx.words,
            limit=RANK_LIMIT,
            query_keys=set(ctx.query_keys),
            candidate_doc_ids=candidates,
            asymmetric=True,
        )

    ranked = [did for did, _ in scored]
    elapsed = (time.perf_counter() - t0) * 1000.0

    return QueryResult(
        qid=ctx.qid,
        route_recall=any(g in route_set for g in gold_ids),
        recall_10=recall_at_k(ranked, rel, 10),
        recall_25=recall_at_k(ranked, rel, 25),
        recall_100=recall_at_k(ranked, rel, 100),
        ndcg_10=ndcg_at_k(ranked, rel, 10),
        gold_rank=gold_rank(ranked, gold_ids) if any(g in route_set for g in gold_ids) else None,
        bucket=gold_bucket(
            knowledge, plane, ctx, gold_ids, route_set,
            cache=cache, degrees=degrees,
        ),
        query_ms=elapsed,
        narrow_pool=len(candidates),
    )


def aggregate_config(cfg: Config, rows: list[QueryResult]) -> ConfigResult:
    n = max(len(rows), 1)
    buckets: dict[str, int] = {}
    ranks: list[int] = []
    for r in rows:
        buckets[r.bucket] = buckets.get(r.bucket, 0) + 1
        if r.gold_rank is not None:
            ranks.append(r.gold_rank)
    return ConfigResult(
        config=cfg,
        per_query=rows,
        route_recall=sum(r.route_recall for r in rows) / n,
        recall_at_10=sum(r.recall_10 for r in rows) / n,
        recall_at_25=sum(r.recall_25 for r in rows) / n,
        recall_at_100=sum(r.recall_100 for r in rows) / n,
        ndcg_at_10=sum(r.ndcg_10 for r in rows) / n,
        mean_query_ms=sum(r.query_ms for r in rows) / n,
        bucket_counts=buckets,
        gold_rank_median=statistics.median(ranks) if ranks else None,
    )


def write_summary_txt(path: Path, results: list[ConfigResult]) -> None:
    lines = ["Glass-box experiment matrix — SciFact test", "=" * 60, ""]
    header = f"{'config':<22} {'route':>6} {'R@10':>6} {'R@25':>6} {'R@100':>7} {'nDCG':>6} {'ms/q':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for cr in sorted(results, key=lambda x: -x.recall_at_10):
        lines.append(
            f"{cr.config.name:<22} "
            f"{cr.route_recall:>6.3f} "
            f"{cr.recall_at_10:>6.3f} "
            f"{cr.recall_at_25:>6.3f} "
            f"{cr.recall_at_100:>7.3f} "
            f"{cr.ndcg_at_10:>6.3f} "
            f"{cr.mean_query_ms:>7.1f}"
        )
    lines.append("")
    for cr in results:
        lines.append(f"{cr.config.name}: buckets={cr.bucket_counts}")
        if cr.gold_rank_median is not None:
            lines.append(f"  gold rank median (in pool): {cr.gold_rank_median:.1f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_rules(
    results: list[ConfigResult],
    queries: dict[str, str],
    out_path: Path,
) -> None:
    by_name = {cr.config.name: cr for cr in results}
    best_r10 = max(results, key=lambda x: x.recall_at_10)
    best_ndcg = max(results, key=lambda x: x.ndcg_at_10)
    best_r100 = max(results, key=lambda x: x.recall_at_100)

    def compare_queries(a: ConfigResult, b: ConfigResult, n: int = 5) -> tuple[list, list]:
        wins: list[dict] = []
        losses: list[dict] = []
        for qa, qb in zip(a.per_query, b.per_query):
            if qa.qid != qb.qid:
                continue
            if qa.recall_10 > qb.recall_10 and qa.recall_10 > 0:
                wins.append({
                    "qid": qa.qid,
                    "query": queries.get(qa.qid, "")[:100],
                    "bucket": qa.bucket,
                    "a_r10": qa.recall_10,
                    "b_r10": qb.recall_10,
                })
            elif qb.recall_10 > qa.recall_10 and qb.recall_10 > 0:
                losses.append({
                    "qid": qa.qid,
                    "query": queries.get(qa.qid, "")[:100],
                    "bucket": qb.bucket,
                    "a_r10": qa.recall_10,
                    "b_r10": qb.recall_10,
                })
        return wins[:n], losses[:n]

    baseline = by_name.get("baseline_kappa")
    witness = by_name.get("witness_1200")
    cascade2 = by_name.get("cascade2_witness")

    lines = [
        "# Glass-box rules — SciFact symbol-plane retrieval",
        "",
        "## Metric winners",
        f"- **Recall@10**: `{best_r10.config.name}` ({best_r10.recall_at_10:.4f})",
        f"- **nDCG@10**: `{best_ndcg.config.name}` ({best_ndcg.ndcg_at_10:.4f})",
        f"- **Recall@100**: `{best_r100.config.name}` ({best_r100.recall_at_100:.4f})",
        "",
        "## Aggregate comparison",
        "",
        "| config | route_recall | R@10 | R@25 | R@100 | nDCG@10 | ms/q |",
        "|--------|-------------|------|------|-------|---------|------|",
    ]
    for cr in sorted(results, key=lambda x: -x.recall_at_10):
        lines.append(
            f"| {cr.config.name} | {cr.route_recall:.3f} | "
            f"{cr.recall_at_10:.3f} | {cr.recall_at_25:.3f} | "
            f"{cr.recall_at_100:.3f} | {cr.ndcg_at_10:.3f} | "
            f"{cr.mean_query_ms:.1f} |"
        )

    lines.extend([
        "",
        "## Recommended production defaults",
        f"- **Stage 1 route**: uniform κ pool, `route_max={ROUTE_MAX}`",
    ])
    if cascade2 and cascade2.recall_at_10 >= (baseline.recall_at_10 if baseline else 0):
        lines.append(
            f"- **Stage 2 narrow**: cascade2 (compound → rarest fill → hub penalty), cap `{NARROW_CAP}`"
        )
        lines.append(
            f"- **Stage 3 rerank**: witness with "
            f"{'all-pair' if cascade2.config.witness_all_pairs else 'rare-only'} meets"
        )
    elif witness and witness.recall_at_10 >= (baseline.recall_at_10 if baseline else 0):
        lines.append(f"- **Stage 2 narrow**: none (full route pool)")
        lines.append("- **Stage 3 rerank**: witness on full pool")
    else:
        lines.append("- **Stage 2 narrow**: none")
        lines.append("- **Stage 3 rerank**: asymmetric κ overlap")

    lines.extend(["", "## Decision rules (if/then)", ""])
    if baseline and witness:
        if witness.recall_at_10 > baseline.recall_at_10:
            lines.append(
                "1. **IF** gold is in route pool **THEN** prefer witness rerank over plain κ "
                f"(R@10 {witness.recall_at_10:.3f} vs {baseline.recall_at_10:.3f})."
            )
        else:
            lines.append(
                "1. **IF** route pool is large and undifferentiated **THEN** asymmetric κ "
                "baseline may match witness; check per-bucket."
            )
    if cascade2 and baseline:
        wins, losses = compare_queries(cascade2, baseline)
        if cascade2.recall_at_10 > baseline.recall_at_10:
            lines.append(
                "2. **IF** query has compound corridor hits (pair-meet > 0) **THEN** cascade2 narrow "
                f"before witness improves R@10 ({cascade2.recall_at_10:.3f} vs baseline "
                f"{baseline.recall_at_10:.3f})."
            )
        comp_buckets = cascade2.bucket_counts.get("compound", 0)
        rare_buckets = cascade2.bucket_counts.get("rarest-only", 0)
        n = len(cascade2.per_query) or 1
        lines.append(
            f"3. **IF** gold bucket is `compound` ({100*comp_buckets/n:.1f}% of queries) **THEN** "
            "prioritize pair-meet docs in narrow pool."
        )
        lines.append(
            f"4. **IF** gold bucket is `rarest-only` ({100*rare_buckets/n:.1f}% of queries) **THEN** "
            "fill narrow pool from rarest-word κ docs after compound tier."
        )
    narrow_rarest = by_name.get("narrow_rarest")
    narrow_pair = by_name.get("narrow_pair_meet")
    if narrow_rarest and narrow_pair:
        lines.append(
            "5. **IF** pair-meet count alone is used for narrow **THEN** compare against "
            f"rarest-only cap (R@10 pair={narrow_pair.recall_at_10:.3f}, "
            f"rarest={narrow_rarest.recall_at_10:.3f})."
        )
    rare_cfg = by_name.get("cascade2_rare_meets")
    all_cfg = by_name.get("cascade2_all_meets")
    if rare_cfg and all_cfg:
        better = rare_cfg if rare_cfg.recall_at_10 >= all_cfg.recall_at_10 else all_cfg
        lines.append(
            f"6. **IF** cascade2 narrow is active **THEN** use "
            f"{'rare-only' if better.config.rare_only_meets else 'all-pair'} meets "
            f"(R@10 {better.recall_at_10:.3f})."
        )
    missed_total = sum(cr.bucket_counts.get("missed", 0) for cr in results) // len(results)
    lines.append(
        f"7. **IF** bucket is `missed` (~{missed_total} queries avg) **THEN** no Stage 2/3 "
        "rule fixes retrieval — expand Stage 1 keys or OOV lattice."
    )
    lines.append(
        "8. **IF** hub words dominate query **THEN** apply hub penalty in narrow scoring "
        "(docs with hub token overlap but no rare token)."
    )
    if witness:
        med = witness.gold_rank_median
        if med is not None:
            lines.append(
                f"9. **IF** gold is in pool **THEN** expect median rank ~{med:.0f} before top-10 cut "
                f"(witness config)."
            )
    lines.append(
        "10. **Never** use BM25 in symbol-plane path — all gains are κ / meet / rare correlation only."
    )

    if cascade2 and baseline:
        wins, losses = compare_queries(cascade2, baseline)
        lines.extend(["", "### cascade2 helps (sample)", ""])
        for w in wins:
            lines.append(f"- Q{w['qid']}: {w['query']!r} bucket={w['bucket']}")
        lines.extend(["", "### cascade2 hurts (sample)", ""])
        for l in losses:
            lines.append(f"- Q{l['qid']}: {l['query']!r} bucket={l['bucket']}")

    if witness and baseline:
        wins, losses = compare_queries(witness, baseline)
        lines.extend(["", "### witness vs baseline helps (sample)", ""])
        for w in wins:
            lines.append(f"- Q{w['qid']}: {w['query']!r}")
        lines.extend(["", "### witness vs baseline hurts (sample)", ""])
        for l in losses:
            lines.append(f"- Q{l['qid']}: {l['query']!r}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Glass-box SciFact symbol-plane experiments")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--brain", default=None)
    p.add_argument("--split", choices=("test", "train", "all"), default="test")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--max-keys", type=int, default=768)
    p.add_argument("--max-corr-neighbors", type=int, default=4)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    brain_name = args.brain or args.dataset
    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)

    print(f"=== Glass-box experiments: {args.dataset} ===", flush=True)
    knowledge, plane = load_brain_and_plane(brain_name)

    queries = load_queries(paths.queries)
    if args.split == "test":
        qrels = load_qrels(paths.qrels_test)
    elif args.split == "train":
        qrels = load_qrels(paths.qrels_train)
    else:
        qrels = merge_qrels(load_qrels(paths.qrels_test), load_qrels(paths.qrels_train))

    qids = [q for q in qrels if q in queries]
    if args.max_queries is not None:
        qids = qids[: args.max_queries]
    print(f"Queries: {len(qids)} (split={args.split})", flush=True)

    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)

    print("Building query contexts ...", flush=True)
    contexts: dict[str, QueryCtx] = {}
    skipped: list[str] = []
    for qid in qids:
        try:
            contexts[qid] = build_query_ctx(
                knowledge, plane, qid, queries[qid],
                cache=cache, degrees=degrees,
                max_keys=args.max_keys,
                max_corr_neighbors=args.max_corr_neighbors,
            )
        except Exception:
            skipped.append(qid)
    qids = [q for q in qids if q in contexts]
    if skipped:
        print(f"  skipped {len(skipped)} queries (token/OOV errors)", flush=True)
    print(f"Evaluable: {len(qids)}", flush=True)

    all_results: list[ConfigResult] = []
    for cfg in CONFIGS:
        print(f"  config: {cfg.name} ...", flush=True)
        rows: list[QueryResult] = []
        for qid in qids:
            try:
                rows.append(run_one_query(
                    knowledge, plane, cfg, contexts[qid], qrels[qid],
                    cache=cache, degrees=degrees,
                    max_keys=args.max_keys,
                    max_corr_neighbors=args.max_corr_neighbors,
                ))
            except Exception:
                continue
        cr = aggregate_config(cfg, rows)
        all_results.append(cr)
        print(
            f"    route={cr.route_recall:.3f} R@10={cr.recall_at_10:.3f} "
            f"R@100={cr.recall_at_100:.3f} nDCG={cr.ndcg_at_10:.3f}",
            flush=True,
        )

    out = Path(args.out or _ROOT / "logs" / "glass_box_experiment_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "brain": brain_name,
        "split": args.split,
        "n_queries": len(qids),
        "n_skipped": len(skipped),
        "skipped_qids": skipped,
        "route_max": ROUTE_MAX,
        "narrow_cap": NARROW_CAP,
        "configs": [
            {
                "name": cr.config.name,
                "label": cr.config.label,
                "narrow": cr.config.narrow,
                "rerank": cr.config.rerank,
                "rare_only_meets": cr.config.rare_only_meets,
                "route_recall": round(cr.route_recall, 6),
                "recall_at_10": round(cr.recall_at_10, 6),
                "recall_at_25": round(cr.recall_at_25, 6),
                "recall_at_100": round(cr.recall_at_100, 6),
                "ndcg_at_10": round(cr.ndcg_at_10, 6),
                "mean_query_ms": round(cr.mean_query_ms, 3),
                "gold_rank_median": cr.gold_rank_median,
                "bucket_counts": cr.bucket_counts,
                "per_query": [
                    {
                        "qid": r.qid,
                        "route_recall": r.route_recall,
                        "recall_10": round(r.recall_10, 4),
                        "recall_25": round(r.recall_25, 4),
                        "recall_100": round(r.recall_100, 4),
                        "ndcg_10": round(r.ndcg_10, 4),
                        "gold_rank": r.gold_rank,
                        "bucket": r.bucket,
                        "query_ms": round(r.query_ms, 2),
                        "narrow_pool": r.narrow_pool,
                    }
                    for r in cr.per_query
                ],
            }
            for cr in all_results
        ],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nJSON: {out}", flush=True)

    summary_path = out.with_suffix(".txt")
    write_summary_txt(summary_path, all_results)
    print(f"Summary: {summary_path}", flush=True)

    rules_path = _ROOT / "logs" / "glass_box_rules.md"
    extract_rules(all_results, queries, rules_path)
    print(f"Rules: {rules_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
