#!/usr/bin/env python3
"""
Full pool-recall sweep — which strategies put gold docs in the candidate pool / top-100?

Measures per strategy (full test split, default nfcorpus):
  - queries_gold_in_pool_pct   any gold in pre-rank candidate pool
  - gold_instances_in_pool_pct fraction of gold doc instances in pool
  - queries_gold_in_top100_pct any gold in ranked top-100
  - gold_instances_in_top100_pct
  - mean_pool_size

Run:
  python scripts/audit_pool_recall_sweep.py nfcorpus
  python scripts/audit_pool_recall_sweep.py scifact --index-mode full
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_expansion
from aethos_encyclopedia_teacher import load_glossary
from aethos_glass_box_search import (
    GlassBoxRetriever,
    GlassBoxSearchConfig,
    _build_route_pool,
    _fuse_pool,
    posting_docs,
    rarest_terms,
    word_idf,
)
from aethos_multi_corpus import IdfCache, MultiCorpusBrain, score_candidates
from aethos_promotion import PromotionRegistry
from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast
from pipeline.bit_04_candidate_router import query_words_for_routing
from scripts.bench_supervised_bridges import load, ndcg10, recall10


@dataclass
class PoolCtx:
    idx: AppendOnlyLatticeIndex
    corpus: dict[str, str]
    br: RelevanceBridges
    N: int
    glossary: dict[str, str]
    kappa_index: object | None = None
    registry: object | None = None
    brain: MultiCorpusBrain | None = None
    branch_name: str = ""


def _gold_sets(rels: dict[str, int]) -> tuple[set[str], int]:
    golds = {d for d, s in rels.items() if s > 0}
    return golds, len(golds)


def _pool_metrics(pool: set[str], golds: set[str], ranked: list[str]) -> dict:
    in_pool = golds & pool
    in_top = golds & set(ranked[:100])
    return {
        "pool_size": len(pool),
        "gold_in_pool": len(in_pool),
        "gold_in_top100": len(in_top),
        "query_gold_in_pool": bool(in_pool),
        "query_gold_in_top100": bool(in_top),
    }


def strat_lex100(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    ranked = ctx.idx.search(query, 100)
    return set(ranked), ranked


def strat_rare_df256(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    pool: set[str] = set()
    for w in set(words(query)):
        p = ctx.idx.token_prime.get(("w", w))
        if p and 0 < ctx.idx.df.get(p, 0) <= 256:
            pl = ctx.idx.postings.get(p)
            if pl:
                pool.update(d for d in pl if d in ctx.idx.alive)
    if not pool:
        return pool, []
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_kappa_route(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    pool: set[str] = set()
    if ctx.kappa_index and ctx.registry:
        from pipeline.bit_04_candidate_router import candidates_from_attractors
        idf = lambda w: word_idf(ctx.idx, w, ctx.N)
        qws = query_words_for_routing(words(query))
        kdocs, _ = candidates_from_attractors(qws, ctx.registry, ctx.kappa_index, idf=idf)
        pool.update(kdocs[:600])
    ranked = sorted(score_candidates(ctx.idx, query, pool), key=lambda d: -score_candidates(ctx.idx, query, pool).get(d, 0))[:100] if pool else []
    return pool, ranked


def strat_glass_route(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    cfg = GlassBoxSearchConfig.scifact_lattice()
    idf_fn = lambda w: word_idf(ctx.idx, w, ctx.N)
    pool = _build_route_pool(
        ctx.idx, query, cfg, ctx.N,
        kappa_index=ctx.kappa_index, registry=ctx.registry, idf_fn=idf_fn,
    )
    scores = score_candidates(ctx.idx, query, pool) if pool else {}
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_scale_search_pool(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    """MultiCorpusBrain scale_search candidate construction (no final teach rerank)."""
    if not ctx.brain or not ctx.branch_name:
        return strat_glass_route(ctx, query)
    branch = ctx.brain._corpora[ctx.branch_name]
    idf = IdfCache(branch.idx, branch.n_docs)
    pool: set[str] = set()
    qws = query_words_for_routing(words(query))
    if branch.kappa_index and qws:
        kdocs, _ = ctx.brain._attractor_route(branch, qws)
        pool.update(kdocs[:600])
    idx = branch.idx
    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p and 0 < idx.df.get(p, 0) <= 256:
            pl = idx.postings.get(p)
            if pl:
                pool.update(d for d in pl if d in idx.alive)
    if branch.pair_bridges:
        exp = bridge_expansion(
            idx, branch.pair_bridges, query,
            idf=idf,
            hub_idf_gate=ctx.brain.HUB_IDF_GATE if ctx.brain.HUB_IDF_GATE < 50 else 0.0,
            hub_blocklist=ctx.brain._learned_hub_blocklist(),
        )
        pool.update(d for d in exp if d in idx.alive)
    scores = score_candidates(idx, query, pool)
    ranked_gids = sorted(scores, key=scores.get, reverse=True)[:100]
    ranked = [g.split("/", 1)[1] for g in ranked_gids if "/" in g]
    pool_local = {g.split("/", 1)[1] for g in pool if "/" in g}
    return pool_local, ranked


def strat_lex100_union_rarest2(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    pool = set(ctx.idx.search(query, 100))
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) >= 2:
        pool |= posting_docs(ctx.idx, rarest[0]) | posting_docs(ctx.idx, rarest[1])
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_rare_idf3_union(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    pool: set[str] = set()
    for w in set(words(query)):
        if word_idf(ctx.idx, w, ctx.N) >= 3.0:
            pool |= posting_docs(ctx.idx, w)
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_bridge_expand_pool(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    idf_fn = lambda w: word_idf(ctx.idx, w, ctx.N)
    exp = bridge_expansion(ctx.idx, ctx.br, query, idf=idf_fn)
    pool = {d for d in exp if d in ctx.idx.alive}
    pool |= set(ctx.idx.search(query, 100))
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_pair_meet_union(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    rare = [w for w in rarest_terms(words(query), ctx.idx, ctx.N)
            if word_idf(ctx.idx, w, ctx.N) >= 2.5][:4]
    pool: set[str] = set()
    for a, b in itertools.combinations(rare, 2):
        pool |= posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
    pool |= set(ctx.idx.search(query, 100))
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


def strat_glass_fuse_pool(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    """Full glass-box fusion pool (lex cand + bridge + pair-meet) before final sort."""
    from aethos_glass_box_search import glossary_expand_query

    cfg = GlassBoxSearchConfig.scifact_lattice()
    expanded = glossary_expand_query(
        query, ctx.idx, ctx.N, ctx.glossary,
        idf_gate=cfg.glossary_idf_gate,
        max_extra=cfg.glossary_max_extra,
    )
    ranked, scores, _ = _fuse_pool(
        ctx.idx, ctx.br, expanded, words(query), ctx.N, cfg,
        corpus=ctx.corpus,
        kappa_index=ctx.kappa_index,
        registry=ctx.registry,
    )
    pool = set(scores.keys())
    return pool, ranked[:100]


def strat_union_recovery_max(ctx: PoolCtx, query: str) -> tuple[set[str], list[str]]:
    """Union every recovery path then BM25-rank top-100."""
    pools = [
        strat_glass_route(ctx, query)[0],
        strat_bridge_expand_pool(ctx, query)[0],
        strat_pair_meet_union(ctx, query)[0],
        strat_rare_idf3_union(ctx, query)[0],
        set(ctx.idx.search(query, 100)),
    ]
    pool: set[str] = set()
    for p in pools:
        pool |= p
    scores = score_candidates(ctx.idx, query, pool)
    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    return pool, ranked


STRATEGIES: list[tuple[str, str, Callable[[PoolCtx, str], tuple[set[str], list[str]]]]] = [
    ("01_lex_top100", "BM25 lexical top-100 only (pool = ranked)", strat_lex100),
    ("02_rare_posting_df256", "Union postings df<=256 per query term", strat_rare_df256),
    ("03_kappa_route_600", "κ plane route candidates (BIT 4/12)", strat_kappa_route),
    ("04_glass_route_pool", "Glass κ-route + rare + lex seed (pool_restrict)", strat_glass_route),
    ("05_scale_search_pool", "Brain scale_search pool (κ+rare+bridges)", strat_scale_search_pool),
    ("06_lex100_union_rarest2", "Lex top-100 ∪ rarest+2nd postings", strat_lex100_union_rarest2),
    ("07_rare_idf3_union", "Union all query terms idf>=3", strat_rare_idf3_union),
    ("08_bridge_expand_union", "Bridge expansion ∪ lex top-100", strat_bridge_expand_pool),
    ("09_pair_meet_union", "Pair-meet rare co-occur ∪ lex top-100", strat_pair_meet_union),
    ("10_glass_fuse_pool", "Full glass-box fuse pool (glossary+bridges+meet)", strat_glass_fuse_pool),
    ("11_union_recovery_max", "Union ALL recovery paths → BM25 top-100", strat_union_recovery_max),
]


def evaluate_strategy(
    name: str,
    fn: Callable[[PoolCtx, str], tuple[set[str], list[str]]],
    ctx: PoolCtx,
    queries: dict[str, str],
    test_q: dict[str, dict[str, int]],
    test_ids: list[str],
) -> dict:
    n = len(test_ids)
    q_pool = q_top = 0
    g_pool = g_top = 0
    g_total = 0
    pool_sizes: list[int] = []
    t0 = time.perf_counter()
    for qid in test_ids:
        q = queries[qid]
        golds, gn = _gold_sets(test_q[qid])
        g_total += gn
        pool, ranked = fn(ctx, q)
        m = _pool_metrics(pool, golds, ranked)
        pool_sizes.append(m["pool_size"])
        g_pool += m["gold_in_pool"]
        g_top += m["gold_in_top100"]
        if m["query_gold_in_pool"]:
            q_pool += 1
        if m["query_gold_in_top100"]:
            q_top += 1
    elapsed = time.perf_counter() - t0
    return {
        "name": name,
        "queries": n,
        "queries_gold_in_pool_pct": round(100 * q_pool / n, 1),
        "gold_instances_in_pool_pct": round(100 * g_pool / max(g_total, 1), 1),
        "queries_gold_in_top100_pct": round(100 * q_top / n, 1),
        "gold_instances_in_top100_pct": round(100 * g_top / max(g_total, 1), 1),
        "mean_pool_size": round(sum(pool_sizes) / max(len(pool_sizes), 1), 1),
        "wall_s": round(elapsed, 2),
    }


def write_report(results: list[dict], baseline: dict, path: Path, dataset: str) -> None:
    lines = [
        f"# Pool recall sweep — {dataset}\n\n",
        "Goal: maximize **gold in candidate pool** and **gold in top-100** "
        "(Stage-1 route recall before rerank fixes rank).\n\n",
        "| strategy | Q gold in pool | gold inst in pool | Q gold top-100 | gold inst top-100 | mean pool | Δ pool vs baseline |\n",
        "|----------|----------------|-------------------|----------------|-------------------|-----------|--------------------|\n",
    ]
    base_pool = baseline["queries_gold_in_pool_pct"]
    for r in sorted(results, key=lambda x: -x["gold_instances_in_pool_pct"]):
        delta = round(r["queries_gold_in_pool_pct"] - base_pool, 1)
        lines.append(
            f"| {r['name']} | {r['queries_gold_in_pool_pct']}% | "
            f"{r['gold_instances_in_pool_pct']}% | {r['queries_gold_in_top100_pct']}% | "
            f"{r['gold_instances_in_top100_pct']}% | {r['mean_pool_size']} | {delta:+.1f} |\n"
        )
    lines.append("\n## Recommendations\n")
    best_pool = max(results, key=lambda x: x["gold_instances_in_pool_pct"])
    best_top = max(results, key=lambda x: x["gold_instances_in_top100_pct"])
    lines.append(f"- **Best gold-in-pool:** `{best_pool['name']}` "
                 f"({best_pool['gold_instances_in_pool_pct']}% instances, "
                 f"{best_pool['queries_gold_in_pool_pct']}% queries)\n")
    lines.append(f"- **Best gold-in-top-100:** `{best_top['name']}` "
                 f"({best_top['gold_instances_in_top100_pct']}% instances)\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Pool recall sweep (gold in pool / top-100)")
    p.add_argument("dataset", nargs="?", default="nfcorpus")
    p.add_argument("--index-mode", default="kappa_primary", choices=("full", "kappa_primary"))
    p.add_argument("--out", default="")
    p.add_argument("--rules-md", default="")
    args = p.parse_args()

    corpus, queries, train_q, test_q = load(args.dataset)
    test_ids = [q for q in test_q if q in queries]
    min_pairs = 1 if args.dataset == "scifact" else 2
    gloss = load_glossary(args.dataset)

    out = Path(args.out or f"logs/pool_recall_sweep_{args.dataset}.json")
    rules = Path(args.rules_md or f"logs/pool_recall_rules_{args.dataset}.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Pool recall sweep: {args.dataset} | {len(test_ids)} test queries | mode={args.index_mode}", flush=True)
    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex(index_mode=args.index_mode)
    for d, t in corpus.items():
        idx.add(d, t)
    idx.finalize()
    N = len(idx.alive)
    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=min_pairs)

    registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
    for text in corpus.values():
        registry.observe_text(text)
    kappa_index = build_attractor_index_fast(
        registry, corpus, lambda w: word_idf(idx, w, N), top_k=10,
    )

    brain = MultiCorpusBrain()
    brain.stack_corpus(
        args.dataset, corpus, queries=queries, train_qrels=train_q,
        index_mode=args.index_mode,
    )

    ctx = PoolCtx(
        idx=idx, corpus=corpus, br=br, N=N, glossary=gloss,
        kappa_index=kappa_index, registry=registry,
        brain=brain, branch_name=args.dataset,
    )
    print(f"  built in {time.perf_counter()-t0:.1f}s", flush=True)

    results = []
    for name, desc, fn in STRATEGIES:
        print(f"  {name} ...", flush=True)
        row = evaluate_strategy(name, fn, ctx, queries, test_q, test_ids)
        row["description"] = desc
        results.append(row)

    baseline = next(r for r in results if r["name"] == "01_lex_top100")
    for r in results:
        r["pool_delta_vs_lex100"] = round(
            r["queries_gold_in_pool_pct"] - baseline["queries_gold_in_pool_pct"], 1
        )

    payload = {
        "dataset": args.dataset,
        "index_mode": args.index_mode,
        "n_queries": len(test_ids),
        "strategies": results,
        "baseline": baseline,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(results, baseline, rules, args.dataset)

    print(f"\n{'='*72}")
    print(f"  POOL RECALL LEADERBOARD ({args.dataset}, {len(test_ids)} queries)")
    print(f"  {'strategy':<28} {'Q pool':>7} {'inst pool':>10} {'Q@100':>7} {'inst@100':>9} {'pool':>7}")
    print("  " + "-" * 68)
    for r in sorted(results, key=lambda x: -x["gold_instances_in_pool_pct"])[:8]:
        print(
            f"  {r['name']:<28} {r['queries_gold_in_pool_pct']:>6.1f}% "
            f"{r['gold_instances_in_pool_pct']:>9.1f}% "
            f"{r['queries_gold_in_top100_pct']:>6.1f}% "
            f"{r['gold_instances_in_top100_pct']:>8.1f}% "
            f"{r['mean_pool_size']:>7.0f}"
        )
    print(f"\n  JSON: {out}")
    print(f"  Rules: {rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
