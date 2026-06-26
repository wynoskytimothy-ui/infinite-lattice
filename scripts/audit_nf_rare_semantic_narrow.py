#!/usr/bin/env python3
"""
NFCorpus rare-word semantic narrowing probes.

Hypothesis: rarest query word brings a wide doc set; correlating RARER words
(co-occur neighbors, SciFact correlates, mined rare anchors from triggered docs)
narrow toward gold.

Each probe: seed pool -> optional narrow filter -> BM25 rank top-100.

Run:
  python scripts/audit_nf_rare_semantic_narrow.py
  python scripts/audit_nf_rare_semantic_narrow.py --max-queries 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_expansion
from aethos_glass_box_search import posting_docs, rarest_terms, word_idf
from aethos_multi_corpus import score_candidates
from scripts.audit_gold_paths_cross_corpus import (
    build_coocc_adjacency,
    build_second_order,
    neighbors_2nd,
    precompute_neighbors_2nd,
)
from scripts.bench_supervised_bridges import load, ndcg10, recall10


@dataclass
class NarrowCtx:
    idx: AppendOnlyLatticeIndex
    corpus: dict[str, str]
    N: int
    idf_fn: Callable[[str], float]
    br: RelevanceBridges
    scifact_knowledge: object | None = None
    adj_nf: dict = field(default_factory=dict)
    nbr2_map: dict[str, list[str]] = field(default_factory=dict)
    rare_gate: float = 2.5
    rare_df_cap: int = 256


def _toks(corpus: dict[str, str], doc_id: str) -> set[str]:
    return set(words(corpus.get(doc_id, "")))


def _gold_sets(rels: dict[str, int]) -> tuple[set[str], int]:
    golds = {d for d, s in rels.items() if s > 0}
    return golds, len(golds)


def _rank_pool(idx, query: str, pool: set[str]) -> list[str]:
    if not pool:
        return []
    scores = score_candidates(idx, query, pool)
    return sorted(scores, key=scores.get, reverse=True)[:100]


def _metrics(pool: set[str], ranked: list[str], golds: set[str]) -> dict:
    in_pool = golds & pool
    in_top = golds & set(ranked[:10])
    in_top100 = golds & set(ranked[:100])
    return {
        "pool_size": len(pool),
        "gold_in_pool": len(in_pool),
        "gold_in_top10": len(in_top),
        "gold_in_top100": len(in_top100),
        "query_gold_in_pool": bool(in_pool),
        "query_gold_in_top10": bool(in_top),
        "query_gold_in_top100": bool(in_top100),
    }


def mine_rare_from_docs(
    corpus: dict[str, str],
    doc_ids: set[str],
    idf_fn: Callable[[str], float],
    gate: float = 2.5,
    top_k: int = 24,
) -> list[str]:
    counts: Counter = Counter()
    for d in doc_ids:
        for w in _toks(corpus, d):
            if idf_fn(w) >= gate:
                counts[w] += 1
    return [w for w, _ in counts.most_common(top_k)]


def scifact_corr_score(
    knowledge,
    rare_query: list[str],
    doc_toks: set[str],
    idf_fn: Callable[[str], float],
) -> float:
    if knowledge is None:
        return 0.0
    doc_rare = [w for w in doc_toks if idf_fn(w) >= 2.0]
    score = 0.0
    for qw in rare_query[:6]:
        for dw in doc_rare[:50]:
            if qw == dw:
                continue
            lk = knowledge.correlates(qw, dw)
            if lk is not None:
                score += lk.strength
    return score


def nf_cooc_score(rarest_w: str, doc_toks: set[str], adj: dict, k: int = 12) -> float:
    nbrs = adj.get(rarest_w) or Counter()
    if not nbrs:
        return 0.0
    score = 0.0
    for w, wt in nbrs.most_common(k):
        if w in doc_toks:
            score += wt
    return score


def overlap_score(doc_toks: set[str], anchor_words: list[str]) -> float:
    return sum(1.0 for w in anchor_words if w in doc_toks)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe_rarest1_only(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = posting_docs(ctx.idx, rarest[0]) if rarest else set()
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest1_then_rarest2(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if len(rarest) < 2:
        return probe_rarest1_only(ctx, query, rarest, rare)
    wide = posting_docs(ctx.idx, rarest[0])
    pool = {d for d in wide if rarest[1] in _toks(ctx.corpus, d)}
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest1_then_2_rare(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest or len(rare) < 2:
        return probe_rarest1_only(ctx, query, rarest, rare)
    wide = posting_docs(ctx.idx, rarest[0])
    pool = set()
    for d in wide:
        toks = _toks(ctx.corpus, d)
        if sum(1 for w in rare if w in toks) >= 2:
            pool.add(d)
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest1_then_neighbor(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    nbr_ctr = ctx.adj_nf.get(rarest[0]) or Counter()
    nbrs = [w for w, _ in nbr_ctr.most_common(10)]
    pool = {d for d in wide if any(w in _toks(ctx.corpus, d) for w in nbrs)}
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest1_then_scifact_corr(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest or ctx.scifact_knowledge is None:
        return probe_rarest1_only(ctx, query, rarest, rare)
    wide = posting_docs(ctx.idx, rarest[0])
    pool = set()
    for d in wide:
        toks = _toks(ctx.corpus, d)
        if scifact_corr_score(ctx.scifact_knowledge, rare or rarest[:4], toks, ctx.idf_fn) > 0:
            pool.add(d)
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest_union_meet2(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if len(rarest) < 2:
        return probe_rarest1_only(ctx, query, rarest, rare)
    pool = posting_docs(ctx.idx, rarest[0]) & posting_docs(ctx.idx, rarest[1])
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_mine_trigger_narrow(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Rarest1+2 postings -> mine rare anchors in triggered docs -> narrow rarest1 pool."""
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    if len(rarest) > 1:
        wide |= posting_docs(ctx.idx, rarest[1])
    mined = mine_rare_from_docs(ctx.corpus, wide, ctx.idf_fn, gate=ctx.rare_gate, top_k=24)
    if not mined:
        return wide, _rank_pool(ctx.idx, query, wide)
    scored = [(overlap_score(_toks(ctx.corpus, d), mined), d) for d in wide]
    scored.sort(key=lambda x: (-x[0], x[1]))
    pool = {d for s, d in scored if s > 0}
    ranked = [d for s, d in scored if s > 0][:100]
    if len(ranked) < 100:
        ranked.extend([d for s, d in scored if s == 0][:100 - len(ranked)])
    return pool, ranked[:100]


def probe_mine_trigger_rank_wide(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Mine rare anchors from rarest1+2 triggered docs; rank FULL rarest1 pool by overlap."""
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    trigger = set(wide)
    if len(rarest) > 1:
        trigger |= posting_docs(ctx.idx, rarest[1])
    mined = mine_rare_from_docs(ctx.corpus, trigger, ctx.idf_fn, gate=ctx.rare_gate, top_k=30)
    if not mined:
        return wide, _rank_pool(ctx.idx, query, wide)
    scored = []
    for d in wide:
        lex = score_candidates(ctx.idx, query, {d}).get(d, 0.0)
        ov = overlap_score(_toks(ctx.corpus, d), mined)
        scored.append((0.35 * lex + 0.65 * ov, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [d for _, d in scored[:100]]
    return wide, ranked


def probe_scifact_corr_rank_rarest1(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    if ctx.scifact_knowledge is None:
        return wide, _rank_pool(ctx.idx, query, wide)
    scored = []
    for d in wide:
        toks = _toks(ctx.corpus, d)
        corr = scifact_corr_score(ctx.scifact_knowledge, rare or rarest[:4], toks, ctx.idf_fn)
        lex = score_candidates(ctx.idx, query, {d}).get(d, 0.0)
        scored.append((0.4 * lex + 0.6 * corr, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return wide, [d for _, d in scored[:100]]


def probe_cooc_rank_rarest1(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    scored = []
    for d in wide:
        toks = _toks(ctx.corpus, d)
        co = nf_cooc_score(rarest[0], toks, ctx.adj_nf)
        lex = score_candidates(ctx.idx, query, {d}).get(d, 0.0)
        scored.append((0.45 * lex + 0.55 * co, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return wide, [d for _, d in scored[:100]]


def probe_2nd_order_narrow(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Rarest1 pool intersect docs containing 2nd-order semantic neighbor of rarest."""
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    nbrs = ctx.nbr2_map.get(rarest[0], [])[:8]
    pool = {d for d in wide if any(w in _toks(ctx.corpus, d) for w in nbrs)}
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest_to_rarest_meet(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if len(rare) < 2:
        return probe_rarest1_only(ctx, query, rarest, rare)
    pool: set[str] = set()
    for a, b in itertools.combinations(rare[:4], 2):
        pool |= posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_bridge_on_rarest1(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Bridge expansion but only docs already in rarest1 posting pool."""
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    br_docs = set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
    pool = wide | (br_docs & wide)  # bridge can't expand outside wide — same as wide
    # real test: bridge docs that intersect rarest1 OR have correlate narrow
    pool = wide | {d for d in br_docs if d in wide}
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_rarest1_plus_bridge_union(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    br_docs = set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
    pool = wide | br_docs
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_mine_corr_words_narrow(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """
    Mine rare words from triggered docs; keep rarest1 docs that share a SciFact
  correlate link between any mined word and any rare query term.
    """
    if not rarest or ctx.scifact_knowledge is None:
        return probe_mine_trigger_narrow(ctx, query, rarest, rare)
    wide = posting_docs(ctx.idx, rarest[0])
    trigger = set(wide)
    if len(rarest) > 1:
        trigger |= posting_docs(ctx.idx, rarest[1])
    mined = mine_rare_from_docs(ctx.corpus, trigger, ctx.idf_fn, gate=ctx.rare_gate, top_k=30)
    pool = set()
    for d in wide:
        toks = _toks(ctx.corpus, d)
        hit = False
        for mw in mined:
            for qw in rare or rarest[:4]:
                if qw == mw:
                    continue
                lk1 = ctx.scifact_knowledge.correlates(qw, mw)
                if lk1 is not None and mw in toks:
                    hit = True
                    break
                lk2 = ctx.scifact_knowledge.correlates(mw, qw)
                if lk2 is not None and mw in toks:
                    hit = True
                    break
            if hit:
                break
        if hit:
            pool.add(d)
    return pool, _rank_pool(ctx.idx, query, pool)


def probe_mine_bridge_trigger_rank(ctx: NarrowCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Mine rare anchors from rarest1 + bridge triggered docs; rank that wide pool."""
    if not rarest:
        return set(), []
    wide = posting_docs(ctx.idx, rarest[0])
    br_docs = set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
    trigger = wide | br_docs
    mined = mine_rare_from_docs(ctx.corpus, trigger, ctx.idf_fn, gate=ctx.rare_gate, top_k=36)
    if not mined:
        return trigger, _rank_pool(ctx.idx, query, trigger)
    scored = []
    for d in trigger:
        lex = score_candidates(ctx.idx, query, {d}).get(d, 0.0)
        ov = overlap_score(_toks(ctx.corpus, d), mined)
        corr = scifact_corr_score(ctx.scifact_knowledge, rare or rarest[:4], _toks(ctx.corpus, d), ctx.idf_fn)
        scored.append((0.35 * lex + 0.45 * ov + 0.20 * corr, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [d for _, d in scored[:100]]
    return trigger, ranked


PROBES: list[tuple[str, str, Callable]] = [
    ("01_rarest1_only", "Rarest query word posting pool", probe_rarest1_only),
    ("02_rarest1_then_rarest2", "Rarest1 pool AND doc has 2nd-rarest word", probe_rarest1_then_rarest2),
    ("03_rarest1_then_2_rare", "Rarest1 pool AND doc has >=2 rare query terms", probe_rarest1_then_2_rare),
    ("04_rarest1_then_neighbor", "Rarest1 pool AND NF co-occur neighbor of rarest", probe_rarest1_then_neighbor),
    ("05_rarest1_then_scifact_corr", "Rarest1 pool AND SciFact correlate link in doc", probe_rarest1_then_scifact_corr),
    ("06_rarest_union_meet2", "Posting(rarest1) INTERSECT posting(rarest2)", probe_rarest_union_meet2),
    ("07_mine_trigger_narrow", "Mine rare anchors from rarest1+2 docs; narrow rarest1", probe_mine_trigger_narrow),
    ("08_mine_trigger_rank", "Mine rare anchors; rank rarest1 pool by overlap+lex", probe_mine_trigger_rank_wide),
    ("09_scifact_corr_rank", "Rank rarest1 pool by SciFact rare correlation score", probe_scifact_corr_rank_rarest1),
    ("10_cooc_rank_rarest1", "Rank rarest1 pool by NF co-occur neighbor score", probe_cooc_rank_rarest1),
    ("11_2nd_order_narrow", "Rarest1 pool AND 2nd-order semantic neighbor in doc", probe_2nd_order_narrow),
    ("12_rarest_to_rarest_meet", "Union pair-meet of rare query terms (idf gate)", probe_rarest_to_rarest_meet),
    ("13_rarest1_plus_bridge", "Rarest1 posting UNION bridge expansion", probe_rarest1_plus_bridge_union),
    ("14_mine_corr_narrow", "Mine triggered rare words + SciFact correlate narrow on rarest1", probe_mine_corr_words_narrow),
    ("15_mine_bridge_trigger_rank", "Mine rare from rarest1+bridge pool; rank by overlap+corr+lex", probe_mine_bridge_trigger_rank),
]


def evaluate_probe(
    name: str,
    fn: Callable,
    ctx: NarrowCtx,
    queries: dict[str, str],
    test_q: dict,
    test_ids: list[str],
) -> dict:
    n_q = len(test_ids)
    q_pool = q_top10 = q_top100 = 0
    g_pool = g_top10 = g_top100 = 0
    g_total = 0
    pool_sizes: list[int] = []
    retention: list[float] = []

    for qid in test_ids:
        q = queries[qid]
        golds, gn = _gold_sets(test_q[qid])
        g_total += gn
        qterms = words(q)
        rarest = rarest_terms(qterms, ctx.idx, ctx.N)
        rare = [w for w in rarest if ctx.idf_fn(w) >= ctx.rare_gate]

        pool, ranked = fn(ctx, q, rarest, rare)
        m = _metrics(pool, ranked, golds)
        pool_sizes.append(m["pool_size"])
        g_pool += m["gold_in_pool"]
        g_top10 += m["gold_in_top10"]
        g_top100 += m["gold_in_top100"]
        if m["query_gold_in_pool"]:
            q_pool += 1
        if m["query_gold_in_top10"]:
            q_top10 += 1
        if m["query_gold_in_top100"]:
            q_top100 += 1

        # retention vs rarest1 wide pool
        if rarest:
            wide = posting_docs(ctx.idx, rarest[0])
            wide_gold = len(golds & wide)
            if wide_gold:
                retention.append(m["gold_in_pool"] / wide_gold)

    def pct(a, b):
        return round(100.0 * a / max(b, 1), 1)

    return {
        "name": name,
        "queries": n_q,
        "queries_gold_in_pool_pct": pct(q_pool, n_q),
        "gold_instances_in_pool_pct": pct(g_pool, g_total),
        "queries_gold_in_top10_pct": pct(q_top10, n_q),
        "gold_instances_in_top10_pct": pct(g_top10, g_total),
        "queries_gold_in_top100_pct": pct(q_top100, n_q),
        "gold_instances_in_top100_pct": pct(g_top100, g_total),
        "mean_pool_size": round(sum(pool_sizes) / max(len(pool_sizes), 1), 1),
        "mean_gold_retention_vs_rarest1": round(
            100 * sum(retention) / max(len(retention), 1), 1
        ),
    }


def write_report(results: list[dict], baseline: dict, path: Path) -> None:
    lines = [
        "# NFCorpus rare semantic narrowing probes\n\n",
        "Seed: rarest-word postings. Narrow: semantic rare-word correlations.\n\n",
        "| probe | Q pool | inst pool | Q@10 | inst@10 | mean pool | retention vs rarest1 |\n",
        "|-------|--------|-----------|------|---------|-----------|----------------------|\n",
    ]
    for r in sorted(results, key=lambda x: -x["gold_instances_in_top10_pct"]):
        lines.append(
            f"| {r['name']} | {r['queries_gold_in_pool_pct']}% | "
            f"{r['gold_instances_in_pool_pct']}% | {r['queries_gold_in_top10_pct']}% | "
            f"{r['gold_instances_in_top10_pct']}% | {r['mean_pool_size']} | "
            f"{r['mean_gold_retention_vs_rarest1']}% |\n"
        )
    best_top10 = max(results, key=lambda x: x["gold_instances_in_top10_pct"])
    best_pool = max(results, key=lambda x: x["gold_instances_in_pool_pct"])
    lines.append(f"\n- Best inst@10: `{best_top10['name']}` ({best_top10['gold_instances_in_top10_pct']}%)\n")
    lines.append(f"- Best inst pool: `{best_pool['name']}` ({best_pool['gold_instances_in_pool_pct']}%)\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="NF rare semantic narrowing probes")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", default="logs/nf_rare_semantic_narrow.json")
    p.add_argument("--rules-md", default="logs/nf_rare_semantic_narrow.md")
    p.add_argument("--index-mode", default="kappa_primary")
    args = p.parse_args()

    corpus, queries, train_q, test_q = load("nfcorpus")
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    print(f"NF rare semantic narrow: {len(test_ids)} queries", flush=True)
    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex(index_mode=args.index_mode)
    for d, t in corpus.items():
        idx.add(d, t)
    idx.finalize()
    N = len(idx.alive)
    idf_fn = lambda w: word_idf(idx, w, N)

    br = RelevanceBridges(idx, N, min_pairs=2).learn(queries, train_q, corpus)
    br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=2)

    co, inv, norm, _ = build_second_order(corpus, idf_fn)
    nbr2_map = precompute_neighbors_2nd(co, inv, norm, idf_fn)
    adj = build_coocc_adjacency(corpus, idf_fn)

    scifact_knowledge = None
    try:
        from eval_beir_symbol import load_brain_and_plane
        scifact_knowledge, _ = load_brain_and_plane("scifact")
        print("  SciFact symbol knowledge loaded for correlate probes", flush=True)
    except Exception as e:
        print(f"  WARN: SciFact knowledge unavailable: {e}", flush=True)

    ctx = NarrowCtx(
        idx=idx, corpus=corpus, N=N, idf_fn=idf_fn, br=br,
        scifact_knowledge=scifact_knowledge, adj_nf=adj, nbr2_map=nbr2_map,
    )
    print(f"  built in {time.perf_counter() - t0:.1f}s", flush=True)

    results = []
    for name, desc, fn in PROBES:
        print(f"  {name} ...", flush=True)
        row = evaluate_probe(name, fn, ctx, queries, test_q, test_ids)
        row["description"] = desc
        results.append(row)

    baseline = next(r for r in results if r["name"] == "01_rarest1_only")
    for r in results:
        r["pool_lift_vs_rarest1"] = round(
            r["gold_instances_in_pool_pct"] - baseline["gold_instances_in_pool_pct"], 1
        )
        r["top10_lift_vs_rarest1"] = round(
            r["gold_instances_in_top10_pct"] - baseline["gold_instances_in_top10_pct"], 1
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"n_queries": len(test_ids), "probes": results, "baseline": baseline}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(results, baseline, Path(args.rules_md))

    print(f"\n{'='*72}")
    print("  NF RARE SEMANTIC NARROWING LEADERBOARD (inst@10)")
    print(f"  {'probe':<28} {'inst@10':>8} {'inst pool':>10} {'pool':>7} {'retain':>8}")
    print("  " + "-" * 60)
    for r in sorted(results, key=lambda x: -x["gold_instances_in_top10_pct"])[:10]:
        print(
            f"  {r['name']:<28} {r['gold_instances_in_top10_pct']:>7.1f}% "
            f"{r['gold_instances_in_pool_pct']:>9.1f}% "
            f"{r['mean_pool_size']:>7.0f} {r['mean_gold_retention_vs_rarest1']:>7.1f}%"
        )
    print(f"\n  JSON: {out}")
    print(f"  Report: {args.rules_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
