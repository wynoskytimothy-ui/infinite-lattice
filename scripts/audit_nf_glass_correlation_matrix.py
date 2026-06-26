#!/usr/bin/env python3
"""
NFCorpus glass-box correlation matrix — subwords, compounds, cross-correlations,
semantic rare-word chains, and multi-signal fusion probes.

Goal: find which inspectable paths push gold instances into top-10.

Run:
  python scripts/audit_nf_glass_correlation_matrix.py
  python scripts/audit_nf_glass_correlation_matrix.py --max-queries 50
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_expansion, corridor_bridge_expansion
from aethos_encyclopedia_teacher import load_glossary
from aethos_glass_box_search import (
    GlassBoxSearchConfig,
    glass_box_search,
    posting_docs,
    rarest_terms,
    word_idf,
    _fuse_pool,
)
from aethos_multi_corpus import score_candidates
from aethos_promotion import _chunk_subwords
from aethos_rare_rank import score_doc_rare_correlations, _DocFreqCache
from scripts.audit_nf_rare_semantic_narrow import mine_rare_from_docs, overlap_score
from scripts.bench_supervised_bridges import load, ndcg10, recall10


@dataclass
class MatrixCtx:
    idx: AppendOnlyLatticeIndex
    corpus: dict[str, str]
    N: int
    idf_fn: Callable[[str], float]
    br: RelevanceBridges
    glossary: dict[str, str]
    scifact_knowledge: object | None = None
    kappa_index: object | None = None
    registry: object | None = None
    rare_gate: float = 2.5
    doc_rare_tokens: dict[str, list[str]] = field(default_factory=dict)
    corr_score_cache: dict[tuple[str, tuple[str, ...]], float] = field(default_factory=dict)


def _toks(doc_id: str, corpus: dict[str, str]) -> set[str]:
    return set(words(corpus.get(doc_id, "")))


def _gold_sets(rels: dict[str, int]) -> tuple[set[str], int]:
    golds = {d for d, s in rels.items() if s > 0}
    return golds, len(golds)


def _metrics(ranked: list[str], golds: set[str], pool: set[str] | None = None) -> dict:
    pool = pool or set(ranked)
    in_top10 = golds & set(ranked[:10])
    in_top100 = golds & set(ranked[:100])
    return {
        "pool_size": len(pool),
        "gold_in_pool": len(golds & pool),
        "gold_in_top10": len(in_top10),
        "gold_in_top100": len(in_top100),
        "query_gold_in_top10": bool(in_top10),
        "query_gold_in_pool": bool(golds & pool),
    }


def subword_pieces(rare_terms: list[str], min_len: int = 3) -> list[str]:
    out: list[str] = []
    for w in rare_terms[:6]:
        for p in _chunk_subwords(w):
            if len(p) >= min_len and p != w:
                out.append(p)
    return list(dict.fromkeys(out))


def subword_posting_pool(ctx: MatrixCtx, rare: list[str]) -> set[str]:
    pool: set[str] = set()
    for p in subword_pieces(rare):
        pool |= posting_docs(ctx.idx, p)
    return pool


def correlate_neighbor_terms(knowledge, word: str, limit: int = 10) -> list[tuple[str, float]]:
    if knowledge is None:
        return []
    w = word.lower()
    links: list[tuple[str, float]] = []
    for link in knowledge.neighbors(w):
        other = link.right if link.left == w else link.left
        links.append((other, link.strength))
    links.sort(key=lambda x: (-x[1], x[0]))
    return links[:limit]


def correlate_neighbor_pool(ctx: MatrixCtx, rare: list[str]) -> set[str]:
    pool: set[str] = set()
    if ctx.scifact_knowledge is None:
        return pool
    for w in rare[:4]:
        for other, _ in correlate_neighbor_terms(ctx.scifact_knowledge, w, limit=8):
            pd = posting_docs(ctx.idx, other)
            if len(pd) <= 800:
                pool |= pd
    return pool


def compound_meet_pool(ctx: MatrixCtx, rare: list[str]) -> set[str]:
    pool: set[str] = set()
    for a, b in itertools.combinations(rare[:5], 2):
        pool |= posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
    return pool


def build_wide_pool(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str], cap: int = 1200) -> set[str]:
    pool = set(ctx.idx.search(query, 100))
    pool |= set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
    if rarest:
        pool |= posting_docs(ctx.idx, rarest[0])
    pool |= subword_posting_pool(ctx, rare)
    pool |= correlate_neighbor_pool(ctx, rare)
    pool |= compound_meet_pool(ctx, rare)
    pool |= set(corridor_bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn).keys())
    if len(pool) > cap:
        # keep bridge + rarest hits first
        br = set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
        keep = br
        if rarest:
            keep |= posting_docs(ctx.idx, rarest[0])
        rest = list(pool - keep)
        pool = keep | set(rest[:max(0, cap - len(keep))])
    return pool


def precompute_doc_rare(ctx: MatrixCtx) -> None:
    for doc_id, text in ctx.corpus.items():
        toks = set(words(text))
        ctx.doc_rare_tokens[doc_id] = [w for w in toks if ctx.idf_fn(w) >= 2.0]


def correlate_score_doc(
    ctx: MatrixCtx,
    doc_id: str,
    query: str,
    rare: list[str],
) -> float:
    if ctx.scifact_knowledge is None:
        return 0.0
    key = (doc_id, tuple(rare[:6]))
    cached = ctx.corr_score_cache.get(key)
    if cached is not None:
        return cached
    text = ctx.corpus.get(doc_id, "")
    cache = _DocFreqCache(ctx.scifact_knowledge)
    sc = score_doc_rare_correlations(
        ctx.scifact_knowledge,
        words(query),
        doc_id,
        text,
        df_cache=cache,
        rare_query=rare,
        rare_doc_tokens=set(ctx.doc_rare_tokens.get(doc_id, [])),
    )
    ctx.corr_score_cache[key] = sc
    return sc


def subword_score_doc(ctx: MatrixCtx, doc_id: str, rare: list[str]) -> float:
    toks = _toks(doc_id, ctx.corpus)
    pieces = subword_pieces(rare)
    return float(sum(1 for p in pieces if p in toks))


def norm_scores(scores: dict[str, float]) -> dict[str, float]:
    mx = max(scores.values()) if scores else 0.0
    if mx <= 0:
        return {d: 0.0 for d in scores}
    return {d: s / mx for d, s in scores.items()}


def rank_multisignal(
    ctx: MatrixCtx,
    query: str,
    pool: set[str],
    rare: list[str],
    weights: dict[str, float],
) -> list[str]:
    if not pool:
        return []
    lex = score_candidates(ctx.idx, query, pool)
    br = bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn)
    corr: dict[str, float] = {}
    sub: dict[str, float] = {}
    pair: dict[str, float] = defaultdict(float)
    for d in pool:
        corr[d] = correlate_score_doc(ctx, d, query, rare)
        sub[d] = subword_score_doc(ctx, d, rare)
    for a, b in itertools.combinations(rare[:4], 2):
        meet = posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
        wt = ctx.idf_fn(a) + ctx.idf_fn(b)
        for d in meet:
            if d in pool:
                pair[d] += wt

    lex_n = norm_scores(lex)
    br_n = norm_scores(br)
    corr_n = norm_scores(corr)
    sub_n = norm_scores(sub)
    pair_n = norm_scores(dict(pair))

    scored: list[tuple[float, str]] = []
    for d in pool:
        s = (
            weights.get("lex", 0.0) * lex_n.get(d, 0.0)
            + weights.get("bridge", 0.0) * br_n.get(d, 0.0)
            + weights.get("corr", 0.0) * corr_n.get(d, 0.0)
            + weights.get("sub", 0.0) * sub_n.get(d, 0.0)
            + weights.get("pair", 0.0) * pair_n.get(d, 0.0)
        )
        scored.append((s, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [d for _, d in scored[:100]]


# ---------------------------------------------------------------------------
# Gold path census (text signals on gold instances)
# ---------------------------------------------------------------------------

def gold_path_census(
    ctx: MatrixCtx,
    query: str,
    gold_local: dict[str, int],
    rarest: list[str],
    rare: list[str],
) -> list[dict]:
    rows = []
    for local_id, score in gold_local.items():
        if score <= 0 or local_id not in ctx.corpus:
            continue
        toks = _toks(local_id, ctx.corpus)

        h1 = bool(rarest) and rarest[0] in toks
        h2 = len(rarest) >= 2 and rarest[0] in toks and rarest[1] in toks
        pairs = sum(
            1 for a, b in itertools.combinations(rare[:4], 2)
            if a in toks and b in toks
        )
        sub_hits = subword_score_doc(ctx, local_id, rare)
        corr_sc = correlate_score_doc(ctx, local_id, query, rare)

        paths = []
        if h1:
            paths.append("rarest_1")
        if h2:
            paths.append("rarest_2")
        if pairs:
            paths.append("compound_pair")
        if sub_hits:
            paths.append("subword")
        if corr_sc > 0:
            paths.append("corr_score")
        if ctx.scifact_knowledge and rare and corr_sc > 0:
            paths.append("correlate_semantic")

        bridge_paths = []
        for qt in set(words(query)):
            for dt, wt in ctx.br.bridge.get(qt, ()):
                if dt in toks:
                    bridge_paths.append({"qt": qt, "dt": dt, "w": round(wt, 3)})
        if bridge_paths:
            paths.append("bridge")

        rows.append({
            "local_id": local_id,
            "paths": paths,
            "corr_score": round(corr_sc, 3),
            "subword_hits": sub_hits,
            "compound_pairs": pairs,
        })
    return rows


# ---------------------------------------------------------------------------
# Retrieval probes
# ---------------------------------------------------------------------------

def probe_bridge_bm25(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = set(bridge_expansion(ctx.idx, ctx.br, query, idf=ctx.idf_fn))
    pool |= set(ctx.idx.search(query, 100))
    ranked = _rank_pool(ctx, query, pool)
    return pool, ranked


def probe_glass_bm25(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    cfg = GlassBoxSearchConfig.scifact_target(polluter_docs=frozenset())
    cfg.use_corridors = False
    ranked = glass_box_search(
        ctx.idx, ctx.br, query, k=100,
        glossary=ctx.glossary, config=cfg, corpus=ctx.corpus,
        kappa_index=ctx.kappa_index, registry=ctx.registry,
    )
    pool = set(ranked)
    return pool, ranked


def probe_glass_corridors(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    cfg = GlassBoxSearchConfig.scifact_target(polluter_docs=frozenset())
    cfg.use_corridors = True
    ranked = glass_box_search(
        ctx.idx, ctx.br, query, k=100,
        glossary=ctx.glossary, config=cfg, corpus=ctx.corpus,
        kappa_index=ctx.kappa_index, registry=ctx.registry,
    )
    return set(ranked), ranked


def probe_glass_lattice_restrict(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    cfg = GlassBoxSearchConfig.scifact_lattice(polluter_docs=frozenset())
    ranked = glass_box_search(
        ctx.idx, ctx.br, query, k=100,
        glossary=ctx.glossary, config=cfg, corpus=ctx.corpus,
        kappa_index=ctx.kappa_index, registry=ctx.registry,
    )
    return set(ranked), ranked


def probe_glass_high_lam(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    cfg = GlassBoxSearchConfig.scifact_target(polluter_docs=frozenset())
    cfg.lam = 0.55
    cfg.lexical_mode = "bm25"
    ranked = glass_box_search(
        ctx.idx, ctx.br, query, k=100,
        glossary=ctx.glossary, config=cfg, corpus=ctx.corpus,
        kappa_index=ctx.kappa_index, registry=ctx.registry,
    )
    return set(ranked), ranked


def probe_wide_multisignal_balanced(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = build_wide_pool(ctx, query, rarest, rare)
    ranked = rank_multisignal(
        ctx, query, pool, rare,
        weights={"lex": 0.30, "bridge": 0.30, "corr": 0.20, "sub": 0.10, "pair": 0.10},
    )
    return pool, ranked


def probe_wide_corr_heavy(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = build_wide_pool(ctx, query, rarest, rare)
    ranked = rank_multisignal(
        ctx, query, pool, rare,
        weights={"lex": 0.20, "bridge": 0.25, "corr": 0.35, "sub": 0.10, "pair": 0.10},
    )
    return pool, ranked


def probe_wide_subword_heavy(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = build_wide_pool(ctx, query, rarest, rare)
    ranked = rank_multisignal(
        ctx, query, pool, rare,
        weights={"lex": 0.25, "bridge": 0.25, "corr": 0.15, "sub": 0.25, "pair": 0.10},
    )
    return pool, ranked


def probe_wide_pair_heavy(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = build_wide_pool(ctx, query, rarest, rare)
    ranked = rank_multisignal(
        ctx, query, pool, rare,
        weights={"lex": 0.25, "bridge": 0.25, "corr": 0.15, "sub": 0.05, "pair": 0.30},
    )
    return pool, ranked


def probe_mine_correlate_rank(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = build_wide_pool(ctx, query, rarest, rare)
    trigger = pool
    mined = mine_rare_from_docs(ctx.corpus, trigger, ctx.idf_fn, gate=ctx.rare_gate, top_k=30)
    scored = []
    for d in pool:
        lex = score_candidates(ctx.idx, query, {d}).get(d, 0.0)
        ov = overlap_score(_toks(d, ctx.corpus), mined)
        corr = correlate_score_doc(ctx, d, query, rare)
        scored.append((0.25 * lex + 0.35 * ov + 0.40 * corr, d))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return pool, [d for _, d in scored[:100]]


def probe_correlate_neighbor_only(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    pool = correlate_neighbor_pool(ctx, rare)
    if rarest:
        pool |= posting_docs(ctx.idx, rarest[0])
    ranked = rank_multisignal(
        ctx, query, pool, rare,
        weights={"lex": 0.35, "bridge": 0.20, "corr": 0.35, "sub": 0.10, "pair": 0.10},
    )
    return pool, ranked


def probe_subword_expand_glass(ctx: MatrixCtx, query: str, rarest: list[str], rare: list[str]) -> tuple[set[str], list[str]]:
    """Glass fuse pool + extra docs from subword postings."""
    cfg = GlassBoxSearchConfig.scifact_target(polluter_docs=frozenset())
    N = ctx.N
    from aethos_glass_box_search import glossary_expand_query
    expanded = glossary_expand_query(
        query, ctx.idx, N, ctx.glossary,
        idf_gate=cfg.glossary_idf_gate, max_extra=cfg.glossary_max_extra,
    )
    ranked, _, _ = _fuse_pool(
        ctx.idx, ctx.br, expanded, words(query), N, cfg,
        corpus=ctx.corpus, kappa_index=ctx.kappa_index, registry=ctx.registry,
    )
    extra = subword_posting_pool(ctx, rare)
    pool = set(ranked) | extra
    if extra:
        ranked = rank_multisignal(
            ctx, query, pool, rare,
            weights={"lex": 0.35, "bridge": 0.30, "corr": 0.15, "sub": 0.15, "pair": 0.05},
        )
    return pool, ranked[:100]


def _rank_pool(ctx: MatrixCtx, query: str, pool: set[str]) -> list[str]:
    if not pool:
        return []
    scores = score_candidates(ctx.idx, query, pool)
    return sorted(scores, key=scores.get, reverse=True)[:100]


PROBES: list[tuple[str, str, Callable]] = [
    ("01_bridge_bm25", "Bridge expansion + BM25", probe_bridge_bm25),
    ("02_glass_bm25", "Glass glossary+bridge+meet BM25", probe_glass_bm25),
    ("03_glass_corridors", "Glass + rarest corridors", probe_glass_corridors),
    ("04_glass_lattice_restrict", "Glass lattice κ-pool restrict", probe_glass_lattice_restrict),
    ("05_glass_high_lam", "Glass BM25 lam=0.55", probe_glass_high_lam),
    ("06_wide_multisignal", "Wide pool: lex+bridge+corr+sub+pair", probe_wide_multisignal_balanced),
    ("07_wide_corr_heavy", "Wide pool correlate-heavy fusion", probe_wide_corr_heavy),
    ("08_wide_subword_heavy", "Wide pool subword-heavy fusion", probe_wide_subword_heavy),
    ("09_wide_pair_heavy", "Wide pool compound-pair heavy", probe_wide_pair_heavy),
    ("10_mine_correlate_rank", "Mine rare + correlate rank on wide pool", probe_mine_correlate_rank),
    ("11_correlate_neighbor_pool", "Correlate neighbor postings + multisignal", probe_correlate_neighbor_only),
    ("12_subword_expand_glass", "Glass fuse + subword posting expand", probe_subword_expand_glass),
]


def summarize_census(all_census: list[list[dict]]) -> dict:
    path_counts: Counter = Counter()
    n_inst = 0
    for rows in all_census:
        for r in rows:
            n_inst += 1
            for p in r["paths"]:
                path_counts[p] += 1
    def pct(n):
        return round(100.0 * n / max(n_inst, 1), 1)
    return {
        "gold_instances": n_inst,
        "path_pct": {p: pct(path_counts[p]) for p in sorted(path_counts)},
        "path_counts": dict(path_counts),
    }


def summarize_misses(
    all_census: list[list[dict]],
    ranked_per_query: list[list[str]],
    gold_per_query: list[set[str]],
) -> dict:
    """Gold instances NOT in top-10: which paths they still have."""
    miss_paths: Counter = Counter()
    n_miss = 0
    for rows, ranked, golds in zip(all_census, ranked_per_query, gold_per_query):
        top10 = set(ranked[:10])
        for r in rows:
            if r["local_id"] in golds and r["local_id"] not in top10:
                n_miss += 1
                for p in r["paths"]:
                    miss_paths[p] += 1
    return {
        "gold_instances_missed_top10": n_miss,
        "miss_path_pct": {
            p: round(100 * miss_paths[p] / max(n_miss, 1), 1)
            for p in sorted(miss_paths)
        },
    }


def evaluate_probes(
    ctx: MatrixCtx,
    probes: list[tuple[str, str, Callable]],
    queries: dict[str, str],
    test_q: dict,
    test_ids: list[str],
    skip_census: bool = False,
) -> tuple[list[dict], dict]:
    results = []
    all_census: list[list[dict]] = []
    best_ranked: list[list[str]] = []
    gold_per_query: list[set[str]] = []

    for name, desc, fn in probes:
        print(f"  probe {name} ...", flush=True)
        g_total = q_top10 = g_top10 = 0
        pool_sizes: list[int] = []
        ndcg_sum = 0.0
        ranked_list: list[list[str]] = []

        for i, qid in enumerate(test_ids):
            q = queries[qid]
            golds, gn = _gold_sets(test_q[qid])
            if name == PROBES[0][0]:
                gold_per_query.append(golds)
            qterms = words(q)
            rarest = rarest_terms(qterms, ctx.idx, ctx.N)
            rare = [w for w in rarest if ctx.idf_fn(w) >= ctx.rare_gate]

            if not skip_census and name == PROBES[0][0]:
                all_census.append(gold_path_census(ctx, q, test_q[qid], rarest, rare))

            pool, ranked = fn(ctx, q, rarest, rare)
            ranked_list.append(ranked)
            m = _metrics(ranked, golds, pool)
            pool_sizes.append(m["pool_size"])
            g_top10 += m["gold_in_top10"]
            g_total += gn
            if m["query_gold_in_top10"]:
                q_top10 += 1
            ndcg_sum += ndcg10(ranked[:10], test_q[qid])

            if (i + 1) % 50 == 0:
                print(f"    {name}: {i+1}/{len(test_ids)}", flush=True)

        n_q = len(test_ids)
        def pct(a, b):
            return round(100.0 * a / max(b, 1), 1)

        results.append({
            "name": name,
            "description": desc,
            "queries": n_q,
            "queries_gold_in_top10_pct": pct(q_top10, n_q),
            "gold_instances_in_top10_pct": pct(g_top10, g_total),
            "mean_pool_size": round(sum(pool_sizes) / max(len(pool_sizes), 1), 1),
            "ndcg10": round(ndcg_sum / max(n_q, 1), 4),
        })

        if name == "06_wide_multisignal":
            best_ranked = ranked_list

    census_summary = summarize_census(all_census) if all_census else {"gold_instances": 0, "path_pct": {}, "path_counts": {}}
    miss_summary = (
        summarize_misses(all_census, best_ranked, gold_per_query)
        if all_census and best_ranked
        else {"gold_instances_missed_top10": 0, "miss_path_pct": {}}
    )
    return results, {"census": census_summary, "misses": miss_summary}


def write_report(results: list[dict], extra: dict, path: Path) -> None:
    lines = [
        "# NFCorpus glass-box correlation matrix\n\n",
        "## Retrieval probes (gold instances in top-10)\n\n",
        "| probe | inst@10 | Q@10 | nDCG@10 | mean pool |\n",
        "|-------|---------|------|---------|----------|\n",
    ]
    for r in sorted(results, key=lambda x: -x["gold_instances_in_top10_pct"]):
        lines.append(
            f"| {r['name']} | {r['gold_instances_in_top10_pct']}% | "
            f"{r['queries_gold_in_top10_pct']}% | {r['ndcg10']} | {r['mean_pool_size']} |\n"
        )
    cs = extra["census"]
    lines.append("\n## Gold text path census (all instances)\n\n")
    for p, v in sorted(cs["path_pct"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{p}**: {v}% ({cs['path_counts'].get(p, 0)} instances)\n")
    ms = extra["misses"]
    lines.append(f"\n## Missed top-10 ({ms['gold_instances_missed_top10']} instances) — paths still present\n\n")
    for p, v in sorted(ms["miss_path_pct"].items(), key=lambda x: -x[1]):
        lines.append(f"- **{p}**: {v}% of misses still have this signal\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="NF glass-box correlation matrix")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", default="logs/nf_glass_correlation_matrix.json")
    p.add_argument("--rules-md", default="logs/nf_glass_correlation_matrix.md")
    p.add_argument("--index-mode", default="kappa_primary")
    p.add_argument("--skip-census", action="store_true", help="Skip gold path census (faster)")
    args = p.parse_args()

    corpus, queries, train_q, test_q = load("nfcorpus")
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    print(f"NF glass correlation matrix: {len(test_ids)} queries", flush=True)
    t0 = time.perf_counter()

    idx = AppendOnlyLatticeIndex(index_mode=args.index_mode)
    for d, t in corpus.items():
        idx.add(d, t)
    idx.finalize()
    N = len(idx.alive)
    idf_fn = lambda w: word_idf(idx, w, N)

    br = RelevanceBridges(idx, N, min_pairs=2).learn(queries, train_q, corpus)
    br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=2)
    gloss = load_glossary("nfcorpus")

    scifact_knowledge = None
    try:
        from eval_beir_symbol import load_brain_and_plane
        scifact_knowledge, _ = load_brain_and_plane("scifact")
        print("  SciFact knowledge loaded", flush=True)
    except Exception as e:
        print(f"  WARN knowledge: {e}", flush=True)

    from aethos_promotion import PromotionRegistry
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast
    registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
    for text in corpus.values():
        registry.observe_text(text)
    kappa_index = build_attractor_index_fast(
        registry, corpus, lambda w: word_idf(idx, w, N), top_k=10,
    )

    ctx = MatrixCtx(
        idx=idx, corpus=corpus, N=N, idf_fn=idf_fn, br=br,
        glossary=gloss, scifact_knowledge=scifact_knowledge,
        kappa_index=kappa_index, registry=registry,
    )
    print("  precomputing doc rare tokens...", flush=True)
    precompute_doc_rare(ctx)
    print(f"  built in {time.perf_counter()-t0:.1f}s", flush=True)

    results, extra = evaluate_probes(ctx, PROBES, queries, test_q, test_ids, skip_census=args.skip_census)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": len(test_ids),
        "probes": results,
        "census_summary": extra["census"],
        "miss_summary": extra["misses"],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(results, extra, Path(args.rules_md))

    print(f"\n{'='*72}")
    print("  GLASS CORRELATION MATRIX — inst@10 leaderboard")
    print(f"  {'probe':<28} {'inst@10':>8} {'Q@10':>7} {'nDCG':>7}")
    print("  " + "-" * 52)
    for r in sorted(results, key=lambda x: -x["gold_instances_in_top10_pct"])[:8]:
        print(
            f"  {r['name']:<28} {r['gold_instances_in_top10_pct']:>7.1f}% "
            f"{r['queries_gold_in_top10_pct']:>6.1f}% {r['ndcg10']:>7.4f}"
        )
    print(f"\n  JSON: {out}")
    print(f"  Report: {args.rules_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
