#!/usr/bin/env python3
"""
Gold-path audit — trace every inspectable route to gold docs.

For each gold doc instance on the test split, records which PATHS reach it:

  TEXT (gold doc explains via):
    rarest_1_text, rarest_2_text, rarest_3_text
    compound_pair_text, compound_triple_text
    bridge_nf_text, bridge_scifact_text, teach_text
    semantic_neighbor_nf, semantic_correlate_scifact
    rarest_to_rarest_text

  POOL (candidate pool would include gold via):
    lex_top100, rarest_1_posting, rarest_2_posting, rare_2nd_only_posting
    rare_posting_df256, kappa_route
    compound_meet, rarest_to_rarest_meet, pair_meet_union
    bridge_nf_pool, bridge_scifact_pool, corridor_nf_pool
    trigger_2nd_order_pool  (semantic neighbors mined from query-triggered docs)

Cross-corpus: SciFact-trained bridges + symbol correlates applied on NFCorpus queries.

Run:
  python scripts/audit_gold_paths_cross_corpus.py nfcorpus
  python scripts/audit_gold_paths_cross_corpus.py nfcorpus scifact
  python scripts/audit_gold_paths_cross_corpus.py nfcorpus --out logs/gold_paths_nfcorpus.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_expansion, corridor_bridge_expansion
from aethos_glass_box_search import posting_docs, rarest_terms, word_idf
from scripts.bench_supervised_bridges import load


# ---------------------------------------------------------------------------
# second-order semantics (rare-anchored, from triggered docs)
# ---------------------------------------------------------------------------

def build_second_order(corpus: dict[str, str], idf_fn: Callable[[str], float], gate: float = 2.5):
    """Rare-anchor context vectors for synonym-style expansion."""
    df: Counter = Counter()
    docterms: list[set[str]] = []
    for txt in corpus.values():
        ts = set(words(txt))
        docterms.append(ts)
        for w in ts:
            df[w] += 1
    N = len(docterms)

    co: dict[str, Counter] = defaultdict(Counter)
    for ts in docterms:
        rare = [w for w in ts if idf_fn(w) >= gate]
        for t in ts:
            ct = co[t]
            for a in rare:
                if a != t:
                    ct[a] += 1

    inv: dict[str, list[str]] = defaultdict(list)
    norm: dict[str, float] = {}
    for t, ctx in co.items():
        ss = 0.0
        for a, c in ctx.items():
            w = c * idf_fn(a)
            ss += w * w
            inv[a].append(t)
        norm[t] = math.sqrt(ss) or 1.0
    return co, inv, norm, N


def neighbors_2nd(
    term: str,
    co: dict,
    inv: dict,
    norm: dict,
    idf_fn: Callable[[str], float],
    k: int = 10,
) -> list[tuple[float, str]]:
    if term not in co:
        return []
    cand: Counter = Counter()
    for a, c in co[term].items():
        wa2 = c * idf_fn(a) * idf_fn(a)
        for t in inv[a]:
            cand[t] += wa2 * co[t][a]
    na = norm[term]
    out = []
    for t, dot in cand.items():
        if t == term:
            continue
        out.append((dot / (na * norm[t]), t))
    out.sort(reverse=True)
    return out[:k]


def build_coocc_adjacency(corpus: dict[str, str], idf_fn: Callable[[str], float], gate: float = 2.0):
    """First-order weighted co-occurrence neighbors."""
    adj: dict[str, Counter] = defaultdict(Counter)
    for txt in corpus.values():
        ts = [w for w in set(words(txt)) if idf_fn(w) >= gate]
        for i, a in enumerate(ts):
            for b in ts[i + 1:]:
                w = idf_fn(a) * idf_fn(b)
                adj[a][b] += w
                adj[b][a] += w
    return adj


# ---------------------------------------------------------------------------
# context
# ---------------------------------------------------------------------------

@dataclass
class PathCtx:
    dataset: str
    idx: AppendOnlyLatticeIndex
    corpus: dict[str, str]
    N: int
    br_nf: RelevanceBridges
    br_scifact: RelevanceBridges | None = None
    kappa_index: object | None = None
    registry: object | None = None
    teach: object | None = None
    scifact_knowledge: object | None = None
    co2: tuple = ()
    nbr2_map: dict[str, list[str]] = field(default_factory=dict)
    adj_nf: dict = field(default_factory=dict)
    idf_fn: Callable[[str], float] = lambda w: 0.0
    rare_df_cap: int = 256


def _doc_toks(corpus: dict[str, str], doc_id: str) -> set[str]:
    return set(words(corpus.get(doc_id, "")))


def _posting_pool(idx: AppendOnlyLatticeIndex, w: str) -> set[str]:
    return posting_docs(idx, w)


def _rare_posting_union(idx: AppendOnlyLatticeIndex, query: str, cap: int) -> set[str]:
    pool: set[str] = set()
    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        dfp = idx.df.get(p, 0)
        if 0 < dfp <= cap:
            pl = idx.postings.get(p)
            if pl:
                pool.update(d for d in pl if d in idx.alive)
    return pool


def _pair_meet_pool(idx: AppendOnlyLatticeIndex, terms: list[str]) -> set[str]:
    pool: set[str] = set()
    for a, b in itertools.combinations(terms, 2):
        pool |= posting_docs(idx, a) & posting_docs(idx, b)
    return pool


def _bridge_pool(idx, br, query: str, idf_fn) -> set[str]:
    if br is None:
        return set()
    return set(bridge_expansion(idx, br, query, idf=idf_fn))


def _corridor_pool(idx, br, query: str, idf_fn) -> set[str]:
    if br is None or not getattr(br, "corridor_bridge", None):
        return set()
    return set(corridor_bridge_expansion(idx, br, query, idf=idf_fn).keys())


def _kappa_pool(ctx: PathCtx, query: str) -> set[str]:
    if ctx.kappa_index is None or ctx.registry is None:
        return set()
    from pipeline.bit_04_candidate_router import (
        candidates_from_attractors,
        query_words_for_routing,
    )
    routed = query_words_for_routing(words(query))
    if not routed:
        return set()
    kdocs, _ = candidates_from_attractors(
        routed, ctx.registry, ctx.kappa_index, idf=ctx.idf_fn,
    )
    return set(kdocs[:600])


def precompute_neighbors_2nd(
    co: dict,
    inv: dict,
    norm: dict,
    idf_fn: Callable[[str], float],
    k: int = 8,
    max_terms: int = 4000,
) -> dict[str, list[str]]:
    """Precompute 2nd-order neighbors for high-idf terms only (bounded)."""
    ranked = sorted(co.keys(), key=lambda t: -idf_fn(t))[:max_terms]
    out: dict[str, list[str]] = {}
    for term in ranked:
        nbrs = neighbors_2nd(term, co, inv, norm, idf_fn, k=k)
        out[term] = [t for _, t in nbrs]
    return out


def _trigger_second_order_pool(
    ctx: PathCtx,
    query: str,
    triggered: set[str],
    gate: float = 2.5,
    max_docs: int = 800,
) -> set[str]:
    nbr_map = ctx.nbr2_map
    if not nbr_map:
        return set()

    anchor_counts: Counter = Counter()
    for d in triggered:
        if d not in ctx.corpus:
            continue
        for w in _doc_toks(ctx.corpus, d):
            if ctx.idf_fn(w) >= gate:
                anchor_counts[w] += 1

    neighbor_terms: set[str] = set()
    q_rare = [w for w in rarest_terms(words(query), ctx.idx, ctx.N)
              if ctx.idf_fn(w) >= gate][:4]
    for w in q_rare:
        neighbor_terms.update(nbr_map.get(w, [])[:8])
    for a, _ in anchor_counts.most_common(12):
        neighbor_terms.update(nbr_map.get(a, [])[:4])

    pool: set[str] = set()
    for t in neighbor_terms:
        pd = posting_docs(ctx.idx, t)
        if len(pd) > 400:
            continue
        pool |= pd
        if len(pool) >= max_docs:
            break
    return pool


def _semantic_neighbor_text(
    rarest_w: str,
    doc_toks: set[str],
    adj: dict,
    k: int = 12,
) -> bool:
    nbrs = adj.get(rarest_w, {})
    if not nbrs:
        return False
    top = [w for w, _ in nbrs.most_common(k)]
    return any(w in doc_toks for w in top)


def _semantic_correlate_scifact(
    knowledge,
    query_rare: list[str],
    doc_toks: set[str],
    idf_fn: Callable[[str], float] | None = None,
) -> list[dict]:
    if knowledge is None:
        return []
    # only rare doc tokens — long NF abstracts blow up correlate checks
    doc_rare = [w for w in doc_toks if idf_fn is None or idf_fn(w) >= 2.0]
    links = []
    for qw in query_rare[:6]:
        for dw in doc_rare[:40]:
            if qw == dw:
                continue
            lk = knowledge.correlates(qw, dw)
            if lk is not None:
                links.append({
                    "query": qw, "doc": dw,
                    "strength": round(lk.strength, 3),
                    "kind": lk.kind,
                })
    links.sort(key=lambda x: -x["strength"])
    return links[:8]


def build_pools(ctx: PathCtx, query: str, rarest: list[str]) -> dict[str, set[str]]:
    idx = ctx.idx
    idf = ctx.idf_fn
    rare_gate = [w for w in rarest if idf(w) >= 2.5][:6]

    lex100 = set(idx.search(query, 100))
    br_nf = _bridge_pool(idx, ctx.br_nf, query, idf)
    br_sf = _bridge_pool(idx, ctx.br_scifact, query, idf)
    triggered = lex100 | br_nf

    pools = {
        "lex_top100": lex100,
        "rarest_1_posting": _posting_pool(idx, rarest[0]) if rarest else set(),
        "rarest_2_posting": _posting_pool(idx, rarest[1]) if len(rarest) > 1 else set(),
        "rare_posting_df256": _rare_posting_union(idx, query, ctx.rare_df_cap),
        "kappa_route": _kappa_pool(ctx, query),
        "compound_meet": _pair_meet_pool(idx, rare_gate),
        "rarest_to_rarest_meet": _pair_meet_pool(idx, rarest[:4]),
        "pair_meet_union": _pair_meet_pool(idx, rare_gate or rarest[:6]),
        "bridge_nf_pool": br_nf,
        "bridge_scifact_pool": br_sf,
        "corridor_nf_pool": _corridor_pool(idx, ctx.br_nf, query, idf),
        "trigger_2nd_order_pool": _trigger_second_order_pool(ctx, query, triggered),
    }
    if rarest:
        r1 = pools["rarest_1_posting"]
        pools["rare_2nd_only_posting"] = pools["rarest_2_posting"] - r1 if len(rarest) > 1 else set()
    else:
        pools["rare_2nd_only_posting"] = set()
    return pools


def audit_gold_paths(
    ctx: PathCtx,
    query: str,
    gold_local: dict[str, int],
    *,
    teach=None,
) -> dict:
    qterms = words(query)
    rarest = rarest_terms(qterms, ctx.idx, ctx.N)
    rare_gate = [w for w in rarest if ctx.idf_fn(w) >= 2.5][:6]
    pools = build_pools(ctx, query, rarest)

    gold_instances = []
    for local_id, score in gold_local.items():
        if score <= 0 or local_id not in ctx.corpus:
            continue
        toks = _doc_toks(ctx.corpus, local_id)

        h1 = bool(rarest) and rarest[0] in toks
        h2 = len(rarest) >= 2 and rarest[0] in toks and rarest[1] in toks
        h3 = len(rarest) >= 3 and all(w in toks for w in rarest[:3])

        compound = _pair_triple_hits(rare_gate or rarest[:6], toks)
        bridge_nf_text = _bridge_text_paths(ctx.br_nf, query, toks)
        bridge_sf_text = _bridge_text_paths(ctx.br_scifact, query, toks)
        teach_text = _teach_text_paths(teach, query, toks)
        sem_nbr = _semantic_neighbor_text(rarest[0], toks, ctx.adj_nf) if rarest else False
        sem_corr = _semantic_correlate_scifact(
            ctx.scifact_knowledge, rare_gate or rarest[:4], toks, ctx.idf_fn,
        )
        r2r_text = len(rare_gate) >= 2 and all(
            any(w in toks for w in rare_gate[:2])
            for _ in [None]
        ) and rare_gate[0] in toks and rare_gate[1] in toks

        text_paths = set()
        if h1:
            text_paths.add("rarest_1_text")
        if h2:
            text_paths.add("rarest_2_text")
        if h3:
            text_paths.add("rarest_3_text")
        if compound["n_rare_pairs_in_doc"]:
            text_paths.add("compound_pair_text")
        if compound["n_rare_triples_in_doc"]:
            text_paths.add("compound_triple_text")
        if bridge_nf_text:
            text_paths.add("bridge_nf_text")
        if bridge_sf_text:
            text_paths.add("bridge_scifact_text")
        if teach_text:
            text_paths.add("teach_text")
        if sem_nbr:
            text_paths.add("semantic_neighbor_nf")
        if sem_corr:
            text_paths.add("semantic_correlate_scifact")
        if r2r_text:
            text_paths.add("rarest_to_rarest_text")

        pool_paths = {name for name, ps in pools.items() if local_id in ps}

        gold_instances.append({
            "local_id": local_id,
            "text_paths": sorted(text_paths),
            "pool_paths": sorted(pool_paths),
            "bridge_nf_paths": bridge_nf_text[:4],
            "bridge_scifact_paths": bridge_sf_text[:4],
            "semantic_correlate_scifact": sem_corr[:4],
            "compound_pairs": compound["sample_pairs"],
        })

    return {
        "rarest_query_words": rarest[:6],
        "rare_gate_terms": rare_gate[:6],
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "gold": gold_instances,
    }


def _bridge_text_paths(br, query: str, doc_toks: set[str]) -> list[dict]:
    if br is None:
        return []
    paths = []
    for qt in set(words(query)):
        for dt, wt in br.bridge.get(qt, ()):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "weight": round(wt, 4)})
    paths.sort(key=lambda x: -x["weight"])
    return paths[:8]


def _teach_text_paths(teach, query: str, doc_toks: set[str]) -> list[dict]:
    if teach is None:
        return []
    paths = []
    for qt in set(words(query)):
        partners = teach.edges.get(qt)
        if not partners:
            continue
        for dt, c in partners.most_common(8):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "count": c})
    return paths


def _pair_triple_hits(rare_sorted: list[str], doc_toks: set[str]) -> dict:
    top = rare_sorted[:6]
    pairs_ok = []
    triples_ok = []
    for a, b in itertools.combinations(top, 2):
        if a in doc_toks and b in doc_toks:
            pairs_ok.append((a, b))
    for a, b, c in itertools.combinations(top, 3):
        if a in doc_toks and b in doc_toks and c in doc_toks:
            triples_ok.append((a, b, c))
    return {
        "n_rare_pairs_in_doc": len(pairs_ok),
        "n_rare_triples_in_doc": len(triples_ok),
        "sample_pairs": pairs_ok[:4],
    }


PATH_ORDER_TEXT = [
    "rarest_1_text", "rarest_2_text", "rarest_3_text",
    "compound_pair_text", "compound_triple_text",
    "rarest_to_rarest_text",
    "bridge_nf_text", "bridge_scifact_text", "teach_text",
    "semantic_neighbor_nf", "semantic_correlate_scifact",
]

PATH_ORDER_POOL = [
    "lex_top100", "rarest_1_posting", "rarest_2_posting", "rare_2nd_only_posting",
    "rare_posting_df256", "kappa_route",
    "compound_meet", "rarest_to_rarest_meet", "pair_meet_union",
    "bridge_nf_pool", "bridge_scifact_pool", "corridor_nf_pool",
    "trigger_2nd_order_pool",
]


def summarize_paths(rows: list[dict], path_order: list[str], key: str) -> dict:
    n_inst = sum(len(r["gold"]) for r in rows)
    n_q = len(rows)
    path_counts: Counter = Counter()
    q_any: Counter = Counter()

    for row in rows:
        for g in row["gold"]:
            for p in g[key]:
                path_counts[p] += 1
        for p in path_order:
            if any(p in g[key] for g in row["gold"]):
                q_any[p] += 1

    def pct(a, b):
        return round(100.0 * a / max(b, 1), 1)

    per_path = {}
    for p in path_order:
        per_path[p] = {
            "gold_instances_pct": pct(path_counts[p], n_inst),
            "gold_instances_n": path_counts[p],
            "queries_any_gold_pct": pct(q_any[p], n_q),
            "queries_any_gold_n": q_any[p],
        }
    return {
        "gold_doc_instances": n_inst,
        "queries": n_q,
        "per_path": per_path,
    }


def scifact_transfer_delta(nf_summary: dict, nf_pool: dict) -> dict:
    """Which SciFact-only paths add gold on NF beyond NF-native paths."""
    pool = nf_pool["per_path"]
    nf_only = pool.get("bridge_nf_pool", {})
    sf_only = pool.get("bridge_scifact_pool", {})
    text_nf = nf_summary.get("text", {}).get("per_path", {}).get("bridge_nf_text", {})
    text_sf = nf_summary.get("text", {}).get("per_path", {}).get("bridge_scifact_text", {})
    corr = nf_summary.get("text", {}).get("per_path", {}).get("semantic_correlate_scifact", {})
    return {
        "bridge_nf_pool_gold_pct": nf_only.get("gold_instances_pct", 0),
        "bridge_scifact_pool_gold_pct": sf_only.get("gold_instances_pct", 0),
        "scifact_bridge_pool_lift_pts": round(
            sf_only.get("gold_instances_pct", 0) - nf_only.get("gold_instances_pct", 0), 1
        ),
        "bridge_nf_text_gold_pct": text_nf.get("gold_instances_pct", 0),
        "bridge_scifact_text_gold_pct": text_sf.get("gold_instances_pct", 0),
        "semantic_correlate_scifact_text_pct": corr.get("gold_instances_pct", 0),
    }


def build_ctx_for_dataset(
    dataset: str,
    index_mode: str,
    min_pairs: int,
    scifact_br: RelevanceBridges | None,
    scifact_knowledge: object | None,
) -> PathCtx:
    corpus, queries, train_q, _ = load(dataset)
    print(f"  indexing {len(corpus)} docs...", flush=True)
    idx = AppendOnlyLatticeIndex(index_mode=index_mode)
    for d, t in corpus.items():
        idx.add(d, t)
    idx.finalize()
    N = len(idx.alive)
    print(f"  learning bridges (min_pairs={min_pairs})...", flush=True)
    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=min_pairs)

    idf_fn = lambda w: word_idf(idx, w, N)
    print("  building 2nd-order neighbor map (high-idf terms)...", flush=True)
    co, inv, norm, _ = build_second_order(corpus, idf_fn)
    nbr2_map = precompute_neighbors_2nd(co, inv, norm, idf_fn)
    adj = build_coocc_adjacency(corpus, idf_fn)

    print("  building kappa index...", flush=True)
    from aethos_promotion import PromotionRegistry
    from aethos_teach_store import TeachStore
    from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast

    registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
    for text in corpus.values():
        registry.observe_text(text)
    kappa_index = build_attractor_index_fast(
        registry, corpus, lambda w: word_idf(idx, w, N), top_k=10,
    )
    teach = TeachStore(idx, N)

    return PathCtx(
        dataset=dataset,
        idx=idx,
        corpus=corpus,
        N=N,
        br_nf=br,
        br_scifact=scifact_br,
        kappa_index=kappa_index,
        registry=registry,
        teach=teach,
        scifact_knowledge=scifact_knowledge,
        co2=(co, inv, norm, N),
        nbr2_map=nbr2_map,
        adj_nf=adj,
        idf_fn=idf_fn,
    )


def load_scifact_transfer(min_pairs: int) -> tuple[RelevanceBridges | None, object | None]:
    """SciFact-trained bridges (on SciFact index) + symbol knowledge for correlate transfer."""
    try:
        corpus, queries, train_q, _ = load("scifact")
        idx = AppendOnlyLatticeIndex(index_mode="kappa_primary")
        for d, t in corpus.items():
            idx.add(d, t)
        idx.finalize()
        N = len(idx.alive)
        br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
        br.learn_rarest_corridors(queries, train_q, corpus, min_pairs=min_pairs)
    except Exception as e:
        print(f"  WARN: SciFact bridge load failed: {e}", flush=True)
        br = None

    knowledge = None
    try:
        from eval_beir_symbol import load_brain_and_plane
        knowledge, _ = load_brain_and_plane("scifact")
    except Exception as e:
        print(f"  WARN: SciFact symbol knowledge load failed: {e}", flush=True)

    return br, knowledge


def run_dataset(
    dataset: str,
    test_ids: list[str],
    queries: dict[str, str],
    test_q: dict,
    ctx: PathCtx,
    teach=None,
) -> tuple[list[dict], dict]:
    rows = []
    t0 = time.perf_counter()
    for i, qid in enumerate(test_ids):
        row = audit_gold_paths(ctx, queries[qid], test_q[qid], teach=teach)
        row["query_id"] = qid
        row["query"] = queries[qid][:100]
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"    {dataset}: {i+1}/{len(test_ids)} queries", flush=True)

    text_sum = summarize_paths(rows, PATH_ORDER_TEXT, "text_paths")
    pool_sum = summarize_paths(rows, PATH_ORDER_POOL, "pool_paths")
    elapsed = time.perf_counter() - t0

    # union pool reach
    n_inst = text_sum["gold_doc_instances"]
    n_q = text_sum["queries"]
    inst_any_pool = sum(
        1 for row in rows for g in row["gold"] if g["pool_paths"]
    )
    q_any_pool = sum(
        1 for row in rows if any(g["pool_paths"] for g in row["gold"])
    )

    summary = {
        "dataset": dataset,
        "queries": n_q,
        "gold_doc_instances": n_inst,
        "queries_any_gold_in_pool_pct": round(100 * q_any_pool / max(n_q, 1), 1),
        "gold_instances_any_pool_pct": round(100 * inst_any_pool / max(n_inst, 1), 1),
        "text": text_sum,
        "pool": pool_sum,
        "wall_s": round(elapsed, 1),
    }
    return rows, summary


def print_summary(summary: dict) -> None:
    ds = summary["dataset"]
    print(f"\n{'='*76}")
    print(f"  GOLD PATH AUDIT — {ds.upper()}")
    print(f"  {summary['queries']} queries | {summary['gold_doc_instances']} gold instances")
    print(f"  any pool path: {summary['queries_any_gold_in_pool_pct']}% queries, "
          f"{summary['gold_instances_any_pool_pct']}% instances")
    print(f"{'='*76}")

    print("\n  TEXT PATHS (gold doc text / semantic link)")
    print(f"  {'path':<32} {'inst%':>7} {'inst_n':>7} {'query%':>8}")
    print("  " + "-" * 56)
    for p in PATH_ORDER_TEXT:
        r = summary["text"]["per_path"].get(p, {})
        print(f"  {p:<32} {r.get('gold_instances_pct', 0):>6.1f}% "
              f"{r.get('gold_instances_n', 0):>7} {r.get('queries_any_gold_pct', 0):>7.1f}%")

    print("\n  POOL PATHS (gold enters candidate pool)")
    print(f"  {'path':<32} {'inst%':>7} {'inst_n':>7} {'query%':>8}")
    print("  " + "-" * 56)
    for p in PATH_ORDER_POOL:
        r = summary["pool"]["per_path"].get(p, {})
        print(f"  {p:<32} {r.get('gold_instances_pct', 0):>6.1f}% "
              f"{r.get('gold_instances_n', 0):>7} {r.get('queries_any_gold_pct', 0):>7.1f}%")


def write_comparison_md(path: Path, summaries: dict[str, dict], transfer: dict | None) -> None:
    lines = ["# Gold path comparison\n\n"]
    datasets = list(summaries.keys())

    lines.append("## Pool paths — gold instances reached (%)\n\n")
    lines.append("| path | " + " | ".join(datasets) + " |\n")
    lines.append("|------|" + "|".join(["---"] * len(datasets)) + "|\n")
    for p in PATH_ORDER_POOL:
        cells = [str(summaries[d]["pool"]["per_path"].get(p, {}).get("gold_instances_pct", 0)) for d in datasets]
        lines.append(f"| {p} | " + " | ".join(cells) + " |\n")

    lines.append("\n## Text paths — gold instances (%)\n\n")
    lines.append("| path | " + " | ".join(datasets) + " |\n")
    lines.append("|------|" + "|".join(["---"] * len(datasets)) + "|\n")
    for p in PATH_ORDER_TEXT:
        cells = [str(summaries[d]["text"]["per_path"].get(p, {}).get("gold_instances_pct", 0)) for d in datasets]
        lines.append(f"| {p} | " + " | ".join(cells) + " |\n")

    if transfer:
        lines.append("\n## SciFact → NFCorpus transfer\n\n")
        for k, v in transfer.items():
            lines.append(f"- **{k}**: {v}\n")

    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Gold path audit across corpora")
    p.add_argument("datasets", nargs="*", default=["nfcorpus"])
    p.add_argument("--index-mode", default="kappa_primary", choices=("kappa_primary", "full"))
    p.add_argument("--out", default="")
    p.add_argument("--comparison-md", default="")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--no-scifact-transfer", action="store_true")
    args = p.parse_args()

    scifact_br, scifact_knowledge = None, None
    if not args.no_scifact_transfer and "nfcorpus" in args.datasets:
        print("Loading SciFact transfer (bridges + symbol correlates)...", flush=True)
        scifact_br, scifact_knowledge = load_scifact_transfer(
            min_pairs=1 if "scifact" in args.datasets else 2
        )

    all_summaries: dict[str, dict] = {}
    all_rows: dict[str, list] = {}
    out_base = Path(args.out or "logs/gold_paths_audit.json")

    for dataset in args.datasets:
        min_pairs = 1 if dataset == "scifact" else 2
        print(f"\nBuilding {dataset} (mode={args.index_mode}, min_pairs={min_pairs})...", flush=True)
        corpus, queries, train_q, test_q = load(dataset)
        test_ids = [q for q in test_q if q in queries]
        if args.max_queries:
            test_ids = test_ids[: args.max_queries]

        br_transfer = scifact_br if dataset == "nfcorpus" else None
        know_transfer = scifact_knowledge if dataset == "nfcorpus" else None
        ctx = build_ctx_for_dataset(
            dataset, args.index_mode, min_pairs, br_transfer, know_transfer,
        )
        teach = ctx.teach

        print(f"  Auditing {len(test_ids)} queries...", flush=True)
        rows, summary = run_dataset(dataset, test_ids, queries, test_q, ctx, teach=teach)
        all_summaries[dataset] = summary
        all_rows[dataset] = rows
        print_summary(summary)

    transfer = None
    if "nfcorpus" in all_summaries:
        transfer = scifact_transfer_delta(
            all_summaries["nfcorpus"], all_summaries["nfcorpus"]["pool"],
        )
        print("\n  SCIFACT -> NFCORPUS TRANSFER")
        for k, v in transfer.items():
            print(f"    {k}: {v}")

    payload = {
        "summaries": all_summaries,
        "scifact_transfer_on_nfcorpus": transfer,
        "queries": {d: all_rows[d] for d in all_rows},
    }
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out_base.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cmp_md = Path(args.comparison_md or "logs/gold_paths_comparison.md")
    write_comparison_md(cmp_md, all_summaries, transfer)
    print(f"\n  JSON: {out_base}")
    print(f"  Comparison: {cmp_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
