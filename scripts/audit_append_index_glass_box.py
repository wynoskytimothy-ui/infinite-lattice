#!/usr/bin/env python3
"""
Glass-box audit for AppendOnlyLatticeIndex — 25+ correlation probes.

Measures how query→gold signal flows through rare words, compound pairs,
top-doc cross-reference, bridges, and multi-view gears. Each probe is a
deterministic rerank/route strategy; we report how many gold docs enter top-10
vs baseline lexical search.

Run:
  python scripts/audit_append_index_glass_box.py
  python scripts/audit_append_index_glass_box.py scifact --index-mode full
  python scripts/audit_append_index_glass_box.py scifact --max-queries 50 --out logs/append_glass_box.json
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

from aethos_append_index import AppendOnlyLatticeIndex, words, GEARS
from aethos_bridges import RelevanceBridges, bridge_search
from aethos_encyclopedia_teacher import load_glossary
from aethos_glass_box_search import glass_box_search, GlassBoxSearchConfig, GlassBoxRetriever
from scripts.bench_active_learning import best_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10

try:
    from scifact_glossary import GLOSSARY as _SCIFACT_GLOSSARY_FALLBACK
except ImportError:
    _SCIFACT_GLOSSARY_FALLBACK = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def word_idf(idx: AppendOnlyLatticeIndex, w: str, N: int) -> float:
    p = idx.token_prime.get(("w", w))
    return idx._idf(p, N) if p else 0.0


def word_df(idx: AppendOnlyLatticeIndex, w: str) -> int:
    p = idx.token_prime.get(("w", w))
    return idx.df.get(p, 0) if p else 0


def rarest_terms(qterms: list[str], idx: AppendOnlyLatticeIndex, N: int) -> list[str]:
    uniq = list(dict.fromkeys(qterms))
    return sorted(uniq, key=lambda w: (word_idf(idx, w, N), w), reverse=True)


def rare_gate_terms(rarest: list[str], idx: AppendOnlyLatticeIndex, N: int, gate: float = 3.0) -> list[str]:
    return [w for w in rarest if word_idf(idx, w, N) >= gate]


def doc_toks(corpus: dict[str, str], doc_id: str) -> set[str]:
    return set(words(corpus.get(doc_id, "")))


def posting_docs(idx: AppendOnlyLatticeIndex, w: str) -> set[str]:
    p = idx.token_prime.get(("w", w))
    if p is None:
        return set()
    pl = idx.postings.get(p)
    if not pl:
        return set()
    return {d for d in pl if d in idx.alive}


def prefix_posting_docs(idx: AppendOnlyLatticeIndex, w: str) -> set[str]:
    pk = ("p", w[:4])
    p = idx.token_prime.get(pk)
    if p is None:
        return set()
    pl = idx.postings.get(p)
    if not pl:
        return set()
    return {d for d in pl if d in idx.alive}


def bm25_rank_pool(idx: AppendOnlyLatticeIndex, query: str, pool: set[str], k: int = 10) -> list[str]:
    if not pool:
        return []
    scores = idx._score(query)
    ranked = sorted(pool, key=lambda d: scores.get(d, 0.0), reverse=True)
    return [d for d in ranked if scores.get(d, 0.0) > 0][:k]


def rrf_merge(rank_a: list[str], rank_b: list[str], k: int = 10, w_a: float = 1.0, w_b: float = 1.0) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for i, d in enumerate(rank_a):
        scores[d] += w_a / (60 + i + 1)
    for i, d in enumerate(rank_b):
        scores[d] += w_b / (60 + i + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]


def idf_jaccard_score(
    idx: AppendOnlyLatticeIndex,
    query_terms: list[str],
    doc_id: str,
    N: int,
) -> float:
    dtoks = idx.doc_words.get(doc_id, set())
    if not dtoks or not query_terms:
        return 0.0
    inter = 0.0
    union = 0.0
    for w in query_terms:
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        iw = word_idf(idx, w, N)
        if p in dtoks:
            inter += iw
        union += iw
    for p in dtoks:
        union += idx._idf(p, N)
    return inter / union if union else 0.0


def glossary_expand(
    query: str,
    idx: AppendOnlyLatticeIndex,
    N: int,
    glossary: dict[str, str],
    max_extra: int = 10,
) -> str:
    extra: list[str] = []
    for t in set(words(query)):
        if t not in glossary:
            continue
        for w in dict.fromkeys(words(glossary[t])):
            if w == t:
                continue
            p = idx.token_prime.get(("w", w))
            if p is not None and idx._idf(p, N) >= 2.5:
                extra.append(w)
    if not extra:
        return query
    return query + " " + " ".join(extra[:max_extra])


@dataclass
class ProbeContext:
    idx: AppendOnlyLatticeIndex
    corpus: dict[str, str]
    br: RelevanceBridges
    br_rare: RelevanceBridges
    N: int
    glossary: dict[str, str] = field(default_factory=dict)
    glass_lattice: GlassBoxRetriever | None = None
    glass_target: GlassBoxRetriever | None = None


# ---------------------------------------------------------------------------
# 25+ glass-box probes (each returns top-k doc ids)
# ---------------------------------------------------------------------------

def probe_01_baseline(ctx: ProbeContext, query: str, k: int) -> list[str]:
    return ctx.idx.search(query, k)


def probe_02_manifold(ctx: ProbeContext, query: str, k: int) -> list[str]:
    return ctx.idx.search_manifold(query, k=k, pool=80, beta=0.2)


def probe_03_bridge_search(ctx: ProbeContext, query: str, k: int) -> list[str]:
    return bridge_search(ctx.idx, ctx.br, query, k=k)


def probe_04_best_search_pool(ctx: ProbeContext, query: str, k: int) -> list[str]:
    return best_search(ctx.idx, ctx.br, query)[:k]


def probe_05_rarest1_posting_bm25(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_01_baseline(ctx, query, k)
    pool = posting_docs(ctx.idx, rarest[0])
    return bm25_rank_pool(ctx.idx, query, pool, k) or probe_01_baseline(ctx, query, k)


def probe_06_rarest2_and_bm25(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 2:
        return probe_05_rarest1_posting_bm25(ctx, query, k)
    pool = posting_docs(ctx.idx, rarest[0]) & posting_docs(ctx.idx, rarest[1])
    return bm25_rank_pool(ctx.idx, query, pool, k) or probe_01_baseline(ctx, query, k)


def probe_07_rarest3_and_bm25(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 3:
        return probe_06_rarest2_and_bm25(ctx, query, k)
    pool = (
        posting_docs(ctx.idx, rarest[0])
        & posting_docs(ctx.idx, rarest[1])
        & posting_docs(ctx.idx, rarest[2])
    )
    return bm25_rank_pool(ctx.idx, query, pool, k) or probe_01_baseline(ctx, query, k)


def probe_08_rarest2_union_bm25(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 2:
        return probe_05_rarest1_posting_bm25(ctx, query, k)
    pool = posting_docs(ctx.idx, rarest[0]) | posting_docs(ctx.idx, rarest[1])
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_09_rare_gate_union_bm25(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=3.0)
    pool: set[str] = set()
    for w in rare:
        pool |= posting_docs(ctx.idx, w)
    if not pool:
        return probe_01_baseline(ctx, query, k)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_10_rarest_only_query(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_01_baseline(ctx, query, k)
    return ctx.idx.search(rarest[0], k)


def probe_11_rarest2_only_query(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 2:
        return probe_10_rarest_only_query(ctx, query, k)
    return ctx.idx.search(f"{rarest[0]} {rarest[1]}", k)


def probe_12_hub_penalty_rerank(ctx: ProbeContext, query: str, k: int) -> list[str]:
    qterms = words(query)
    rarest = rarest_terms(qterms, ctx.idx, ctx.N)
    hub = {w for w in qterms if word_idf(ctx.idx, w, ctx.N) < 2.0}
    hub_query = " ".join([w for w in qterms if w not in hub])
    if not hub_query.strip():
        return probe_01_baseline(ctx, query, k)
    base = ctx.idx._score(hub_query)
    rare_boost = defaultdict(float)
    for w in rare_gate_terms(rarest, ctx.idx, ctx.N, gate=3.0):
        for d in posting_docs(ctx.idx, w):
            rare_boost[d] += word_idf(ctx.idx, w, ctx.N)
    mx = max(base.values()) if base else 1.0
    rbmx = max(rare_boost.values()) if rare_boost else 1.0
    final = {
        d: base.get(d, 0.0) / mx + 0.35 * rare_boost.get(d, 0.0) / rbmx
        for d in set(base) | set(rare_boost)
    }
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def probe_13_prefix_gear_rarest(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_01_baseline(ctx, query, k)
    pool = prefix_posting_docs(ctx.idx, rarest[0]) | posting_docs(ctx.idx, rarest[0])
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_14_top1_shared_rare_expand(ctx: ProbeContext, query: str, k: int) -> list[str]:
    base = ctx.idx.search(query, 1)
    if not base:
        return probe_01_baseline(ctx, query, k)
    top1 = base[0]
    toks = doc_toks(ctx.corpus, top1)
    rare_in_top = [w for w in toks if word_idf(ctx.idx, w, ctx.N) >= 3.0]
    pool = set(base)
    for w in rare_in_top[:8]:
        pool |= posting_docs(ctx.idx, w)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_15_top3_shared_rare_expand(ctx: ProbeContext, query: str, k: int) -> list[str]:
    base = ctx.idx.search(query, 3)
    pool = set(base)
    for doc_id in base:
        toks = doc_toks(ctx.corpus, doc_id)
        for w in toks:
            if word_idf(ctx.idx, w, ctx.N) >= 3.0:
                pool |= posting_docs(ctx.idx, w)
    return bm25_rank_pool(ctx.idx, query, pool, k) if pool else probe_01_baseline(ctx, query, k)


def probe_16_bridge_rarest_qt_only(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_03_bridge_search(ctx, query, k)
    qt = rarest[0]
    lex = ctx.idx._score(query)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    exp = defaultdict(float)
    for dt, w in ctx.br.bridge.get(qt, ()):
        p = ctx.idx.token_prime.get(("w", dt))
        if p is None:
            continue
        for d, tf in ctx.idx.postings.get(p, {}).items():
            if d in ctx.idx.alive:
                exp[d] += w * tf / (tf + 1.0)
    pool = list(dict.fromkeys(cand + sorted(exp, key=lambda d: exp[d], reverse=True)[:20]))
    lmax = max(lex.values()) if lex else 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + 0.25 * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def probe_17_rarest_corridor_bridges(ctx: ProbeContext, query: str, k: int) -> list[str]:
    return bridge_search(ctx.idx, ctx.br_rare, query, k=k)


def probe_18_pair_cooccur_boost(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=2.5)[:6]
    lex = ctx.idx._score(query)
    boost = defaultdict(float)
    for a, b in itertools.combinations(rare, 2):
        inter = posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
        wt = word_idf(ctx.idx, a, ctx.N) + word_idf(ctx.idx, b, ctx.N)
        for d in inter:
            boost[d] += wt
    mx = max(lex.values()) if lex else 1.0
    bmx = max(boost.values()) if boost else 1.0
    pool = set(lex) | set(boost)
    final = {d: lex.get(d, 0.0) / mx + 0.4 * boost.get(d, 0.0) / bmx for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def probe_19_triple_cooccur_boost(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=2.5)[:5]
    if len(rare) < 3:
        return probe_18_pair_cooccur_boost(ctx, query, k)
    lex = ctx.idx._score(query)
    boost = defaultdict(float)
    for a, b, c in itertools.combinations(rare, 3):
        inter = posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b) & posting_docs(ctx.idx, c)
        wt = word_idf(ctx.idx, a, ctx.N) + word_idf(ctx.idx, b, ctx.N) + word_idf(ctx.idx, c, ctx.N)
        for d in inter:
            boost[d] += wt * 1.5
    mx = max(lex.values()) if lex else 1.0
    bmx = max(boost.values()) if boost else 1.0
    pool = set(lex) | set(boost)
    final = {d: lex.get(d, 0.0) / mx + 0.5 * boost.get(d, 0.0) / bmx for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def probe_20_jaccard_rare_weighted(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=2.5)
    if not rare:
        return probe_01_baseline(ctx, query, k)
    pool: set[str] = set()
    for w in rare:
        pool |= posting_docs(ctx.idx, w)
    pool |= set(ctx.idx.search(query, 50))
    scores = {d: idf_jaccard_score(ctx.idx, rare, d, ctx.N) for d in pool}
    return sorted(scores, key=lambda d: scores[d], reverse=True)[:k]


def probe_21_rrf_lexical_rarest_posting(ctx: ProbeContext, query: str, k: int) -> list[str]:
    lex = ctx.idx.search(query, 50)
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return lex[:k]
    pool = posting_docs(ctx.idx, rarest[0])
    rare_rank = bm25_rank_pool(ctx.idx, query, pool, k=50)
    return rrf_merge(lex, rare_rank, k=k)


def probe_22_glossary_expand_bridge(ctx: ProbeContext, query: str, k: int) -> list[str]:
    qx = glossary_expand(query, ctx.idx, ctx.N, ctx.glossary)
    return bridge_search(ctx.idx, ctx.br, qx, k=k)


def probe_31_glass_box_target(ctx: ProbeContext, query: str, k: int) -> list[str]:
    if ctx.glass_target is None:
        return probe_03_bridge_search(ctx, query, k)
    return ctx.glass_target.search(query, k=k)


def probe_32_glass_lattice_pool(ctx: ProbeContext, query: str, k: int) -> list[str]:
    if ctx.glass_lattice is None:
        return probe_31_glass_box_target(ctx, query, k)
    return ctx.glass_lattice.search(query, k=k)


def probe_23_morph_prefix5_rarest(ctx: ProbeContext, query: str, k: int) -> list[str]:
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_01_baseline(ctx, query, k)
    r = rarest[0]
    pref = r[:5] if len(r) >= 5 else r[:4]
    pool = {d for d in ctx.idx.alive if any(t.startswith(pref) for t in doc_toks(ctx.corpus, d))}
    pool |= posting_docs(ctx.idx, r)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_24_top10_cluster_rare_cross(ctx: ProbeContext, query: str, k: int) -> list[str]:
    """Cross-ref top-10: rare words shared between top hits → expand pool."""
    top10 = ctx.idx.search(query, 10)
    shared_rare: Counter[str] = Counter()
    for doc_id in top10:
        for w in doc_toks(ctx.corpus, doc_id):
            if word_idf(ctx.idx, w, ctx.N) >= 3.0:
                shared_rare[w] += 1
    # words appearing in 2+ top docs are cluster anchors
    anchors = [w for w, c in shared_rare.items() if c >= 2]
    pool = set(top10)
    for w in anchors[:6]:
        pool |= posting_docs(ctx.idx, w)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_25_false_hub_subtract(ctx: ProbeContext, query: str, k: int) -> list[str]:
    lex = ctx.idx._score(query)
    ranked = sorted(lex, key=lambda d: lex[d], reverse=True)
    top = ranked[:30]
    qterms = words(query)
    hubs = {w for w in qterms if word_idf(ctx.idx, w, ctx.N) < 2.0}
    false_like = []
    for d in top[:10]:
        dt = doc_toks(ctx.corpus, d)
        if hubs and sum(1 for h in hubs if h in dt) >= len(hubs) * 0.6:
            false_like.append(d)
    penalized = {}
    for d in top:
        pen = 0.15 if d in false_like else 0.0
        penalized[d] = lex[d] * (1.0 - pen)
    return sorted(penalized, key=lambda d: penalized[d], reverse=True)[:k]


def probe_26_compound_concat_substring(ctx: ProbeContext, query: str, k: int) -> list[str]:
    """Compound: concatenation of two rarest words appears as substring in doc token."""
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 2:
        return probe_01_baseline(ctx, query, k)
    a, b = rarest[0], rarest[1]
    compound = a + b
    pool = set()
    for d in ctx.idx.alive:
        toks = doc_toks(ctx.corpus, d)
        if any(compound in t or (a in t and b in t) for t in toks):
            pool.add(d)
    pool |= posting_docs(ctx.idx, a) | posting_docs(ctx.idx, b)
    return bm25_rank_pool(ctx.idx, query, pool, k) or probe_01_baseline(ctx, query, k)


def probe_27_lex100_union_rarest2_postings(ctx: ProbeContext, query: str, k: int) -> list[str]:
    lex = ctx.idx._score(query)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    pool = set(cand)
    for w in rarest[:2]:
        pool |= posting_docs(ctx.idx, w)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_28_sim_lift_rarest_in_gold_neighbor(ctx: ProbeContext, query: str, k: int) -> list[str]:
    """If gold-like docs share rare-2 with a high lexical doc, boost that corridor."""
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if len(rarest) < 2:
        return probe_01_baseline(ctx, query, k)
    r2 = rarest[1]
    anchor_pool = posting_docs(ctx.idx, r2)
    lex = ctx.idx._score(query)
    boost = defaultdict(float)
    for d in anchor_pool:
        dt = doc_toks(ctx.corpus, d)
        for w in rare_gate_terms(rarest, ctx.idx, ctx.N, gate=3.0):
            if w in dt:
                boost[d] += word_idf(ctx.idx, w, ctx.N)
    mx = max(lex.values()) if lex else 1.0
    bmx = max(boost.values()) if boost else 1.0
    pool = set(lex) | anchor_pool
    final = {d: lex.get(d, 0.0) / mx + 0.3 * boost.get(d, 0.0) / bmx for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)[:k]


def probe_29_subword_chunk_rarest(ctx: ProbeContext, query: str, k: int) -> list[str]:
    from aethos_promotion import _chunk_subwords

    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    if not rarest:
        return probe_01_baseline(ctx, query, k)
    pool: set[str] = set()
    for w in rarest[:3]:
        pool |= posting_docs(ctx.idx, w)
        for piece in _chunk_subwords(w):
            if len(piece) >= 3:
                pool |= posting_docs(ctx.idx, piece)
    return bm25_rank_pool(ctx.idx, query, pool, k)


def probe_30_bridge_plus_pair_meet_pool(ctx: ProbeContext, query: str, k: int) -> list[str]:
    base = best_search(ctx.idx, ctx.br, query)[:50]
    rarest = rarest_terms(words(query), ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=2.5)[:4]
    meet_pool: set[str] = set()
    for a, b in itertools.combinations(rare, 2):
        meet_pool |= posting_docs(ctx.idx, a) & posting_docs(ctx.idx, b)
    pool = list(dict.fromkeys(base + sorted(meet_pool, key=lambda d: ctx.idx._score(query).get(d, 0), reverse=True)[:30]))
    return bm25_rank_pool(ctx.idx, query, set(pool), k) or base[:k]


PROBES: list[tuple[str, str, Callable[[ProbeContext, str, int], list[str]]]] = [
    ("01_baseline_lexical", "Current idx.search (multi-view BM25)", probe_01_baseline),
    ("02_manifold_rerank", "Meet-overlap cluster boost on BM25 pool", probe_02_manifold),
    ("03_bridge_search", "Supervised bridge rerank+expand", probe_03_bridge_search),
    ("04_best_search_pool", "Lexical pool-100 + bridge expand (active learning path)", probe_04_best_search_pool),
    ("05_rarest1_posting_bm25", "BM25 within rarest-word posting list", probe_05_rarest1_posting_bm25),
    ("06_rarest2_and_bm25", "AND rarest+2nd posting, BM25 rerank", probe_06_rarest2_and_bm25),
    ("07_rarest3_and_bm25", "AND top-3 rarest postings", probe_07_rarest3_and_bm25),
    ("08_rarest2_union_bm25", "UNION rarest+2nd postings, BM25 rerank", probe_08_rarest2_union_bm25),
    ("09_rare_idf3_union_bm25", "UNION all idf≥3 query terms", probe_09_rare_gate_union_bm25),
    ("10_rarest_only_query", "Search query = rarest word only", probe_10_rarest_only_query),
    ("11_rarest2_only_query", "Search query = rarest two words", probe_11_rarest2_only_query),
    ("12_hub_penalty_rerank", "Drop hub words + rare posting boost", probe_12_hub_penalty_rerank),
    ("13_prefix_gear_rarest", "Prefix gear + word posting for rarest", probe_13_prefix_gear_rarest),
    ("14_top1_shared_rare_expand", "Expand from lexical top-1 rare words", probe_14_top1_shared_rare_expand),
    ("15_top3_shared_rare_expand", "Expand from lexical top-3 rare words", probe_15_top3_shared_rare_expand),
    ("16_bridge_rarest_qt_only", "Bridges from rarest query term only", probe_16_bridge_rarest_qt_only),
    ("17_rarest_corridor_bridges", "learn_rarest_corridors bridge set", probe_17_rarest_corridor_bridges),
    ("18_pair_cooccur_boost", "Boost docs with 2+ rare query words", probe_18_pair_cooccur_boost),
    ("19_triple_cooccur_boost", "Boost docs with 3+ rare query words", probe_19_triple_cooccur_boost),
    ("20_jaccard_rare_weighted", "Idf-weighted Jaccard on rare terms", probe_20_jaccard_rare_weighted),
    ("21_rrf_lexical_rarest", "RRF lexical + rarest posting rank", probe_21_rrf_lexical_rarest_posting),
    ("22_glossary_expand_bridge", "Glossary query expand + bridges", probe_22_glossary_expand_bridge),
    ("23_morph_prefix5", "Morph: doc token prefix of rarest", probe_23_morph_prefix5_rarest),
    ("24_top10_cluster_rare", "Cross-ref top-10 shared rare anchors", probe_24_top10_cluster_rare_cross),
    ("25_false_hub_subtract", "Penalize hub-heavy false-like top docs", probe_25_false_hub_subtract),
    ("26_compound_concat", "Compound substring / dual-token morph", probe_26_compound_concat_substring),
    ("27_lex100_rarest_union", "Lex top-100 ∪ rarest postings", probe_27_lex100_union_rarest2_postings),
    ("28_rarest2_corridor_boost", "Rarest-2 posting + rare co-occur boost", probe_28_sim_lift_rarest_in_gold_neighbor),
    ("29_subword_chunk_rarest", "Subword pieces of rarest terms", probe_29_subword_chunk_rarest),
    ("30_bridge_pair_meet_pool", "Bridge pool + pair-meet intersection expand", probe_30_bridge_plus_pair_meet_pool),
    ("31_glass_box_target", "Glass-box BM25 target + glossary + bridges", probe_31_glass_box_target),
    ("32_glass_lattice_pool", "Glass-box lattice κ-pool (no BM25)", probe_32_glass_lattice_pool),
]


# ---------------------------------------------------------------------------
# per-query glass profile (gold vs false)
# ---------------------------------------------------------------------------

def gold_false_profile(
    ctx: ProbeContext,
    query: str,
    gold_local: dict[str, int],
    ranked: list[str],
) -> dict[str, object]:
    qterms = words(query)
    rarest = rarest_terms(qterms, ctx.idx, ctx.N)
    rare = rare_gate_terms(rarest, ctx.idx, ctx.N, gate=3.0)
    top10 = set(ranked[:10])
    gold_ids = [d for d, s in gold_local.items() if s > 0]
    false_ids = [d for d in ranked[:10] if d not in gold_ids]

    def profile_doc(doc_id: str, is_gold: bool) -> dict[str, object]:
        toks = doc_toks(ctx.corpus, doc_id)
        h1 = rarest[0] in toks if rarest else False
        h2 = len(rarest) >= 2 and rarest[0] in toks and rarest[1] in toks
        pairs = []
        for a, b in itertools.combinations(rare[:6], 2):
            if a in toks and b in toks:
                pairs.append((a, b))
        bridge_paths = []
        for qt in set(qterms):
            for dt, wt in ctx.br.bridge.get(qt, ()):
                if dt in toks:
                    bridge_paths.append({"qt": qt, "dt": dt, "w": round(wt, 3)})
        bridge_paths.sort(key=lambda x: -x["w"])
        return {
            "doc_id": doc_id,
            "is_gold": is_gold,
            "rank": ranked.index(doc_id) + 1 if doc_id in ranked else None,
            "in_top10": doc_id in top10,
            "has_rarest_1": h1,
            "has_rarest_2": h2,
            "rare_pairs_in_doc": pairs[:6],
            "n_rare_pairs": len(pairs),
            "bridge_paths": bridge_paths[:8],
            "n_bridge_paths": len(bridge_paths),
        }

    gold_profiles = [profile_doc(g, True) for g in gold_ids if g in ctx.corpus]
    false_profiles = [profile_doc(f, False) for f in false_ids[:5]]

    # cross-ref: rare words in top-10 that also appear in any gold doc
    cluster_rare: Counter[str] = Counter()
    for d in ranked[:10]:
        for w in doc_toks(ctx.corpus, d):
            if word_idf(ctx.idx, w, ctx.N) >= 3.0:
                cluster_rare[w] += 1
    gold_rare_union: set[str] = set()
    for g in gold_ids:
        for w in doc_toks(ctx.corpus, g):
            if word_idf(ctx.idx, w, ctx.N) >= 3.0:
                gold_rare_union.add(w)
    cross_help = [w for w, c in cluster_rare.items() if c >= 2 and w in gold_rare_union]

    return {
        "rarest_query_words": rarest[:8],
        "rare_idf3_terms": rare[:6],
        "gold": gold_profiles,
        "false_top": false_profiles,
        "cross_ref_rare_in_top10_and_gold": cross_help[:10],
        "n_gold_in_top10": sum(1 for g in gold_ids if g in top10),
    }


# ---------------------------------------------------------------------------
# evaluate all probes
# ---------------------------------------------------------------------------

def evaluate_probes(
    ctx: ProbeContext,
    queries: dict[str, str],
    test_q: dict[str, dict[str, int]],
    test_ids: list[str],
) -> tuple[dict[str, dict], list[dict]]:
    n = len(test_ids)
    probe_stats: dict[str, dict] = {}

    for name, desc, fn in PROBES:
        nd = rc = 0.0
        gold_inst_top10 = 0
        gold_inst_total = 0
        q_hit = 0
        t0 = time.perf_counter()
        for qid in test_ids:
            rels = test_q[qid]
            ranked = fn(ctx, queries[qid], 10)
            nd += ndcg10(ranked, rels)
            rc += recall10(ranked, rels)
            golds = {d for d, s in rels.items() if s > 0}
            gold_inst_total += len(golds)
            hit = len(set(ranked[:10]) & golds)
            gold_inst_top10 += hit
            if hit:
                q_hit += 1
        elapsed = time.perf_counter() - t0
        probe_stats[name] = {
            "description": desc,
            "ndcg_at_10": round(nd / n, 4),
            "recall_at_10": round(rc / n, 4),
            "queries_gold_in_top10_pct": round(100.0 * q_hit / n, 1),
            "gold_instances_in_top10_pct": round(100.0 * gold_inst_top10 / max(gold_inst_total, 1), 1),
            "wall_s": round(elapsed, 2),
        }

    baseline_nd = probe_stats["01_baseline_lexical"]["ndcg_at_10"]
    baseline_rc = probe_stats["01_baseline_lexical"]["recall_at_10"]
    for st in probe_stats.values():
        st["ndcg_delta_vs_baseline"] = round(st["ndcg_at_10"] - baseline_nd, 4)
        st["recall_delta_vs_baseline"] = round(st["recall_at_10"] - baseline_rc, 4)

    # per-query profiles on baseline only
    profiles = []
    for qid in test_ids:
        ranked = probe_01_baseline(ctx, queries[qid], 10)
        prof = gold_false_profile(ctx, queries[qid], test_q[qid], ranked)
        prof["query_id"] = qid
        prof["query"] = queries[qid][:140]
        profiles.append(prof)

    return probe_stats, profiles


def summarize_coverage(profiles: list[dict]) -> dict[str, object]:
    n_gold = sum(len(p["gold"]) for p in profiles)
    n_false = sum(len(p["false_top"]) for p in profiles)
    agg = Counter()
    for p in profiles:
        for g in p["gold"]:
            if g["has_rarest_1"]:
                agg["gold_rarest_1"] += 1
            if g["has_rarest_2"]:
                agg["gold_rarest_2"] += 1
            if g["n_rare_pairs"]:
                agg["gold_compound_pair"] += 1
            if g["n_bridge_paths"]:
                agg["gold_bridge"] += 1
            if g["in_top10"]:
                agg["gold_top10"] += 1
        for f in p["false_top"]:
            if f["has_rarest_1"]:
                agg["false_rarest_1"] += 1
            if f["n_bridge_paths"]:
                agg["false_bridge"] += 1
        if p["cross_ref_rare_in_top10_and_gold"]:
            agg["queries_cross_ref_help"] += 1

    def pct(a, b):
        return round(100.0 * a / max(b, 1), 1)

    return {
        "gold_doc_instances": n_gold,
        "false_doc_instances": n_false,
        "gold_has_rarest_1_pct": pct(agg["gold_rarest_1"], n_gold),
        "gold_has_rarest_2_pct": pct(agg["gold_rarest_2"], n_gold),
        "gold_compound_pair_pct": pct(agg["gold_compound_pair"], n_gold),
        "gold_bridge_path_pct": pct(agg["gold_bridge"], n_gold),
        "gold_in_top10_pct": pct(agg["gold_top10"], n_gold),
        "false_has_rarest_1_pct": pct(agg["false_rarest_1"], n_false),
        "false_bridge_path_pct": pct(agg["false_bridge"], n_false),
        "queries_with_cross_ref_rare_help": agg["queries_cross_ref_help"],
    }


def print_leaderboard(probe_stats: dict[str, dict], coverage: dict) -> None:
    print(f"\n{'='*78}")
    print("  APPEND-ONLY LATTICE INDEX — GLASS-BOX PROBE LEADERBOARD (30 probes)")
    print(f"{'='*78}")
    print("\n  GOLD vs FALSE — baseline lexical profile")
    print(f"    gold has rarest query word:     {coverage['gold_has_rarest_1_pct']}%")
    print(f"    gold has rarest 1+2 words:      {coverage['gold_has_rarest_2_pct']}%")
    print(f"    gold has rare query pair (AND): {coverage['gold_compound_pair_pct']}%")
    print(f"    gold has bridge qt->dt path:     {coverage['gold_bridge_path_pct']}%")
    print(f"    gold in baseline top-10:        {coverage['gold_in_top10_pct']}%")
    print(f"    false top has rarest word:      {coverage['false_has_rarest_1_pct']}%")
    print(f"    false top has bridge path:      {coverage['false_bridge_path_pct']}%")
    print(f"    queries w/ cross-ref rare help: {coverage['queries_with_cross_ref_rare_help']}")

    rows = sorted(probe_stats.items(), key=lambda x: (-x[1]["ndcg_at_10"], -x[1]["recall_at_10"]))
    print(f"\n  {'probe':<28} {'nDCG':>7} {'dNDCG':>7} {'R@10':>7} {'dR@10':>7} {'gold%':>6}")
    print("  " + "-" * 72)
    for name, st in rows[:15]:
        print(
            f"  {name:<28} {st['ndcg_at_10']:>7.4f} {st['ndcg_delta_vs_baseline']:>+7.4f} "
            f"{st['recall_at_10']:>7.4f} {st['recall_delta_vs_baseline']:>+7.4f} "
            f"{st['gold_instances_in_top10_pct']:>5.1f}%"
        )
    print(f"\n  ... {len(rows)} probes total. Full table in JSON output.")


def write_rules_md(probe_stats: dict, coverage: dict, path: Path) -> None:
    top = sorted(probe_stats.items(), key=lambda x: -x[1]["ndcg_at_10"])[:8]
    lines = [
        "# AppendOnlyLatticeIndex glass-box probes\n",
        "\n## Coverage (baseline lexical)\n",
        f"- Gold has rarest query word: **{coverage['gold_has_rarest_1_pct']}%**\n",
        f"- Gold has rarest 1+2: **{coverage['gold_has_rarest_2_pct']}%**\n",
        f"- Gold rare pair co-occur: **{coverage['gold_compound_pair_pct']}%**\n",
        f"- Gold bridge path: **{coverage['gold_bridge_path_pct']}%**\n",
        f"- False top rarest word: **{coverage['false_has_rarest_1_pct']}%**\n",
        "\n## Top probes by nDCG@10\n",
        "| probe | nDCG | Δ vs baseline | R@10 | note |\n",
        "|-------|------|---------------|------|------|\n",
    ]
    base_nd = probe_stats["01_baseline_lexical"]["ndcg_at_10"]
    for name, st in top:
        lines.append(
            f"| {name} | {st['ndcg_at_10']:.4f} | {st['ndcg_delta_vs_baseline']:+.4f} | "
            f"{st['recall_at_10']:.4f} | {st['description'][:50]} |\n"
        )
    lines.append(
        "\n## Glass-box rules\n"
        "1. **IF** gold lacks rarest word → bridges/glossary are the recovery path.\n"
        "2. **IF** gold has rare pair co-occur → pair-meet boost probes help R@10.\n"
        "3. **IF** false docs share rarest word → hub penalty / false-subtract probes.\n"
        "4. **IF** top-10 cluster shares rare w/ gold → cross-ref expand (probe 24).\n"
        "5. **Never** AND-narrow to 3 rarest words alone — too aggressive for pool.\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Glass-box audit: AppendOnlyLatticeIndex")
    p.add_argument("dataset", nargs="?", default="scifact")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--index-mode", default="full", choices=("full", "kappa_primary"))
    p.add_argument("--out", default="logs/append_glass_box.json")
    p.add_argument("--rules-md", default="logs/append_glass_box_rules.md")
    p.add_argument("--min-pairs", type=int, default=0, help="0 = auto (scifact=1, else=2)")
    p.add_argument("--no-glass-box", action="store_true", help="Skip probes 31-32")
    args = p.parse_args()

    corpus, queries, train_q, test_q = load(args.dataset)
    if not train_q:
        print(f"  WARNING: {args.dataset} has no train qrels — bridge probes will be weak", flush=True)
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    min_pairs = args.min_pairs or (1 if args.dataset == "scifact" else 2)
    gloss = load_glossary(args.dataset)
    if not gloss and args.dataset == "scifact":
        gloss = dict(_SCIFACT_GLOSSARY_FALLBACK)
    if not gloss:
        print(f"  (no glossary module for {args.dataset} — glossary probes use empty dict)", flush=True)

    print(f"Building AppendOnlyLatticeIndex ({args.dataset}, mode={args.index_mode}) ...", flush=True)
    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex(index_mode=args.index_mode)
    for d, txt in corpus.items():
        idx.add(d, txt)
    idx.finalize()
    N = len(idx.alive)
    print(f"  {N} docs, {len(idx.token_prime)} vocab, finalize {time.perf_counter()-t0:.1f}s", flush=True)

    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    br_rare = RelevanceBridges(idx, N, min_pairs=min_pairs)
    br_rare.learn_rarest_corridors(queries, train_q, corpus)

    glass_target = None
    glass_lattice = None
    if train_q and not args.no_glass_box:
        from aethos_lattice_lexical import lattice_lexical_scorer

        glass_target = GlassBoxRetriever(
            idx=idx, bridges=br, glossary=gloss,
            config=GlassBoxSearchConfig.scifact_target(),
            corpus=corpus,
        )
        lat_cfg = GlassBoxSearchConfig.scifact_lattice()
        kappa_index = None
        registry = None
        if args.index_mode == "kappa_primary":
            from aethos_promotion import PromotionRegistry
            from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast

            registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
            for text in corpus.values():
                registry.observe_text(text)
            kappa_index = build_attractor_index_fast(
                registry, corpus, lambda w: word_idf(idx, w, N),
            )
        glass_lattice = GlassBoxRetriever(
            idx=idx, bridges=br, glossary=gloss,
            config=lat_cfg,
            lexical_scorer=lattice_lexical_scorer(idx, mode="lattice_pure", pair_w=0.0),
            corpus=corpus,
            kappa_index=kappa_index,
            registry=registry,
        )

    ctx = ProbeContext(
        idx=idx, corpus=corpus, br=br, br_rare=br_rare, N=N,
        glossary=gloss,
        glass_target=glass_target,
        glass_lattice=glass_lattice,
    )

    print(f"Running {len(PROBES)} probes on {len(test_ids)} queries ...", flush=True)
    probe_stats, profiles = evaluate_probes(ctx, queries, test_q, test_ids)
    coverage = summarize_coverage(profiles)

    payload = {
        "dataset": args.dataset,
        "index_mode": args.index_mode,
        "n_queries": len(test_ids),
        "coverage": coverage,
        "probes": probe_stats,
        "query_profiles": profiles,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rules = Path(args.rules_md)
    write_rules_md(probe_stats, coverage, rules)

    print_leaderboard(probe_stats, coverage)
    print(f"\n  JSON: {out}")
    print(f"  Rules: {rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
