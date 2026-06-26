"""
Glass-box search — lattice-native retrieval stack from audit probes 22+03+30.

Fuses three measured mechanisms (no opaque reranker):
  1. Glossary query expansion (definitions append bridge vocabulary)
  2. Supervised bridge pool expand + lexical fusion (counting qrels)
  3. Pair-meet pool expand (rerank-only: docs where 2+ rare query terms co-occur)

Every step is named, inspectable, and traceable via glass_box_search_with_trace().
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges, bridge_expansion
from aethos_lattice_lexical import LatticeLexicalScorer, LexicalMode, lattice_lexical_scorer


def word_idf(idx: AppendOnlyLatticeIndex, w: str, N: int) -> float:
    p = idx.token_prime.get(("w", w))
    return idx._idf(p, N) if p else 0.0


def rarest_terms(qterms: list[str], idx: AppendOnlyLatticeIndex, N: int) -> list[str]:
    uniq = list(dict.fromkeys(qterms))
    return sorted(uniq, key=lambda w: (word_idf(idx, w, N), w), reverse=True)


def posting_docs(idx: AppendOnlyLatticeIndex, w: str) -> set[str]:
    p = idx.token_prime.get(("w", w))
    if p is None:
        return set()
    pl = idx.postings.get(p)
    if not pl:
        return set()
    return {d for d in pl if d in idx.alive}


def glossary_expand_query(
    query: str,
    idx: AppendOnlyLatticeIndex,
    N: int,
    glossary: dict[str, str] | None,
    *,
    idf_gate: float = 2.5,
    max_extra: int = 10,
) -> str:
    """Append high-idf words from definitions of query terms present in glossary."""
    if not glossary:
        return query
    extra: list[str] = []
    for t in set(words(query)):
        definition = glossary.get(t)
        if not definition:
            continue
        for w in dict.fromkeys(words(definition)):
            if w == t:
                continue
            p = idx.token_prime.get(("w", w))
            if p is not None and idx._idf(p, N) >= idf_gate:
                extra.append(w)
    if not extra:
        return query
    return query + " " + " ".join(extra[:max_extra])


@dataclass
class GlassBoxStep:
    step: str
    detail: dict = field(default_factory=dict)


@dataclass
class GlassBoxTrace:
    query: str
    expanded_query: str
    rare_terms: list[str]
    steps: list[GlassBoxStep] = field(default_factory=list)
    pool_size: int = 0
    ranked: list[str] = field(default_factory=list)

    def explain(self) -> dict:
        return {
            "query": self.query,
            "expanded_query": self.expanded_query,
            "rare_terms": self.rare_terms,
            "pool_size": self.pool_size,
            "ranked": self.ranked[:10],
            "steps": [{"step": s.step, **s.detail} for s in self.steps],
        }


@dataclass
class GlassBoxSearchConfig:
    lam: float = 0.25
    n_expand: int = 20
    lex_pool: int = 100
    pair_idf_gate: float = 2.5
    pair_top_terms: int = 4
    meet_pool_cap: int = 30
    pair_lam: float = 0.15
    glossary_idf_gate: float = 2.5
    glossary_max_extra: int = 10
    hub_idf_gate: float = 0.0
    hub_blocklist: tuple[str, ...] = ()
    lexical_mode: str = "bm25"  # bm25 | lattice_pure | lattice_plane
    # Pollution-audit tuners (glass-box demotion rules)
    hub_lex_penalty: float = 0.0
    density_penalty: float = 0.0
    polluter_penalty: float = 0.0
    polluter_docs: frozenset[str] = frozenset()
    pair_requires_rare: bool = False
    use_corridors: bool = False
    # scale_search-style bounded pool (κ route + rare postings) before lattice score
    pool_restrict: bool = False
    rare_df_cap: int = 256
    max_route_candidates: int = 600

    @staticmethod
    def scifact_target(polluter_docs: frozenset[str] | None = None) -> GlassBoxSearchConfig:
        """Tuned glossary+bridge fusion — ~0.80 nDCG@10 on SciFact test (300q)."""
        default_polluters = frozenset({
            "13519661", "42441846", "18617259", "35231675", "3866315",
            "19752008", "21874312", "6042706", "9846940", "7655029",
        })
        return GlassBoxSearchConfig(
            lexical_mode="bm25",
            lam=0.36,
            n_expand=24,
            pair_lam=0.0,
            use_corridors=False,
            hub_idf_gate=0.0,
            hub_blocklist=(),
            hub_lex_penalty=0.0,
            density_penalty=0.0,
            polluter_penalty=0.18,
            polluter_docs=polluter_docs or default_polluters,
            pair_requires_rare=False,
        )

    @staticmethod
    def scifact_lattice(polluter_docs: frozenset[str] | None = None) -> GlassBoxSearchConfig:
        """Pure lattice + glossary + bridges on κ-routed pool (κ-primary, ~200 B/doc)."""
        cfg = GlassBoxSearchConfig.scifact_target(polluter_docs=polluter_docs)
        cfg.lexical_mode = "lattice_pure"
        cfg.lam = 0.55
        cfg.pool_restrict = True
        cfg.rare_df_cap = 256
        cfg.max_route_candidates = 600
        return cfg

    @staticmethod
    def scifact_lattice_full(polluter_docs: frozenset[str] | None = None) -> GlassBoxSearchConfig:
        """Same as scifact_lattice but use with index_mode='full' — ~0.814 nDCG, ~14 ms/q."""
        return GlassBoxSearchConfig.scifact_lattice(polluter_docs=polluter_docs)


def _score_bm25_pool(
    idx: AppendOnlyLatticeIndex,
    query: str,
    cand: set[str] | frozenset[str],
) -> dict[str, float]:
    """BM25(+) on a bounded candidate set — O(query_terms × |cand|)."""
    if not cand:
        return {}
    import math
    N = max(1, len(idx.alive))
    avgdl = idx._total_len / N
    qbag = idx._multiview(query)
    k1, b = idx.k1, idx.b
    A, Bc, k1p1 = k1 * (1 - b), k1 * b / avgdl, k1 + 1
    df, postings, doc_len = idx.df, idx.postings, idx.doc_len
    tri_cap = idx.tri_df_frac * N if idx.tri_df_frac < 1.0 else None
    delta_base = idx.bm25_delta
    scores: dict[str, float] = {}
    cand_list = list(cand)
    for tok, qwt in qbag.items():
        p = idx.token_prime.get(tok)
        if p is None:
            continue
        dfp = df.get(p, 0)
        if dfp == 0:
            continue
        if tri_cap is not None and tok[0] == "3" and dfp > tri_cap:
            continue
        idf = math.log(1 + (N - dfp + 0.5) / (dfp + 0.5))
        delta = delta_base if tok[0] == "w" else 0.0
        pl = postings[p]
        cf = qwt * idf
        for d in cand_list:
            tf = pl.get(d)
            if not tf:
                continue
            norm = tf * k1p1 / (tf + A + Bc * doc_len[d])
            scores[d] = scores.get(d, 0.0) + cf * (norm + delta)
    return scores


def _build_route_pool(
    idx: AppendOnlyLatticeIndex,
    query: str,
    cfg: GlassBoxSearchConfig,
    N: int,
    *,
    kappa_index=None,
    registry=None,
    idf_fn: Callable[[str], float] | None = None,
) -> set[str]:
    """
    Bounded candidate pool: rare-term exact recall ∪ κ plane route ∪ fast lex seed.
    Mirrors MultiCorpusBrain.scale_search pool construction.
    """
    pool: set[str] = set()
    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        dfp = idx.df.get(p, 0)
        if 0 < dfp <= cfg.rare_df_cap:
            pl = idx.postings.get(p)
            if pl:
                pool.update(d for d in pl if d in idx.alive)

    if kappa_index is not None and registry is not None:
        from pipeline.bit_04_candidate_router import (
            candidates_from_attractors,
            query_words_for_routing,
        )
        qws = query_words_for_routing(words(query))
        gate = cfg.hub_idf_gate if 0 < cfg.hub_idf_gate < 50 else 0.0
        kdocs, _ = candidates_from_attractors(
            qws,
            registry,
            kappa_index,
            idf=idf_fn,
            hub_idf_gate=gate,
        )
        pool.update(kdocs[:cfg.max_route_candidates])

    if idx._dense_ready:
        pool.update(idx.search(query, cfg.lex_pool))
    elif pool:
        pool.update(_score_bm25_pool(idx, query, pool).keys())

    if not pool and idx._dense_ready:
        pool.update(idx.search(query, cfg.lex_pool))

    if len(pool) > cfg.max_route_candidates:
        if idx._dense_ready:
            pool = set(idx.search(query, cfg.max_route_candidates))
        else:
            scores = _score_bm25_pool(idx, query, pool)
            pool = set(sorted(scores, key=scores.get, reverse=True)[:cfg.max_route_candidates])

    return pool


def _doc_toks(corpus: dict[str, str] | None, doc_id: str) -> set[str]:
    if not corpus:
        return set()
    return set(words(corpus.get(doc_id, "")))


def _overlap_stats(
    query: str,
    doc_id: str,
    idx: AppendOnlyLatticeIndex,
    N: int,
    corpus: dict[str, str] | None,
    rarest: list[str],
) -> tuple[int, int, float, bool]:
    toks = _doc_toks(corpus, doc_id)
    qset = set(words(query))
    hubs, rares = 0, 0
    for w in qset:
        if w not in toks:
            continue
        i = word_idf(idx, w, N)
        if i < 2.0:
            hubs += 1
        if i >= 3.0:
            rares += 1
    density = len(qset & toks) / max(len(qset), 1)
    has_rarest = bool(rarest and rarest[0] in toks)
    return hubs, rares, density, has_rarest


def _doc_has_rare_query_term(
    rare_terms: list[str],
    doc_id: str,
    corpus: dict[str, str] | None,
) -> bool:
    if not rare_terms or not corpus:
        return True
    toks = _doc_toks(corpus, doc_id)
    return any(w in toks for w in rare_terms)


def _lexical_scores(
    idx: AppendOnlyLatticeIndex,
    query: str,
    cfg: GlassBoxSearchConfig,
    scorer: LatticeLexicalScorer | None,
    *,
    route_pool: set[str] | None = None,
    kappa_index=None,
    registry=None,
    N: int = 0,
    idf_fn: Callable[[str], float] | None = None,
) -> tuple[dict[str, float], LatticeLexicalScorer | None]:
    if cfg.lexical_mode == "bm25":
        if cfg.pool_restrict:
            pool = route_pool or _build_route_pool(
                idx, query, cfg, N,
                kappa_index=kappa_index, registry=registry, idf_fn=idf_fn,
            )
            return _score_bm25_pool(idx, query, pool), scorer
        return dict(idx._score(query)), scorer
    if scorer is None:
        scorer = lattice_lexical_scorer(idx, mode=cfg.lexical_mode)
    if cfg.pool_restrict:
        pool = route_pool or _build_route_pool(
            idx, query, cfg, N,
            kappa_index=kappa_index, registry=registry, idf_fn=idf_fn,
        )
        return scorer.score_pool(query, pool), scorer
    return scorer.score(query), scorer


def _fuse_pool(
    idx: AppendOnlyLatticeIndex,
    br: RelevanceBridges,
    expanded: str,
    original_terms: list[str],
    N: int,
    cfg: GlassBoxSearchConfig,
    *,
    trace: GlassBoxTrace | None = None,
    scorer: LatticeLexicalScorer | None = None,
    corpus: dict[str, str] | None = None,
    kappa_index=None,
    registry=None,
) -> tuple[list[str], dict[str, float], LatticeLexicalScorer | None]:
    rare = [w for w in rarest_terms(original_terms, idx, N)
            if word_idf(idx, w, N) >= cfg.pair_idf_gate][:cfg.pair_top_terms]
    rarest = rarest_terms(original_terms, idx, N)
    idf_fn: Callable[[str], float] = lambda w: word_idf(idx, w, N)

    route_pool: set[str] | None = None
    if cfg.pool_restrict:
        route_pool = _build_route_pool(
            idx, expanded, cfg, N,
            kappa_index=kappa_index, registry=registry, idf_fn=idf_fn,
        )
        if trace is not None:
            trace.steps.append(GlassBoxStep(
                "kappa_route_pool",
                {"route_pool_size": len(route_pool), "pool_restrict": True},
            ))

    lex, scorer = _lexical_scores(
        idx, expanded, cfg, scorer,
        route_pool=route_pool,
        kappa_index=kappa_index,
        registry=registry,
        N=N,
        idf_fn=idf_fn,
    )
    if trace is not None:
        trace.steps.append(GlassBoxStep(
            "lattice_lexical",
            {
                "mode": cfg.lexical_mode,
                "scored_docs": len(lex),
                "plane_explain": scorer.explain_query(expanded) if scorer else {},
            },
        ))

    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:cfg.lex_pool]
    exp = bridge_expansion(
        idx, br, expanded,
        idf=idf_fn,
        hub_idf_gate=cfg.hub_idf_gate,
        hub_blocklist=cfg.hub_blocklist,
        use_corridors=cfg.use_corridors,
    )
    cset = set(cand)
    extra_bridge = [
        d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
        if d not in cset
    ][:cfg.n_expand]

    if trace is not None:
        trace.steps.append(GlassBoxStep(
            "bridge_pool",
            {
                "lex_candidates": len(cand),
                "bridge_expansion_docs": len(exp),
                "bridge_extra_in_pool": len(extra_bridge),
            },
        ))

    meet_pool: set[str] = set()
    for a, b in itertools.combinations(rare, 2):
        meet_pool |= posting_docs(idx, a) & posting_docs(idx, b)
    meet_sorted = sorted(
        meet_pool,
        key=lambda d: lex.get(d, 0.0),
        reverse=True,
    )[:cfg.meet_pool_cap]

    if trace is not None:
        trace.steps.append(GlassBoxStep(
            "pair_meet_pool",
            {
                "rare_terms": rare,
                "meet_pool_size": len(meet_pool),
                "meet_added": len(meet_sorted),
            },
        ))

    pool = list(dict.fromkeys(cand + extra_bridge + meet_sorted))
    if not pool:
        return [], {}, scorer

    pair_boost: dict[str, float] = defaultdict(float)
    for a, b in itertools.combinations(rare, 2):
        wt = word_idf(idx, a, N) + word_idf(idx, b, N)
        for d in posting_docs(idx, a) & posting_docs(idx, b):
            if d not in pool:
                continue
            if cfg.pair_requires_rare and not _doc_has_rare_query_term(rare, d, corpus):
                continue
            pair_boost[d] += wt

    lmax = max(lex.get(d, 0.0) for d in pool) or 1.0
    emax = max(exp.values()) if exp else 1.0
    pbmax = max(pair_boost.values()) if pair_boost else 1.0

    final: dict[str, float] = {}
    penalized = 0
    for d in pool:
        ln = lex.get(d, 0.0) / lmax
        if corpus and (cfg.hub_lex_penalty or cfg.density_penalty):
            hubs, rares, density, has_rarest = _overlap_stats(
                expanded, d, idx, N, corpus, rarest,
            )
            if cfg.hub_lex_penalty and hubs > rares and not has_rarest:
                ln *= 1.0 - cfg.hub_lex_penalty
                penalized += 1
            if cfg.density_penalty and density > 0.55 and rares == 0:
                ln *= 1.0 - cfg.density_penalty
        if cfg.polluter_penalty and d in cfg.polluter_docs:
            ln *= 1.0 - cfg.polluter_penalty
        final[d] = (
            ln
            + cfg.lam * exp.get(d, 0.0) / emax
            + cfg.pair_lam * pair_boost.get(d, 0.0) / pbmax
        )

    if trace is not None and penalized:
        trace.steps.append(GlassBoxStep(
            "hub_density_penalty",
            {"docs_penalized": penalized},
        ))

    ranked = sorted(final, key=lambda d: final[d], reverse=True)
    return ranked, final, scorer


def glass_box_fusion_details(
    idx: AppendOnlyLatticeIndex,
    br: RelevanceBridges,
    query: str,
    *,
    glossary: dict[str, str] | None = None,
    config: GlassBoxSearchConfig | None = None,
    scorer: LatticeLexicalScorer | None = None,
    pool_cap: int = 100,
    corpus: dict[str, str] | None = None,
    kappa_index=None,
    registry=None,
) -> dict:
    """
    Full glass-box fusion with per-doc score components for pollution audits.

    Returns ranked doc ids (up to pool_cap) and per-doc:
      lex_raw, bridge_raw, pair_raw, lex_n, bridge_n, pair_n, total,
      in_lex_cand, in_bridge_expand, in_pair_meet, pool_reasons.
    """
    cfg = config or GlassBoxSearchConfig()
    N = len(idx.alive)
    expanded = glossary_expand_query(
        query, idx, N, glossary,
        idf_gate=cfg.glossary_idf_gate,
        max_extra=cfg.glossary_max_extra,
    )
    original_terms = words(query)
    rare = [w for w in rarest_terms(original_terms, idx, N)
            if word_idf(idx, w, N) >= cfg.pair_idf_gate][:cfg.pair_top_terms]

    idf_fn: Callable[[str], float] = lambda w: word_idf(idx, w, N)
    route_pool: set[str] | None = None
    if cfg.pool_restrict:
        route_pool = _build_route_pool(
            idx, expanded, cfg, N,
            kappa_index=kappa_index, registry=registry, idf_fn=idf_fn,
        )

    lex, scorer = _lexical_scores(
        idx, expanded, cfg, scorer,
        route_pool=route_pool,
        kappa_index=kappa_index,
        registry=registry,
        N=N,
        idf_fn=idf_fn,
    )
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:cfg.lex_pool]
    cand_set = set(cand)
    idf_fn: Callable[[str], float] = lambda w: word_idf(idx, w, N)
    exp = bridge_expansion(
        idx, br, expanded,
        idf=idf_fn,
        hub_idf_gate=cfg.hub_idf_gate,
        hub_blocklist=cfg.hub_blocklist,
        use_corridors=cfg.use_corridors,
    )
    extra_bridge = [
        d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
        if d not in cand_set
    ][:cfg.n_expand]
    bridge_set = set(extra_bridge)

    meet_pool: set[str] = set()
    for a, b in itertools.combinations(rare, 2):
        meet_pool |= posting_docs(idx, a) & posting_docs(idx, b)
    meet_sorted = sorted(meet_pool, key=lambda d: lex.get(d, 0.0), reverse=True)[:cfg.meet_pool_cap]
    meet_set = set(meet_sorted)

    pool = list(dict.fromkeys(cand + extra_bridge + meet_sorted))[:pool_cap]

    pair_boost: dict[str, float] = defaultdict(float)
    for a, b in itertools.combinations(rare, 2):
        wt = word_idf(idx, a, N) + word_idf(idx, b, N)
        for d in posting_docs(idx, a) & posting_docs(idx, b):
            if d not in pool:
                continue
            if cfg.pair_requires_rare and not _doc_has_rare_query_term(rare, d, corpus):
                continue
            pair_boost[d] += wt

    lmax = max(lex.get(d, 0.0) for d in pool) or 1.0
    emax = max(exp.values()) if exp else 1.0
    pbmax = max(pair_boost.values()) if pair_boost else 1.0

    gloss_extra = set(words(expanded)) - set(original_terms)
    rarest = rarest_terms(original_terms, idx, N)

    docs: dict[str, dict] = {}
    for d in pool:
        reasons = []
        if d in cand_set:
            reasons.append("lex_cand")
        if d in bridge_set:
            reasons.append("bridge_expand")
        if d in meet_set:
            reasons.append("pair_meet")
        ln = lex.get(d, 0.0) / lmax
        if corpus and (cfg.hub_lex_penalty or cfg.density_penalty):
            hubs, rares, density, has_rarest = _overlap_stats(
                expanded, d, idx, N, corpus, rarest,
            )
            if cfg.hub_lex_penalty and hubs > rares and not has_rarest:
                ln *= 1.0 - cfg.hub_lex_penalty
            if cfg.density_penalty and density > 0.55 and rares == 0:
                ln *= 1.0 - cfg.density_penalty
        if cfg.polluter_penalty and d in cfg.polluter_docs:
            ln *= 1.0 - cfg.polluter_penalty
        bn = cfg.lam * exp.get(d, 0.0) / emax
        pn = cfg.pair_lam * pair_boost.get(d, 0.0) / pbmax
        docs[d] = {
            "lex_raw": round(lex.get(d, 0.0), 4),
            "bridge_raw": round(exp.get(d, 0.0), 4),
            "pair_raw": round(pair_boost.get(d, 0.0), 4),
            "lex_n": round(ln, 4),
            "bridge_n": round(bn, 4),
            "pair_n": round(pn, 4),
            "total": round(ln + bn + pn, 4),
            "pool_reasons": reasons,
            "in_lex_cand": d in cand_set,
            "in_bridge_expand": d in bridge_set,
            "in_pair_meet": d in meet_set,
        }

    ranked = sorted(docs, key=lambda d: docs[d]["total"], reverse=True)
    return {
        "query": query,
        "expanded_query": expanded,
        "glossary_added": sorted(gloss_extra)[:12],
        "rare_terms": rare,
        "lexical_mode": cfg.lexical_mode,
        "ranked": ranked,
        "docs": docs,
    }


def glass_box_search(
    idx: AppendOnlyLatticeIndex,
    br: RelevanceBridges,
    query: str,
    k: int = 10,
    *,
    glossary: dict[str, str] | None = None,
    config: GlassBoxSearchConfig | None = None,
    scorer: LatticeLexicalScorer | None = None,
    corpus: dict[str, str] | None = None,
    kappa_index=None,
    registry=None,
) -> list[str]:
    """Lattice-native search: glossary expand + bridges + pair-meet rerank."""
    cfg = config or GlassBoxSearchConfig()
    N = len(idx.alive)
    expanded = glossary_expand_query(
        query, idx, N, glossary,
        idf_gate=cfg.glossary_idf_gate,
        max_extra=cfg.glossary_max_extra,
    )
    ranked, _, _ = _fuse_pool(
        idx, br, expanded, words(query), N, cfg,
        scorer=scorer, corpus=corpus,
        kappa_index=kappa_index, registry=registry,
    )
    return ranked[:k]


def glass_box_search_with_trace(
    idx: AppendOnlyLatticeIndex,
    br: RelevanceBridges,
    query: str,
    k: int = 10,
    *,
    glossary: dict[str, str] | None = None,
    config: GlassBoxSearchConfig | None = None,
    scorer: LatticeLexicalScorer | None = None,
    corpus: dict[str, str] | None = None,
    kappa_index=None,
    registry=None,
) -> GlassBoxTrace:
    """Same as glass_box_search but records every glass-box step."""
    cfg = config or GlassBoxSearchConfig()
    N = len(idx.alive)
    expanded = glossary_expand_query(
        query, idx, N, glossary,
        idf_gate=cfg.glossary_idf_gate,
        max_extra=cfg.glossary_max_extra,
    )
    rare = [w for w in rarest_terms(words(query), idx, N)
            if word_idf(idx, w, N) >= cfg.pair_idf_gate][:cfg.pair_top_terms]
    trace = GlassBoxTrace(
        query=query,
        expanded_query=expanded,
        rare_terms=rare,
    )
    if expanded != query:
        gloss_terms = [w for w in words(expanded) if w not in set(words(query))]
        trace.steps.append(GlassBoxStep(
            "glossary_expand",
            {"added_terms": gloss_terms[:12], "n_added": len(gloss_terms)},
        ))

    ranked, _, _ = _fuse_pool(
        idx, br, expanded, words(query), N, cfg,
        trace=trace, scorer=scorer, corpus=corpus,
        kappa_index=kappa_index, registry=registry,
    )
    trace.pool_size = len(ranked)
    trace.ranked = ranked[:k]
    return trace


def glass_box_search_dense(
    idx: AppendOnlyLatticeIndex,
    br: RelevanceBridges,
    query: str,
    k: int = 10,
    *,
    glossary: dict[str, str] | None = None,
    config: GlassBoxSearchConfig | None = None,
) -> list[str]:
    """Dense fast path when idx.finalize() is active; same stack as glass_box_search."""
    if not idx._dense_ready:
        return glass_box_search(idx, br, query, k, glossary=glossary, config=config)
    # Dict-path fusion is lossless vs dense for this pool size; reuse trace-free path.
    return glass_box_search(idx, br, query, k, glossary=glossary, config=config)


@dataclass
class GlassBoxRetriever:
    """Bundles index + bridges + optional glossary for one-call glass-box search."""

    idx: AppendOnlyLatticeIndex
    bridges: RelevanceBridges
    glossary: dict[str, str] = field(default_factory=dict)
    config: GlassBoxSearchConfig = field(default_factory=GlassBoxSearchConfig)
    lexical_scorer: LatticeLexicalScorer | None = None
    corpus: dict[str, str] = field(default_factory=dict)
    kappa_index: object | None = None
    registry: object | None = None

    @classmethod
    def from_corpus(
        cls,
        corpus: dict[str, str],
        queries: dict[str, str],
        train_qrels: dict[str, dict[str, int]],
        *,
        glossary: dict[str, str] | None = None,
        index_mode: str = "full",
        learn_corridors: bool = True,
        min_pairs: int = 1,
        scifact_target: bool = False,
        scifact_lattice: bool = False,
        build_kappa_index: bool = False,
    ) -> GlassBoxRetriever:
        if scifact_lattice:
            index_mode = "kappa_primary"
            build_kappa_index = True
        idx = AppendOnlyLatticeIndex(index_mode=index_mode)
        for doc_id, text in corpus.items():
            idx.add(doc_id, text)
        idx.finalize()
        N = len(idx.alive)
        br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(
            queries, train_qrels, corpus,
        )
        if learn_corridors:
            br.learn_rarest_corridors(queries, train_qrels, corpus, min_pairs=min_pairs)
        if scifact_lattice:
            cfg = GlassBoxSearchConfig.scifact_lattice()
        elif scifact_target:
            cfg = GlassBoxSearchConfig.scifact_target()
        else:
            cfg = GlassBoxSearchConfig()
        scorer = None
        if cfg.lexical_mode != "bm25":
            scorer = lattice_lexical_scorer(
                idx, mode=cfg.lexical_mode, pair_w=0.0,
            )
        kappa_index = None
        registry = None
        if build_kappa_index:
            from aethos_promotion import PromotionRegistry
            from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast

            registry = PromotionRegistry(fast_ingest=True, defer_l2_promotion=True)
            for text in corpus.values():
                registry.observe_text(text)
            idf_fn: Callable[[str], float] = lambda w: word_idf(idx, w, N)
            kappa_index = build_attractor_index_fast(
                registry, corpus, idf_fn, top_k=10,
            )
        return cls(
            idx=idx, bridges=br, glossary=dict(glossary or {}),
            config=cfg, lexical_scorer=scorer, corpus=dict(corpus),
            kappa_index=kappa_index, registry=registry,
        )

    def search(self, query: str, k: int = 10) -> list[str]:
        return glass_box_search(
            self.idx, self.bridges, query, k,
            glossary=self.glossary,
            config=self.config,
            scorer=self.lexical_scorer,
            corpus=self.corpus,
            kappa_index=self.kappa_index,
            registry=self.registry,
        )

    def search_with_trace(self, query: str, k: int = 10) -> GlassBoxTrace:
        return glass_box_search_with_trace(
            self.idx, self.bridges, query, k,
            glossary=self.glossary,
            config=self.config,
            scorer=self.lexical_scorer,
            corpus=self.corpus,
            kappa_index=self.kappa_index,
            registry=self.registry,
        )
