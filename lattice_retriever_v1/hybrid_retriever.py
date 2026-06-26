"""
Unified dual-lattice zero-shot retriever — three layers on one hotel.

Phase A (pool):
  Stage 08 ``LatticeRetriever.route_pool`` — corridor pins, compound-first,
  lift-pin widen, FTA letter fallback.
  Optional pair-meet expand — docs where 2+ rare query terms co-occur (append index).
  Optional walk-pair expand — docs from oriented bigram-origin intersect (walker index).

Phase B (score on bounded pool only):
  Layer 0 — ``LatticeLexicalScorer`` multiview append index (the ~0.7 floor).
  Layer 1 — ``Lattice2CorrelationPass`` shell MaxSim on L4–L6 wing cages.
  Layer 2 — ``WalkerMaxSimPass.walk_maxsim`` geometric dot witness on pair walks.

Fusion:
  norm:     λ_lex × norm(lex) + λ_l2 × norm(l2) + λ_walk × norm(walk)
  additive: λ_lex × lex + λ_l2 × l2 + λ_walk × walk  (anchor-dominant — default)

Optional Stage 08 blind rerank (lexical_bridge + cage_anchor) on preliminary rank.

``UnifiedDualLatticeRetriever`` is an alias for ``HybridZeroShotRetriever`` with all
three layers enabled via ``build_unified_dual_lattice_retriever``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Literal

from aethos_append_index import AppendOnlyLatticeIndex
from aethos_complex_plane import missing_member
from aethos_glass_box_search import posting_docs, rarest_terms, word_idf
from aethos_lattice_lexical import lattice_lexical_scorer
from aethos_promotion import PromotionRegistry
from aethos_teach_store import TeachStore

from lattice_retriever_v1.corpus_lattice import CorpusLattice, CorpusLatticeBuilder
from lattice_retriever_v1.corpus_prime import corpus_scope
from lattice_retriever_v1.doc_lattice_codec import (
    DocPrimePool,
    build_doc_correlation_shells,
    build_rare_correlation_shells,
    encode_doc,
)
from lattice_retriever_v1.rare_shell_lattice import RareShellLatticeIndex
from lattice_retriever_v1.glass_box_demote import apply_lexical_demotion, scifact_polluter_docs
from lattice_retriever_v1.lattice2_correlation import Lattice2CorrelationPass
from lattice_retriever_v1.pool_rerank import (
    cross_hit_mutual_rerank,
    gravity_cascade_rerank,
    missing_neighbor_rerank,
)
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever, RetrieveHit, RouteMode
from lattice_retriever_v1.walker_maxsim_retriever import (
    OrientedPairIndex,
    WalkerMaxSimPass,
    word_pair_walk,
)

_TOKEN_RE = re.compile(r"[a-z]+")
LexicalModeStr = Literal["bm25", "append_index", "lattice_pure", "lattice_plane"]
FuseModeStr = Literal["norm", "additive", "lex_mult_rare_l2"]
CageIngestModeStr = Literal["positional", "rare_combo"]


def resolve_rccm_config(cfg: HybridConfig) -> HybridConfig:
    """Apply RCCM Phase-1 preset when ``enable_rccm`` is set."""
    if not cfg.enable_rccm:
        return cfg
    fuse = cfg.fuse_mode
    if fuse == "additive":
        fuse = "lex_mult_rare_l2"
    return replace(
        cfg,
        cage_ingest_mode="rare_combo",
        enable_corpus_lattice=True,
        enable_rare_shell_lattice=True,
        rare_shell_pool_mode="widen",
        enable_append_pool_union=True,
        fuse_mode=fuse,
    )


def needs_corpus_lattice_build(cfg: HybridConfig) -> bool:
    """Ingest-time corpus lattice skeleton for meet-vector / EQ-RAG / RCCM."""
    return (
        cfg.enable_corpus_lattice
        or cfg.enable_rccm
        or cfg.enable_eq_rag_expand
        or cfg.enable_meet_vector_pair
    )


def resolve_eq_rag_config(cfg: HybridConfig) -> HybridConfig:
    """EQ-RAG uses corpus lattice for complement recovery (no preset changes)."""
    return cfg


def resolve_meet_vector_config(cfg: HybridConfig) -> HybridConfig:
    """Meet-vector pair routing uses corpus lattice at ingest (no preset changes)."""
    return cfg


@dataclass(frozen=True)
class HybridConfig:
    lam_lex: float = 1.0
    lam_l2: float = 0.0
    lam_walk: float = 0.0
    fuse_mode: FuseModeStr = "additive"
    max_pool: int | None = 4000
    sat_a: float = 1.0
    lpow: float = 0.35
    # append_index: multiview BM25 on pool (~0.7 SciFact floor); lattice_plane: lattice formula fallback
    lexical_mode: LexicalModeStr = "append_index"
    enable_pair_meet: bool = True
    pair_idf_gate: float = 2.5
    pair_top_terms: int = 4
    pair_meet_cap: int = 80
    # Union append-index BM25 top-K into Phase A pool — recovers gold corridor routing misses
    enable_append_pool_union: bool = True
    append_pool_k: int = 200
    enable_walk_pool_expand: bool = True
    walk_pool_cap: int = 80
    enable_stage08_rerank: bool = True
    enable_missing_neighbor_rerank: bool = False
    # Default 0.35; SciFact 50q MNCR-only hurts nDCG — use lower lambda on real corpora
    mn_rerank_lambda: float = 0.35
    enable_cross_hit_rerank: bool = False
    ch_rerank_lambda: float = 0.25
    ch_consensus_k: int = 20
    enable_gravity_cascade_rerank: bool = False
    gw_rerank_lambda: float = 0.30
    gw_anchor_count: int = 3
    rerank_prelim: int = 120
    enable_demotion: bool = True
    polluter_penalty: float = 0.18
    polluter_docs: frozenset[str] = field(default_factory=scifact_polluter_docs)
    hub_lex_penalty: float = 0.0
    density_penalty: float = 0.0
    lam_teach: float = 0.20
    teach_query_rewrite: bool = True
    enable_rare_shell_lattice: bool = False
    # intersect shrinks corridor pool (~0.49 recall on SciFact); widen unions rare anchors only
    rare_shell_pool_mode: Literal["intersect", "widen"] = "widen"
    rare_shell_widen_cap: int = 120
    rare_k_per_doc: int = 8
    rare_max_df_frac: float = 0.05
    lam_rare: float = 0.15
    cage_ingest_mode: CageIngestModeStr = "positional"
    enable_corpus_lattice: bool = False
    # RCCM Phase 1 — rare-combo corpus mesh (default off for backward compat)
    enable_rccm: bool = False
    rccm_eps: float = 0.05
    rccm_rare_term_cap: int = 4
    ugf_boost: float = 0.03
    # Soft EQ-RAG — complement/sunflower term expansion for Phase A routing only
    enable_eq_rag_expand: bool = False
    eq_rag_expand_cap: int = 4
    eq_rag_idf_gate: float = 2.5
    # Meet-vector pair routing — union global_3way docs into pair_meet (needs corpus lattice)
    enable_meet_vector_pair: bool = False


@dataclass(frozen=True)
class HybridHit:
    doc_id: str
    score: float
    lex_score: float
    l2_score: float
    l2_trace: dict
    walk_score: float = 0.0
    walk_trace: tuple[dict, ...] = ()
    rerank_boost: float = 0.0

    def explain(self) -> dict:
        out = {
            "doc_id": self.doc_id,
            "score": round(self.score, 6),
            "lex_score": round(self.lex_score, 6),
            "l2_score": round(self.l2_score, 6),
            "walk_score": round(self.walk_score, 6),
            "l2_trace": self.l2_trace,
        }
        if self.walk_trace:
            out["walk_witnesses"] = list(self.walk_trace)
        if self.rerank_boost:
            out["rerank_boost"] = round(self.rerank_boost, 6)
        return out


@dataclass(frozen=True)
class HybridRetrieveTrace:
    query: str
    route_mode: RouteMode
    pool_size: int
    pool_docs: frozenset[str]
    filter_steps: tuple[dict, ...]
    hits: tuple[HybridHit, ...]
    query_walk: tuple[dict, ...] = ()

    def explain(self) -> dict:
        out = {
            "query": self.query,
            "route_mode": self.route_mode,
            "pool_size": self.pool_size,
            "pool_docs": sorted(self.pool_docs),
            "filter_steps": list(self.filter_steps),
            "hits": [h.explain() for h in self.hits],
        }
        if self.query_walk:
            out["query_walk"] = list(self.query_walk)
        return out


@dataclass
class HybridZeroShotRetriever:
    """Stage 08 routing + lexical floor + shell MaxSim + walk MaxSim fusion."""

    router: LatticeRetriever
    append_idx: AppendOnlyLatticeIndex
    lexical: object  # LatticeLexicalScorer after bind()
    corpus: dict[str, str] = field(default_factory=dict)
    teach: TeachStore | None = None
    placements: dict = field(default_factory=dict)
    shell_index: dict[str, tuple] = field(default_factory=dict)
    doc_walks: dict[str, tuple] = field(default_factory=dict)
    pair_index: OrientedPairIndex = field(default_factory=OrientedPairIndex)
    rare_lattice: RareShellLatticeIndex | None = None
    corpus_prime: int | None = None
    corpus_lattice: CorpusLattice | None = None
    config: HybridConfig = field(default_factory=HybridConfig)

    def with_config(self, **kwargs) -> HybridZeroShotRetriever:
        """Return self with updated config (for bench sweeps without rebuild)."""
        self.config = replace(self.config, **kwargs)
        return self

    def _query_terms(self, query: str) -> list[str]:
        return [w for w in _TOKEN_RE.findall(query.lower()) if len(w) >= 2]

    def _cap_pool(
        self, pool: set[str], *, must_keep: frozenset[str] = frozenset()
    ) -> set[str]:
        cap = self.config.max_pool
        if cap is None or len(pool) <= cap:
            return pool
        must = must_keep & pool
        if len(must) >= cap:
            return set(must)
        remainder = sorted(pool - must)
        return must | set(remainder[: cap - len(must)])

    def _append_pool_union(
        self, query: str, pool: set[str]
    ) -> tuple[set[str], frozenset[str], tuple[dict, ...]]:
        cfg = self.config
        if not cfg.enable_append_pool_union or cfg.append_pool_k <= 0:
            return pool, frozenset(), ()
        append_top = self.lexical.append_top_k(cfg.append_pool_k)
        if not append_top:
            append_top = self.append_idx.search(query, k=cfg.append_pool_k)
        append_set = frozenset(append_top)
        if not append_set:
            return pool, frozenset(), ()
        expanded = pool | append_set
        step = {
            "step": "append_pool_union",
            "append_pool_k": cfg.append_pool_k,
            "append_top": len(append_top),
            "added_docs": len(append_set - pool),
            "pool_size_before": len(pool),
            "pool_size": len(expanded),
        }
        return expanded, append_set, (step,)

    def _eq_rag_soft_expand_terms(
        self, query_terms: list[str]
    ) -> tuple[list[str], list[str], tuple[dict, ...]]:
        """Soft complement expand — add recovered rare terms to Phase A routing set."""
        cfg = self.config
        if not cfg.enable_eq_rag_expand:
            return query_terms, [], ()
        idx = self.append_idx
        N = len(idx.alive)
        rare = [
            w
            for w in rarest_terms(query_terms, idx, N)
            if word_idf(idx, w, N) >= cfg.eq_rag_idf_gate
        ][: cfg.pair_top_terms]
        if len(rare) < 2:
            return query_terms, [], ()
        sem = self.router.semantic
        query_set = {t.lower() for t in query_terms}
        recovered: list[str] = []
        recovered_set: set[str] = set()
        for a, b in combinations(rare, 2):
            if len(recovered) >= cfg.eq_rag_expand_cap:
                break
            pa, pb = sorted([sem._prime_for_term(a), sem._prime_for_term(b)])
            candidates: set[str] = set()
            if self.corpus_lattice is not None:
                keys_a = self.corpus_lattice.term_to_meet_keys.get(a.lower(), frozenset())
                keys_b = self.corpus_lattice.term_to_meet_keys.get(b.lower(), frozenset())
                for mk in keys_a & keys_b:
                    rec = self.corpus_lattice.global_3way.get(mk)
                    if rec is not None:
                        candidates |= rec.correlated_terms
                docs_ab = (
                    self.corpus_lattice.term_to_docs.get(a.lower(), frozenset())
                    & self.corpus_lattice.term_to_docs.get(b.lower(), frozenset())
                )
                for doc_id in docs_ab:
                    entry = self.corpus_lattice.doc_registry.get(doc_id)
                    if entry is not None:
                        candidates |= set(entry.rare_terms)
            else:
                for doc_id in posting_docs(idx, a) & posting_docs(idx, b):
                    for w in sem.doc_freq:
                        if w in (a, b) or w in query_set or w in recovered_set:
                            continue
                        if sem.is_rare(w, max_df=max(2, int(sem.n_docs * 0.05))):
                            candidates.add(w)
            for t in sorted(candidates):
                if len(recovered) >= cfg.eq_rag_expand_cap:
                    break
                tl = t.lower()
                if tl in query_set or tl in recovered_set:
                    continue
                pc = sem._prime_for_term(t)
                chain = tuple(sorted({pa, pb, pc}))
                if len(chain) < 3:
                    continue
                try:
                    pred = int(round(missing_member(chain, (pa, pb))))
                except ValueError:
                    continue
                if pred == pc:
                    recovered.append(tl)
                    recovered_set.add(tl)
        if not recovered:
            return query_terms, [], ()
        expanded = list(query_terms) + recovered
        return expanded, recovered, (
            {
                "step": "eq_rag_expanded_terms",
                "rare_query_terms": rare,
                "recovered_terms": recovered,
                "expand_cap": cfg.eq_rag_expand_cap,
            },
        )

    def _pair_meet_expand(
        self,
        query: str,
        pool: set[str],
        *,
        route_terms: list[str] | None = None,
    ) -> tuple[set[str], tuple[dict, ...]]:
        cfg = self.config
        if not cfg.enable_pair_meet:
            return pool, ()
        idx = self.append_idx
        N = len(idx.alive)
        terms = route_terms if route_terms is not None else self.router._query_terms(query)
        rare = [
            w
            for w in rarest_terms(terms, idx, N)
            if word_idf(idx, w, N) >= cfg.pair_idf_gate
        ][: cfg.pair_top_terms]
        meet: set[str] = set()
        meet_key_docs = 0
        lat = self.corpus_lattice
        use_meet_keys = cfg.enable_meet_vector_pair and lat is not None
        sem = self.router.semantic
        for a, b in combinations(rare, 2):
            meet |= posting_docs(idx, a) & posting_docs(idx, b)
            if use_meet_keys:
                pa = sem._prime_for_term(a)
                pb = sem._prime_for_term(b)
                before = len(meet)
                meet |= lat.lookup_pair_meet(pa, pb)
                meet |= lat.lookup_pair_meet_terms(a, b)
                meet_key_docs += len(meet) - before
        if not meet:
            return pool, ()
        # Prefer docs already in corridor pool when capping meet additions
        new_only = meet - pool
        if len(new_only) > cfg.pair_meet_cap:
            new_only = set(sorted(new_only)[: cfg.pair_meet_cap])
        expanded = pool | new_only
        step = {
            "step": "pair_meet_expand",
            "rare_terms": rare,
            "meet_key_docs": meet_key_docs,
            "added_docs": len(new_only),
            "pool_size": len(expanded),
        }
        return self._cap_pool(expanded), (step,)

    def _walk_pool_expand(
        self, query: str, pool: set[str]
    ) -> tuple[set[str], tuple[dict, ...]]:
        cfg = self.config
        if not cfg.enable_walk_pool_expand or not self.doc_walks:
            return pool, ()
        query_walk = word_pair_walk(query, self.router.semantic)
        if not query_walk:
            return pool, ()
        walker_pool, route_steps = self.pair_index.route_pool(query_walk)
        new_only = walker_pool - pool
        if len(new_only) > cfg.walk_pool_cap:
            new_only = set(sorted(new_only)[: cfg.walk_pool_cap])
        if not new_only:
            return pool, ()
        expanded = pool | new_only
        step = {
            "step": "walk_pair_expand",
            "added_docs": len(new_only),
            "pool_size": len(expanded),
        }
        return self._cap_pool(expanded), (step,) + tuple(route_steps)

    def _corpus_lattice_pool_expand(
        self,
        query: str,
        pool: set[str],
        *,
        route_terms: list[str] | None = None,
    ) -> tuple[set[str], set[str], tuple[dict, ...]]:
        cfg = self.config
        if (
            not (cfg.enable_corpus_lattice or cfg.enable_rccm)
            or self.corpus_lattice is None
        ):
            return pool, set(), ()
        terms = route_terms if route_terms is not None else self._query_terms(query)
        lat_pool, lat_steps = self.corpus_lattice.route_pool(
            terms, semantic=self.router.semantic
        )
        if not lat_pool:
            return pool, set(), tuple(lat_steps)
        new_only = lat_pool - pool
        expanded = pool | lat_pool
        step = {
            "step": "corpus_lattice_pool",
            "lattice_pool_size": len(lat_pool),
            "added_docs": len(new_only),
            "pool_size": len(expanded),
        }
        return expanded, lat_pool, tuple(lat_steps) + (step,)

    def _rare_shell_pool_sidecar(
        self, query: str, pool: set[str]
    ) -> tuple[set[str], tuple[dict, ...]]:
        cfg = self.config
        if not cfg.enable_rare_shell_lattice or self.rare_lattice is None:
            return pool, ()
        terms = self._query_terms(query)
        sem = self.router.semantic
        if cfg.rare_shell_pool_mode == "widen":
            rare_pool, rare_steps = self.rare_lattice.route_pool_widen(
                terms, semantic=sem
            )
            if not rare_pool:
                return pool, tuple(rare_steps)
            new_only = rare_pool - pool
            if len(new_only) > cfg.rare_shell_widen_cap:
                new_only = set(sorted(new_only)[: cfg.rare_shell_widen_cap])
            expanded = pool | new_only
            step = {
                "step": "rare_shell_widen",
                "rare_pool_size": len(rare_pool),
                "added_docs": len(new_only),
                "pool_size": len(expanded),
            }
            return expanded, tuple(rare_steps) + (step,)

        rare_pool, rare_steps = self.rare_lattice.route_pool(terms, semantic=sem)
        if not rare_pool:
            return pool, tuple(rare_steps)
        narrowed = pool & rare_pool
        step = {
            "step": "rare_shell_intersect",
            "rare_pool_size": len(rare_pool),
            "intersect_size": len(narrowed),
            "pool_size": len(narrowed) if narrowed else len(pool),
        }
        return (narrowed if narrowed else pool), tuple(rare_steps) + (step,)

    def _phase_a_pool(
        self, query: str
    ) -> tuple[set[str], RouteMode, tuple[dict, ...], dict[str, int]]:
        base_terms = self._query_terms(query)
        route_terms, _, eq_steps = self._eq_rag_soft_expand_terms(base_terms)
        pool_list, route_mode, steps, _, _ = self.router.route_pool(query)
        corridor_pool = set(pool_list)
        pool = set(corridor_pool)
        meet_pool, meet_steps = self._pair_meet_expand(
            query, pool, route_terms=route_terms
        )
        walk_pool, walk_steps = self._walk_pool_expand(query, meet_pool)
        union_pool, append_must, append_steps = self._append_pool_union(query, walk_pool)
        lat_pool, corpus_lat_docs, lat_steps = self._corpus_lattice_pool_expand(
            query, union_pool, route_terms=route_terms
        )
        rare_pool, rare_steps = self._rare_shell_pool_sidecar(query, lat_pool)
        all_steps = (
            tuple(steps)
            + eq_steps
            + meet_steps
            + walk_steps
            + append_steps
            + lat_steps
            + rare_steps
        )
        pool = self._cap_pool(rare_pool, must_keep=append_must)

        gate_counts: dict[str, int] = {}
        if self.config.enable_rccm:
            for doc_id in corridor_pool:
                gate_counts[doc_id] = gate_counts.get(doc_id, 0) + 1
            for doc_id in append_must:
                gate_counts[doc_id] = gate_counts.get(doc_id, 0) + 1
            for doc_id in corpus_lat_docs:
                gate_counts[doc_id] = gate_counts.get(doc_id, 0) + 1
            ugf_hits = sum(1 for c in gate_counts.values() if c >= 2)
            all_steps = all_steps + (
                {
                    "step": "ugf_lite_gate_counts",
                    "docs_2of3": ugf_hits,
                    "corridor_size": len(corridor_pool),
                    "append_must_size": len(append_must),
                    "corpus_lattice_size": len(corpus_lat_docs),
                },
            )

        return pool, route_mode, all_steps, gate_counts

    @staticmethod
    def _fuse(
        lex: dict[str, float],
        l2: dict[str, float],
        walk: dict[str, float],
        pool: set[str],
        *,
        lam_lex: float,
        lam_l2: float,
        lam_walk: float,
        fuse_mode: FuseModeStr,
        l2_rare: dict[str, float] | None = None,
        rccm_eps: float = 0.05,
        gate_counts: dict[str, int] | None = None,
        ugf_boost: float = 0.03,
    ) -> dict[str, float]:
        if fuse_mode == "lex_mult_rare_l2":
            l2r = l2_rare or {}
            l2r_max = max(l2r.values(), default=0.0) or 1.0
            out: dict[str, float] = {}
            for d in pool:
                norm_rare = l2r.get(d, 0.0) / l2r_max if l2r_max else 0.0
                score = lam_lex * lex.get(d, 0.0) * (1.0 + rccm_eps * norm_rare)
                score += lam_walk * walk.get(d, 0.0)
                if gate_counts and gate_counts.get(d, 0) >= 2:
                    score += ugf_boost * lex.get(d, 0.0)
                out[d] = score
            return out
        if fuse_mode == "additive":
            return {
                d: (
                    lam_lex * lex.get(d, 0.0)
                    + lam_l2 * l2.get(d, 0.0)
                    + lam_walk * walk.get(d, 0.0)
                )
                for d in pool
            }
        lmax = max(lex.values(), default=0.0) or 1.0
        l2max = max(l2.values(), default=0.0) or 1.0
        wmax = max(walk.values(), default=0.0) or 1.0
        return {
            d: (
                lam_lex * lex.get(d, 0.0) / lmax
                + lam_l2 * l2.get(d, 0.0) / l2max
                + lam_walk * walk.get(d, 0.0) / wmax
            )
            for d in pool
        }

    def _rare_l2_scores(
        self,
        terms: list[str],
        pool: set[str],
        *,
        shell_idx: dict[str, tuple],
        idf_fn,
        cap: int,
    ) -> dict[str, float]:
        sem = self.router.semantic
        rare_terms = [t for t in terms if sem.is_rare(t)][:cap]
        if not rare_terms:
            return {d: 0.0 for d in pool}
        out: dict[str, float] = {}
        for doc_id in pool:
            score, _ = Lattice2CorrelationPass.score(
                rare_terms,
                doc_id,
                shell_idx,
                idf_fn,
                placements=self.placements,
            )
            out[doc_id] = score
        return out

    def _apply_stage08_rerank(
        self,
        hits: list[HybridHit],
        query: str,
    ) -> list[HybridHit]:
        cfg = self.config
        if not cfg.enable_stage08_rerank or len(hits) < 2:
            return hits
        terms = self._query_terms(query)
        by_id = {h.doc_id: h for h in hits}
        working = list(hits)
        if cfg.enable_missing_neighbor_rerank:
            working = missing_neighbor_rerank(
                working,
                terms,
                self.shell_index,
                self.placements,
                self.router.semantic.idf,
                lambda_mn=cfg.mn_rerank_lambda,
            )
        if cfg.enable_cross_hit_rerank:
            working = cross_hit_mutual_rerank(
                working,
                self.router.semantic.idf,
                lambda_ch=cfg.ch_rerank_lambda,
                consensus_k=cfg.ch_consensus_k,
            )
        working_by_id = {h.doc_id: h for h in working}
        rh = [
            RetrieveHit(doc_id=h.doc_id, score=h.score, reasons=())
            for h in working
        ]
        rh = self.router._lexical_bridge_rerank(rh)
        if cfg.enable_gravity_cascade_rerank:
            bridged_hits: list[HybridHit] = []
            for r in rh:
                staged = working_by_id.get(r.doc_id, by_id[r.doc_id])
                bridged_hits.append(
                    HybridHit(
                        doc_id=staged.doc_id,
                        score=r.score,
                        lex_score=staged.lex_score,
                        l2_score=staged.l2_score,
                        l2_trace=staged.l2_trace,
                        walk_score=staged.walk_score,
                        walk_trace=staged.walk_trace,
                        rerank_boost=staged.rerank_boost,
                    )
                )
            gw_out = gravity_cascade_rerank(
                bridged_hits,
                terms,
                self.shell_index,
                self.router.semantic.idf,
                self.router.semantic.is_rare,
                lambda_gw=cfg.gw_rerank_lambda,
                anchor_count=cfg.gw_anchor_count,
            )
            working_by_id = {h.doc_id: h for h in gw_out}
            rh = [
                RetrieveHit(doc_id=h.doc_id, score=h.score, reasons=())
                for h in gw_out
            ]
        rh = self.router._cage_anchor_rerank(rh, terms)
        out: list[HybridHit] = []
        for r in rh:
            base = by_id[r.doc_id]
            staged = working_by_id.get(r.doc_id, base)
            boost = r.score - base.score
            out.append(
                HybridHit(
                    doc_id=base.doc_id,
                    score=r.score,
                    lex_score=base.lex_score,
                    l2_score=base.l2_score,
                    l2_trace=staged.l2_trace,
                    walk_score=base.walk_score,
                    walk_trace=base.walk_trace,
                    rerank_boost=boost,
                )
            )
        return out

    def retrieve_with_trace(self, query: str, *, limit: int = 10) -> HybridRetrieveTrace:
        cfg = self.config
        score_query = query
        if self.teach and cfg.teach_query_rewrite:
            score_query = self.teach.rewrite_query(query)

        terms = self._query_terms(score_query)
        if self.append_idx._dense_ready:
            self.lexical.cache_dense_scores(score_query)
        pool, route_mode, steps, gate_counts = self._phase_a_pool(score_query)

        lex = self.lexical.score_pool(score_query, frozenset(pool))
        l2_scores: dict[str, float] = {}
        l2_traces: dict[str, dict] = {}
        l2_rare_scores: dict[str, float] = {}
        walk_scores: dict[str, float] = {}
        walk_traces: dict[str, tuple[dict, ...]] = {}
        idf_fn = self.router.semantic.idf
        query_walk = word_pair_walk(score_query, self.router.semantic)
        query_walk_explain = tuple(d.explain() for d in query_walk)
        pair_idf_fn = self.pair_index.idf_pair if self.doc_walks else None
        use_rare = cfg.enable_rare_shell_lattice and self.rare_lattice is not None
        shell_idx = (
            self.rare_lattice.doc_shells if use_rare else self.shell_index
        )
        lam_l2 = cfg.lam_rare if use_rare else cfg.lam_l2
        use_mrl2 = cfg.fuse_mode == "lex_mult_rare_l2" or cfg.enable_rccm
        if use_mrl2:
            l2_rare_scores = self._rare_l2_scores(
                terms,
                pool,
                shell_idx=shell_idx,
                idf_fn=idf_fn,
                cap=cfg.rccm_rare_term_cap,
            )

        for doc_id in pool:
            score, trace = Lattice2CorrelationPass.score(
                terms,
                doc_id,
                shell_idx,
                idf_fn,
                placements=self.placements,
            )
            l2_scores[doc_id] = score
            l2_traces[doc_id] = trace.explain()

            if self.doc_walks:
                wscore, witnesses = WalkerMaxSimPass.walk_maxsim(
                    query_walk,
                    self.doc_walks.get(doc_id, ()),
                    idf_fn,
                    pair_idf_fn=pair_idf_fn,
                )
                walk_scores[doc_id] = wscore
                walk_traces[doc_id] = tuple(
                    {
                        "query_dot": w.query_dot,
                        "best_witness": round(w.best_witness, 4),
                        "best_doc_dot": w.best_doc_dot,
                        "idf": round(w.idf, 4),
                        "weighted": round(w.weighted, 4),
                    }
                    for w in witnesses
                )

        fuse_mode = cfg.fuse_mode
        if cfg.enable_rccm and fuse_mode == "additive":
            fuse_mode = "lex_mult_rare_l2"
        fused = self._fuse(
            lex,
            l2_scores,
            walk_scores,
            pool,
            lam_lex=cfg.lam_lex,
            lam_l2=lam_l2,
            lam_walk=cfg.lam_walk,
            fuse_mode=fuse_mode,
            l2_rare=l2_rare_scores if use_mrl2 else None,
            rccm_eps=cfg.rccm_eps,
            gate_counts=gate_counts if cfg.enable_rccm else None,
            ugf_boost=cfg.ugf_boost,
        )

        if self.teach and cfg.lam_teach > 0:
            teach_exp = self.teach.expand_scores(query)
            tmax = max(teach_exp.values(), default=0.0) or 1.0
            for d in pool:
                fused[d] = fused.get(d, 0.0) + cfg.lam_teach * teach_exp.get(d, 0.0) / tmax

        if cfg.enable_demotion and self.corpus:
            fused, _ = apply_lexical_demotion(
                fused,
                score_query,
                pool,
                self.append_idx,
                self.corpus,
                polluter_docs=cfg.polluter_docs,
                polluter_penalty=cfg.polluter_penalty,
                hub_lex_penalty=cfg.hub_lex_penalty,
                density_penalty=cfg.density_penalty,
            )

        prelim_n = max(cfg.rerank_prelim, limit)
        prelim_ids = sorted(fused.keys(), key=lambda d: (-fused[d], d))[:prelim_n]
        prelim_hits = [
            HybridHit(
                doc_id=d,
                score=fused[d],
                lex_score=lex.get(d, 0.0),
                l2_score=l2_scores.get(d, 0.0),
                l2_trace=l2_traces.get(d, {}),
                walk_score=walk_scores.get(d, 0.0),
                walk_trace=walk_traces.get(d, ()),
            )
            for d in prelim_ids
        ]
        reranked = self._apply_stage08_rerank(prelim_hits, score_query)
        hits = tuple(reranked[:limit])

        return HybridRetrieveTrace(
            query=query,
            route_mode=route_mode,
            pool_size=len(pool),
            pool_docs=frozenset(pool),
            filter_steps=steps,
            hits=hits,
            query_walk=query_walk_explain,
        )

    def retrieve(self, query: str, *, limit: int = 10) -> list[HybridHit]:
        return list(self.retrieve_with_trace(query, limit=limit).hits)


def build_hybrid_retriever(
    corpus: dict[str, str],
    *,
    registry: Stage04Registry | None = None,
    config: HybridConfig | None = None,
    fast_ingest: bool = False,
    corpus_name: str = "default",
) -> HybridZeroShotRetriever:
    """Single-pass ingest: Stage 08 + append multiview + L2 shells + pair walks."""
    cfg = resolve_rccm_config(config or HybridConfig())
    cfg = resolve_eq_rag_config(cfg)
    cfg = resolve_meet_vector_config(cfg)
    inner = PromotionRegistry(fast_ingest=fast_ingest, defer_l2_promotion=True)
    reg = registry or Stage04Registry(registry=inner)
    scope = corpus_scope(corpus_name, reg)
    corpus_prime = scope.corpus_prime
    semantic = SemanticLightIndex(registry=reg)
    router = LatticeRetriever(semantic=semantic)
    if cfg.lexical_mode == "append_index":
        # BM25 prelim dominates; bridge/cage may only ε-tie-break within δ of top-1.
        router.bridge_rerank_tiebreak_frac = 0.05
        router.cage_anchor_rerank_tiebreak_frac = 0.05
    append_idx = AppendOnlyLatticeIndex()
    doc_pool = DocPrimePool()
    pair_index = OrientedPairIndex()

    placements: dict = {}
    shell_index: dict[str, tuple] = {}
    doc_walks: dict[str, tuple] = {}
    rare_lattice: RareShellLatticeIndex | None = (
        RareShellLatticeIndex()
        if cfg.enable_rare_shell_lattice or cfg.enable_rccm
        else None
    )
    corpus_lattice_builder: CorpusLatticeBuilder | None = (
        CorpusLatticeBuilder(
            corpus_prime,
            reg,
            semantic,
            doc_pool,
            k_rare=cfg.rare_k_per_doc,
            max_df_frac=cfg.rare_max_df_frac,
        )
        if needs_corpus_lattice_build(cfg)
        else None
    )
    ingest_mode = cfg.cage_ingest_mode

    for doc_id, text in corpus.items():
        if corpus_lattice_builder is not None:
            corpus_lattice_builder.observe_doc(doc_id, text)
        reg.observe_text(text)
        router.index_doc(
            doc_id,
            text,
            cage_ingest_mode=ingest_mode,
            k_rare=cfg.rare_k_per_doc,
            max_df_frac=cfg.rare_max_df_frac,
        )
        append_idx.add(doc_id, text)
        placement = encode_doc(
            doc_id,
            text,
            reg,
            doc_pool,
            semantic=semantic,
            corpus_prime=corpus_prime,
        )
        shells = build_doc_correlation_shells(
            text,
            reg,
            semantic=semantic,
            mode=ingest_mode,
            k_rare=cfg.rare_k_per_doc,
            max_df_frac=cfg.rare_max_df_frac,
        )
        walk = word_pair_walk(text, semantic)
        placements[doc_id] = placement
        shell_index[doc_id] = shells
        doc_walks[doc_id] = walk
        pair_index.index_doc(doc_id, walk)
        if rare_lattice is not None:
            rare_shells = build_rare_correlation_shells(
                text,
                reg,
                semantic,
                k=cfg.rare_k_per_doc,
                max_df_frac=cfg.rare_max_df_frac,
                mode=ingest_mode,
            )
            rare_lattice.observe_doc(doc_id, rare_shells)

    append_idx.finalize()
    corpus_lattice = (
        corpus_lattice_builder.finalize() if corpus_lattice_builder is not None else None
    )

    lexical = lattice_lexical_scorer(
        append_idx,
        mode=cfg.lexical_mode,
        sat_a=cfg.sat_a,
        lpow=cfg.lpow,
    )

    return HybridZeroShotRetriever(
        router=router,
        append_idx=append_idx,
        lexical=lexical,
        corpus=dict(corpus),
        placements=placements,
        shell_index=shell_index,
        doc_walks=doc_walks,
        pair_index=pair_index,
        rare_lattice=rare_lattice,
        corpus_prime=corpus_prime,
        corpus_lattice=corpus_lattice,
        config=cfg,
    )


# Alias — same class, three-layer defaults via build_unified_dual_lattice_retriever.
UnifiedDualLatticeRetriever = HybridZeroShotRetriever


def build_unified_dual_lattice_retriever(
    corpus: dict[str, str],
    *,
    registry: Stage04Registry | None = None,
    config: HybridConfig | None = None,
    fast_ingest: bool = False,
    lam_lex: float = 1.0,
    lam_l2: float = 0.25,
    lam_walk: float = 0.20,
) -> UnifiedDualLatticeRetriever:
    """Three-layer zero-shot retriever — lexical floor + shell + walk MaxSim."""
    cfg = config or HybridConfig(lam_lex=lam_lex, lam_l2=lam_l2, lam_walk=lam_walk)
    return build_hybrid_retriever(corpus, registry=registry, config=cfg, fast_ingest=fast_ingest)
