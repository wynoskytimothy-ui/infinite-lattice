#!/usr/bin/env python3
"""
aethos_edge_rag3.py — AETHOS Revolutionary EdgeRAG

Full synthesis of 7 months measured research + 3D plane formulas:

PROVEN SPINE (EdgeRAG2):
  M1 gamma-di/BIC codec | M2 WAND router | M3 NPMI bake | M4 counting-bridges
  M5 governor | elimination ledger

REVOLUTIONARY LAYER (pool-stage only — never geometry-as-full-corpus-ranker):
  R1 Deep pool (depth 200–300) + meet3 governor
  R2 Supervised bridge POOL expansion (+0.057 nDCG append path)
  R2b §5.29 error-driven bridge promotion in learn_bridges (governor-gated)
  R3 NPMI drift POOL expansion (governor-gated, semantic corpora)
  R4 Geometric enrich (sovereign unlock, intersection, grand-cross) — your formulas
  R5 Miniverse: query-time intersection replicas + N-line store (lazy) + 3-way fusion
  R6 Bounded fuse: lex + bridges + geo boost on |pool| only
  R6b §5.9.4 tier-stack precision dial rerank (decode_tiers + merge4, governor-gated)
  R7 v7 lattice: state4 + corridor triple points + VA4 precision + composite grid (§3.9a, §5.28-29)
"""
from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np

from aethos_edge_rag2 import ELIMINATION_LEDGER, EdgeRAG2
from aethos_geometric_enrich import GeometricEnricher
from edge_rag3_pinwire import build_hub_inverted, build_pin_wire_layer, layer_bytes_per_doc, pool_expand_pin_wire
from edge_rag3_pool import (
    bridge_scores_tf_vec,
    pool_expand_bridges,
    pool_expand_drift,
    prune_query_terms,
    topk_from_ordered,
)
from edge_rag3_rerank import SCIFACT_POLLUTERS, fuse_scores_with_rerank, tier_stack_precision_vec
from edge_rag3_miniverse import (
    build_miniverse_trainer,
    build_transgressor_miniverse_store,
    pool_expand_intersection_miniverse,
    pool_expand_miniverse_fusion,
    pool_expand_transgressor_store,
    transgressor_meet_expand,
)
from edge_rag3_v7 import V7LatticeLayer
from edge_rag3_bridge_boost import (
    learn_bridges_error_driven,
    pool_expand_corridor_bridge,
    restore_base_bridges,
)

REVOLUTIONARY_LEDGER = ELIMINATION_LEDGER


class EdgeRAG3(EdgeRAG2):
    """EdgeRAG2 + pool-stage fusion + geometric enrich (revolutionary tier)."""

    def __init__(
        self,
        stem: bool = True,
        bake_M: int = 16,
        bake_mode: str = "npmi",
        margin: float = 0.002,
        few_term: int = 4,
        *,
        pool_depth: int = 300,
        bridge_pool_n: int = 50,
        drift_pool_n: int = 40,
        max_pool: int = 280,
        lam_bridge: float = 0.42,
        geo_lam: float = 0.0,
        bridge_top_per: int = 21,
        use_drift_pool: str = "govern",
        use_geo: bool = True,
        use_pin_wire: bool = True,
        pin_wire_k: int = 12,
        pin_wire_expand_n: int = 20,
        use_continuum: bool = False,
        use_lattice: bool = False,
        use_tf_bridges: bool = False,
        use_demotion: bool = True,
        use_lex_bridge_rerank: bool = False,
        polluter_penalty: float = 0.18,
        lex_bridge_lam: float = 0.15,
        use_miniverse_pool: str = "off",
        use_transgressor_meet: str = "on",
        use_transgressor_store: str = "off",
        use_intersection_miniverse: bool = True,
        use_v7_lattice: str = "govern",
        use_error_bridge_promotion: str = "govern",
        use_tier_stack_rerank: str = "govern",
        v7_expand_n: int = 28,
        v7_boost_lam: float = 0.22,
        tier_stack_lam: float = 0.14,
        tier_stack_depth: int = 2,
        miniverse_expand_n: int = 25,
        transgressor_expand_n: int = 30,
        transgressor_store_expand_n: int = 35,
        miniverse_idf_gate: float = 2.5,
        drift_margin: float = 0.0,
    ) -> None:
        super().__init__(stem=stem, bake_M=bake_M, bake_mode=bake_mode, margin=margin, few_term=few_term)
        self.pool_depth = int(pool_depth)
        self.bridge_pool_n = int(bridge_pool_n)
        self.drift_pool_n = int(drift_pool_n)
        self.max_pool = int(max_pool)
        self.lam_bridge = float(lam_bridge)
        self.geo_lam = float(geo_lam)
        self.bridge_top_per = int(bridge_top_per)
        self.use_drift_pool = use_drift_pool
        self.use_geo = use_geo
        self.use_pin_wire = use_pin_wire
        self.pin_wire_k = int(pin_wire_k)
        self.pin_wire_expand_n = int(pin_wire_expand_n)
        self._use_tf_bridges = use_tf_bridges
        self.use_demotion = use_demotion
        self.use_lex_bridge_rerank = use_lex_bridge_rerank
        self.polluter_penalty = float(polluter_penalty)
        self.lex_bridge_lam = float(lex_bridge_lam)
        self.use_miniverse_pool = use_miniverse_pool
        self.use_transgressor_meet = use_transgressor_meet
        self.use_transgressor_store = use_transgressor_store
        self.use_intersection_miniverse = use_intersection_miniverse
        self.use_v7_lattice = use_v7_lattice
        self.use_error_bridge_promotion = use_error_bridge_promotion
        self.use_tier_stack_rerank = use_tier_stack_rerank
        self.v7_expand_n = int(v7_expand_n)
        self.v7_boost_lam = float(v7_boost_lam)
        self.tier_stack_lam = float(tier_stack_lam)
        self.tier_stack_depth = int(tier_stack_depth)
        self.miniverse_expand_n = int(miniverse_expand_n)
        self.transgressor_expand_n = int(transgressor_expand_n)
        self.transgressor_store_expand_n = int(transgressor_store_expand_n)
        self.miniverse_idf_gate = float(miniverse_idf_gate)
        self.drift_margin = float(drift_margin)
        self._use_continuum_geo = use_continuum
        self._use_lattice = use_lattice
        self._drift_pool: Optional[dict] = None
        self._drift_pool_on = False
        self._geo: Optional[GeometricEnricher] = None
        self._rev_enabled = False
        self._rev_log = ""
        self._pin_layer = None
        self._hub_inv: Optional[dict] = None
        self._miniverse_trainer = None
        self._miniverse_on = False
        self._transgressor_meet_on = False
        self._transgressor_store = None
        self._transgressor_sem = None
        self._transgressor_reg = None
        self._transgressor_store_on = False
        self._v7: Optional[V7LatticeLayer] = None
        self._v7_on = False
        self._error_bridge_on = False
        self._tier_stack_on = False

    def build(self, corpus: Dict[Any, str]) -> "EdgeRAG3":
        super().build(corpus)
        assert self.lex is not None
        self.lex.sort_segments()
        src = self._enriched or corpus
        if self.use_pin_wire:
            self._pin_layer = build_pin_wire_layer(src, k=self.pin_wire_k)
            self._hub_inv = build_hub_inverted(self._pin_layer, self.lex._d2i)
        if self.use_geo:
            self._geo = GeometricEnricher(self.lex, src)
            if self._use_continuum_geo:
                from aethos_edge_continuum import attach_continuum

                cont = attach_continuum(self.lex, corpus)
                self._geo.attach_continuum(cont)
                self._continuum = cont
        if self._use_lattice:
            self.lex.attach_lattice_chambers(self._corpus_raw or corpus, use_lattice_enrich=True)
        self._polluter_docs = SCIFACT_POLLUTERS if len(corpus) > 4000 else frozenset()
        self._attach_drift_graph()
        if self.use_v7_lattice != "off":
            self._v7 = V7LatticeLayer(
                self.lex,
                self._enriched or corpus,
                expand_n=self.v7_expand_n,
                va4_lam=self.v7_boost_lam * 0.82,
                grid_lam=self.v7_boost_lam * 0.55,
            )
        return self

    def _ensure_miniverse_trainer(self) -> None:
        if self._miniverse_trainer is None:
            raw = self._corpus_raw or self._enriched
            self._miniverse_trainer = build_miniverse_trainer(raw, self.lex)

    def _ensure_transgressor_store(self) -> None:
        if self._transgressor_store is None and self.use_transgressor_store != "off":
            raw = self._corpus_raw or self._enriched
            try:
                self._transgressor_store, self._transgressor_sem, self._transgressor_reg = (
                    build_transgressor_miniverse_store(raw, self.lex)
                )
            except Exception:
                self._transgressor_store = None

    def _attach_drift_graph(self) -> None:
        if self.use_drift_pool == "off":
            return
        from _fast_tok import words
        from _o1_drift import build_drift_fast

        seeds = {w for t in self._enriched.values() for w in words(t)}
        self._drift_pool, _, _ = build_drift_fast(self._enriched, seeds, mode="npmi")

    def learn(self, queries, train_qrels, tiers=("bridged",)) -> "EdgeRAG3":
        if "bridged" in tiers:
            if self.use_error_bridge_promotion != "off":
                learn_bridges_error_driven(
                    self.lex,
                    queries,
                    train_qrels,
                    self._enriched,
                    top_per=self.bridge_top_per,
                    pool_depth=min(self.pool_depth, self.lex.N),
                    expand_n=self.bridge_pool_n,
                )
            else:
                self.lex.learn_bridges(
                    queries, train_qrels, self._enriched, top_per=self.bridge_top_per
                )
        self._meet3_log = self.lex.fit_tier_governor(queries, train_qrels)
        self._govern(queries, train_qrels)
        self._govern_drift_pool(queries, train_qrels)
        self._govern_transgressor_store(queries, train_qrels)
        self._govern_miniverse_pool(queries, train_qrels)
        self._govern_v7_lattice(queries, train_qrels)
        if self._v7 is not None and self._v7_on:
            self._v7.learn(queries, train_qrels, self._enriched)
        self._govern_error_bridges(queries, train_qrels)
        self._govern_tier_stack_rerank(queries, train_qrels)
        self._govern_revolutionary(queries, train_qrels)
        return self

    def _govern_tier_stack_rerank(self, queries, train_qrels) -> None:
        """Governor for §5.9.4 tier-stack precision dial at fuse time."""
        if self.use_tier_stack_rerank == "off":
            self._tier_stack_on = False
            return
        if self.use_tier_stack_rerank == "on":
            self._tier_stack_on = True
            return

        from scripts.bench_supervised_bridges import ndcg10

        ids = self._train_ids(queries, train_qrels)[:35]
        if not ids:
            self._tier_stack_on = False
            return

        def mean_nd(on: bool) -> float:
            self._tier_stack_on = on
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base, exp = mean_nd(False), mean_nd(True)
        self._tier_stack_on = exp >= base + self.margin

    def _govern_error_bridges(self, queries, train_qrels) -> None:
        """Governor for §5.29 error-driven bridge promotion (bridged tier only)."""
        if self.use_error_bridge_promotion == "off":
            self._error_bridge_on = False
            return
        if self.use_error_bridge_promotion == "on":
            self._error_bridge_on = getattr(self.lex, "_bridge_error_promoted", False)
            return
        if not getattr(self.lex, "_bridge_error_promoted", False):
            self._error_bridge_on = False
            return

        from scripts.bench_supervised_bridges import ndcg10

        ids = self._train_ids(queries, train_qrels)[:35]
        if not ids:
            self._error_bridge_on = False
            restore_base_bridges(self.lex)
            return

        promoted = deepcopy(self.lex.bridge)
        promoted_corridor = deepcopy(getattr(self.lex, "corridor_bridge", {}))

        def mean_nd(use_promoted: bool) -> float:
            if use_promoted:
                self.lex.bridge = promoted
                self.lex.corridor_bridge = promoted_corridor
            else:
                restore_base_bridges(self.lex)
            self._error_bridge_on = use_promoted
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base, exp = mean_nd(False), mean_nd(True)
        if exp >= base + self.margin:
            self.lex.bridge = promoted
            self.lex.corridor_bridge = promoted_corridor
            self._error_bridge_on = True
        else:
            restore_base_bridges(self.lex)
            self._error_bridge_on = False

    def _govern_v7_lattice(self, queries, train_qrels) -> None:
        if self.use_v7_lattice == "off" or self._v7 is None:
            self._v7_on = False
            return
        if self.use_v7_lattice == "on":
            self._v7_on = True
            return
        from scripts.bench_supervised_bridges import ndcg10

        ids = self._train_ids(queries, train_qrels)[:30]
        if not ids:
            self._v7_on = False
            return

        def mean_nd(on: bool) -> float:
            self._v7_on = on
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base, exp = mean_nd(False), mean_nd(True)
        self._v7_on = exp >= base - 1e-6

    def _govern_transgressor_store(self, queries, train_qrels) -> None:
        """Governor for full N-line TransgressorMiniverseStore pool expand."""
        if self.use_transgressor_store == "off":
            self._transgressor_store_on = False
            return
        if self.use_transgressor_store == "on":
            self._ensure_transgressor_store()
            self._transgressor_store_on = self._transgressor_store is not None
            return
        from scripts.bench_supervised_bridges import ndcg10

        self._ensure_transgressor_store()
        if self._transgressor_store is None:
            self._transgressor_store_on = False
            return

        ids = self._train_ids(queries, train_qrels)[:30]
        if not ids:
            self._transgressor_store_on = False
            return

        def mean_nd(on: bool) -> float:
            self._transgressor_store_on = on
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base, exp = mean_nd(False), mean_nd(True)
        self._transgressor_store_on = exp >= base - 1e-6

    def _govern_miniverse_pool(self, queries, train_qrels) -> None:
        """Governor for transgressor meets + miniverse fusion pool expand."""
        from scripts.bench_supervised_bridges import ndcg10

        def _set_flags(mv: bool, tr: bool) -> None:
            self._miniverse_on = mv
            self._transgressor_meet_on = tr

        forced_tr = self.use_transgressor_meet == "on"
        forced_mv = self.use_miniverse_pool == "on"
        if self.use_miniverse_pool == "off" and self.use_transgressor_meet == "off":
            _set_flags(False, False)
            return
        if forced_mv:
            self._miniverse_on = True
        if forced_tr:
            self._transgressor_meet_on = True
        if not (
            (self.use_miniverse_pool == "govern" and not forced_mv)
            or (self.use_transgressor_meet == "govern" and not forced_tr)
        ):
            return

        ids = self._train_ids(queries, train_qrels)[:40]
        if not ids:
            _set_flags(False, False)
            return

        def mean_nd(mv: bool, tr: bool) -> float:
            _set_flags(forced_mv or mv, forced_tr or tr)
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base = mean_nd(False, False)
        both = mean_nd(True, True)
        if both >= base - 1e-6:
            _set_flags(True, True)
        else:
            tr_only = mean_nd(False, True)
            mv_only = mean_nd(True, False)
            if max(tr_only, mv_only) >= base - 1e-6:
                _set_flags(mv_only >= tr_only, tr_only > mv_only)
            else:
                _set_flags(False, False)

    def _govern_drift_pool(self, queries, train_qrels) -> None:
        if self.use_drift_pool == "off" or not self._drift_pool:
            self._drift_pool_on = False
            return
        if self.use_drift_pool == "on":
            self._drift_pool_on = True
            return
        from scripts.bench_supervised_bridges import ndcg10

        ids = self._train_ids(queries, train_qrels)[:60]
        if not ids:
            self._drift_pool_on = False
            return

        def mean_nd(on: bool) -> float:
            self._drift_pool_on = on
            return sum(
                ndcg10(self._search_revolutionary(queries[q], 10), train_qrels[q]) for q in ids
            ) / len(ids)

        base, exp = mean_nd(False), mean_nd(True)
        self._drift_pool_on = exp >= base + self.drift_margin

    def _train_ids(self, queries, train_qrels) -> list:
        return [
            q
            for q in train_qrels
            if q in queries
            and any(train_qrels[q].get(d, 0) > 0 and d in self.lex._d2i for d in train_qrels[q])
        ]

    def _govern_revolutionary(self, queries, train_qrels, k: int = 10) -> None:
        from scripts.bench_supervised_bridges import ndcg10

        ids = self._train_ids(queries, train_qrels)[:40]
        if not ids:
            self._rev_enabled = False
            self._rev_log = "revolutionary: no train qrels — OFF"
            return

        base = sum(
            ndcg10(self.lex.search_bridged(queries[q], k), train_qrels[q]) for q in ids
        ) / len(ids)
        rev = sum(
            ndcg10(self._search_revolutionary(queries[q], k), train_qrels[q]) for q in ids
        ) / len(ids)
        self._rev_enabled = rev >= base - 1e-6
        if self._rev_enabled and rev > base + self.margin:
            self._auto_tier = "revolutionary"
        self._rev_log = (
            f"revolutionary: bridged={base:.4f} rev={rev:.4f} drift={'ON' if self._drift_pool_on else 'OFF'} "
            f"pin_wire={'ON' if self.use_pin_wire else 'OFF'} meet3={getattr(self.lex, '_use_meet3', False)} "
            f"inter_miniverse={'ON' if self.use_intersection_miniverse else 'OFF'} "
            f"transgressor={'ON' if self._transgressor_meet_on else 'OFF'} "
            f"tmv_store={'ON' if self._transgressor_store_on else 'OFF'} "
            f"miniverse={'ON' if self._miniverse_on else 'OFF'} "
            f"v7={'ON' if self._v7_on else 'OFF'} "
            f"err_bridge={'ON' if self._error_bridge_on else 'OFF'} "
            f"tier_stack={'ON' if self._tier_stack_on else 'OFF'} -> "
            f"{'ENABLED' if self._rev_enabled else 'held'}"
        )

    def _deep_lex_pool(self, query: str) -> np.ndarray:
        depth = min(self.pool_depth, self.lex.N)
        q = " ".join(prune_query_terms(self.lex, query)) or query
        if getattr(self.lex, "_use_meet3", False):
            return np.asarray(self.lex._lex_topk_idx_expanded(q, depth), np.int64)
        return np.asarray(self.lex._lex_topk_idx(q, depth), np.int64)

    def _search_revolutionary(self, query: str, k: int = 10) -> List[Any]:
        """Full revolutionary stack: deep pool → bridge expand → drift expand → geo enrich → fuse."""
        lex = self.lex
        cand = self._deep_lex_pool(query)
        if cand.size == 0:
            return []

        cand = pool_expand_bridges(lex, query, cand, expand_n=self.bridge_pool_n, tf_weighted=self._use_tf_bridges)
        if self._error_bridge_on:
            cand = pool_expand_corridor_bridge(
                lex, query, cand, expand_n=max(18, self.bridge_pool_n // 2),
            )
        if self.use_intersection_miniverse:
            cand = pool_expand_intersection_miniverse(
                lex,
                query,
                cand,
                self._enriched or self._corpus_raw,
                expand_n=self.transgressor_store_expand_n,
                idf_gate=self.miniverse_idf_gate,
            )
        if self._transgressor_store_on:
            self._ensure_transgressor_store()
        if self._transgressor_store_on and self._transgressor_store is not None:
            cand = pool_expand_transgressor_store(
                lex,
                query,
                cand,
                self._transgressor_store,
                self._transgressor_sem,
                self._transgressor_reg,
                expand_n=self.transgressor_store_expand_n,
            )
        elif self._transgressor_meet_on:
            cand = transgressor_meet_expand(
                lex,
                query,
                cand,
                expand_n=self.transgressor_expand_n,
                idf_gate=self.miniverse_idf_gate,
            )
        if self._miniverse_on:
            self._ensure_miniverse_trainer()
        if self._miniverse_on and self._miniverse_trainer is not None:
            cand = pool_expand_miniverse_fusion(
                lex,
                query,
                cand,
                self._miniverse_trainer,
                expand_n=self.miniverse_expand_n,
                idf_gate=self.miniverse_idf_gate,
            )
        if self._hub_inv and self.use_pin_wire:
            cand = pool_expand_pin_wire(
                lex, query, cand, self._hub_inv, expand_n=self.pin_wire_expand_n
            )
        if self._drift_pool_on and self._drift_pool:
            cand = pool_expand_drift(lex, query, cand, self._drift_pool, expand_n=self.drift_pool_n)
        if self._v7_on and self._v7 is not None:
            cand = self._v7.expand_pool(query, cand)

        if self._geo is not None and self.use_geo:
            cand = self._geo.enrich_candidate_idx(
                query,
                cand,
                max_pool=self.max_pool,
                use_continuum=self._use_continuum_geo,
            )

        cand = np.asarray(cand, np.int64)
        lex_sc = lex._lex_scores(query, cand)
        bs = (
            bridge_scores_tf_vec(lex, query, cand)
            if self._use_tf_bridges
            else (
                lex._bridge_scores_vec(query, cand)
                if hasattr(lex, "bridge") and lex.bridge
                else np.zeros(cand.size)
            )
        )
        geo_sc = None
        if self._geo is not None and self.use_geo and self.geo_lam > 0:
            geo_sc = self._geo.geo_boost_vec(query, cand)
        v7_sc = None
        if self._v7_on and self._v7 is not None and self.v7_boost_lam > 0:
            v7_sc = self._v7.boost_vec(query, cand)
        tier_sc = None
        if self._tier_stack_on and self.tier_stack_lam > 0:
            tier_sc = tier_stack_precision_vec(
                lex,
                query,
                cand,
                depth=self.tier_stack_depth,
                idf_gate=self.miniverse_idf_gate,
            )

        final = fuse_scores_with_rerank(
            lex,
            query,
            cand,
            lex_sc,
            bs,
            lam_bridge=self.lam_bridge,
            geo_sc=geo_sc,
            geo_lam=self.geo_lam,
            extra_sc=v7_sc,
            extra_lam=self.v7_boost_lam if v7_sc is not None else 0.0,
            tier_sc=tier_sc,
            tier_lam=self.tier_stack_lam if tier_sc is not None else 0.0,
            polluter_docs=self._polluter_docs if self.use_demotion else None,
            polluter_penalty=self.polluter_penalty,
            lex_bridge_lam=self.lex_bridge_lam,
            use_demotion=self.use_demotion,
            use_lex_bridge_rerank=self.use_lex_bridge_rerank,
        )
        ordered = cand[np.argsort(final)[::-1]]

        return topk_from_ordered(ordered, lex, k)

    def footprint_report(self) -> str:
        codec = self.bytes_per_doc()
        hub = layer_bytes_per_doc(self._pin_layer) if self._pin_layer else 0.0
        return f"postings={codec:.1f} B/doc + pin_wire={hub:.1f} B/doc = {codec + hub:.1f} B/doc total"

    def retrieve(self, query: str, k: int = 10, tier: str = "auto") -> List[Any]:
        if tier in REVOLUTIONARY_LEDGER:
            raise ValueError(f"tier '{tier}' blocked by elimination ledger")
        if tier == "revolutionary":
            return self._search_revolutionary(query, k)
        if tier == "auto":
            if self._rev_enabled and self._auto_tier == "revolutionary":
                return self._search_revolutionary(query, k)
        return super().retrieve(query, k, tier=tier)


def _proof(name: str = "scifact") -> None:
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    from scripts.bench_supervised_bridges import load, ndcg10, recall10

    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    print("=" * 96)
    print(f"EdgeRAG-3 REVOLUTIONARY — {name}: {len(corpus):,} docs, {len(test_ids)} test q")
    print("=" * 96)

    def _run(label: str, fn):
        lat: List[float] = []
        nd = rc = 0.0
        for qid in test_ids:
            t0 = time.perf_counter()
            r = fn(queries[qid])
            lat.append((time.perf_counter() - t0) * 1000)
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        med = float(np.median(lat))
        print(f"  {label:<40} nDCG {nd/n:.4f}  R@10 {rc/n:.4f}  {med:.2f}ms")
        return nd / n, med

    eng2 = EdgeRAG2(stem=True, bake_M=16).build(corpus)
    eng2.learn(queries, train_q)
    b_nd, _ = _run("EdgeRAG2 bridged", lambda q: eng2.retrieve(q, 10, tier="bridged"))

    eng3 = EdgeRAG3(stem=True, bake_M=16).build(corpus)
    eng3.learn(queries, train_q)
    r_nd, _ = _run("EdgeRAG3 revolutionary", lambda q: eng3.retrieve(q, 10, tier="revolutionary"))
    print(f"\n  {eng3._rev_log}")
    print(f"  Δ revolutionary vs bridged: {r_nd - b_nd:+.4f} nDCG")
    print(f"  Footprint: {eng3.footprint_report()}")
    print("=" * 96)


if __name__ == "__main__":
    import sys

    _proof(sys.argv[1] if len(sys.argv) > 1 else "scifact")
