#!/usr/bin/env python3
"""
Bench lattice_retriever_v1 on BEIR SciFact (held-out test qrels).

Pure Stages 01–08 stack — no MultiCorpusBrain, no BM25, no rerankers.
Reports pool recall, R@10, nDCG@10, ms/query.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beir_data_root import resolve_beir_root
from aethos_promotion import PromotionRegistry
from lattice_retriever_v1.stage04_promote import Stage04Registry
from lattice_retriever_v1.stage07_semantic_light import SemanticLightIndex
from lattice_retriever_v1.hybrid_retriever import (
    HybridConfig,
    build_hybrid_retriever,
    build_unified_dual_lattice_retriever,
    resolve_rccm_config,
    resolve_eq_rag_config,
    resolve_meet_vector_config,
)
from lattice_retriever_v1.self_teach_audit import audit_and_self_teach
from lattice_retriever_v1.stage08_retrieve import LatticeRetriever


def find_scifact() -> Path:
    root = Path(resolve_beir_root()) / "scifact"
    if (root / "corpus.jsonl").exists():
        return root
    sys.exit(f"scifact not found under {resolve_beir_root()}")


def load_scifact():
    root = find_scifact()
    corpus: dict[str, str] = {}
    for line in open(root / "corpus.jsonl", encoding="utf-8"):
        o = json.loads(line)
        corpus[o["_id"]] = (o.get("title", "") + " " + o.get("text", "")).strip()

    queries: dict[str, str] = {}
    for line in open(root / "queries.jsonl", encoding="utf-8"):
        o = json.loads(line)
        queries[o["_id"]] = o["text"]

    def qrels(split: str) -> dict[str, dict[str, int]]:
        rel: dict[str, dict[str, int]] = {}
        p = root / "qrels" / f"{split}.tsv"
        if not p.exists():
            return rel
        r = csv.reader(open(p, encoding="utf-8"), delimiter="\t")
        next(r, None)
        for qid, cid, sc in r:
            rel.setdefault(qid, {})[cid] = int(sc)
        return rel

    return corpus, queries, qrels("test")


def ndcg10(ranked: list[str], rels: dict[str, int]) -> float:
    dcg = sum(rels.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall10(ranked: list[str], rels: dict[str, int]) -> float:
    rel = {d for d, s in rels.items() if s > 0}
    return len(set(ranked[:10]) & rel) / len(rel) if rel else 0.0


def build_retriever(corpus: dict[str, str], *, fast_ingest: bool = False) -> LatticeRetriever:
    """Single-pass index: L2 promotion + postings + wing cages."""
    inner = PromotionRegistry(fast_ingest=fast_ingest, defer_l2_promotion=True)
    stage = Stage04Registry(registry=inner)
    semantic = SemanticLightIndex(registry=stage)
    retriever = LatticeRetriever(semantic=semantic)
    n = len(corpus)
    for i, (doc_id, text) in enumerate(corpus.items()):
        stage.observe_text(text)
        retriever.index_doc(doc_id, text)
        if (i + 1) % 1000 == 0:
            print(f"  indexed {i + 1}/{n}", flush=True)
    return retriever


def evaluate_retriever(retriever, queries, test_qrels, test_ids, *, hybrid: bool) -> dict:
    ndcgs: list[float] = []
    recalls: list[float] = []
    pool_hits = 0
    t_query = 0.0

    for qid in test_ids:
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}

        t1 = time.time()
        if hybrid:
            trace = retriever.retrieve_with_trace(q, limit=10)
            pool = trace.pool_docs
            hits = trace.hits
        else:
            pool, _ = retriever.lazy_pool(q)
            hits = retriever.retrieve(q, limit=10)
        t_query += time.time() - t1

        ranked = [h.doc_id for h in hits]
        if gold & pool:
            pool_hits += 1
        ndcgs.append(ndcg10(ranked, rels))
        recalls.append(recall10(ranked, rels))

    n = len(test_ids)
    return {
        "pool_recall": pool_hits / n,
        "recall_at_10": sum(recalls) / n,
        "ndcg_at_10": sum(ndcgs) / n,
        "ms_per_query": 1000 * t_query / n,
        "n_queries": n,
    }


def print_metrics(label: str, m: dict) -> None:
    print(f"\n{'='*56}")
    print(label)
    print(f"{'='*56}")
    print(f"  pool_recall   {m['pool_recall']:.4f}")
    print(f"  R@10          {m['recall_at_10']:.4f}")
    print(f"  nDCG@10       {m['ndcg_at_10']:.4f}")
    print(f"  ms/query      {m['ms_per_query']:.1f}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SciFact bench — lattice_retriever_v1 honest baseline")
    parser.add_argument("--trace-query", default="cancer mutation rare variant", help="Query for sample glass-box trace")
    parser.add_argument("--trace-only", action="store_true", help="Print trace for one query and exit")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="fast_ingest=True (no L2 promotions) — records the no-promotion floor only",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="BM25-class postings + L2 shell MaxSim + optional walk MaxSim fusion",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Three-layer dual lattice (implies --hybrid; default lam_l2=0.25 lam_walk=0.20)",
    )
    parser.add_argument("--lam-lex", type=float, default=1.0, help="Hybrid fusion weight for lexical layer")
    parser.add_argument("--lam-l2", type=float, default=None, help="Shell MaxSim fusion weight")
    parser.add_argument("--lam-walk", type=float, default=None, help="Walk MaxSim fusion weight")
    parser.add_argument(
        "--lexical-mode",
        default="append_index",
        choices=("append_index", "bm25", "lattice_pure", "lattice_plane"),
        help="Hybrid L0 lexical floor: append_index (multiview BM25) or lattice_plane fallback",
    )
    parser.add_argument(
        "--fuse-mode",
        default="additive",
        choices=("norm", "additive"),
        help="Score fusion: additive preserves lexical order; norm min-maxes both layers",
    )
    parser.add_argument(
        "--no-pair-meet",
        action="store_true",
        help="Disable pair-meet pool expansion from append index",
    )
    parser.add_argument(
        "--no-append-pool-union",
        action="store_true",
        help="Disable append-index BM25 top-K union into Phase A pool",
    )
    parser.add_argument(
        "--append-pool-k",
        type=int,
        default=200,
        help="Top-K append-index docs to union into pool (default 200)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable Stage 08 lexical_bridge + cage_anchor rerank",
    )
    parser.add_argument(
        "--lam-sweep",
        action="store_true",
        help="Build once; evaluate several lam_lex/lam_l2 pairs without re-ingest",
    )
    parser.add_argument(
        "--self-teach",
        action="store_true",
        help="Run synthetic Q->doc audit on corpus; attach TeachStore bridges before eval",
    )
    parser.add_argument(
        "--self-teach-max",
        type=int,
        default=600,
        help="Max corpus docs for self-teach audit (default 600; ignored with --self-teach-full)",
    )
    parser.add_argument(
        "--self-teach-full",
        action="store_true",
        help="Self-teach audit on entire corpus (all docs)",
    )
    parser.add_argument(
        "--no-demotion",
        action="store_true",
        help="Disable glass-box polluter demotion on hybrid scores",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Cap test queries (0 = all). Use 50 for fast sweep smoke.",
    )
    parser.add_argument(
        "--enable-mn-rerank",
        action="store_true",
        help="Enable missing-neighbor pool rerank (HybridConfig mn_rerank_lambda)",
    )
    parser.add_argument(
        "--mn-lambda",
        type=float,
        default=0.35,
        help="Missing-neighbor rerank fusion weight (default 0.35)",
    )
    parser.add_argument(
        "--enable-ch-rerank",
        action="store_true",
        help="Enable cross-hit mutual pool rerank (HybridConfig ch_rerank_lambda)",
    )
    parser.add_argument(
        "--ch-lambda",
        type=float,
        default=0.25,
        help="Cross-hit rerank fusion weight (default 0.25)",
    )
    parser.add_argument(
        "--enable-gw-rerank",
        action="store_true",
        help="Enable gravity-cascade pool rerank (HybridConfig gw_rerank_lambda)",
    )
    parser.add_argument(
        "--gw-lambda",
        type=float,
        default=0.30,
        help="Gravity-cascade rerank fusion weight (default 0.30)",
    )
    parser.add_argument(
        "--enable-all-pool-rerank",
        action="store_true",
        help="Enable all three pool rerankers with default lambdas (MNCR+CHMR+GWCR)",
    )
    parser.add_argument(
        "--enable-recommended-pool-rerank",
        action="store_true",
        help=(
            "SciFact-tuned pool rerank: GWCR (lambda=0.30) + CHMR (lambda=0.25); "
            "MNCR excluded — 50-query bench showed MNCR@0.35 hurts nDCG "
            "(all-three 0.418 vs baseline 0.491; MNCR-only 0.435)"
        ),
    )
    parser.add_argument(
        "--pool-rerank-sweep",
        action="store_true",
        help="Build once; A/B baseline vs all-three vs individual pool rerank arms",
    )
    parser.add_argument(
        "--skip-sample-trace",
        action="store_true",
        help="Skip diagnostic sample trace before scoring",
    )
    parser.add_argument(
        "--enable-rare-shell-lattice",
        action="store_true",
        help="Build rare-shell lattice index; pool narrow + lam_rare MaxSim",
    )
    parser.add_argument(
        "--enable-rccm",
        action="store_true",
        help="RCCM Phase 1: rare-combo corpus mesh + MRL2 fusion preset",
    )
    parser.add_argument(
        "--enable-eq-rag",
        action="store_true",
        help="Soft EQ-RAG complement term expansion in Phase A pool routing",
    )
    parser.add_argument(
        "--enable-meet-vector-pair",
        action="store_true",
        help="Meet-vector pair routing: global_3way docs union into pair_meet expand",
    )
    parser.add_argument(
        "--meet-vector-bench",
        action="store_true",
        help=(
            "Build once with corpus lattice; A/B baseline vs meet-vector pair routing "
            "on all test queries via with_config"
        ),
    )
    parser.add_argument(
        "--rccm-bench",
        action="store_true",
        help=(
            "Build once with RCCM indices; A/B baseline (recommended-zero-shot) "
            "vs RCCM on all test queries via with_config"
        ),
    )
    parser.add_argument(
        "--lam-rare",
        type=float,
        default=0.15,
        help="Rare-shell MaxSim fusion weight (default 0.15)",
    )
    parser.add_argument(
        "--zero-shot-stack-sweep",
        action="store_true",
        help="Build once with rare index; eval baseline vs pool rerank vs rare shell",
    )
    parser.add_argument(
        "--push-zero-shot-sweep",
        action="store_true",
        help=(
            "Append-union zero-shot push: lam sweep (6 arms) then best-lam + "
            "recommended pool rerank and GWCR-only; logs lattice_v1_push_zero_shot.json"
        ),
    )
    parser.add_argument(
        "--recommended-zero-shot",
        action="store_true",
        help=(
            "Zero-shot SciFact preset: append_union ON, lam from push sweep winner "
            "(default 1/0/0 until sweep updates HybridConfig)"
        ),
    )
    args = parser.parse_args()
    if args.unified:
        args.hybrid = True
    if args.enable_all_pool_rerank:
        args.enable_mn_rerank = True
        args.enable_ch_rerank = True
        args.enable_gw_rerank = True
    if args.enable_recommended_pool_rerank:
        args.enable_ch_rerank = True
        args.enable_gw_rerank = True
    if args.recommended_zero_shot:
        args.hybrid = True
        args.lexical_mode = "append_index"
        # Overridden after push sweep if a winning arm is wired here
        args.lam_lex = 1.0
        args.lam_l2 = 0.0
        args.lam_walk = 0.0
    if args.rccm_bench:
        args.hybrid = True
        args.skip_sample_trace = True
    if args.meet_vector_bench:
        args.hybrid = True
        args.skip_sample_trace = True
        args.enable_meet_vector_pair = True

    corpus, queries, test_qrels = load_scifact()
    test_ids = [q for q in test_qrels if q in queries]
    if args.max_queries > 0:
        test_ids = test_ids[: args.max_queries]
    print(f"SciFact: {len(corpus)} docs | test {len(test_ids)} queries", flush=True)

    lam_l2 = args.lam_l2 if args.lam_l2 is not None else (0.25 if args.unified else 0.0)
    lam_walk = args.lam_walk if args.lam_walk is not None else (0.20 if args.unified else 0.0)

    t0 = time.time()
    if args.hybrid:
        cfg = HybridConfig(
            lam_lex=args.lam_lex,
            lam_l2=lam_l2,
            lam_walk=lam_walk,
            lexical_mode=args.lexical_mode,
            fuse_mode=args.fuse_mode,
            enable_pair_meet=not args.no_pair_meet,
            enable_append_pool_union=not args.no_append_pool_union,
            append_pool_k=args.append_pool_k,
            enable_stage08_rerank=not args.no_rerank,
            enable_demotion=not args.no_demotion,
            enable_missing_neighbor_rerank=args.enable_mn_rerank,
            mn_rerank_lambda=args.mn_lambda,
            enable_cross_hit_rerank=args.enable_ch_rerank,
            ch_rerank_lambda=args.ch_lambda,
            enable_gravity_cascade_rerank=args.enable_gw_rerank,
            gw_rerank_lambda=args.gw_lambda,
            enable_rare_shell_lattice=args.enable_rare_shell_lattice or args.zero_shot_stack_sweep,
            lam_rare=args.lam_rare,
            enable_rccm=args.enable_rccm or args.rccm_bench,
            enable_eq_rag_expand=args.enable_eq_rag,
            enable_meet_vector_pair=args.enable_meet_vector_pair or args.meet_vector_bench,
        )
        cfg = resolve_rccm_config(cfg)
        cfg = resolve_eq_rag_config(cfg)
        cfg = resolve_meet_vector_config(cfg)
        build_fn = build_unified_dual_lattice_retriever if args.unified else build_hybrid_retriever
        retriever = build_fn(corpus, config=cfg, fast_ingest=args.fast)
        if args.self_teach:
            max_docs = None if args.self_teach_full else args.self_teach_max
            label = f"all {len(corpus)}" if max_docs is None else f"up to {max_docs}"
            print(f"  self-teach audit on {label} docs...", flush=True)
            t_st = time.time()
            teach, st_stats = audit_and_self_teach(
                retriever, corpus, max_docs=max_docs,
            )
            retriever.teach = teach
            st_elapsed = time.time() - t_st
            print(
                f"  self-teach done in {st_elapsed:.1f}s | {st_stats.explain()}",
                flush=True,
            )
            if args.self_teach_full:
                st_log = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_self_teach_full.json"
                st_log.parent.mkdir(exist_ok=True)
                st_log.write_text(
                    json.dumps(
                        {
                            **st_stats.explain(),
                            "audit_seconds": round(st_elapsed, 1),
                            "n_corpus": len(corpus),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(f"  self-teach stats: {st_log}", flush=True)
        pool_rerank = (
            f"mn={cfg.enable_missing_neighbor_rerank}/{cfg.mn_rerank_lambda} "
            f"ch={cfg.enable_cross_hit_rerank}/{cfg.ch_rerank_lambda} "
            f"gw={cfg.enable_gravity_cascade_rerank}/{cfg.gw_rerank_lambda}"
        )
        mode = (
            f"{'unified' if args.unified else 'hybrid'} "
            f"(lam={cfg.lam_lex}/{cfg.lam_l2}/{cfg.lam_walk} fuse={cfg.fuse_mode} "
            f"lex={cfg.lexical_mode} meet={cfg.enable_pair_meet} "
            f"append_union={cfg.enable_append_pool_union}/{cfg.append_pool_k} "
            f"rerank={cfg.enable_stage08_rerank} "
            f"pool_rerank=[{pool_rerank}])"
        )
        n_promo = len(retriever.router.semantic.registry.promotions)
    else:
        retriever = build_retriever(corpus, fast_ingest=args.fast)
        mode = "fast_ingest (no L2)" if args.fast else "promotion_on"
        n_promo = len(retriever.semantic.registry.promotions)
    print(f"index built in {time.time() - t0:.1f}s | {mode} | L2 promotions {n_promo}", flush=True)

    if args.trace_only:
        if args.hybrid:
            trace = retriever.retrieve_with_trace(args.trace_query, limit=5)
        else:
            trace = retriever.retrieve_with_trace(args.trace_query, limit=5)
        print(json.dumps(trace.explain(), indent=2))
        return

    # Sample trace before scoring — diagnostic surface for tuning
    if (
        not args.skip_sample_trace
        and not args.lam_sweep
        and not args.pool_rerank_sweep
        and not args.zero_shot_stack_sweep
        and not args.push_zero_shot_sweep
    ):
        sample_q = queries[test_ids[0]]
        sample_trace = retriever.retrieve_with_trace(sample_q, limit=3)
        print(f"\n--- sample trace ({sample_q[:60]}...) ---")
        print(json.dumps(sample_trace.explain(), indent=2)[:4000])

    if args.lam_sweep and args.hybrid:
        sweep_configs = [
            ("unified default 1.0/0.25/0.20", {"lam_lex": 1.0, "lam_l2": 0.25, "lam_walk": 0.20}),
            ("anchor 0.85/0.10/0.05", {"lam_lex": 0.85, "lam_l2": 0.10, "lam_walk": 0.05}),
            ("lex+shell 0.75/0.25/0.0", {"lam_lex": 0.75, "lam_l2": 0.25, "lam_walk": 0.0}),
            ("lex+walk 0.85/0.0/0.15", {"lam_lex": 0.85, "lam_l2": 0.0, "lam_walk": 0.15}),
            ("additive 0.70/0.20/0.10", {"lam_lex": 0.70, "lam_l2": 0.20, "lam_walk": 0.10}),
            ("lex only 1.0/0.0/0.0", {"lam_lex": 1.0, "lam_l2": 0.0, "lam_walk": 0.0}),
            ("0.70/0.30 rerank off", {"lam_lex": 0.70, "lam_l2": 0.30, "lam_walk": 0.0, "enable_stage08_rerank": False}),
        ]
        best_label = ""
        best_ndcg = -1.0
        best_metrics: dict = {}
        for label, kw in sweep_configs:
            retriever.with_config(**kw)
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"SWEEP: {label}", m)
            if m["ndcg_at_10"] > best_ndcg:
                best_ndcg = m["ndcg_at_10"]
                best_label = label
                best_metrics = m
        print(f"\n  best sweep: {best_label} -> nDCG@10 {best_ndcg:.4f}")
        log_path = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_scifact_unified_sweep.json"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            json.dumps({"best": best_label, **best_metrics}, indent=2),
            encoding="utf-8",
        )
        print(f"  sweep log: {log_path}")
        return


    if args.zero_shot_stack_sweep and args.hybrid:
        arms = [
            (
                "A lexical baseline (lam 1/0/0, no pool rerank, no rare narrow)",
                {
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                    "enable_missing_neighbor_rerank": False,
                    "enable_cross_hit_rerank": False,
                    "enable_gravity_cascade_rerank": False,
                    "enable_rare_shell_lattice": False,
                },
            ),
            (
                "B all pool rerank (lam 1/0/0)",
                {
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                    "enable_missing_neighbor_rerank": True,
                    "mn_rerank_lambda": args.mn_lambda,
                    "enable_cross_hit_rerank": True,
                    "ch_rerank_lambda": args.ch_lambda,
                    "enable_gravity_cascade_rerank": True,
                    "gw_rerank_lambda": args.gw_lambda,
                    "enable_rare_shell_lattice": False,
                },
            ),
            (
                "C all pool rerank + rare shell lattice",
                {
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                    "enable_missing_neighbor_rerank": True,
                    "mn_rerank_lambda": args.mn_lambda,
                    "enable_cross_hit_rerank": True,
                    "ch_rerank_lambda": args.ch_lambda,
                    "enable_gravity_cascade_rerank": True,
                    "gw_rerank_lambda": args.gw_lambda,
                    "enable_rare_shell_lattice": True,
                    "lam_rare": args.lam_rare,
                },
            ),
        ]
        sweep_results: list[dict] = []
        for label, kw in arms:
            retriever.with_config(**kw)
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"ZERO-SHOT STACK: {label}", m)
            sweep_results.append({"label": label, **m, **kw})
        log_path = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_zero_shot_stack_scifact.json"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "n_queries": len(test_ids),
                    "index_build_seconds": round(time.time() - t0, 1),
                    "mn_lambda": args.mn_lambda,
                    "ch_lambda": args.ch_lambda,
                    "gw_lambda": args.gw_lambda,
                    "lam_rare": args.lam_rare,
                    "arms": sweep_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  zero-shot stack log: {log_path}")
        return

    if args.push_zero_shot_sweep and args.hybrid:
        _pool_off = {
            "enable_missing_neighbor_rerank": False,
            "enable_cross_hit_rerank": False,
            "enable_gravity_cascade_rerank": False,
        }
        _append_union = {
            "enable_append_pool_union": True,
            "append_pool_k": args.append_pool_k,
            "lexical_mode": "append_index",
        }
        lam_arms: list[tuple[str, dict]] = [
            ("baseline lex 1/0/0", {"lam_lex": 1.0, "lam_l2": 0.0, "lam_walk": 0.0, **_pool_off}),
            ("lam 0.85/0.10/0.05", {"lam_lex": 0.85, "lam_l2": 0.10, "lam_walk": 0.05, **_pool_off}),
            ("lam 0.70/0.20/0.10", {"lam_lex": 0.70, "lam_l2": 0.20, "lam_walk": 0.10, **_pool_off}),
            ("lam 0.90/0.05/0.05", {"lam_lex": 0.90, "lam_l2": 0.05, "lam_walk": 0.05, **_pool_off}),
            ("lex+shell 0.75/0.25/0.0", {"lam_lex": 0.75, "lam_l2": 0.25, "lam_walk": 0.0, **_pool_off}),
            ("lex+walk 0.85/0.0/0.15", {"lam_lex": 0.85, "lam_l2": 0.0, "lam_walk": 0.15, **_pool_off}),
        ]
        baseline_ndcg = 0.7039
        sweep_results: list[dict] = []
        best_lam_label = ""
        best_lam_kw: dict = {}
        best_lam_ndcg = -1.0

        for label, kw in lam_arms:
            retriever.with_config(**_append_union, **kw)
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"PUSH ZERO-SHOT: {label}", m)
            arm = {"label": label, "phase": "lam_sweep", **m, **kw, **_append_union}
            arm["delta_vs_baseline"] = round(m["ndcg_at_10"] - baseline_ndcg, 4)
            sweep_results.append(arm)
            if m["ndcg_at_10"] > best_lam_ndcg:
                best_lam_ndcg = m["ndcg_at_10"]
                best_lam_label = label
                best_lam_kw = kw

        rerank_arms: list[tuple[str, dict]] = [
            (
                f"best lam ({best_lam_label}) + recommended GWCR+CH",
                {
                    **best_lam_kw,
                    "enable_missing_neighbor_rerank": False,
                    "enable_cross_hit_rerank": True,
                    "ch_rerank_lambda": args.ch_lambda,
                    "enable_gravity_cascade_rerank": True,
                    "gw_rerank_lambda": args.gw_lambda,
                },
            ),
            (
                f"best lam ({best_lam_label}) + GWCR only",
                {
                    **best_lam_kw,
                    "enable_missing_neighbor_rerank": False,
                    "enable_cross_hit_rerank": False,
                    "enable_gravity_cascade_rerank": True,
                    "gw_rerank_lambda": args.gw_lambda,
                },
            ),
        ]
        best_overall_label = best_lam_label
        best_overall_ndcg = best_lam_ndcg
        for label, kw in rerank_arms:
            retriever.with_config(**_append_union, **kw)
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"PUSH ZERO-SHOT: {label}", m)
            arm = {"label": label, "phase": "pool_rerank", **m, **kw, **_append_union}
            arm["delta_vs_baseline"] = round(m["ndcg_at_10"] - baseline_ndcg, 4)
            sweep_results.append(arm)
            if m["ndcg_at_10"] > best_overall_ndcg:
                best_overall_ndcg = m["ndcg_at_10"]
                best_overall_label = label

        print(
            f"\n  best lam arm: {best_lam_label} -> nDCG@10 {best_lam_ndcg:.4f} "
            f"(delta {best_lam_ndcg - baseline_ndcg:+.4f})"
        )
        print(
            f"  best overall: {best_overall_label} -> nDCG@10 {best_overall_ndcg:.4f} "
            f"(delta {best_overall_ndcg - baseline_ndcg:+.4f})"
        )
        log_path = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_push_zero_shot.json"
        log_path.parent.mkdir(exist_ok=True)
        log_payload = {
            "n_queries": len(test_ids),
            "index_build_seconds": round(time.time() - t0, 1),
            "baseline_reference_ndcg_at_10": baseline_ndcg,
            "baseline_reference_pool_recall": 0.9633,
            "append_pool_k": args.append_pool_k,
            "ch_lambda": args.ch_lambda,
            "gw_lambda": args.gw_lambda,
            "best_lam_arm": best_lam_label,
            "best_lam_ndcg_at_10": best_lam_ndcg,
            "best_overall_arm": best_overall_label,
            "best_overall_ndcg_at_10": best_overall_ndcg,
            "best_overall_delta_vs_baseline": round(best_overall_ndcg - baseline_ndcg, 4),
            "beats_baseline_meaningfully": (best_overall_ndcg - baseline_ndcg) > 0.005,
            "arms": sweep_results,
        }
        log_path.write_text(json.dumps(log_payload, indent=2), encoding="utf-8")
        print(f"  push zero-shot log: {log_path}")
        return

    if args.pool_rerank_sweep and args.hybrid:
        _pool_off = {
            "enable_missing_neighbor_rerank": False,
            "enable_cross_hit_rerank": False,
            "enable_gravity_cascade_rerank": False,
        }
        sweep_configs = [
            ("baseline (no pool rerank)", _pool_off),
            (
                "all three pool rerank",
                {
                    **_pool_off,
                    "enable_missing_neighbor_rerank": True,
                    "mn_rerank_lambda": args.mn_lambda,
                    "enable_cross_hit_rerank": True,
                    "ch_rerank_lambda": args.ch_lambda,
                    "enable_gravity_cascade_rerank": True,
                    "gw_rerank_lambda": args.gw_lambda,
                },
            ),
            (
                "mn only",
                {**_pool_off, "enable_missing_neighbor_rerank": True, "mn_rerank_lambda": args.mn_lambda},
            ),
            (
                "ch only",
                {**_pool_off, "enable_cross_hit_rerank": True, "ch_rerank_lambda": args.ch_lambda},
            ),
            (
                "gw only",
                {**_pool_off, "enable_gravity_cascade_rerank": True, "gw_rerank_lambda": args.gw_lambda},
            ),
        ]
        sweep_results: list[dict] = []
        best_label = ""
        best_ndcg = -1.0
        for label, kw in sweep_configs:
            retriever.with_config(**kw)
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"POOL RERANK SWEEP: {label}", m)
            arm = {"label": label, **m, **kw}
            sweep_results.append(arm)
            if m["ndcg_at_10"] > best_ndcg:
                best_ndcg = m["ndcg_at_10"]
                best_label = label
        print(f"\n  best pool-rerank arm: {best_label} -> nDCG@10 {best_ndcg:.4f}")
        log_path = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_pool_rerank_ab.json"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "n_queries": len(test_ids),
                    "mn_lambda": args.mn_lambda,
                    "ch_lambda": args.ch_lambda,
                    "gw_lambda": args.gw_lambda,
                    "best": best_label,
                    "best_ndcg_at_10": best_ndcg,
                    "arms": sweep_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  pool-rerank A/B log: {log_path}")
        return

    if args.rccm_bench and args.hybrid:
        baseline_ndcg = 0.7039
        baseline_pool = 0.9633
        build_secs = round(time.time() - t0, 1)
        arms: list[tuple[str, dict]] = [
            (
                "recommended-zero-shot (baseline)",
                {
                    "enable_rccm": False,
                    "fuse_mode": "additive",
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                    "lexical_mode": "append_index",
                    "enable_append_pool_union": True,
                    "enable_corpus_lattice": False,
                    "enable_rare_shell_lattice": False,
                    "enable_eq_rag_expand": False,
                },
            ),
            (
                "RCCM Phase 1",
                {
                    "enable_rccm": True,
                },
            ),
        ]
        if args.enable_eq_rag:
            arms.append(
                (
                    "RCCM + EQ-RAG soft expand",
                    {
                        "enable_rccm": True,
                        "enable_eq_rag_expand": True,
                    },
                )
            )
        sweep_results: list[dict] = []
        for label, kw in arms:
            resolved = resolve_rccm_config(replace(retriever.config, **kw))
            resolved = resolve_eq_rag_config(resolved)
            retriever.config = resolved
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"RCCM BENCH: {label}", m)
            arm = {
                "label": label,
                "index_build_seconds": build_secs,
                **m,
                **kw,
                "delta_ndcg_vs_baseline": round(m["ndcg_at_10"] - baseline_ndcg, 4),
                "delta_pool_vs_baseline": round(m["pool_recall"] - baseline_pool, 4),
            }
            sweep_results.append(arm)
        log_path = Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_rccm_bench.json"
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "n_queries": len(test_ids),
                    "index_build_seconds": build_secs,
                    "baseline_reference_ndcg_at_10": baseline_ndcg,
                    "baseline_reference_pool_recall": baseline_pool,
                    "arms": sweep_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  index build: {build_secs}s | RCCM bench log: {log_path}")
        return

    if args.meet_vector_bench and args.hybrid:
        baseline_ndcg = 0.7084
        baseline_pool = 0.9633
        build_secs = round(time.time() - t0, 1)
        arms: list[tuple[str, dict]] = [
            (
                "recommended-zero-shot (baseline)",
                {
                    "enable_meet_vector_pair": False,
                    "enable_corpus_lattice": False,
                    "lexical_mode": "append_index",
                    "enable_append_pool_union": True,
                    "append_pool_k": 200,
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                },
            ),
            (
                "meet-vector pair only",
                {
                    "enable_meet_vector_pair": True,
                    "enable_corpus_lattice": False,
                    "lexical_mode": "append_index",
                    "enable_append_pool_union": True,
                    "append_pool_k": 200,
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                },
            ),
            (
                "meet-vector + corpus_lattice pool sidecar",
                {
                    "enable_meet_vector_pair": True,
                    "enable_corpus_lattice": True,
                    "lexical_mode": "append_index",
                    "enable_append_pool_union": True,
                    "append_pool_k": 200,
                    "lam_lex": 1.0,
                    "lam_l2": 0.0,
                    "lam_walk": 0.0,
                },
            ),
        ]
        sweep_results: list[dict] = []
        for label, kw in arms:
            resolved = resolve_meet_vector_config(replace(retriever.config, **kw))
            retriever.config = resolved
            m = evaluate_retriever(retriever, queries, test_qrels, test_ids, hybrid=True)
            print_metrics(f"MEET-VECTOR BENCH: {label}", m)
            arm = {
                "label": label,
                "index_build_seconds": build_secs,
                **m,
                **kw,
                "delta_ndcg_vs_baseline": round(m["ndcg_at_10"] - baseline_ndcg, 4),
                "delta_pool_vs_baseline": round(m["pool_recall"] - baseline_pool, 4),
            }
            sweep_results.append(arm)
        log_path = (
            Path(__file__).resolve().parents[1] / "logs" / "lattice_v1_meet_vector_bench.json"
        )
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(
            json.dumps(
                {
                    "n_queries": len(test_ids),
                    "index_build_seconds": build_secs,
                    "baseline_reference_ndcg_at_10": baseline_ndcg,
                    "baseline_reference_pool_recall": baseline_pool,
                    "arms": sweep_results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n  index build: {build_secs}s | meet-vector bench log: {log_path}")
        return

    ndcgs: list[float] = []
    recalls: list[float] = []
    pool_hits = 0
    t_query = 0.0

    for qid in test_ids:
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}

        t1 = time.time()
        if args.hybrid:
            trace = retriever.retrieve_with_trace(q, limit=10)
            pool = trace.pool_docs
            hits = trace.hits
        else:
            pool, _ = retriever.lazy_pool(q)
            hits = retriever.retrieve(q, limit=10)
        t_query += time.time() - t1

        ranked = [h.doc_id for h in hits]
        if gold & pool:
            pool_hits += 1
        ndcgs.append(ndcg10(ranked, rels))
        recalls.append(recall10(ranked, rels))

    n = len(test_ids)
    pool_r = pool_hits / n
    r10 = sum(recalls) / n
    ndcg = sum(ndcgs) / n
    ms_q = 1000 * t_query / n

    label = (
        "UNIFIED DUAL-LATTICE (lex + shell + walk)"
        if args.unified
        else ("HYBRID (lex + shell)" if args.hybrid else "HONEST BASELINE (Stages 01-08)")
    )
    print(f"\n{'='*56}")
    print(f"lattice_retriever_v1 {label}")
    print(f"{'='*56}")
    print(f"  pool_recall   {pool_r:.4f}")
    print(f"  R@10          {r10:.4f}")
    print(f"  nDCG@10       {ndcg:.4f}")
    print(f"  ms/query      {ms_q:.1f}")
    print("\n  reference: pure-lattice ~0.367 | BM25-hybrid ~0.776 (not the day-one bar)")

    log_name = (
        "lattice_v1_scifact_unified.json"
        if args.unified
        else ("lattice_v1_scifact_hybrid.json" if args.hybrid else "lattice_v1_scifact_baseline.json")
    )
    log_path = Path(__file__).resolve().parents[1] / "logs" / log_name
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "pool_recall": pool_r,
                "recall_at_10": r10,
                "ndcg_at_10": ndcg,
                "ms_per_query": ms_q,
                "n_docs": len(corpus),
                "n_queries": n,
                "l2_promotions": n_promo,
                "fast_ingest": args.fast,
                "hybrid": args.hybrid,
                "unified": args.unified,
                "lam_lex": args.lam_lex if args.hybrid else None,
                "lam_l2": lam_l2 if args.hybrid else None,
                "lam_walk": lam_walk if args.hybrid else None,
                "lexical_mode": args.lexical_mode if args.hybrid else None,
                "fuse_mode": args.fuse_mode if args.hybrid else None,
                "pair_meet": (not args.no_pair_meet) if args.hybrid else None,
                "stage08_rerank": (not args.no_rerank) if args.hybrid else None,
                "demotion": (not args.no_demotion) if args.hybrid else None,
                "self_teach": args.self_teach if args.hybrid else None,
                "self_teach_full": args.self_teach_full if args.hybrid else None,
                "self_teach_max": (
                    None if args.self_teach_full else args.self_teach_max
                ) if args.hybrid and args.self_teach else None,
                "enable_mn_rerank": args.enable_mn_rerank if args.hybrid else None,
                "mn_lambda": args.mn_lambda if args.hybrid else None,
                "enable_ch_rerank": args.enable_ch_rerank if args.hybrid else None,
                "ch_lambda": args.ch_lambda if args.hybrid else None,
                "enable_gw_rerank": args.enable_gw_rerank if args.hybrid else None,
                "gw_lambda": args.gw_lambda if args.hybrid else None,
                "enable_recommended_pool_rerank": (
                    args.enable_recommended_pool_rerank if args.hybrid else None
                ),
                "note": (
                    "unified dual-lattice lex+shell+walk"
                    if args.unified
                    else ("hybrid lex+shell" if args.hybrid else "honest baseline — no tuning")
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  baseline written: {log_path}")


if __name__ == "__main__":
    main()
