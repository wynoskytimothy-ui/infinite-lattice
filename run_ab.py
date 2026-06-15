#!/usr/bin/env python3
"""
SciFact quality A/B with two stages (build once, score twice).

Stage 1 (build): ingest + multi-pass + anchor training → saves:
  - brains/scifact_quality.distilled.json  (promoted primes only)
  - brains/scifact_quality.brain.json      (anchor weights + λ_coord/λ_neighbor)
  - brains/scifact_quality.eval.pkl        (full scoring bundle)

Stage 2 (ab): load .eval.pkl → A/B comparison (scoring only).

Usage:
  python run_ab.py --stage all
  python run_ab.py --stage build
  python run_ab.py --stage ab --ab kappa
  python run_ab.py --stage ab --ab pf
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from beir_data_root import resolve_beir_root
from eval_beir import evaluate_dataset, load_paths, score_from_bundle
from eval_checkpoint import checkpoint_path, load_checkpoint
from core.learning_engine import BadCorrelationStore, bad_correlation_path


def stage_build(
    *,
    dataset: str = "scifact",
    max_docs: int | None = None,
    max_queries: int | None = None,
    max_convergence_rounds: int = 4,
    skip_training: bool = False,
) -> Path:
    root = Path(resolve_beir_root())
    paths = load_paths(root, dataset)
    ckpt = checkpoint_path(dataset, "quality")
    print(f"\n=== Stage 1: BUILD ({dataset}) ===", flush=True)
    print(f"  checkpoint -> {ckpt}", flush=True)
    t0 = time.perf_counter()
    evaluate_dataset(
        paths,
        mode="quality",
        max_docs=max_docs,
        max_queries=max_queries,
        build_only=True,
        save_checkpoint=ckpt,
        max_convergence_rounds=max_convergence_rounds,
        n_passes=2,
        skip_training=skip_training,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Stage 1 done in {elapsed / 60:.1f} min", flush=True)
    return ckpt


def stage_ab_pf(
    *,
    dataset: str = "scifact",
    max_queries: int | None = None,
    progress_every: int = 1,
) -> int:
    ckpt = checkpoint_path(dataset, "quality")
    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}")
        print("Run:  python run_ab.py --stage build")
        return 1

    bundle = load_checkpoint(ckpt)
    if max_queries is not None:
        bundle.qids = bundle.qids[:max_queries]

    bad_store = BadCorrelationStore.load(bad_correlation_path(dataset, "quality"))

    print(f"\n=== Stage 2: A/B Signal 5b ({dataset}) ===", flush=True)
    print(f"  checkpoint: {ckpt.name}  queries={len(bundle.qids)}", flush=True)

    t0 = time.perf_counter()
    baseline = score_from_bundle(
        bundle,
        lambda_prime_factor=0.0,
        bad_store=bad_store,
        progress_every=progress_every,
        arm_label="baseline λ_pf=0",
    )
    treatment = score_from_bundle(
        bundle,
        lambda_prime_factor=0.35,
        bad_store=bad_store,
        progress_every=progress_every,
        arm_label="treatment λ_pf=0.35",
    )
    elapsed = time.perf_counter() - t0

    delta = treatment.ndcg10 - baseline.ndcg10
    ref = baseline.bm25_ref
    ref_line = ""
    if ref is not None:
        ref_line = f"  BM25 ref:     {ref:.3f}  (baseline vs BM25 {baseline.ndcg10 - ref:+.3f})\n"

    print("\n" + "=" * 56)
    print("  SciFact Signal 5b A/B  (quality, test qrels)")
    print("=" * 56)
    print(f"  λ_pf=0.00   NDCG@10 = {baseline.ndcg10:.4f}   R@10 = {baseline.r10:.4f}")
    print(f"  λ_pf=0.35   NDCG@10 = {treatment.ndcg10:.4f}   R@10 = {treatment.r10:.4f}")
    print(f"  ΔNDCG@10    {delta:+.4f}   (Signal 5b lift)")
    print(ref_line, end="")
    print(f"  Stage 2 wall time: {elapsed / 60:.1f} min  (scoring only)")
    print("=" * 56)

    return 0


def stage_ab_kappa(
    *,
    dataset: str = "scifact",
    max_queries: int | None = None,
    progress_every: int = 1,
) -> int:
    ckpt = checkpoint_path(dataset, "quality")
    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}")
        print("Run:  python run_ab.py --stage build")
        return 1

    bundle = load_checkpoint(ckpt)
    if max_queries is not None:
        bundle.qids = bundle.qids[:max_queries]

    bad_store = BadCorrelationStore.load(bad_correlation_path(dataset, "quality"))

    print(f"\n=== Stage 2: A/B Signal 8a kappa ({dataset}) ===", flush=True)
    print(f"  checkpoint: {ckpt.name}  queries={len(bundle.qids)}", flush=True)

    t0 = time.perf_counter()
    baseline = score_from_bundle(
        bundle,
        lambda_kappa=0.0,
        bad_store=bad_store,
        progress_every=progress_every,
        arm_label="baseline λ_κ=0 (routing only)",
    )
    treatment = score_from_bundle(
        bundle,
        lambda_kappa=0.25,
        bad_store=bad_store,
        progress_every=progress_every,
        arm_label="treatment λ_κ=0.25 (+8a)",
    )
    elapsed = time.perf_counter() - t0

    delta = treatment.ndcg10 - baseline.ndcg10
    ref = baseline.bm25_ref
    ref_line = ""
    if ref is not None:
        ref_line = f"  BM25 ref:     {ref:.3f}  (baseline vs BM25 {baseline.ndcg10 - ref:+.3f})\n"

    print("\n" + "=" * 56)
    print("  SciFact Signal 8a kappa Jaccard A/B  (quality, test qrels)")
    print("=" * 56)
    print(f"  lambda_k=0.00   NDCG@10 = {baseline.ndcg10:.4f}   R@10 = {baseline.r10:.4f}")
    print(f"  lambda_k=0.25   NDCG@10 = {treatment.ndcg10:.4f}   R@10 = {treatment.r10:.4f}")
    print(f"  dNDCG@10        {delta:+.4f}   (kappa signal lift)")
    print(ref_line, end="")
    print(f"  Stage 2 wall time: {elapsed / 60:.1f} min  (scoring only)")
    print("=" * 56)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SciFact quality A/B (build + score)")
    parser.add_argument(
        "--stage",
        choices=("build", "ab", "all"),
        default="all",
        help="build=stage1 only, ab=stage2 only, all=both",
    )
    parser.add_argument(
        "--ab",
        choices=("kappa", "pf"),
        default="kappa",
        help="A/B comparison: kappa=Signal 8a (default), pf=Signal 5b",
    )
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--max-convergence-rounds",
        type=int,
        default=4,
        help="Fewer rounds in stage 1 for faster build (default 4)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip anchor training/convergence (full corpus indices only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Less progress output (summary every 50 queries)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=None,
        metavar="N",
        help="Print progress every N queries (default: 1, or 50 with --quiet)",
    )
    args = parser.parse_args()
    progress_every = args.progress_every
    if progress_every is None:
        progress_every = 50 if args.quiet else 1

    if args.stage in ("build", "all"):
        stage_build(
            dataset=args.dataset,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
            max_convergence_rounds=args.max_convergence_rounds,
            skip_training=args.skip_training,
        )
    if args.stage in ("ab", "all"):
        if args.ab == "kappa":
            return stage_ab_kappa(
                dataset=args.dataset,
                max_queries=args.max_queries,
                progress_every=progress_every,
            )
        return stage_ab_pf(
            dataset=args.dataset,
            max_queries=args.max_queries,
            progress_every=progress_every,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
