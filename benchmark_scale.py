#!/usr/bin/env python3
"""
Corpus scale benchmark — latency + fingerprint vs targets.

Targets:
  p99 ingest <= 50 ms/doc (typical web prose length)
  fingerprint <= 100 bytes/doc metadata

Usage:
  python benchmark_scale.py
  python benchmark_scale.py --docs 2000 --words 120
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from aethos_pipeline import AethosPipeline
from aethos_scale import ScaleConfig, ScaleMetrics, fingerprint_document, timed_ingest_one
from diagnose_corpus import QUERIES, SMALL_CORPUS


def _synthetic_doc(rng: random.Random, n_words: int = 80) -> str:
    vocab = (
        "phone technical chip software hardware network apple fruit pie orchard "
        "banana dessert fresh released support called about runs fast better "
        "the and with from for has new cat bat tab mat salad recipe"
    ).split()
    words = [rng.choice(vocab) for _ in range(n_words)]
    return " ".join(words)


def _accuracy_probe(_pipe: AethosPipeline) -> dict[str, bool]:
    """Gold accuracy on SMALL_CORPUS (not random synthetic stream)."""
    pipe = AethosPipeline(rebuild_every=2)
    pipe.ingest(*SMALL_CORPUS)
    tech = pipe.resolve("apple", ["phone", "chip"])
    food = pipe.resolve("apple", ["fruit", "pie"])
    return {
        "apple_disambiguation": tech["cluster_id"] != food["cluster_id"] and bool(tech["cluster_id"]),
        "phone_stable": pipe.resolve("phone", ["fruit", "pie"])["cluster_id"] == "theme_phone",
        "oov_empty": pipe.resolve("zebra")["cluster_id"] == "",
    }


def run_benchmark(*, n_docs: int, words_per_doc: int, seed: int = 0) -> tuple[ScaleMetrics, dict[str, bool]]:
    rng = random.Random(seed)
    cfg = ScaleConfig(
        rebuild_every=128,
        lazy_clusters=True,
        max_window_tokens=48,
        max_corr_pairs_per_doc=256,
    )
    pipe = AethosPipeline(rebuild_every=cfg.rebuild_every)
    pipe.apply_scale_config(cfg)

    metrics = ScaleMetrics()
    t_batch = time.perf_counter()
    for i in range(n_docs):
        doc = _synthetic_doc(rng, words_per_doc)
        timing, fp = timed_ingest_one(pipe, i, doc)
        metrics.record(timing, fp)
    pipe.flush()
    batch_ms = (time.perf_counter() - t_batch) * 1000.0

    acc = _accuracy_probe(pipe)
    print(f"  batch total: {batch_ms:.0f} ms ({batch_ms / max(n_docs, 1):.2f} ms/doc incl. flush)")
    return metrics, acc


def run_small_corpus_accuracy() -> dict[str, bool]:
    pipe = AethosPipeline(rebuild_every=2)
    pipe.ingest(*SMALL_CORPUS)
    out: dict[str, bool] = {}
    for word, ctx in QUERIES:
        r = pipe.resolve(word, ctx)
        out[f"{word}|{','.join(ctx)}"] = bool(r["cluster_id"]) or word == "zebra"
    tech = pipe.resolve("apple", ["phone", "chip"])
    food = pipe.resolve("apple", ["fruit", "pie"])
    out["apple_split"] = tech["cluster_id"] != food["cluster_id"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="AETHOS scale benchmark")
    parser.add_argument("--docs", type=int, default=500)
    parser.add_argument("--words", type=int, default=80)
    parser.add_argument("--target-ms", type=float, default=50.0)
    parser.add_argument("--target-bytes", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("AETHOS SCALE BENCHMARK")
    print("=" * 60)
    print(f"  targets: {args.target_ms} ms/doc (p99), {args.target_bytes} B/doc fingerprint\n")

    metrics, acc = run_benchmark(n_docs=args.docs, words_per_doc=args.words)
    print(metrics.summary(target_ms=args.target_ms, target_bytes=args.target_bytes))
    print("\nAccuracy probes (scale corpus):")
    for k, v in acc.items():
        print(f"  {'ok' if v else 'FAIL'}: {k}")

    print("\nSmall-corpus routing (gold):")
    gold = run_small_corpus_accuracy()
    gold_ok = all(gold.values())
    for k, v in gold.items():
        if k == "apple_split":
            print(f"  {'ok' if v else 'FAIL'}: apple tech vs food clusters differ")
    print(f"  gold queries ok: {sum(gold.values())}/{len(gold)}")

    ok = metrics.pass_latency(args.target_ms) and metrics.pass_fingerprint(args.target_bytes) and all(acc.values()) and gold_ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
