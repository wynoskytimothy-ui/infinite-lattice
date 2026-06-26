#!/usr/bin/env python3
"""Compare hub vs fast kappa index build time vs N."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import words
from aethos_multi_corpus import MultiCorpusBrain, IdfCache
from pipeline.bit_03_doc_attractor_set import build_attractor_index_fast
from scripts.bench_kappa_scale import distractors
from scripts.bench_supervised_bridges import load


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    corpus, queries, train_q, _ = load(name)
    base_n = len(corpus)

    print(f"Kappa ingest build time — {name} (fast path vs hub path)\n")
    print(f"{'N':>8} {'fast s':>8} {'hub s':>8} {'speedup':>8} {'buckets':>8}")

    for target in [base_n, 20000]:
        c = dict(corpus)
        if target > base_n:
            c.update(distractors(corpus, target - base_n))

        brain = MultiCorpusBrain()
        branch = brain.stack_corpus(name, c, build_kappa=False, finalize=False)
        idf = IdfCache(branch.idx, branch.n_docs)
        gtexts = {branch.global_id(k): v for k, v in branch.texts.items()}

        t0 = time.perf_counter()
        fast_idx = build_attractor_index_fast(
            brain._registry, gtexts, idf, top_k=brain.KAPPA_TOP_K,
        )
        fast_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        hub_idx = brain._build_kappa_index_hub(branch)
        hub_s = time.perf_counter() - t1

        sp = hub_s / fast_s if fast_s else 0
        print(
            f"{target:>8,} {fast_s:>8.1f} {hub_s:>8.1f} {sp:>7.1f}x "
            f"{len(fast_idx.by_key):>8}",
            flush=True,
        )
        del brain, fast_idx, hub_idx


if __name__ == "__main__":
    main()
