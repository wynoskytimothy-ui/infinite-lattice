"""Benchmark SciFact symbol-knowledge ingest."""

from __future__ import annotations

import argparse
import time

from aethos_symbol_knowledge import SymbolKnowledgeIndex, load_beir_corpus_text


def timed(label: str, fn):
    t0 = time.perf_counter()
    result = fn()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=500)
    p.add_argument("--full", action="store_true", help="run full 5183 doc build")
    args = p.parse_args()

    max_docs = None if args.full else args.max_docs

    print("Loading SciFact corpus texts ...", flush=True)
    t_load, corpus = timed("load", lambda: load_beir_corpus_text("scifact", max_docs=max_docs))
    n_docs = len(corpus)
    print(f"  corpus load: {t_load:.0f} ms  docs={n_docs}")

    print("Full build (morph + cellular + lazy chamber ingest) ...", flush=True)
    t_build, idx = timed(
        "build",
        lambda: SymbolKnowledgeIndex.build_from_corpus(
            corpus,
            dataset="scifact_bench",
            subjects={1, 9, 10},
        ),
    )
    s = idx.summary()
    print(f"  build: {t_build:.0f} ms  ({t_build / n_docs:.1f} ms/doc)")
    print(f"  master links: {s['total_cross_links']}")
    print(f"  chambers built: {s['n_chambers']}  lazy pending: {len(idx._chamber_dirty)}")
    print(f"  reported build_ms: {s['build_ms']}")

    # Lazy chamber first-touch
    if idx._chamber_dirty:
        ch = min(idx._chamber_dirty)
        t_ch, _ = timed(f"ensure_chamber_{ch}", lambda: idx._ensure_chamber(ch))
        print(f"  first lazy chamber {ch}: {t_ch:.0f} ms")

    # Incremental stack: 50 more docs
    extra = load_beir_corpus_text("scifact", max_docs=50)
    extra = {f"extra_{k}": v for k, v in extra.items()}
    print(f"compound_learn +50 docs ...", flush=True)
    t_stack, rep = timed(
        "stack",
        lambda: idx.compound_learn(extra, subjects={1, 9, 10}),
    )
    print(f"  stack: {t_stack:.0f} ms  ingest_ms={rep.get('ingest_ms')}")
    print(f"  links added: {rep['links_added']}")

    if args.full:
        total_s = t_build / 1000
        print(f"\nFULL SciFact: {total_s:.1f}s total ({n_docs} docs)")


if __name__ == "__main__":
    main()
