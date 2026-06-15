#!/usr/bin/env python3
"""
SciFact Markov perplexity — before/after strengthen passes.

  python scripts/bench_markov_scifact.py
  python scripts/bench_markov_scifact.py --passes 3 --eval-docs 400 --train-docs 800
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path
from aethos_symbol_markov import MarkovCorrelationBrain, build_markov_brain, markov_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact_compound")
    p.add_argument("--eval-docs", type=int, default=300)
    p.add_argument("--train-docs", type=int, default=600)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-plane", action="store_true")
    args = p.parse_args()

    paths = [
        knowledge_path(args.dataset),
        knowledge_path("scifact"),
    ]
    kpath = next((x for x in paths if x.is_file()), None)
    if kpath is None:
        print("No scifact knowledge found — run test_pretrain_brain_memory.py first")
        return 1

    print(f"Loading {kpath} ...")
    t0 = time.perf_counter()
    knowledge = SymbolKnowledgeIndex.load(args.dataset, path=kpath)
    print(f"  {len(knowledge.corpus)} docs in {(time.perf_counter()-t0)*1000:.0f} ms")

    print("Building Markov brain ...")
    t0 = time.perf_counter()
    brain = MarkovCorrelationBrain(knowledge=knowledge)
    brain.ingest_corpus()
    if not args.no_plane:
        plane_pkl = knowledge_path(args.dataset).parent / f"{args.dataset.replace('_compound','')}_plane.pkl"
        if not plane_pkl.is_file():
            plane_pkl = knowledge_path("scifact").parent / "scifact_plane.pkl"
        if plane_pkl.is_file():
            import pickle
            _k, plane = pickle.load(open(plane_pkl, "rb"))
            brain.attach_plane(plane)
            print(f"  attached plane from {plane_pkl.name}")
    print(f"  bigram={len(brain.bigram)} in {(time.perf_counter()-t0)*1000:.0f} ms")

    rng = random.Random(args.seed)
    all_ids = sorted(knowledge.corpus.keys())
    rng.shuffle(all_ids)
    eval_ids = all_ids[: args.eval_docs]
    train_ids = all_ids[args.eval_docs : args.eval_docs + args.train_docs]
    eval_corpus = {k: knowledge.corpus[k] for k in eval_ids}
    train_corpus = {k: knowledge.corpus[k] for k in train_ids}

    print(f"\nEval holdout: {len(eval_ids)} docs  Train: {len(train_ids)} docs")

    # Reset accuracy counters
    brain.hit_top1 = brain.hit_top5 = brain.total_steps = 0
    before = brain.eval_corpus_perplexity(eval_corpus, strengthen_on_miss=False)
    print("\nBEFORE strengthen (holdout eval):")
    print(f"  perplexity={before['perplexity']:.1f}  steps={before['steps']}")
    print(f"  top1={before['top1']:.3f}  top5={before['top5']:.3f}")

    print(f"\nTraining {args.passes} pass(es) on {len(train_ids)} docs ...")
    history: list[dict] = [{"phase": "before", **before}]
    for n in range(args.passes):
        t0 = time.perf_counter()
        brain.hit_top1 = brain.hit_top5 = brain.total_steps = 0
        for did in train_ids:
            brain.eval_text_perplexity(
                train_corpus[did], strengthen_on_miss=True,
            )
        train_ms = (time.perf_counter() - t0) * 1000.0
        brain.hit_top1 = brain.hit_top5 = brain.total_steps = 0
        after = brain.eval_corpus_perplexity(eval_corpus, strengthen_on_miss=False)
        row = {
            "phase": f"pass_{n+1}",
            "train_ms": round(train_ms, 1),
            "mismatch_strengthen": brain.mismatch_strengthen,
            "bigram_edges": len(brain.bigram),
            **after,
        }
        history.append(row)
        print(
            f"  pass {n+1}: perplexity={after['perplexity']:.1f}  "
            f"top1={after['top1']:.3f}  strengthened_total={brain.mismatch_strengthen}  "
            f"train_ms={train_ms:.0f}"
        )

    improved = history[-1]["perplexity"] < history[0]["perplexity"]
    print(f"\nPerplexity improved: {improved}")
    print(f"  {history[0]['perplexity']:.1f} -> {history[-1]['perplexity']:.1f}")

    out = _ROOT / "logs" / "markov_scifact_perplexity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "eval_docs": len(eval_ids),
        "train_docs": len(train_ids),
        "passes": args.passes,
        "history": history,
        "improved": improved,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {out}")

    brain.save(markov_path(args.dataset))
    print(f"Saved brain: {markov_path(args.dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
