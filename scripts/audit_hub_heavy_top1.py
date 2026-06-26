#!/usr/bin/env python3
"""
Hub-heavy query split — does hub×rare compound ingest lift top-1?

Compares ENABLE_HUB_COMPOUNDS on vs off on SciFact held-out, split by:
  - hub-heavy: query has known polluter hub OR any content word with idf < HUB_IDF_GATE
  - non-hub: otherwise

Output: logs/hub_heavy_top1_audit.json

Run:  python scripts/audit_hub_heavy_top1.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_append_index import words
from aethos_multi_corpus import IdfCache, MultiCorpusBrain
from pipeline.bit_04_candidate_router import query_words_for_routing
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def _is_hub_heavy(query: str, idf, hub_words: frozenset[str], gate: float) -> tuple[bool, list[str]]:
    hubs: list[str] = []
    for w in query_words_for_routing(words(query)):
        if w in hub_words or idf(w) < gate:
            hubs.append(w)
    return bool(hubs), hubs


def _eval_split(brain, test_ids, queries, test_q, idf, hub_words, gate):
    hub_rows: list[dict] = []
    other_rows: list[dict] = []

    for qid in test_ids:
        query = queries[qid]
        heavy, hub_list = _is_hub_heavy(query, idf, hub_words, gate)
        r = brain.search(query, corpus="scifact", k=10)
        rel = test_q[qid]
        local = r.local_ids
        top1_gold = bool(local and local[0] in rel)
        gold_ranks = [i + 1 for i, d in enumerate(local) if d in rel]
        best = min(gold_ranks) if gold_ranks else None
        row = {
            "query_id": qid,
            "query": query[:90],
            "hub_words": hub_list[:6],
            "top1_gold": top1_gold,
            "best_gold_rank": best,
            "ndcg10": round(ndcg10(local, rel), 4),
            "recall10": round(recall10(local, rel), 4),
        }
        (hub_rows if heavy else other_rows).append(row)

    def _summ(rows: list[dict]) -> dict:
        n = len(rows)
        if not n:
            return {"n": 0}
        ranks = [r["best_gold_rank"] for r in rows if r["best_gold_rank"]]
        return {
            "n": n,
            "top1_gold_pct": round(100 * sum(r["top1_gold"] for r in rows) / n, 1),
            "mean_ndcg10": round(sum(r["ndcg10"] for r in rows) / n, 4),
            "mean_recall10": round(sum(r["recall10"] for r in rows) / n, 4),
            "mean_gold_rank": round(statistics.mean(ranks), 2) if ranks else None,
            "median_gold_rank": round(statistics.median(ranks), 1) if ranks else None,
        }

    return {
        "hub_heavy": _summ(hub_rows),
        "non_hub": _summ(other_rows),
        "hub_rows": hub_rows,
        "other_rows": other_rows,
    }


def _flip_analysis(on_rows: list[dict], off_rows: list[dict]) -> dict:
    on_map = {r["query_id"]: r for r in on_rows}
    off_map = {r["query_id"]: r for r in off_rows}
    helped = hurt = same = 0
    helped_ids: list[str] = []
    hurt_ids: list[str] = []
    for qid in on_map:
        a, b = on_map[qid]["top1_gold"], off_map[qid]["top1_gold"]
        if a and not b:
            helped += 1
            helped_ids.append(qid)
        elif b and not a:
            hurt += 1
            hurt_ids.append(qid)
        else:
            same += 1
    return {
        "compounds_helped_top1": helped,
        "compounds_hurt_top1": hurt,
        "unchanged": same,
        "helped_ids": helped_ids[:20],
        "hurt_ids": hurt_ids[:20],
    }


def main() -> None:
    corpus, queries, train_q, test_q = load("scifact")
    test_ids = [q for q in test_q if q in queries]

    results: dict = {"dataset": "scifact", "n_test": len(test_ids), "configs": {}}
    hub_words = MultiCorpusBrain.HUB_COMPOUND_WORDS
    gate = MultiCorpusBrain.HUB_IDF_GATE

    per_config_rows: dict[str, list] = {}

    for label, compounds in [("solo_rare_kappa", False), ("hub_x_rare_compounds", True)]:
        print(f"Stack + eval: {label}...", flush=True)
        brain = MultiCorpusBrain()
        brain.ENABLE_HUB_COMPOUNDS = compounds
        brain.stack_corpus("scifact", corpus, queries=queries, train_qrels=train_q)
        branch = brain._corpora["scifact"]
        idf = IdfCache(branch.idx, branch.n_docs)
        split = _eval_split(brain, test_ids, queries, test_q, idf, hub_words, gate)
        results["configs"][label] = {
            "hub_heavy": split["hub_heavy"],
            "non_hub": split["non_hub"],
        }
        per_config_rows[label] = split["hub_rows"] + split["other_rows"]
        hh = split["hub_heavy"]
        nh = split["non_hub"]
        print(
            f"  hub-heavy ({hh['n']}q): top1={hh['top1_gold_pct']}% "
            f"nDCG={hh['mean_ndcg10']} mean_rank={hh['mean_gold_rank']}",
            flush=True,
        )
        print(
            f"  non-hub   ({nh['n']}q): top1={nh['top1_gold_pct']}% "
            f"nDCG={nh['mean_ndcg10']} mean_rank={nh['mean_gold_rank']}",
            flush=True,
        )

    on_all = per_config_rows["hub_x_rare_compounds"]
    off_all = per_config_rows["solo_rare_kappa"]
    results["top1_flips"] = _flip_analysis(on_all, off_all)

    on_hub = [r for r in on_all if r["hub_words"]]
    off_hub = [r for r in off_all if r["hub_words"]]
    results["top1_flips_hub_heavy_only"] = _flip_analysis(on_hub, off_hub)

    out = Path("logs/hub_heavy_top1_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Summary ===", flush=True)
    print(f"All queries top-1 flips: helped={results['top1_flips']['compounds_helped_top1']} "
          f"hurt={results['top1_flips']['compounds_hurt_top1']}", flush=True)
    print(f"Hub-heavy only:        helped={results['top1_flips_hub_heavy_only']['compounds_helped_top1']} "
          f"hurt={results['top1_flips_hub_heavy_only']['compounds_hurt_top1']}", flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
