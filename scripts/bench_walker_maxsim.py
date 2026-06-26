#!/usr/bin/env python3
"""Bench WalkerMaxSimRetriever on BEIR SciFact (held-out test qrels)."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from beir_data_root import resolve_beir_root
from lattice_retriever_v1.walker_maxsim_retriever import build_walker_maxsim_retriever


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


def build_retriever(corpus: dict[str, str]):
    return build_walker_maxsim_retriever(corpus)


def main() -> None:
    corpus, queries, test_qrels = load_scifact()
    test_ids = [q for q in test_qrels if q in queries]
    print(f"SciFact: {len(corpus)} docs, {len(test_ids)} test queries")
    retriever = build_retriever(corpus)
    ndcgs: list[float] = []
    recalls: list[float] = []
    pool_hits = 0
    t0 = time.perf_counter()
    for i, qid in enumerate(test_ids):
        q = queries[qid]
        rels = test_qrels[qid]
        gold = {d for d, s in rels.items() if s > 0}
        trace = retriever.retrieve_with_trace(q, limit=10)
        ranked = [h.doc_id for h in trace.hits]
        if gold & set(trace.lit_docs):
            pool_hits += 1
        ndcgs.append(ndcg10(ranked, rels))
        recalls.append(recall10(ranked, rels))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_ids)}", flush=True)
    elapsed = time.perf_counter() - t0
    n = len(test_ids)
    print(
        f"pool_recall={pool_hits/n:.3f} R@10={sum(recalls)/n:.3f} "
        f"nDCG@10={sum(ndcgs)/n:.3f} ms/q={1000*elapsed/n:.1f}"
    )


if __name__ == "__main__":
    main()
