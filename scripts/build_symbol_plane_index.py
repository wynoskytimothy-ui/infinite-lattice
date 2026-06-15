#!/usr/bin/env python3
"""
Build + save symbol-plane κ index from symbol knowledge.

  python scripts/build_symbol_plane_index.py --dataset scifact
  python scripts/build_symbol_plane_index.py --dataset scifact --query cancer treatment
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path
from pipeline.bit_12_symbol_plane_index import (
    build_symbol_plane_index,
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)


def plane_index_path(dataset: str) -> Path:
    return knowledge_path(dataset).parent / f"{dataset}_plane.pkl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--rebuild-knowledge", action="store_true")
    p.add_argument(
        "--slim",
        action="store_true",
        help="rare-only pair meets + rare adjacency capped at 4 per word",
    )
    p.add_argument("--max-adjacency", type=int, default=None)
    p.add_argument("--query", nargs="*", default=None)
    args = p.parse_args()

    kpath = knowledge_path(args.dataset)
    if args.rebuild_knowledge or not kpath.is_file():
        print(f"Building knowledge for {args.dataset} (ingest_rare_weight=True) ...")
        knowledge = SymbolKnowledgeIndex.build_from_beir(args.dataset, download=True)
        knowledge.ingest_rare_weight = True
        knowledge.save()
    else:
        print(f"Loading knowledge {kpath} ...")
        knowledge = SymbolKnowledgeIndex.load(args.dataset)

    slim = args.slim
    max_adj = 4 if slim else args.max_adjacency
    print(
        f"Building symbol-plane kappa index ({len(knowledge.corpus)} docs) "
        f"slim={slim} max_adjacency={max_adj} ...",
    )
    t0 = time.perf_counter()
    plane = build_symbol_plane_index(
        knowledge,
        rare_pairs_only=slim,
        rare_adjacency_only=slim,
        max_adjacency_per_word=max_adj,
        index_doc_meet_keys=True,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    adj_edges = sum(len(v) for v in plane.word_adjacency.values())
    print(
        f"  plane keys={len(plane.by_key)}  docs={len(plane.doc_keys)}  "
        f"pair_meets={plane.n_pair_keys}  adj_edges={adj_edges}  build_ms={ms:.0f}"
    )

    out = plane_index_path(args.dataset)
    with open(out, "wb") as f:
        pickle.dump((knowledge, plane), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved: {out}")

    if args.query:
        words = args.query
        t0 = time.perf_counter()
        route = route_symbol_plane_candidates(knowledge, plane, words)
        qms = (time.perf_counter() - t0) * 1000.0
        print(f"\nQuery {words!r}:")
        print(f"  kappa keys={route.n_query_keys}  candidates={len(route.doc_ids)}  {qms:.2f} ms")
        ranked = rank_symbol_plane_docs(knowledge, plane, words, limit=10)
        for did, score in ranked[:8]:
            print(f"    {did}  overlap={score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
