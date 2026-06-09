#!/usr/bin/env python3
"""
Full-text report: query, gold/relevant docs, top-5 ranked per query.

  python scripts/report_query_top5_fulltext.py
  python scripts/report_query_top5_fulltext.py --max-queries 30 --out logs/query_top5_fulltext.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k, recall_at_k
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)


def _wrap(text: str, width: int = 100) -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    n = 0
    for w in words:
        if n + len(w) + 1 > width and line:
            lines.append(" ".join(line))
            line = [w]
            n = len(w)
        else:
            line.append(w)
            n += len(w) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def build_report(
    knowledge,
    plane,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    qids: list[str],
    *,
    max_keys: int,
    max_corr_neighbors: int,
    top_k: int,
) -> tuple[list[dict], str]:
    rows: list[dict] = []
    md_parts: list[str] = [
        "# SciFact Query Report — Full Text + Top 5",
        "",
        f"Queries: {len(qids)}  |  mode: kappa  |  top_k: {top_k}",
        "",
    ]

    for i, qid in enumerate(qids, 1):
        qtext = queries[qid]
        rel = qrels[qid]
        words = query_words(qtext)

        route = route_symbol_plane_candidates(
            knowledge, plane, words,
            max_candidates=600,
            max_keys=max_keys,
            max_corr_neighbors=max_corr_neighbors,
        )
        ranked = rank_symbol_plane_docs(
            knowledge, plane, words, limit=top_k,
            query_keys=set(route.query_keys),
            candidate_doc_ids=route.doc_ids,
        )

        gold_ids = list(rel.keys())
        top5 = [{"doc_id": did, "score": round(sc, 5), "is_gold": did in rel} for did, sc in ranked]

        gold_docs = [
            {
                "doc_id": gid,
                "text": knowledge.corpus.get(gid, "(missing from corpus)"),
            }
            for gid in gold_ids
        ]

        row = {
            "query_id": qid,
            "query": qtext,
            "query_words": words,
            "gold_doc_ids": gold_ids,
            "gold_docs": gold_docs,
            "top5": top5,
            "ndcg_at_10": round(ndcg_at_k([d for d, _ in ranked], rel, 10), 4),
            "recall_at_10": round(recall_at_k([d for d, _ in ranked], rel, 10), 4),
            "gold_in_route": any(g in route.doc_ids for g in gold_ids),
        }
        rows.append(row)

        md_parts.append(f"## {i}. Query `{qid}`")
        md_parts.append("")
        md_parts.append("### Full query")
        md_parts.append(f"> {qtext}")
        md_parts.append("")
        md_parts.append(f"**Tokens:** `{' | '.join(words)}`")
        md_parts.append("")
        md_parts.append(
            f"**Metrics:** nDCG@10={row['ndcg_at_10']}  Recall@10={row['recall_at_10']}  "
            f"gold_in_route={row['gold_in_route']}"
        )
        md_parts.append("")
        md_parts.append("### Gold / relevant docs")
        md_parts.append("")
        for g in gold_docs:
            md_parts.append(f"#### Gold `{g['doc_id']}`")
            md_parts.append("")
            md_parts.append("```")
            md_parts.append(_wrap(g["text"]))
            md_parts.append("```")
            md_parts.append("")
        md_parts.append(f"### Top {top_k} ranked")
        md_parts.append("")
        md_parts.append("| Rank | Doc ID | Score | Gold? |")
        md_parts.append("|------|--------|-------|-------|")
        for rank, item in enumerate(top5, 1):
            flag = "YES" if item["is_gold"] else ""
            md_parts.append(
                f"| {rank} | {item['doc_id']} | {item['score']:.5f} | {flag} |"
            )
        md_parts.append("")
        for rank, item in enumerate(top5, 1):
            did = item["doc_id"]
            text = knowledge.corpus.get(did, "(missing)")
            label = "GOLD" if item["is_gold"] else "rank"
            md_parts.append(f"#### Top {rank} `{did}` ({label}) score={item['score']:.5f}")
            md_parts.append("")
            md_parts.append("```")
            md_parts.append(_wrap(text))
            md_parts.append("```")
            md_parts.append("")
        md_parts.append("---")
        md_parts.append("")

    return rows, "\n".join(md_parts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-keys", type=int, default=1024)
    p.add_argument("--max-corr-neighbors", type=int, default=2)
    p.add_argument("--out-md", default="logs/query_top5_fulltext.md")
    p.add_argument("--out-json", default="logs/query_top5_fulltext.json")
    args = p.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    qids = [q for q in qrels if q in queries][: args.max_queries]
    print(f"Loading brain + plane, reporting {len(qids)} queries ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)

    rows, md = build_report(
        knowledge, plane, queries, qrels, qids,
        max_keys=args.max_keys,
        max_corr_neighbors=args.max_corr_neighbors,
        top_k=args.top_k,
    )

    md_path = Path(args.out_md)
    json_path = Path(args.out_json)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Markdown: {md_path.resolve()}")
    print(f"JSON:     {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
