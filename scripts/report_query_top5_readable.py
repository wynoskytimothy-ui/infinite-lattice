#!/usr/bin/env python3
"""
Readable markdown report — clean full text + token/subword/corr summary per block.

  python scripts/report_query_top5_readable.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_rare_rank import morph_trigger_pieces
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from eval_beir_symbol import load_brain_and_plane

_TOKEN_RE = re.compile(r"[a-z]{2,}", re.I)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text.lower())}


def _summarize_block(
    knowledge: SymbolKnowledgeIndex,
    query_words: list[str],
    text: str,
    *,
    is_query: bool = False,
) -> dict[str, list[str]]:
    doc_toks = _tokenize(text)
    query_hits: list[str] = []
    subwords: list[str] = []
    correlated: list[str] = []
    morph = knowledge.morph

    for qw in query_words:
        ql = qw.lower()
        if is_query or ql in doc_toks:
            query_hits.append(ql)

        for piece in morph_trigger_pieces(knowledge, ql):
            pl = piece.lower()
            if pl in morph.subwords or pl in morph.composites:
                if re.search(re.escape(pl), text, re.I) and pl not in subwords:
                    if is_query or pl != ql or pl in doc_toks:
                        subwords.append(pl)
            elif is_query and len(pl) >= 3 and pl in ql:
                if re.search(re.escape(pl), text, re.I) and pl not in subwords:
                    subwords.append(pl)

        if not is_query:
            for dt in sorted(doc_toks):
                if dt == ql:
                    continue
                lk = knowledge.correlates(ql, dt)
                if lk is not None and lk.strength >= 2.0 and dt not in correlated:
                    correlated.append(dt)

    return {
        "query_hits": query_hits,
        "subwords": subwords,
        "correlated": correlated[:24],
    }


def _fmt_list(items: list[str], empty: str = "(none)") -> str:
    return ", ".join(f"`{w}`" for w in items) if items else empty


def build_md(knowledge: SymbolKnowledgeIndex, rows: list[dict]) -> str:
    lines: list[str] = [
        "# SciFact — Queries, Gold Docs, Top 5 (readable)",
        "",
        "Legend per block:",
        "- **Query tokens** = literal match in text",
        "- **Subwords** = morph/L2 pieces from query tokens",
        "- **Correlated** = cross-link from query token to doc token (top 24)",
        "",
        "---",
        "",
    ]

    for i, row in enumerate(rows, 1):
        qid = row["query_id"]
        qwords = row["query_words"]
        qtext = row["query"]
        qs = _summarize_block(knowledge, qwords, qtext, is_query=True)

        lines.append(f"## {i}. Query `{qid}`")
        lines.append("")
        lines.append(f"**Query:** {qtext}")
        lines.append("")
        lines.append(f"nDCG@10={row['ndcg_at_10']} | Recall@10={row['recall_at_10']} | gold_in_route={row['gold_in_route']}")
        lines.append("")
        lines.append(f"- Query tokens: {_fmt_list(qs['query_hits'])}")
        lines.append(f"- Subwords: {_fmt_list(qs['subwords'])}")
        lines.append("")

        lines.append("### Gold / relevant")
        lines.append("")
        for g in row["gold_docs"]:
            gid = g["doc_id"]
            gtext = g["text"]
            gs = _summarize_block(knowledge, qwords, gtext)
            lines.append(f"#### Gold `{gid}`")
            lines.append("")
            lines.append(f"- Query tokens in doc: {_fmt_list(gs['query_hits'])}")
            lines.append(f"- Subwords in doc: {_fmt_list(gs['subwords'])}")
            lines.append(f"- Correlated tokens: {_fmt_list(gs['correlated'])}")
            lines.append("")
            lines.append(gtext)
            lines.append("")

        lines.append("### Top 5 ranked")
        lines.append("")
        lines.append("| Rank | Doc ID | Score | Gold? |")
        lines.append("|------|--------|-------|-------|")
        for rank, item in enumerate(row["top5"], 1):
            flag = "yes" if item["is_gold"] else ""
            lines.append(f"| {rank} | {item['doc_id']} | {item['score']:.5f} | {flag} |")
        lines.append("")

        for rank, item in enumerate(row["top5"], 1):
            did = item["doc_id"]
            dtext = knowledge.corpus.get(did, "(missing)")
            ds = _summarize_block(knowledge, qwords, dtext)
            label = "GOLD" if item["is_gold"] else "rank"
            lines.append(f"#### Top {rank} `{did}` ({label}) — score {item['score']:.5f}")
            lines.append("")
            lines.append(f"- Query tokens in doc: {_fmt_list(ds['query_hits'])}")
            lines.append(f"- Subwords in doc: {_fmt_list(ds['subwords'])}")
            lines.append(f"- Correlated tokens: {_fmt_list(ds['correlated'])}")
            lines.append("")
            lines.append(dtext)
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--in-json", default="logs/query_top5_fulltext.json")
    p.add_argument("--out", default="logs/query_top5_fulltext.md")
    args = p.parse_args()

    rows = json.loads(Path(args.in_json).read_text(encoding="utf-8"))
    print("Loading brain ...", flush=True)
    knowledge, _ = load_brain_and_plane(args.dataset)
    md = build_md(knowledge, rows)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"Wrote: {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
