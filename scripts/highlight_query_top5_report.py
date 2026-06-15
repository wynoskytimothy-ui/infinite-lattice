#!/usr/bin/env python3
"""
Build HTML report with subwords (blue) and related tokens highlighted.

Reads logs/query_top5_fulltext.json + symbol brain morph/correlations.

  python scripts/highlight_query_top5_report.py
"""

from __future__ import annotations

import argparse
import html
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

# CSS class priority (higher wins on overlap)
_PRIORITY = {"hl-query": 3, "hl-corr": 2, "hl-subword": 1}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text.lower())]


def _build_highlight_map(
    knowledge: SymbolKnowledgeIndex,
    query_words: list[str],
    text: str,
    *,
    is_query_text: bool = False,
) -> dict[str, str]:
    """term -> css class for this text block."""
    terms: dict[str, str] = {}
    doc_toks = set(_tokenize(text))
    morph = knowledge.morph

    for qw in query_words:
        ql = qw.lower()
        if is_query_text:
            terms[ql] = "hl-query"
        elif ql in doc_toks:
            terms[ql] = "hl-query"

        for piece in morph_trigger_pieces(knowledge, ql):
            pl = piece.lower()
            if pl in morph.subwords or pl in morph.composites:
                if pl != ql or is_query_text:
                    if _appears_in_text(text, pl):
                        if pl not in terms or _PRIORITY["hl-subword"] >= _PRIORITY.get(terms[pl], 0):
                            terms[pl] = "hl-subword"
            elif len(pl) >= 3 and pl in ql and is_query_text:
                if _appears_in_text(text, pl):
                    terms[pl] = "hl-subword"

        if not is_query_text:
            for dt in doc_toks:
                if dt == ql:
                    continue
                lk = knowledge.correlates(ql, dt)
                if lk is not None and lk.strength >= 2.0:
                    if "hl-query" not in (terms.get(dt),) and dt not in terms:
                        terms[dt] = "hl-corr"
                    elif terms.get(dt) == "hl-corr":
                        pass
                    elif dt not in terms:
                        terms[dt] = "hl-corr"

    # Promoted morph subwords appearing anywhere in text (linked to query via decomposition)
    if not is_query_text:
        for qw in query_words:
            for piece in morph_trigger_pieces(knowledge, qw):
                pl = piece.lower()
                if len(pl) >= 3 and _appears_in_text(text, pl):
                    if terms.get(pl) != "hl-query":
                        terms[pl] = "hl-subword"

    return terms


def _appears_in_text(text: str, term: str) -> bool:
    return bool(re.search(re.escape(term), text, re.I))


def _highlight_html(text: str, term_classes: dict[str, str]) -> str:
    if not term_classes or not text:
        return html.escape(text)

    escaped = html.escape(text)
    intervals: list[tuple[int, int, str]] = []

    for term, cls in sorted(term_classes.items(), key=lambda x: -len(x[0])):
        if len(term) < 2:
            continue
        for m in re.finditer(re.escape(term), escaped, re.I):
            intervals.append((m.start(), m.end(), cls))

    if not intervals:
        return escaped

    intervals.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: list[tuple[int, int, str]] = []
    for start, end, cls in intervals:
        if merged and start < merged[-1][1]:
            prev_s, prev_e, prev_c = merged[-1]
            if _PRIORITY[cls] > _PRIORITY[prev_c]:
                if start > prev_s:
                    merged[-1] = (prev_s, start, prev_c)
                    merged.append((start, end, cls))
                else:
                    merged[-1] = (start, max(end, prev_e), cls)
            else:
                if start >= prev_e:
                    merged.append((start, end, cls))
                elif end > prev_e:
                    merged[-1] = (prev_s, end, prev_c)
            continue
        merged.append((start, end, cls))

    # Re-merge adjacent same class
    cleaned: list[tuple[int, int, str]] = []
    for iv in merged:
        if cleaned and iv[0] <= cleaned[-1][1] and iv[2] == cleaned[-1][2]:
            cleaned[-1] = (cleaned[-1][0], max(cleaned[-1][1], iv[1]), iv[2])
        else:
            cleaned.append(iv)

    out: list[str] = []
    pos = 0
    for start, end, cls in sorted(cleaned, key=lambda x: x[0]):
        if start < pos:
            continue
        out.append(escaped[pos:start])
        out.append(f'<span class="{cls}">{escaped[start:end]}</span>')
        pos = end
    out.append(escaped[pos:])
    return "".join(out)


def _legend_block() -> str:
    return """
<div class="legend">
  <span class="hl-query">query token</span> literal match in text
  <span class="hl-subword">subword</span> morph/L2 piece (blue)
  <span class="hl-corr">correlated</span> cross-link from query token
</div>
"""


def _term_list(term_classes: dict[str, str]) -> str:
    if not term_classes:
        return "<p class='meta'>No highlighted terms</p>"
    by_cls: dict[str, list[str]] = {}
    for t, c in sorted(term_classes.items(), key=lambda x: (x[1], x[0])):
        by_cls.setdefault(c, []).append(t)
    parts = []
    labels = {
        "hl-query": "Query tokens",
        "hl-subword": "Subwords (blue)",
        "hl-corr": "Correlated tokens",
    }
    for cls in ("hl-query", "hl-subword", "hl-corr"):
        if cls in by_cls:
            tags = " ".join(
                f'<span class="{cls}">{html.escape(w)}</span>' for w in by_cls[cls]
            )
            parts.append(f"<p><b>{labels[cls]}:</b> {tags}</p>")
    return "\n".join(parts)


def build_html(knowledge: SymbolKnowledgeIndex, rows: list[dict]) -> str:
    sections: list[str] = []
    sections.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SciFact Query Report — Highlighted</title>
<style>
body { font-family: Georgia, serif; max-width: 920px; margin: 2rem auto; padding: 0 1rem;
       background: #1a1a1e; color: #e8e8ec; line-height: 1.55; }
h1 { color: #f0f0f5; border-bottom: 1px solid #444; }
h2 { color: #c8d8ff; margin-top: 2.5rem; }
h3 { color: #a0b0d0; }
h4 { color: #90a0c0; }
.meta { color: #888; font-size: 0.9rem; }
.doc-block { background: #242428; border-left: 4px solid #555; padding: 1rem 1.2rem;
             margin: 1rem 0; border-radius: 4px; }
.gold-block { border-left-color: #4a9; }
.false-block { border-left-color: #a64; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #444; padding: 0.4rem 0.6rem; text-align: left; }
th { background: #2a2a32; }
tr.gold-row { background: #1e3228; }
.legend { margin: 1.5rem 0; padding: 0.8rem; background: #2a2a32; border-radius: 4px; }
.hl-subword { background: #2563eb; color: #fff; padding: 0 2px; border-radius: 2px; }
.hl-query { background: #059669; color: #fff; padding: 0 2px; border-radius: 2px; }
.hl-corr { background: #d97706; color: #fff; padding: 0 2px; border-radius: 2px; }
hr { border: none; border-top: 1px solid #333; margin: 2rem 0; }
</style>
</head>
<body>
<h1>SciFact — 30 Queries (highlighted)</h1>
""")
    sections.append(_legend_block())

    for i, row in enumerate(rows, 1):
        qid = row["query_id"]
        qwords = row["query_words"]
        qtext = row["query"]
        q_hl = _build_highlight_map(knowledge, qwords, qtext, is_query_text=True)

        sections.append(f"<h2>{i}. Query <code>{html.escape(qid)}</code></h2>")
        sections.append(
            f"<p class='meta'>nDCG@10={row['ndcg_at_10']} | Recall@10={row['recall_at_10']} | "
            f"gold_in_route={row['gold_in_route']}</p>"
        )
        sections.append("<h3>Full query</h3>")
        sections.append(_term_list(q_hl))
        sections.append(
            f'<div class="doc-block">{_highlight_html(qtext, q_hl)}</div>'
        )

        sections.append("<h3>Gold / relevant docs</h3>")
        for g in row["gold_docs"]:
            gid = g["doc_id"]
            gtext = g["text"]
            g_hl = _build_highlight_map(knowledge, qwords, gtext)
            sections.append(f'<h4>Gold <code>{html.escape(gid)}</code></h4>')
            sections.append(_term_list(g_hl))
            sections.append(
                f'<div class="doc-block gold-block">{_highlight_html(gtext, g_hl)}</div>'
            )

        sections.append("<h3>Top 5 ranked</h3>")
        sections.append("<table><tr><th>Rank</th><th>Doc</th><th>Score</th><th>Gold?</th></tr>")
        for rank, item in enumerate(row["top5"], 1):
            cls = ' class="gold-row"' if item["is_gold"] else ""
            flag = "YES" if item["is_gold"] else ""
            sections.append(
                f"<tr{cls}><td>{rank}</td><td>{html.escape(item['doc_id'])}</td>"
                f"<td>{item['score']:.5f}</td><td>{flag}</td></tr>"
            )
        sections.append("</table>")

        for rank, item in enumerate(row["top5"], 1):
            did = item["doc_id"]
            dtext = knowledge.corpus.get(did, "(missing)")
            d_hl = _build_highlight_map(knowledge, qwords, dtext)
            label = "GOLD" if item["is_gold"] else "rank"
            sections.append(
                f'<h4>Top {rank} <code>{html.escape(did)}</code> ({label}) '
                f'score={item["score"]:.5f}</h4>'
            )
            sections.append(_term_list(d_hl))
            sections.append(
                f'<div class="doc-block false-block">{_highlight_html(dtext, d_hl)}</div>'
            )

        sections.append("<hr>")

    sections.append("</body></html>")
    return "\n".join(sections)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--in-json", default="logs/query_top5_fulltext.json")
    p.add_argument("--out", default="logs/query_top5_fulltext_highlighted.html")
    args = p.parse_args()

    in_path = Path(args.in_json)
    if not in_path.is_file():
        print(f"Missing {in_path} — run report_query_top5_fulltext.py first", file=sys.stderr)
        return 1

    rows = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"Loading brain ({args.dataset}) for morph + correlations ...", flush=True)
    knowledge, _plane = load_brain_and_plane(args.dataset)

    html_doc = build_html(knowledge, rows)
    out_path = Path(args.out)
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"Wrote: {out_path.resolve()} ({len(rows)} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
