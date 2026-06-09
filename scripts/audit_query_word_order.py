#!/usr/bin/env python3
"""
Audit query token ORDER vs gold and false docs.

Lattice view: primes meet on the imag line in sequence — does doc text preserve
query word order, invert it, or scatter tokens?

  python scripts/audit_query_word_order.py
  python scripts/audit_query_word_order.py --max-queries 30 --top-false 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)

_TOKEN_RE = re.compile(r"[a-z]+")
_MIN_LEN = 3


def _doc_token_list(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _DEFAULT_MEMBRANE and len(t) >= _MIN_LEN
    ]


def _query_token_list(text: str) -> list[str]:
    return [w.lower() for w in query_words(text)]


def _positions_in_doc(doc_toks: list[str], query_toks: list[str]) -> dict[str, list[int]]:
    pos: dict[str, list[int]] = {w: [] for w in query_toks}
    for i, t in enumerate(doc_toks):
        if t in pos:
            pos[t].append(i)
    return pos


def _first_positions(doc_toks: list[str], query_toks: list[str]) -> list[int | None]:
    pos = _positions_in_doc(doc_toks, query_toks)
    out: list[int | None] = []
    for w in query_toks:
        out.append(pos[w][0] if pos[w] else None)
    return out


def _ordered_subsequence(first_pos: list[int | None]) -> bool:
    """All query tokens present and strictly increasing positions."""
    if any(p is None for p in first_pos):
        return False
    return all(first_pos[i] < first_pos[i + 1] for i in range(len(first_pos) - 1))


def _exact_adjacent(doc_toks: list[str], query_toks: list[str]) -> bool:
    """Query tokens appear as consecutive run in doc (same order)."""
    if not query_toks:
        return False
    n = len(query_toks)
    for i in range(len(doc_toks) - n + 1):
        if doc_toks[i : i + n] == query_toks:
            return True
    return False


def _inversion_count(first_pos: list[int | None]) -> int | None:
    """Pairwise inversions among tokens that appear (query order vs doc order)."""
    present = [(i, p) for i, p in enumerate(first_pos) if p is not None]
    if len(present) < 2:
        return 0 if present else None
    inv = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            qi, pi = present[i]
            qj, pj = present[j]
            if qi < qj and pi > pj:
                inv += 1
    return inv


def _mean_gap(first_pos: list[int | None]) -> float | None:
    present = [p for p in first_pos if p is not None]
    if len(present) < 2:
        return None
    gaps = [present[i + 1] - present[i] for i in range(len(present) - 1)]
    return sum(gaps) / len(gaps)


def _order_class(
    doc_toks: list[str],
    query_toks: list[str],
    first_pos: list[int | None],
) -> str:
    n = len(query_toks)
    present = sum(1 for p in first_pos if p is not None)
    if present == 0:
        return "none_present"
    if _exact_adjacent(doc_toks, query_toks):
        return "exact_adjacent_phrase"
    if present < n:
        if _ordered_subsequence(first_pos):
            return "partial_same_order"
        return "partial_scattered"
    if _ordered_subsequence(first_pos):
        gaps = _mean_gap(first_pos)
        if gaps is not None and gaps == 1.0:
            return "exact_adjacent_phrase"
        return "ordered_subsequence"
    return "same_set_inverted_order"


def analyze_doc(
    doc_id: str,
    text: str,
    query_toks: list[str],
    *,
    label: str,
) -> dict[str, object]:
    doc_toks = _doc_token_list(text)
    first_pos = _first_positions(doc_toks, query_toks)
    inv = _inversion_count(first_pos)
    n_present = sum(1 for p in first_pos if p is not None)
    order_class = _order_class(doc_toks, query_toks, first_pos)
    missing = [query_toks[i] for i, p in enumerate(first_pos) if p is None]

    # Doc-order of present query tokens (by first occurrence position)
    present_pairs = [(query_toks[i], first_pos[i]) for i in range(len(query_toks)) if first_pos[i] is not None]
    present_pairs.sort(key=lambda x: x[1])  # type: ignore[arg-type]
    doc_order = [w for w, _ in present_pairs]

    return {
        "doc_id": doc_id,
        "label": label,
        "n_query_tokens": len(query_toks),
        "n_present": n_present,
        "order_class": order_class,
        "exact_adjacent": order_class == "exact_adjacent_phrase",
        "ordered_subsequence": order_class in (
            "exact_adjacent_phrase", "ordered_subsequence", "partial_same_order",
        ),
        "inverted": order_class == "same_set_inverted_order",
        "inversion_count": inv,
        "mean_token_gap": round(_mean_gap(first_pos), 2) if _mean_gap(first_pos) is not None else None,
        "query_order": query_toks,
        "doc_order_of_query_tokens": doc_order,
        "missing_tokens": missing,
        "first_positions": first_pos,
    }


def _pair_order_stats(profiles: list[dict]) -> dict[str, object]:
    pair_same = pair_inv = pair_total = 0
    present_counts: list[int] = []
    query_counts: list[int] = []
    for p in profiles:
        present_counts.append(int(p["n_present"]))
        query_counts.append(int(p["n_query_tokens"]))
        fp = p["first_positions"]
        for i in range(len(fp) - 1):
            if fp[i] is None or fp[i + 1] is None:
                continue
            pair_total += 1
            if fp[i] < fp[i + 1]:
                pair_same += 1
            elif fp[i] > fp[i + 1]:
                pair_inv += 1
    multi = [p for p in profiles if int(p["n_present"]) >= 2]
    multi_ordered = sum(
        1 for p in multi
        if _ordered_subsequence(p["first_positions"])
    )
    return {
        "avg_query_tokens": round(sum(query_counts) / max(len(query_counts), 1), 2),
        "avg_present_in_doc": round(sum(present_counts) / max(len(present_counts), 1), 2),
        "pct_query_tokens_present": round(
            100 * sum(present_counts) / max(sum(query_counts), 1), 1,
        ),
        "adjacent_pairs_both_present": pair_total,
        "adjacent_pairs_same_order": pair_same,
        "adjacent_pairs_inverted": pair_inv,
        "pct_pairs_same_order": round(100 * pair_same / max(pair_total, 1), 1),
        "pct_pairs_inverted": round(100 * pair_inv / max(pair_total, 1), 1),
        "docs_with_2plus_present": len(multi),
        "docs_2plus_all_in_query_order": multi_ordered,
        "pct_2plus_in_query_order": round(
            100 * multi_ordered / max(len(multi), 1), 1,
        ),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    def _count(profiles: list[dict], key: str) -> dict[str, int]:
        c: dict[str, int] = {}
        for p in profiles:
            val = p.get(key)
            if isinstance(val, bool):
                k = str(val)
            else:
                k = str(val)
            c[k] = c.get(k, 0) + 1
        return c

    gold_profiles: list[dict] = []
    false_profiles: list[dict] = []
    for row in rows:
        gold_profiles.extend(row["gold"])
        false_profiles.extend(row["false_top"])

    def _agg(profiles: list[dict], label: str) -> dict[str, object]:
        n = len(profiles)
        if n == 0:
            return {"instances": 0}
        exact = sum(1 for p in profiles if p["exact_adjacent"])
        ordered = sum(1 for p in profiles if p["ordered_subsequence"])
        inverted = sum(1 for p in profiles if p["inverted"])
        all_present = sum(1 for p in profiles if p["n_present"] == p["n_query_tokens"])
        gaps = [p["mean_token_gap"] for p in profiles if p["mean_token_gap"] is not None]
        invs = [p["inversion_count"] for p in profiles if p["inversion_count"] is not None]
        return {
            "instances": n,
            "exact_adjacent_phrase": exact,
            "pct_exact_adjacent": round(100 * exact / n, 1),
            "ordered_subsequence": ordered,
            "pct_ordered_subsequence": round(100 * ordered / n, 1),
            "inverted_order": inverted,
            "pct_inverted": round(100 * inverted / n, 1),
            "all_tokens_present": all_present,
            "pct_all_present": round(100 * all_present / n, 1),
            "order_class_counts": _count(profiles, "order_class"),
            "mean_gap_avg": round(sum(gaps) / len(gaps), 2) if gaps else None,
            "inversion_avg": round(sum(invs) / len(invs), 2) if invs else None,
        }

    # Query-level: any gold with exact phrase / ordered
    q_exact = q_ordered = q_inverted = 0
    for row in rows:
        if any(g["exact_adjacent"] for g in row["gold"]):
            q_exact += 1
        if any(g["ordered_subsequence"] for g in row["gold"]):
            q_ordered += 1
        if any(g["inverted"] for g in row["gold"]):
            q_inverted += 1

    return {
        "queries": len(rows),
        "query_level_gold_exact_adjacent": q_exact,
        "query_level_gold_ordered_subsequence": q_ordered,
        "query_level_gold_inverted": q_inverted,
        "gold": {**_agg(gold_profiles, "gold"), **_pair_order_stats(gold_profiles)},
        "false": {**_agg(false_profiles, "false"), **_pair_order_stats(false_profiles)},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--max-keys", type=int, default=1024)
    p.add_argument("--max-corr-neighbors", type=int, default=2)
    p.add_argument("--top-false", type=int, default=1)
    p.add_argument("--out", default="logs/query_word_order_audit.json")
    args = p.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    print(f"Loading brain + plane ({args.dataset}) ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)

    qids = [q for q in qrels if q in queries][: args.max_queries]
    rows: list[dict[str, object]] = []

    for qid in qids:
        qtext = queries[qid]
        query_toks = _query_token_list(qtext)
        gold_ids = list(qrels[qid].keys())

        route = route_symbol_plane_candidates(
            knowledge, plane, query_words(qtext),
            max_candidates=600,
            max_keys=args.max_keys,
            max_corr_neighbors=args.max_corr_neighbors,
        )
        ranked = rank_symbol_plane_docs(
            knowledge, plane, query_words(qtext), limit=100,
            query_keys=set(route.query_keys),
            candidate_doc_ids=route.doc_ids,
        )
        ranked_ids = [d for d, _ in ranked]
        gold_set = set(gold_ids)
        false_ids = [d for d in ranked_ids if d not in gold_set][: args.top_false]

        gold_profiles = [
            analyze_doc(
                gid, knowledge.corpus.get(gid, ""), query_toks, label="gold",
            )
            for gid in gold_ids
        ]
        false_profiles = [
            analyze_doc(
                fid, knowledge.corpus.get(fid, ""), query_toks, label="false",
            )
            for fid in false_ids
        ]

        rows.append({
            "query_id": qid,
            "query": qtext[:120],
            "n_query_tokens": len(query_toks),
            "query_token_order": query_toks,
            "gold": gold_profiles,
            "false_top": false_profiles,
        })

    summary = summarize(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "queries": rows}, indent=2),
        encoding="utf-8",
    )

    g = summary["gold"]
    f = summary["false"]
    print("\n=== Query word ORDER audit (SciFact test) ===")
    print(f"Queries: {summary['queries']}  |  gold instances: {g['instances']}  |  false top-{args.top_false}: {f['instances']}")
    print()
    print("Exact phrase (query tokens consecutive, same order in doc):")
    print(f"  Gold:  {g['exact_adjacent_phrase']}/{g['instances']} ({g['pct_exact_adjacent']}%)")
    print(f"  False: {f['exact_adjacent_phrase']}/{f['instances']} ({f['pct_exact_adjacent']}%)")
    print(f"  Queries with ANY gold exact: {summary['query_level_gold_exact_adjacent']}/{summary['queries']}")
    print()
    print("Ordered subsequence (all present tokens keep query order, may have gaps):")
    print(f"  Gold:  {g['ordered_subsequence']}/{g['instances']} ({g['pct_ordered_subsequence']}%)")
    print(f"  False: {f['ordered_subsequence']}/{f['instances']} ({f['pct_ordered_subsequence']}%)")
    print()
    print("Inverted (all tokens present but NOT in query order):")
    print(f"  Gold:  {g['inverted_order']}/{g['instances']} ({g['pct_inverted']}%)")
    print(f"  False: {f['inverted_order']}/{f['instances']} ({f['pct_inverted']}%)")
    print()
    print("All query tokens present in doc:")
    print(f"  Gold:  {g['all_tokens_present']}/{g['instances']} ({g['pct_all_present']}%)")
    print(f"  False: {f['all_tokens_present']}/{f['instances']} ({f['pct_all_present']}%)")
    print()
    print("Avg token gap (positions between consecutive query words in doc):")
    print(f"  Gold:  {g['mean_gap_avg']}  |  False: {f['mean_gap_avg']}")
    print()
    print("Token coverage:")
    print(f"  Gold  avg {g['avg_present_in_doc']}/{g['avg_query_tokens']} query tokens in doc ({g['pct_query_tokens_present']}%)")
    print(f"  False avg {f['avg_present_in_doc']}/{f['avg_query_tokens']} ({f['pct_query_tokens_present']}%)")
    print()
    print("Adjacent query PAIRS both in doc (order preserved vs inverted):")
    print(f"  Gold  same-order pairs: {g['adjacent_pairs_same_order']}/{g['adjacent_pairs_both_present']} ({g['pct_pairs_same_order']}%)")
    print(f"  Gold  inverted pairs:   {g['adjacent_pairs_inverted']}/{g['adjacent_pairs_both_present']} ({g['pct_pairs_inverted']}%)")
    print(f"  False same-order pairs: {f['adjacent_pairs_same_order']}/{f['adjacent_pairs_both_present']} ({f['pct_pairs_same_order']}%)")
    print(f"  False inverted pairs:   {f['adjacent_pairs_inverted']}/{f['adjacent_pairs_both_present']} ({f['pct_pairs_inverted']}%)")
    print()
    print("Order class breakdown (gold):", g.get("order_class_counts"))
    print("Order class breakdown (false):", f.get("order_class_counts"))
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
