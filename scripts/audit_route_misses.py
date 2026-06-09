#!/usr/bin/env python3
"""
Audit BEIR symbol eval routing misses — gold not in kappa candidate pool.

  python scripts/audit_route_misses.py
  python scripts/audit_route_misses.py --eval-json logs/eval_beir_symbol_scifact_test.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex
from aethos_tokenize import tokenize_words
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    query_symbol_plane_keys,
    symbol_word_chain,
)

_TOKEN_RE = re.compile(r"[a-z]+")


def _content_tokens(text: str, *, min_len: int = 3) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= min_len}


def diagnose_miss(
    knowledge: SymbolKnowledgeIndex,
    plane,
    query_id: str,
    query_text: str,
    gold_ids: list[str],
) -> dict[str, object]:
    words = query_words(query_text)
    qset = set(words)
    keys = query_symbol_plane_keys(knowledge, plane, words)

    gold_reports: list[dict[str, object]] = []
    for gid in gold_ids:
        gtext = knowledge.corpus.get(gid, "")
        gtoks = _content_tokens(gtext)
        overlap = sorted(qset & gtoks)
        missing = sorted(qset - gtoks)[:12]

        # query term links in brain
        term_links: list[dict[str, object]] = []
        for w in words[:8]:
            nbrs = knowledge.neighbors(w, kinds={"direct"})[:5]
            term_links.append({
                "word": w,
                "degree": len(knowledge.neighbors(w)),
                "top_neighbors": [
                    (lk.right if lk.left == w else lk.left, lk.strength)
                    for lk in nbrs
                ],
            })

        # gold doc kappa keys vs query keys
        gold_keys = plane.doc_keys.get(gid, set())
        key_overlap = len(keys & gold_keys)
        key_union = len(keys | gold_keys) or 1
        jaccard = key_overlap / key_union

        # do any gold tokens share kappa with query?
        gold_word_hits: list[str] = []
        for w in sorted(overlap)[:10]:
            if plane.word_keys.get(w) and keys & plane.word_keys.get(w, set()):
                gold_word_hits.append(w)

        # classify failure
        if not overlap:
            kind = "vocabulary_zero_overlap"
        elif key_overlap == 0:
            if len(overlap) <= 1:
                kind = "kappa_miss_sparse_overlap"
            else:
                kind = "kappa_miss_despite_overlap"
        else:
            kind = "rank_miss_gold_in_keys"

        gold_reports.append({
            "gold_doc_id": gid,
            "gold_snippet": gtext[:200].replace("\n", " "),
            "token_overlap": overlap,
            "query_terms_missing_in_gold": missing,
            "query_kappa_keys": len(keys),
            "gold_kappa_keys": len(gold_keys),
            "kappa_key_overlap": key_overlap,
            "kappa_jaccard": round(jaccard, 4),
            "overlap_words_in_plane": gold_word_hits,
            "failure_kind": kind,
        })

    return {
        "query_id": query_id,
        "query": query_text,
        "query_words": words,
        "gold_reports": gold_reports,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-json", default=str(_ROOT / "logs" / "eval_beir_symbol_scifact_test.json"))
    p.add_argument("--brain", default="scifact")
    p.add_argument("--route-only", action="store_true", default=True)
    p.add_argument("--out", default=str(_ROOT / "logs" / "route_miss_audit.json"))
    args = p.parse_args()

    eval_path = Path(args.eval_json)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    failures = data.get("failures", [])
    misses = [f for f in failures if not f.get("gold_in_route")]
    print(f"Loaded {eval_path.name}: {len(misses)} routing misses / {data.get('n_queries')} queries")

    knowledge, plane = load_brain_and_plane(args.brain)

    reports: list[dict[str, object]] = []
    kind_counts: dict[str, int] = {}

    for row in misses:
        rep = diagnose_miss(
            knowledge,
            plane,
            str(row["query_id"]),
            row["query"],
            list(row["gold"]),
        )
        reports.append(rep)
        for gr in rep["gold_reports"]:
            k = gr["failure_kind"]
            kind_counts[k] = kind_counts.get(k, 0) + 1

        print("\n" + "=" * 70)
        print(f"Q{rep['query_id']}: {rep['query'][:90]}...")
        print(f"  query_words: {rep['query_words']}")
        for gr in rep["gold_reports"]:
            print(f"  gold {gr['gold_doc_id']}: {gr['failure_kind']}")
            print(f"    overlap={gr['token_overlap']}")
            print(f"    missing_in_gold={gr['query_terms_missing_in_gold']}")
            print(f"    kappa overlap={gr['kappa_key_overlap']} / q={gr['query_kappa_keys']} g={gr['gold_kappa_keys']}")
            print(f"    snippet: {gr['gold_snippet'][:100]}...")

    summary = {
        "source_eval": str(eval_path),
        "n_route_misses": len(misses),
        "failure_kinds": kind_counts,
        "reports": reports,
        "notes": {
            "vocabulary_zero_overlap": "Query content words never appear in gold abstract — claim uses different terms.",
            "kappa_miss_sparse_overlap": "Few shared words; no shared kappa buckets between query and gold.",
            "kappa_miss_despite_overlap": "Shared words exist but plane keys do not meet — correlation/window gap.",
            "rank_miss_gold_in_keys": "Gold shares kappa keys but not in top-600 route pool (ranking cap).",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 70)
    print(f"Failure kinds: {kind_counts}")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
