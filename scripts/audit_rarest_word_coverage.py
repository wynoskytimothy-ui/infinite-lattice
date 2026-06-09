#!/usr/bin/env python3
"""
Audit rarest-query-word coverage in gold vs false (top non-gold) docs.

For each SciFact test query:
  - Rank query tokens by corpus doc-frequency (rarest first)
  - Check gold docs vs top-ranked false doc for 1/2/3 rarest word presence
  - Trace whether κ overlap comes from direct word cells vs cross-correlation meets

  python scripts/audit_rarest_word_coverage.py
  python scripts/audit_rarest_word_coverage.py --max-queries 30 --top-false 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_rare_rank import _DocFreqCache, is_rare_word, degree_map_from_plane
from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)
from scripts.audit_false_correlations import trace_query_key_sources

_TOKEN_RE = re.compile(r"[a-z]+")


def _doc_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3}


def _rarest_query_words(
    knowledge,
    words: list[str],
    cache: _DocFreqCache,
) -> list[str]:
    """Query tokens sorted rarest-first (lowest doc freq, excluding membrane)."""
    routed = [
        w.lower() for w in words
        if w.lower() not in _DEFAULT_MEMBRANE and len(w) >= 3
    ]
    unique = list(dict.fromkeys(routed))
    return sorted(unique, key=lambda w: (cache.get(w), w))


def _prefix_hits(doc_toks: set[str], rarest: list[str], n: int) -> tuple[int, list[str]]:
    want = rarest[:n]
    hit = [w for w in want if w in doc_toks]
    return len(hit), hit


def _cross_corr_hits(
    knowledge,
    query_rare: list[str],
    doc_toks: set[str],
) -> dict[str, object]:
    """Direct rare overlap + stored cross-link strengths query→doc."""
    overlap = [w for w in query_rare if w in doc_toks]
    pair_links: list[dict[str, object]] = []
    for qw in query_rare:
        for dw in doc_toks:
            if qw == dw:
                continue
            lk = knowledge.correlates(qw, dw)
            if lk is not None:
                pair_links.append({
                    "query": qw,
                    "doc": dw,
                    "strength": round(lk.strength, 2),
                    "kind": lk.kind,
                })
    pair_links.sort(key=lambda x: (-x["strength"], x["query"], x["doc"]))
    return {
        "direct_overlap": overlap,
        "n_direct": len(overlap),
        "cross_links": pair_links[:12],
        "n_cross_links": len(pair_links),
        "has_cross_corr": len(pair_links) > 0,
    }


def _kappa_source_profile(
    plane,
    doc_id: str,
    keys: set,
    key_sources: dict,
) -> dict[str, object]:
    doc_k = plane.doc_keys.get(doc_id, set())
    hit_keys = keys & doc_k
    word_cell = corr_meet = oov = nb = 0
    for k in hit_keys:
        for src in key_sources.get(str(k), []):
            kind = src.get("kind", "")
            if kind == "word_cell":
                word_cell += 1
            elif kind == "corr_meet":
                corr_meet += 1
            elif kind.startswith("oov"):
                oov += 1
            elif "+nb" in kind:
                nb += 1
    total = word_cell + corr_meet + oov + nb
    return {
        "hit_keys": len(hit_keys),
        "word_cell_hits": word_cell,
        "corr_meet_hits": corr_meet,
        "oov_hits": oov,
        "neighbor_hits": nb,
        "from_cross_corr": corr_meet > 0,
        "cross_corr_dominant": corr_meet > word_cell,
        "total_source_hits": total,
    }


def audit_one(
    knowledge,
    plane,
    qid: str,
    query_text: str,
    gold_ids: list[str],
    *,
    max_keys: int,
    max_corr_neighbors: int,
    top_false: int,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> dict[str, object]:
    words = query_words(query_text)
    rarest = _rarest_query_words(knowledge, words, cache)
    rare_query = [
        w for w in rarest
        if is_rare_word(knowledge, w, df_cache=cache, degrees=degrees)
    ]

    trace = trace_query_key_sources(
        knowledge, plane, words,
        max_keys=max_keys, max_corr_neighbors=max_corr_neighbors,
    )
    keys = trace["keys"]
    key_sources = trace["key_sources"]

    route = route_symbol_plane_candidates(
        knowledge, plane, words,
        max_candidates=600,
        max_keys=max_keys,
        max_corr_neighbors=max_corr_neighbors,
    )
    ranked = rank_symbol_plane_docs(
        knowledge, plane, words, limit=100,
        query_keys=set(route.query_keys),
        candidate_doc_ids=route.doc_ids,
    )
    ranked_ids = [d for d, _ in ranked]
    gold_set = set(gold_ids)

    false_ids = [d for d in ranked_ids if d not in gold_set][:top_false]

    def _profile(doc_id: str, label: str) -> dict[str, object]:
        text = knowledge.corpus.get(doc_id, "")
        toks = _doc_tokens(text)
        h1, w1 = _prefix_hits(toks, rarest, 1)
        h2, w2 = _prefix_hits(toks, rarest, 2)
        h3, w3 = _prefix_hits(toks, rarest, 3)
        return {
            "doc_id": doc_id,
            "label": label,
            "has_rarest_1": h1 >= 1,
            "has_rarest_2": h2 >= 2,
            "has_rarest_3": h3 >= 3,
            "rarest_words": rarest[:3],
            "hit_rarest_1": w1,
            "hit_rarest_2": w2,
            "hit_rarest_3": w3,
            "cross_corr": _cross_corr_hits(knowledge, rare_query or rarest[:3], toks),
            "kappa_sources": _kappa_source_profile(plane, doc_id, keys, key_sources),
        }

    gold_profiles = [_profile(g, "gold") for g in gold_ids]
    false_profiles = [_profile(d, "false") for d in false_ids]

    return {
        "query_id": qid,
        "query": query_text[:100],
        "rarest_query_words": rarest[:6],
        "rare_query_words": rare_query[:6],
        "n_gold": len(gold_ids),
        "gold": gold_profiles,
        "false_top": false_profiles,
        "gold_in_route": any(g in route.doc_ids for g in gold_set),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate counts across all gold doc instances and false doc instances."""
    agg = {
        "queries": len(rows),
        "gold_doc_instances": 0,
        "false_doc_instances": 0,
        "gold_has_rarest_1": 0,
        "gold_has_rarest_2": 0,
        "gold_has_rarest_3": 0,
        "false_has_rarest_1": 0,
        "false_has_rarest_2": 0,
        "false_has_rarest_3": 0,
        "gold_has_cross_corr_link": 0,
        "false_has_cross_corr_link": 0,
        "gold_kappa_from_cross_corr": 0,
        "false_kappa_from_cross_corr": 0,
        "gold_cross_corr_dominant": 0,
        "false_cross_corr_dominant": 0,
        "gold_direct_rare_overlap": 0,
        "false_direct_rare_overlap": 0,
    }

    for row in rows:
        for g in row["gold"]:
            agg["gold_doc_instances"] += 1
            if g["has_rarest_1"]:
                agg["gold_has_rarest_1"] += 1
            if g["has_rarest_2"]:
                agg["gold_has_rarest_2"] += 1
            if g["has_rarest_3"]:
                agg["gold_has_rarest_3"] += 1
            if g["cross_corr"]["has_cross_corr"]:
                agg["gold_has_cross_corr_link"] += 1
            if g["cross_corr"]["n_direct"] > 0:
                agg["gold_direct_rare_overlap"] += 1
            ks = g["kappa_sources"]
            if ks["from_cross_corr"]:
                agg["gold_kappa_from_cross_corr"] += 1
            if ks["cross_corr_dominant"]:
                agg["gold_cross_corr_dominant"] += 1

        for f in row["false_top"]:
            agg["false_doc_instances"] += 1
            if f["has_rarest_1"]:
                agg["false_has_rarest_1"] += 1
            if f["has_rarest_2"]:
                agg["false_has_rarest_2"] += 1
            if f["has_rarest_3"]:
                agg["false_has_rarest_3"] += 1
            if f["cross_corr"]["has_cross_corr"]:
                agg["false_has_cross_corr_link"] += 1
            if f["cross_corr"]["n_direct"] > 0:
                agg["false_direct_rare_overlap"] += 1
            ks = f["kappa_sources"]
            if ks["from_cross_corr"]:
                agg["false_kappa_from_cross_corr"] += 1
            if ks["cross_corr_dominant"]:
                agg["false_cross_corr_dominant"] += 1

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / max(d, 1), 1)

    agg["gold_pct_rarest_1"] = pct(agg["gold_has_rarest_1"], agg["gold_doc_instances"])
    agg["gold_pct_rarest_2"] = pct(agg["gold_has_rarest_2"], agg["gold_doc_instances"])
    agg["gold_pct_rarest_3"] = pct(agg["gold_has_rarest_3"], agg["gold_doc_instances"])
    agg["false_pct_rarest_1"] = pct(agg["false_has_rarest_1"], agg["false_doc_instances"])
    agg["false_pct_rarest_2"] = pct(agg["false_has_rarest_2"], agg["false_doc_instances"])
    agg["false_pct_rarest_3"] = pct(agg["false_has_rarest_3"], agg["false_doc_instances"])
    agg["gold_pct_cross_corr_link"] = pct(
        agg["gold_has_cross_corr_link"], agg["gold_doc_instances"],
    )
    agg["false_pct_cross_corr_link"] = pct(
        agg["false_has_cross_corr_link"], agg["false_doc_instances"],
    )
    agg["gold_pct_kappa_cross_corr"] = pct(
        agg["gold_kappa_from_cross_corr"], agg["gold_doc_instances"],
    )
    agg["false_pct_kappa_cross_corr"] = pct(
        agg["false_kappa_from_cross_corr"], agg["false_doc_instances"],
    )
    return agg


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--max-keys", type=int, default=1024)
    p.add_argument("--max-corr-neighbors", type=int, default=2)
    p.add_argument("--top-false", type=int, default=1,
                   help="top-ranked non-gold docs per query (default: rank-1 false)")
    p.add_argument("--out", default="logs/rarest_word_coverage_audit.json")
    args = p.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)

    print(f"Loading brain + plane ({args.dataset}) ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)

    qids = [q for q in qrels if q in queries][: args.max_queries]
    rows: list[dict[str, object]] = []
    for qid in qids:
        rows.append(audit_one(
            knowledge, plane, qid, queries[qid], list(qrels[qid].keys()),
            max_keys=args.max_keys,
            max_corr_neighbors=args.max_corr_neighbors,
            top_false=args.top_false,
            cache=cache,
            degrees=degrees,
        ))

    summary = summarize(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "queries": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    s = summary
    gi = s["gold_doc_instances"]
    fi = s["false_doc_instances"]
    print("\n=== Rarest query-word coverage (SciFact test queries) ===")
    print(f"Gold doc instances: {gi}  |  False top-{args.top_false} instances: {fi}")
    print()
    print("Contains rarest query words in doc text:")
    print(f"  Gold  - 1 rarest: {s['gold_has_rarest_1']}/{gi} ({s['gold_pct_rarest_1']}%)")
    print(f"  Gold  - 2 rarest: {s['gold_has_rarest_2']}/{gi} ({s['gold_pct_rarest_2']}%)")
    print(f"  Gold  - 3 rarest: {s['gold_has_rarest_3']}/{gi} ({s['gold_pct_rarest_3']}%)")
    print(f"  False - 1 rarest: {s['false_has_rarest_1']}/{fi} ({s['false_pct_rarest_1']}%)")
    print(f"  False - 2 rarest: {s['false_has_rarest_2']}/{fi} ({s['false_pct_rarest_2']}%)")
    print(f"  False - 3 rarest: {s['false_has_rarest_3']}/{fi} ({s['false_pct_rarest_3']}%)")
    print()
    print("Cross-correlation signal (stored brain link query_rare -> doc_token):")
    print(f"  Gold  has link: {s['gold_has_cross_corr_link']}/{gi} ({s['gold_pct_cross_corr_link']}%)")
    print(f"  False has link: {s['false_has_cross_corr_link']}/{fi} ({s['false_pct_cross_corr_link']}%)")
    print()
    print("kappa overlap from correlation meet keys (not just direct word cell):")
    print(f"  Gold  kappa via cross-corr: {s['gold_kappa_from_cross_corr']}/{gi} ({s['gold_pct_kappa_cross_corr']}%)")
    print(f"  False kappa via cross-corr: {s['false_kappa_from_cross_corr']}/{fi} ({s['false_pct_kappa_cross_corr']}%)")
    print(f"\nJSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
