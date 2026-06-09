#!/usr/bin/env python3
"""
Audit false-correlation flooding on 30 SciFact test queries.

Traces which hub vs rare query expansions drive top-ranked docs and
whether gold is drowned by spurious correlation meets.

  python scripts/audit_false_correlations.py
  python scripts/audit_false_correlations.py --out logs/false_correlation_audit_30.json
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

from aethos_query_oov import expand_oov_query_word, word_needs_oov_build
from aethos_rare_rank import degree_map_from_plane, is_hub_word
from aethos_symbol_knowledge import SymbolKnowledgeIndex
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries, ndcg_at_k, recall_at_k
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_02_attractor_key import attractor_neighbors
from pipeline.bit_04_candidate_router import query_words_for_routing
from pipeline.bit_12_symbol_plane_index import (
    _chamber_adjacency_neighbors,
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)

_TOKEN_RE = re.compile(r"[a-z]+")
_HUB_STRENGTH = 500  # degree threshold aligned with rare ranker


def _token_set(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3}


def _classify_word(
    word: str,
    degrees: dict[str, int],
    knowledge: SymbolKnowledgeIndex,
) -> str:
    w = word.lower()
    if is_hub_word(knowledge, w, degrees=degrees):
        return "hub"
    prof = knowledge.cellular.profiles.get(w)
    if prof is not None and prof.role.value == "membrane":
        return "membrane"
    return "signal"


def trace_query_key_sources(
    knowledge: SymbolKnowledgeIndex,
    plane,
    words: list[str],
    *,
    max_keys: int = 1024,
    max_corr_neighbors: int = 2,
    radius: int = 1,
) -> dict[str, object]:
    """Build κ keys with provenance: which word/correlation produced each key."""
    degrees = degree_map_from_plane(plane)
    routed = query_words_for_routing(words)
    active_chambers = knowledge.active_chambers_for_query(routed)

    keys: set = set()
    key_sources: dict = defaultdict(list)  # key -> [{word, via, kind, hub}]

    def _record(key, *, word: str, kind: str, via: str | None = None) -> bool:
        if len(keys) >= max_keys:
            return False
        keys.add(key)
        hub = _classify_word(word, degrees, knowledge) == "hub"
        if via:
            hub = hub or _classify_word(via, degrees, knowledge) == "hub"
        key_sources[key].append({
            "word": word,
            "via": via,
            "kind": kind,
            "hub": hub,
        })
        return True

    def _add_key(key, **meta) -> bool:
        ok = _record(key, **meta)
        if not ok:
            return False
        if radius > 0:
            for nk in attractor_neighbors(key, radius=radius):
                if len(keys) >= max_keys:
                    return False
                keys.add(nk)
                key_sources[nk].append({**meta, "kind": meta.get("kind", "") + "+nb"})
        return True

    expansion_log: list[dict[str, object]] = []

    for w in routed:
        wclass = _classify_word(w, degrees, knowledge)
        if word_needs_oov_build(knowledge, plane, w):
            node = knowledge.ensure_query_lattice(w, plane)
            expansion_log.append({
                "word": w,
                "class": wclass,
                "oov": True,
                "anchors": list(node.anchors)[:8],
            })
            from aethos_query_oov import ephemeral_word_kappa_keys

            for k in ephemeral_word_kappa_keys(knowledge, w, quantize=plane.quantize):
                _add_key(k, word=w, kind="oov_cell", via=None)
            for anchor in node.anchors:
                for k in plane.keys_for_word(anchor):
                    _add_key(k, word=w, kind="oov_anchor", via=anchor)
            continue

        expansion_log.append({"word": w, "class": wclass, "oov": False})
        for k in plane.keys_for_word(w):
            if not _add_key(k, word=w, kind="word_cell", via=None):
                break

        neighbors = _chamber_adjacency_neighbors(
            knowledge, w, active_chambers, max_corr_neighbors, plane=plane,
        )
        corr_rows: list[dict[str, object]] = []
        for other, strength, kind in neighbors:
            oclass = _classify_word(other, degrees, knowledge)
            corr_rows.append({
                "other": other,
                "strength": round(strength, 2),
                "link_kind": kind,
                "class": oclass,
            })
            meet = plane.pair_keys.get(tuple(sorted((w, other))))
            if meet:
                for mk in meet:
                    if not _add_key(mk, word=w, kind="corr_meet", via=other):
                        break
        expansion_log[-1]["correlations"] = corr_rows

    hub_key_count = sum(
        1 for k in keys
        if any(s.get("hub") for s in key_sources.get(k, []))
    )
    return {
        "keys": keys,
        "key_sources": {str(k): v for k, v in key_sources.items()},
        "expansion_log": expansion_log,
        "n_keys": len(keys),
        "n_hub_sourced_keys": hub_key_count,
        "hub_key_fraction": round(hub_key_count / max(len(keys), 1), 4),
    }


def doc_hit_profile(
    plane,
    doc_id: str,
    keys: set,
    key_sources: dict,
) -> dict[str, object]:
    doc_k = plane.doc_keys.get(doc_id, set())
    hit_keys = keys & doc_k
    hub_hits = 0
    rare_hits = 0
    via_counter: Counter[str] = Counter()
    for k in hit_keys:
        for src in key_sources.get(str(k), []):
            if src.get("hub"):
                hub_hits += 1
                if src.get("via"):
                    via_counter[f"{src['word']}->{src['via']}"] += 1
                else:
                    via_counter[src["word"]] += 1
            else:
                rare_hits += 1
                if src.get("via"):
                    via_counter[f"{src['word']}->{src['via']}"] += 1
    jaccard = len(hit_keys) / len(keys | doc_k) if (keys | doc_k) else 0.0
    return {
        "doc_id": doc_id,
        "hit_keys": len(hit_keys),
        "hub_hits": hub_hits,
        "rare_hits": rare_hits,
        "hub_driven": hub_hits > rare_hits,
        "jaccard": round(jaccard, 5),
        "top_via": via_counter.most_common(6),
    }


def audit_query(
    knowledge: SymbolKnowledgeIndex,
    plane,
    query_id: str,
    query_text: str,
    gold_ids: list[str],
    *,
    max_keys: int,
    max_corr_neighbors: int,
) -> dict[str, object]:
    words = query_words(query_text)
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

    top5_profiles = [
        doc_hit_profile(plane, did, keys, key_sources)
        for did, _ in ranked[:5]
    ]
    gold_profiles = [
        doc_hit_profile(plane, gid, keys, key_sources)
        for gid in gold_ids
    ]
    gold_rank = next(
        (i + 1 for i, did in enumerate(ranked_ids) if did in gold_set),
        None,
    )

    # false flood: top docs hub-driven while gold has low hub hits
    flood_pattern = None
    if ranked_ids and gold_rank and gold_rank > 10:
        top = top5_profiles[0]
        gold_p = gold_profiles[0] if gold_profiles else {}
        if top.get("hub_driven") and not gold_p.get("hub_driven"):
            flood_pattern = "hub_correlation_drowns_gold"
        elif top.get("hit_keys", 0) > gold_p.get("hit_keys", 0) * 2:
            flood_pattern = "hub_key_count_drowns_gold"

    # spurious correlation list from expansion
    spurious_corr: list[dict[str, object]] = []
    for row in trace["expansion_log"]:
        for c in row.get("correlations", []) or []:
            if c.get("class") == "hub":
                spurious_corr.append({
                    "query_word": row["word"],
                    "hub_neighbor": c["other"],
                    "strength": c["strength"],
                })

    return {
        "query_id": query_id,
        "query": query_text[:140],
        "query_words": words,
        "ndcg_at_10": round(ndcg_at_k(ranked_ids, {g: 1 for g in gold_ids}, 10), 4),
        "recall_at_100": recall_at_k(ranked_ids, {g: 1 for g in gold_ids}, 100),
        "gold_rank": gold_rank,
        "gold_in_route": any(g in route.doc_ids for g in gold_ids),
        "n_query_keys": trace["n_keys"],
        "hub_key_fraction": trace["hub_key_fraction"],
        "expansion_log": trace["expansion_log"],
        "spurious_hub_correlations": spurious_corr[:12],
        "top5_docs": top5_profiles,
        "gold_docs": gold_profiles,
        "flood_pattern": flood_pattern,
    }


def aggregate(reports: list[dict[str, object]]) -> dict[str, object]:
    patterns: Counter[str] = Counter()
    hub_neighbors: Counter[str] = Counter()
    flooded = 0
    zero_ndcg = 0
    for r in reports:
        if r.get("flood_pattern"):
            flooded += 1
            patterns[str(r["flood_pattern"])] += 1
        if r.get("ndcg_at_10") == 0:
            zero_ndcg += 1
        for s in r.get("spurious_hub_correlations", []):
            hub_neighbors[f"{s['query_word']}->{s['hub_neighbor']}"] += 1

    return {
        "n_queries": len(reports),
        "zero_ndcg_at_10": zero_ndcg,
        "hub_flood_cases": flooded,
        "flood_patterns": dict(patterns),
        "top_spurious_hub_expansions": hub_neighbors.most_common(15),
        "mean_hub_key_fraction": round(
            sum(r.get("hub_key_fraction", 0) for r in reports) / max(len(reports), 1),
            4,
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="False correlation flood audit (30 queries)")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--max-keys", type=int, default=1024)
    p.add_argument("--max-corr", type=int, default=2)
    p.add_argument("--out", default=str(_ROOT / "logs" / "false_correlation_audit_30.json"))
    args = p.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    qids = [q for q in qrels if q in queries][: args.max_queries]

    print(f"Loading brain ({args.dataset}) ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)
    print(f"Auditing {len(qids)} queries ...", flush=True)

    reports = [
        audit_query(
            knowledge, plane, qid, queries[qid], list(qrels[qid].keys()),
            max_keys=args.max_keys,
            max_corr_neighbors=args.max_corr,
        )
        for qid in qids
    ]
    summary = aggregate(reports)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "max_keys": args.max_keys,
        "max_corr_neighbors": args.max_corr,
        "summary": summary,
        "queries": reports,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n--- false correlation audit ---")
    print(f"queries          : {summary['n_queries']}")
    print(f"zero nDCG@10     : {summary['zero_ndcg_at_10']}")
    print(f"hub flood cases  : {summary['hub_flood_cases']}")
    print(f"mean hub key frac: {summary['mean_hub_key_fraction']}")
    print("top spurious hub expansions:")
    for pair, cnt in summary["top_spurious_hub_expansions"][:8]:
        print(f"  {pair}  ({cnt}x)")
    print(f"JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
