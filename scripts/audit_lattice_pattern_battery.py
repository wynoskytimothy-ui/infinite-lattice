#!/usr/bin/env python3
"""
Lattice pattern battery — interpretable tests (not RAG metrics).

Tests subword quality, cross-correlation dilution, meet witnesses, triples.

  python scripts/audit_lattice_pattern_battery.py
  python scripts/audit_lattice_pattern_battery.py --max-queries 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_rare_rank import (
    _DocFreqCache,
    degree_map_from_plane,
    is_hub_word,
    is_rare_word,
    morph_trigger_pieces,
)
from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from beir_data_root import resolve_beir_root
from eval_beir import load_paths, load_qrels, load_queries
from eval_beir_symbol import load_brain_and_plane, query_words
from pipeline.bit_12_symbol_plane_index import (
    correlation_meet_keys,
    rank_symbol_plane_docs,
    route_symbol_plane_candidates,
)
from scripts.audit_false_correlations import doc_hit_profile, trace_query_key_sources

_TOKEN_RE = re.compile(r"[a-z]{2,}", re.I)


def _doc_toks(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _DEFAULT_MEMBRANE and len(t) >= 3
    }


def _first_positions(doc_toks_list: list[str], query_toks: list[str]) -> list[int | None]:
    pos = {t: i for i, t in enumerate(doc_toks_list)}
    return [pos.get(w) for w in query_toks]


def _promoted_morph_pieces(knowledge, token: str) -> list[str]:
    morph = knowledge.morph
    out: list[str] = []
    for piece in morph_trigger_pieces(knowledge, token):
        pl = piece.lower()
        if pl in morph.subwords or pl in morph.composites:
            out.append(pl)
    return list(dict.fromkeys(out))


def _junk_morph_pieces(knowledge, token: str) -> list[str]:
    """Substring pieces not in promoted morph registry."""
    promoted = set(_promoted_morph_pieces(knowledge, token))
    w = token.lower()
    junk: list[str] = []
    for piece in morph_trigger_pieces(knowledge, token):
        pl = piece.lower()
        if pl not in promoted and len(pl) >= 3 and pl in w:
            junk.append(pl)
    return list(dict.fromkeys(junk))


def _corr_profile(
    knowledge,
    query_words: list[str],
    doc_toks: set[str],
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> dict[str, object]:
    hub_n = rare_n = 0
    hub_strength = rare_strength = 0.0
    top_hub: list[tuple[str, str, float]] = []
    top_rare: list[tuple[str, str, float]] = []

    for qw in query_words:
        ql = qw.lower()
        for dt in doc_toks:
            lk = knowledge.correlates(ql, dt)
            if lk is None:
                continue
            q_hub = is_hub_word(knowledge, ql, df_cache=cache, degrees=degrees)
            d_hub = is_hub_word(knowledge, dt, df_cache=cache, degrees=degrees)
            if q_hub or d_hub:
                hub_n += 1
                hub_strength += lk.strength
                top_hub.append((ql, dt, lk.strength))
            else:
                rare_n += 1
                rare_strength += lk.strength
                top_rare.append((ql, dt, lk.strength))

    top_hub.sort(key=lambda x: -x[2])
    top_rare.sort(key=lambda x: -x[2])
    total = hub_n + rare_n
    return {
        "hub_link_count": hub_n,
        "rare_link_count": rare_n,
        "hub_strength_sum": round(hub_strength, 2),
        "rare_strength_sum": round(rare_strength, 2),
        "hub_dilution_ratio": round(hub_n / max(total, 1), 3),
        "top_hub_links": [f"{a}->{b}({s:.0f})" for a, b, s in top_hub[:5]],
        "top_rare_links": [f"{a}->{b}({s:.0f})" for a, b, s in top_rare[:5]],
    }


def _morph_profile(knowledge, query_words: list[str], text: str) -> dict[str, object]:
    doc_list = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3]
    doc_set = set(doc_list)
    promoted_hits: list[str] = []
    junk_hits: list[str] = []
    for qw in query_words:
        for pl in _promoted_morph_pieces(knowledge, qw):
            if re.search(re.escape(pl), text, re.I):
                if pl not in promoted_hits:
                    promoted_hits.append(pl)
        for jl in _junk_morph_pieces(knowledge, qw):
            if re.search(re.escape(jl), text, re.I):
                if jl not in junk_hits:
                    junk_hits.append(jl)
    return {
        "promoted_subword_hits": promoted_hits,
        "junk_substring_hits": junk_hits,
        "promoted_count": len(promoted_hits),
        "junk_count": len(junk_hits),
        "morph_signal_ratio": round(
            len(promoted_hits) / max(len(promoted_hits) + len(junk_hits), 1), 3,
        ),
    }


def _pair_order_profile(query_words: list[str], text: str) -> dict[str, object]:
    doc_list = [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _DEFAULT_MEMBRANE and len(t) >= 3
    ]
    fp = _first_positions(doc_list, [w.lower() for w in query_words])
    same = inv = total = 0
    for i in range(len(fp) - 1):
        if fp[i] is None or fp[i + 1] is None:
            continue
        total += 1
        if fp[i] < fp[i + 1]:
            same += 1
        elif fp[i] > fp[i + 1]:
            inv += 1
    return {
        "adjacent_pairs_present": total,
        "pairs_same_order": same,
        "pairs_inverted": inv,
        "pair_order_ratio": round(same / max(total, 1), 3),
    }


def _meet_witness_score(
    knowledge,
    plane,
    query_words: list[str],
    doc_id: str,
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> dict[str, object]:
    rare_q = [
        w.lower() for w in query_words
        if is_rare_word(knowledge, w, df_cache=cache, degrees=degrees)
    ]
    doc_k = plane.doc_keys.get(doc_id, set())
    word_overlap = 0
    meet_overlap = 0.0
    meet_pairs = 0

    for w in query_words:
        wl = w.lower()
        for k in plane.keys_for_word(wl):
            if k in doc_k:
                word_overlap += 1

    for a, b in combinations(rare_q, 2):
        lk = knowledge.correlates(a, b)
        pair = tuple(sorted((a, b)))
        meet = plane.pair_keys.get(pair)
        if not meet:
            meet = correlation_meet_keys(knowledge, a, b, link=lk, quantize=plane.quantize)
        hits = len(meet & doc_k)
        if hits:
            meet_pairs += 1
            meet_overlap += hits * (lk.strength if lk else 1.0)

    return {
        "rare_query_tokens": len(rare_q),
        "word_cell_key_overlap": word_overlap,
        "rare_pair_meet_hits": meet_pairs,
        "meet_witness_score": round(meet_overlap, 2),
    }


def _triple_witness(
    knowledge,
    query_words: list[str],
    doc_toks: set[str],
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
) -> dict[str, object]:
    rare_q = [
        w.lower() for w in query_words
        if is_rare_word(knowledge, w, df_cache=cache, degrees=degrees)
    ]
    rare_d = [
        t for t in doc_toks
        if is_rare_word(knowledge, t, df_cache=cache, degrees=degrees)
        and t not in rare_q
    ]
    triples: list[str] = []
    for a, b in combinations(rare_q, 2):
        ab = knowledge.correlates(a, b)
        if ab is None:
            continue
        for c in rare_d:
            ac = knowledge.correlates(a, c)
            bc = knowledge.correlates(b, c)
            if ac and bc:
                strength = min(ab.strength, ac.strength, bc.strength)
                triples.append(f"{a}+{b}+{c}({strength:.0f})")
    triples.sort(key=lambda x: -float(x.split("(")[1].rstrip(")")))
    return {
        "triple_witness_count": len(triples),
        "top_triples": triples[:4],
    }


def analyze_doc(
    knowledge,
    plane,
    doc_id: str,
    text: str,
    query_words: list[str],
    *,
    cache: _DocFreqCache,
    degrees: dict[str, int],
    keys,
    key_sources: dict,
    label: str,
) -> dict[str, object]:
    doc_toks = _doc_toks(text)
    kappa = doc_hit_profile(plane, doc_id, keys, key_sources)
    return {
        "doc_id": doc_id,
        "label": label,
        "corr": _corr_profile(knowledge, query_words, doc_toks, cache=cache, degrees=degrees),
        "morph": _morph_profile(knowledge, query_words, text),
        "pair_order": _pair_order_profile(query_words, text),
        "meet": _meet_witness_score(
            knowledge, plane, query_words, doc_id, cache=cache, degrees=degrees,
        ),
        "triple": _triple_witness(
            knowledge, query_words, doc_toks, cache=cache, degrees=degrees,
        ),
        "kappa": {
            "hit_keys": kappa["hit_keys"],
            "hub_hits": kappa["hub_hits"],
            "rare_hits": kappa["rare_hits"],
            "hub_driven": kappa["hub_driven"],
            "jaccard": kappa["jaccard"],
            "top_via": kappa["top_via"],
        },
    }


def _aggregate(profiles: list[dict], label: str) -> dict[str, object]:
    n = len(profiles)
    if n == 0:
        return {"label": label, "n": 0}

    def avg(path: str) -> float:
        vals = []
        for p in profiles:
            cur = p
            for part in path.split("."):
                cur = cur[part]
            vals.append(float(cur))
        return round(sum(vals) / len(vals), 3)

    hub_driven = sum(1 for p in profiles if p["kappa"]["hub_driven"])
    has_triple = sum(1 for p in profiles if p["triple"]["triple_witness_count"] > 0)
    has_meet = sum(1 for p in profiles if p["meet"]["rare_pair_meet_hits"] > 0)

    return {
        "label": label,
        "n": n,
        "avg_hub_dilution_ratio": avg("corr.hub_dilution_ratio"),
        "avg_rare_strength_sum": avg("corr.rare_strength_sum"),
        "avg_hub_strength_sum": avg("corr.hub_strength_sum"),
        "avg_morph_signal_ratio": avg("morph.morph_signal_ratio"),
        "avg_junk_substring_hits": avg("morph.junk_count"),
        "avg_promoted_subword_hits": avg("morph.promoted_count"),
        "avg_pair_order_ratio": avg("pair_order.pair_order_ratio"),
        "avg_meet_witness_score": avg("meet.meet_witness_score"),
        "pct_hub_driven_kappa": round(100 * hub_driven / n, 1),
        "pct_has_triple_witness": round(100 * has_triple / n, 1),
        "pct_has_rare_pair_meet": round(100 * has_meet / n, 1),
        "avg_jaccard": avg("kappa.jaccard"),
    }


def run_battery(
    knowledge,
    plane,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    qids: list[str],
    *,
    max_keys: int,
    max_corr_neighbors: int,
) -> dict[str, object]:
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)

    per_query: list[dict] = []
    all_gold: list[dict] = []
    all_false1: list[dict] = []

    for qid in qids:
        words = query_words(queries[qid])
        trace = trace_query_key_sources(
            knowledge, plane, words,
            max_keys=max_keys, max_corr_neighbors=max_corr_neighbors,
        )
        keys = trace["keys"]
        key_sources = trace["key_sources"]

        route = route_symbol_plane_candidates(
            knowledge, plane, words,
            max_candidates=600, max_keys=max_keys,
            max_corr_neighbors=max_corr_neighbors,
        )
        ranked = rank_symbol_plane_docs(
            knowledge, plane, words, limit=10,
            query_keys=set(route.query_keys),
            candidate_doc_ids=route.doc_ids,
        )
        ranked_ids = [d for d, _ in ranked]
        gold_ids = list(qrels[qid].keys())
        gold_set = set(gold_ids)
        false1 = ranked_ids[0] if ranked_ids and ranked_ids[0] not in gold_set else None

        gold_profiles = [
            analyze_doc(
                knowledge, plane, gid, knowledge.corpus.get(gid, ""),
                words, cache=cache, degrees=degrees,
                keys=keys, key_sources=key_sources, label="gold",
            )
            for gid in gold_ids
        ]
        false_profile = None
        if false1:
            false_profile = analyze_doc(
                knowledge, plane, false1, knowledge.corpus.get(false1, ""),
                words, cache=cache, degrees=degrees,
                keys=keys, key_sources=key_sources, label="false_top1",
            )

        all_gold.extend(gold_profiles)
        if false_profile:
            all_false1.append(false_profile)

        gold_rank = next(
            (i + 1 for i, d in enumerate(ranked_ids) if d in gold_set), None,
        )

        per_query.append({
            "query_id": qid,
            "query": queries[qid][:100],
            "gold_rank": gold_rank,
            "gold_in_route": any(g in route.doc_ids for g in gold_set),
            "gold": gold_profiles,
            "false_top1": false_profile,
            "hidden_pattern_flags": _pattern_flags(gold_profiles, false_profile, gold_rank),
        })

    summary = {
        "n_queries": len(qids),
        "gold_aggregate": _aggregate(all_gold, "gold"),
        "false_top1_aggregate": _aggregate(all_false1, "false_top1"),
        "pattern_flag_counts": Counter(
            f for row in per_query for f in row["hidden_pattern_flags"]
        ),
        "tests": {
            "T1_hub_dilution": "hub_link_count / total cross-links query->doc",
            "T2_morph_quality": "promoted L2 subwords vs junk substring hits",
            "T3_pair_order": "adjacent query token pairs preserved in doc order",
            "T4_meet_witness": "rare pair meet key overlap x link strength",
            "T5_triple_witness": "rare q_a + rare q_b + rare doc_c all linked",
            "T6_kappa_decompose": "hub_hits vs rare_hits in kappa overlap",
        },
        "tuning_signals": _tuning_signals(
            _aggregate(all_gold, "gold"),
            _aggregate(all_false1, "false_top1"),
        ),
    }
    return {"summary": summary, "queries": per_query}


def _pattern_flags(
    gold: list[dict],
    false1: dict | None,
    gold_rank: int | None,
) -> list[str]:
    flags: list[str] = []
    if not gold or not false1:
        return flags
    g = gold[0]
    f = false1

    if f["corr"]["hub_dilution_ratio"] > g["corr"]["hub_dilution_ratio"] + 0.15:
        flags.append("false_more_hub_diluted")
    if f["morph"]["junk_count"] > g["morph"]["junk_count"]:
        flags.append("false_more_junk_morph")
    if g["triple"]["triple_witness_count"] > 0 and f["triple"]["triple_witness_count"] == 0:
        flags.append("gold_has_triple_false_lacks")
    if g["meet"]["meet_witness_score"] > f["meet"]["meet_witness_score"]:
        flags.append("gold_stronger_meet_witness")
    if f["kappa"]["hub_driven"] and not g["kappa"]["hub_driven"]:
        flags.append("kappa_hub_flood")
    if gold_rank and gold_rank > 5:
        if f["pair_order"]["pair_order_ratio"] < g["pair_order"]["pair_order_ratio"]:
            flags.append("gold_better_pair_order_but_loses_rank")
    if g["meet"]["word_cell_key_overlap"] == 0 and g["triple"]["triple_witness_count"] > 0:
        flags.append("bridge_only_gold_no_literal")
    return flags


def _tuning_signals(gold_agg: dict, false_agg: dict) -> list[str]:
    signals: list[str] = []
    if not gold_agg.get("n") or not false_agg.get("n"):
        return signals

    if false_agg["avg_hub_dilution_ratio"] > gold_agg["avg_hub_dilution_ratio"]:
        signals.append("Penalize hub cross-links: false top-1 more hub-diluted than gold")
    if false_agg["avg_junk_substring_hits"] > gold_agg["avg_junk_substring_hits"]:
        signals.append("Use promoted L2 morph only; junk substrings higher in false docs")
    if gold_agg["avg_pair_order_ratio"] > false_agg["avg_pair_order_ratio"]:
        signals.append("Boost ordered rare bigrams; gold preserves pair order more")
    if gold_agg["avg_meet_witness_score"] < false_agg["avg_meet_witness_score"]:
        signals.append("Meet witnesses not yet discriminating — wire pair_keys into rank score")
    else:
        signals.append("Meet witness score higher on gold — strengthen in rank formula")
    if gold_agg["pct_has_triple_witness"] > false_agg["pct_has_triple_witness"]:
        signals.append("Triple witness separates gold — add TRIPLE_BONUS to rank")
    if false_agg["pct_hub_driven_kappa"] > gold_agg["pct_hub_driven_kappa"]:
        signals.append("Kappa hub-driven on false more often — downweight hub-sourced keys")
    return signals


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=30)
    p.add_argument("--max-keys", type=int, default=1024)
    p.add_argument("--max-corr-neighbors", type=int, default=2)
    p.add_argument("--out", default="logs/lattice_pattern_battery.json")
    args = p.parse_args()

    root = Path(resolve_beir_root())
    paths = load_paths(root, args.dataset)
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    qids = [q for q in qrels if q in queries][: args.max_queries]

    print(f"Loading brain + running {len(qids)}-query pattern battery ...", flush=True)
    knowledge, plane = load_brain_and_plane(args.dataset)
    report = run_battery(
        knowledge, plane, queries, qrels, qids,
        max_keys=args.max_keys, max_corr_neighbors=args.max_corr_neighbors,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Counter not JSON serializable
    report["summary"]["pattern_flag_counts"] = dict(
        report["summary"]["pattern_flag_counts"]
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = report["summary"]
    g = s["gold_aggregate"]
    f = s["false_top1_aggregate"]
    print("\n=== Lattice Pattern Battery ===\n")
    print("Tests:", ", ".join(s["tests"].keys()))
    print()
    print(f"{'Metric':<32} {'Gold':>10} {'False#1':>10}")
    print("-" * 54)
    for key in (
        "avg_hub_dilution_ratio",
        "avg_rare_strength_sum",
        "avg_hub_strength_sum",
        "avg_morph_signal_ratio",
        "avg_junk_substring_hits",
        "avg_promoted_subword_hits",
        "avg_pair_order_ratio",
        "avg_meet_witness_score",
        "pct_hub_driven_kappa",
        "pct_has_triple_witness",
        "pct_has_rare_pair_meet",
        "avg_jaccard",
    ):
        print(f"{key:<32} {g.get(key, '-'):>10} {f.get(key, '-'):>10}")
    print()
    print("Hidden pattern flags (count):")
    for flag, cnt in sorted(s["pattern_flag_counts"].items(), key=lambda x: -x[1]):
        print(f"  {flag}: {cnt}")
    print()
    print("Tuning signals:")
    for line in s["tuning_signals"]:
        print(f"  - {line}")
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
