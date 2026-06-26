#!/usr/bin/env python3
"""
Glass-box pollution audit — why gold is NOT rank #1, and how to separate noise.

Runs multiple distinguishability tests on gold vs false docs (especially false
ABOVE gold and wrong top-1). Not gold-only coverage — focuses on what pollutes.

Tests (each is a named pattern we count across miss queries):
  T01 hub_heavy_top1      false #1 matches more hub words (idf<2) than gold
  T02 bridge_hub_dilute   false #1 has bridge paths from hub qt terms only
  T03 rarest_word_gap     gold has rarest query word, false #1 lacks it
  T04 rare_pair_gap       gold has rare pair co-occur, false #1 lacks it
  T05 lex_wins_false      false #1 lex_n > gold lex_n (formula floor pollution)
  T06 bridge_wins_false   false #1 bridge_n > gold bridge_n
  T07 pair_wins_false     false #1 pair_n > gold pair_n
  T08 global_polluter     false #1 is chronic wrong-top-1 doc
  T09 low_density_noise   false #1 high hub hits, low query_density
  T10 long_doc_dilute     false #1 long doc_len, weak query_density
  T11 glossary_only_gold  gold hit via glossary-expanded terms only
  T12 bridge_pool_only    false entered pool via bridge_expand not lex_cand
  T13 pair_meet_only      false entered via pair_meet only
  T14 multi_weak_bridge   false #1 has 2+ bridge paths, gold has fewer
  T15 score_gap_tiny      gold within 0.08 total of false #1 (rank fixable)
  T16 score_gap_huge      gold total < 50% of false #1 (route/pool issue)
  T17 false_has_rarest    false #1 still has rarest word (hub+rarest dilution)
  T18 gold_missing_bridge false #1 has bridge path, gold has none
  T19 plane_band_mismatch query rare band != false #1 dominant band (if plane mode)
  T20 trigram_gear_only   false matches query via prefix/trigram not word gear

Output:
  logs/glass_box_pollution_audit.json
  logs/glass_box_pollution_rules.md

Run:
  python scripts/audit_glass_box_pollution.py
  python scripts/audit_glass_box_pollution.py --max-queries 50
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_append_index import words
from aethos_encyclopedia_teacher import load_glossary
from aethos_glass_box_search import (
    GlassBoxRetriever,
    glass_box_fusion_details,
    posting_docs,
    rarest_terms,
    word_idf,
)
from aethos_lattice_lexical import word_band_profile
from scripts.bench_supervised_bridges import load, ndcg10, recall10

HUB_IDF = 2.0
RARE_IDF = 3.0

# Named distinguishability tests
TEST_NAMES = [
    "T01_hub_heavy_top1",
    "T02_bridge_hub_dilute",
    "T03_rarest_word_gap",
    "T04_rare_pair_gap",
    "T05_lex_wins_false",
    "T06_bridge_wins_false",
    "T07_pair_wins_false",
    "T08_global_polluter",
    "T09_low_density_noise",
    "T10_long_doc_dilute",
    "T11_glossary_only_gold",
    "T12_bridge_pool_only",
    "T13_pair_meet_only",
    "T14_multi_weak_bridge",
    "T15_score_gap_tiny",
    "T16_score_gap_huge",
    "T17_false_has_rarest",
    "T18_gold_missing_bridge",
    "T19_plane_band_mismatch",
    "T20_trigram_gear_only",
]


def _doc_toks(corpus: dict[str, str], doc_id: str) -> set[str]:
    return set(words(corpus.get(doc_id, "")))


def _hub_rare_hits(query: str, toks: set[str], idx, N: int) -> tuple[list[str], list[str]]:
    hubs, rares = [], []
    for w in set(words(query)):
        if w in toks:
            i = word_idf(idx, w, N)
            if i < HUB_IDF:
                hubs.append(w)
            if i >= RARE_IDF:
                rares.append(w)
    return hubs, rares


def _rare_pairs_in_doc(rare_terms: list[str], toks: set[str]) -> int:
    import itertools
    return sum(
        1 for a, b in itertools.combinations(rare_terms[:4], 2)
        if a in toks and b in toks
    )


def _bridge_paths(br, query: str, toks: set[str]) -> list[dict]:
    paths = []
    for qt in set(words(query)):
        for dt, wt in br.bridge.get(qt, ()):
            if dt in toks:
                paths.append({"qt": qt, "dt": dt, "w": round(wt, 3)})
    paths.sort(key=lambda x: -x["w"])
    return paths


def _dominant_band(toks: set[str]) -> int | None:
    if not toks:
        return None
    bands = Counter()
    for w in toks:
        bid, _ = word_band_profile(w)
        bands[bid] += 1
    return bands.most_common(1)[0][0] if bands else None


def profile_doc(
    doc_id: str,
    fusion: dict,
    corpus: dict[str, str],
    query: str,
    idx,
    br,
    N: int,
    rare_terms: list[str],
    gloss_extra: set[str],
) -> dict:
    toks = _doc_toks(corpus, doc_id)
    hubs, rare_hits = _hub_rare_hits(query, toks, idx, N)
    rarest = rarest_terms(words(query), idx, N)
    bp = _bridge_paths(br, query, toks)
    hub_bp = [p for p in bp if word_idf(idx, p["qt"], N) < HUB_IDF]
    rare_bp = [p for p in bp if word_idf(idx, p["qt"], N) >= RARE_IDF]

    gloss_hits = [w for w in gloss_extra if w in toks]
    fd = fusion["docs"].get(doc_id, {})

    return {
        "doc_id": doc_id,
        **fd,
        "n_hub_hits": len(hubs),
        "n_rare_hits": len(rare_hits),
        "hub_hits": hubs[:6],
        "rare_hits": rare_hits[:6],
        "has_rarest_1": bool(rarest and rarest[0] in toks),
        "has_rarest_2": len(rarest) >= 2 and rarest[0] in toks and rarest[1] in toks,
        "rare_pair_count": _rare_pairs_in_doc(rare_terms, toks),
        "n_bridge_paths": len(bp),
        "n_hub_bridge_paths": len(hub_bp),
        "n_rare_bridge_paths": len(rare_bp),
        "bridge_paths": bp[:6],
        "glossary_hits": gloss_hits[:6],
        "doc_len": round(idx.doc_len.get(doc_id, 0.0), 1),
        "query_density": round(
            len(set(words(query)) & toks) / max(len(set(words(query))), 1), 3,
        ),
        "dominant_band": _dominant_band(toks),
    }


def run_distinguishability_tests(
    gold_prof: dict,
    false_prof: dict,
    query_rare_bands: list[int],
    polluter_score: int,
) -> dict[str, bool]:
    """Which pollution patterns explain false beating gold."""
    if not gold_prof or not false_prof:
        return {}

    g, f = gold_prof, false_prof
    gap = f["total"] - g["total"]

    return {
        "T01_hub_heavy_top1": f["n_hub_hits"] > g["n_hub_hits"],
        "T02_bridge_hub_dilute": f["n_hub_bridge_paths"] > g["n_rare_bridge_paths"],
        "T03_rarest_word_gap": g["has_rarest_1"] and not f["has_rarest_1"],
        "T04_rare_pair_gap": g["rare_pair_count"] > f["rare_pair_count"],
        "T05_lex_wins_false": f["lex_n"] > g["lex_n"],
        "T06_bridge_wins_false": f["bridge_n"] > g["bridge_n"],
        "T07_pair_wins_false": f["pair_n"] > g["pair_n"],
        "T08_global_polluter": polluter_score >= 5,
        "T09_low_density_noise": (
            f["n_hub_hits"] > f["n_rare_hits"] and f["query_density"] < 0.35
        ),
        "T10_long_doc_dilute": f["doc_len"] > 120 and f["query_density"] < g["query_density"],
        "T11_glossary_only_gold": bool(g["glossary_hits"]) and not f["glossary_hits"],
        "T12_bridge_pool_only": f["in_bridge_expand"] and not f["in_lex_cand"],
        "T13_pair_meet_only": f["in_pair_meet"] and not f["in_lex_cand"] and not f["in_bridge_expand"],
        "T14_multi_weak_bridge": f["n_bridge_paths"] >= 2 and f["n_bridge_paths"] > g["n_bridge_paths"],
        "T15_score_gap_tiny": 0 < gap <= 0.08,
        "T16_score_gap_huge": gap > 0.25,
        "T17_false_has_rarest": f["has_rarest_1"] and not g["has_rarest_1"],
        "T18_gold_missing_bridge": f["n_bridge_paths"] > 0 and g["n_bridge_paths"] == 0,
        "T19_plane_band_mismatch": (
            query_rare_bands
            and f.get("dominant_band") is not None
            and f["dominant_band"] not in query_rare_bands
        ),
        "T20_trigram_gear_only": (
            f["lex_n"] > 0 and not f["has_rarest_1"] and g["has_rarest_1"]
        ),
    }


def audit_query(
    retriever: GlassBoxRetriever,
    corpus: dict[str, str],
    qid: str,
    query: str,
    gold_local: dict[str, int],
    polluter_scores: dict[str, int],
) -> dict:
    idx = retriever.idx
    br = retriever.bridges
    N = len(idx.alive)

    fusion = glass_box_fusion_details(
        idx, br, query,
        glossary=retriever.glossary,
        config=retriever.config,
        scorer=retriever.lexical_scorer,
        pool_cap=100,
        corpus=corpus,
        kappa_index=retriever.kappa_index,
        registry=retriever.registry,
    )
    ranked = fusion["ranked"]
    gloss_extra = set(fusion.get("glossary_added", []))
    rare_terms = fusion.get("rare_terms", [])

    gold_ids = [d for d, s in gold_local.items() if s > 0 and d in corpus]
    gold_ranks = {g: ranked.index(g) + 1 for g in gold_ids if g in ranked}
    best_gold = min(gold_ranks.values()) if gold_ranks else None
    top1 = ranked[0] if ranked else None
    top1_gold = top1 in gold_ids if top1 else False

    gold_profiles = [
        profile_doc(g, fusion, corpus, query, idx, br, N, rare_terms, gloss_extra)
        for g in gold_ids
    ]
    # primary gold for comparisons (best ranked)
    gold_prof = None
    if gold_ranks:
        best_gid = min(gold_ranks, key=gold_ranks.get)
        gold_prof = next(p for p in gold_profiles if p["doc_id"] == best_gid)

    false_top1_prof = (
        profile_doc(top1, fusion, corpus, query, idx, br, N, rare_terms, gloss_extra)
        if top1 and not top1_gold else None
    )

    false_above = []
    if best_gold and best_gold > 1:
        for d in ranked[: best_gold - 1]:
            if d not in gold_ids:
                false_above.append(
                    profile_doc(d, fusion, corpus, query, idx, br, N, rare_terms, gloss_extra)
                )

    tests: dict[str, bool] = {}
    if gold_prof and false_top1_prof:
        q_bands = []
        for w in rare_terms[:3]:
            bid, _ = word_band_profile(w)
            q_bands.append(bid)
        tests = run_distinguishability_tests(
            gold_prof,
            false_top1_prof,
            q_bands,
            polluter_scores.get(top1, 0),
        )

    return {
        "query_id": qid,
        "query": query[:120],
        "top1_doc": top1,
        "top1_is_gold": top1_gold,
        "best_gold_rank": best_gold,
        "gold_ranks": gold_ranks,
        "ndcg10": round(ndcg10(ranked[:10], gold_local), 4),
        "recall10": round(recall10(ranked[:10], gold_local), 4),
        "ranked_top10": ranked[:10],
        "gold_profile": gold_prof,
        "false_top1_profile": false_top1_prof,
        "false_above_gold": false_above[:8],
        "distinguish_tests": tests,
        "fusion_summary": {
            "lexical_mode": fusion["lexical_mode"],
            "glossary_added": fusion.get("glossary_added", []),
            "rare_terms": rare_terms,
        },
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    miss_top1 = [r for r in rows if not r["top1_is_gold"]]
    test_counts = Counter()
    test_on_miss = Counter()
    gold_feats: list[dict] = []
    false_feats: list[dict] = []

    for r in miss_top1:
        for tname, fired in r.get("distinguish_tests", {}).items():
            if fired:
                test_on_miss[tname] += 1
        if r.get("gold_profile"):
            gold_feats.append(r["gold_profile"])
        if r.get("false_top1_profile"):
            false_feats.append(r["false_top1_profile"])

    for r in rows:
        for tname, fired in r.get("distinguish_tests", {}).items():
            if fired:
                test_counts[tname] += 1

    def med(profiles, key, default=0.0):
        vals = [p[key] for p in profiles if key in p]
        return round(statistics.median(vals), 3) if vals else default

    feature_compare = []
    for key in (
        "total", "lex_n", "bridge_n", "pair_n",
        "n_hub_hits", "n_rare_hits", "n_bridge_paths",
        "rare_pair_count", "query_density", "doc_len",
    ):
        gm, fm = med(gold_feats, key), med(false_feats, key)
        feature_compare.append({
            "feature": key,
            "gold_median": gm,
            "false_top1_median": fm,
            "delta": round(gm - fm, 3),
        })

    top1_gold = sum(1 for r in rows if r["top1_is_gold"])
    ranks = [r["best_gold_rank"] for r in rows if r["best_gold_rank"]]

    return {
        "queries": n,
        "top1_gold_pct": round(100 * top1_gold / max(n, 1), 1),
        "miss_top1_count": len(miss_top1),
        "mean_best_gold_rank": round(statistics.mean(ranks), 2) if ranks else None,
        "test_fire_counts_all": dict(test_counts),
        "test_fire_on_miss_top1": dict(test_on_miss),
        "test_fire_pct_on_miss": {
            k: round(100 * v / max(len(miss_top1), 1), 1)
            for k, v in test_on_miss.items()
        },
        "feature_medians_gold_vs_false_top1": sorted(
            feature_compare, key=lambda x: -abs(x["delta"]),
        ),
    }


def find_polluters(rows: list[dict], top_n: int = 20) -> list[dict]:
    top1_false = Counter()
    above = Counter()
    for r in rows:
        if not r["top1_is_gold"] and r["top1_doc"]:
            top1_false[r["top1_doc"]] += 1
        for fp in r.get("false_above_gold", []):
            above[fp["doc_id"]] += 1
    merged = Counter()
    for d, c in top1_false.items():
        merged[d] += c * 3
    for d, c in above.items():
        merged[d] += c * 2
    return [
        {
            "doc_id": d,
            "pollution_score": s,
            "wrong_top1": top1_false[d],
            "above_gold": above[d],
        }
        for d, s in merged.most_common(top_n)
    ]


def write_rules_md(path: Path, summary: dict, polluters: list[dict], rows: list[dict]) -> None:
    lines = [
        "# Glass-box pollution audit\n",
        f"- Queries: {summary['queries']}\n",
        f"- Top-1 gold: {summary['top1_gold_pct']}%\n",
        f"- Miss top-1: {summary['miss_top1_count']}\n",
        f"- Mean best gold rank: {summary['mean_best_gold_rank']}\n",
        "\n## Distinguishability tests (fire rate when top-1 is wrong)\n",
        "| Test | % of miss-top-1 | Meaning |\n",
        "|------|-----------------|--------|\n",
    ]
    meanings = {
        "T01_hub_heavy_top1": "False #1 matches more hub words than gold",
        "T02_bridge_hub_dilute": "False #1 bridged via hub qt, not rare qt",
        "T03_rarest_word_gap": "Gold has rarest word, false #1 lacks it",
        "T04_rare_pair_gap": "Gold has more rare query pairs in doc",
        "T05_lex_wins_false": "Lexical/formula floor favors false",
        "T06_bridge_wins_false": "Bridge component favors false",
        "T07_pair_wins_false": "Pair-meet component favors false",
        "T08_global_polluter": "Chronic wrong-top-1 document",
        "T09_low_density_noise": "Hub-heavy loose match",
        "T10_long_doc_dilute": "Long doc beats focused gold",
        "T11_glossary_only_gold": "Gold needs glossary terms false lacks",
        "T12_bridge_pool_only": "False entered via bridge expand only",
        "T13_pair_meet_only": "False entered via pair-meet only",
        "T14_multi_weak_bridge": "Many weak bridges on false",
        "T15_score_gap_tiny": "Gold close — rerank fixable",
        "T16_score_gap_huge": "Gold far behind — pool/route issue",
        "T17_false_has_rarest": "False has rarest word too — hub+rarest dilution",
        "T18_gold_missing_bridge": "False has bridge path, gold none",
        "T19_plane_band_mismatch": "False band off query rare bands",
        "T20_trigram_gear_only": "False wins via gears not rarest word",
    }
    pct = summary.get("test_fire_pct_on_miss", {})
    for t in TEST_NAMES:
        if t in pct:
            lines.append(f"| {t} | {pct[t]}% | {meanings.get(t, '')} |\n")

    lines.append("\n## Gold vs false top-1 medians\n")
    lines.append("| Feature | Gold | False #1 | Delta |\n")
    lines.append("|---------|------|----------|-------|\n")
    for fc in summary.get("feature_medians_gold_vs_false_top1", [])[:12]:
        lines.append(
            f"| {fc['feature']} | {fc['gold_median']} | {fc['false_top1_median']} | {fc['delta']} |\n"
        )

    lines.append("\n## Top polluter docs\n")
    for p in polluters[:12]:
        lines.append(
            f"- `{p['doc_id']}` score={p['pollution_score']} "
            f"wrong#1={p['wrong_top1']} above_gold={p['above_gold']}\n"
        )

  # sample miss queries
    lines.append("\n## Sample miss-top-1 (actionable)\n")
    miss = [r for r in rows if not r["top1_is_gold"]][:8]
    for r in miss:
        fired = [t for t, v in r.get("distinguish_tests", {}).items() if v]
        lines.append(
            f"- Q{r['query_id']}: gold_rank={r['best_gold_rank']} "
            f"false=`{r['top1_doc']}` tests={fired[:4]}\n"
        )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Glass-box pollution / distinguishability audit")
    p.add_argument("dataset", nargs="?", default="scifact")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", default="logs/glass_box_pollution_audit.json")
    p.add_argument("--rules-out", default="logs/glass_box_pollution_rules.md")
    p.add_argument("--lattice", action="store_true", help="Use scifact_lattice config (κ-pool, no BM25)")
    p.add_argument("--index-mode", default="full", choices=("full", "kappa_primary"))
    p.add_argument("--min-pairs", type=int, default=0)
    args = p.parse_args()

    corpus, queries, train_q, test_q = load(args.dataset)
    if not train_q:
        print(f"  WARNING: {args.dataset} has no train qrels", flush=True)
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    gloss = load_glossary(args.dataset)
    min_pairs = args.min_pairs or (1 if args.dataset == "scifact" else 2)
    mode = args.index_mode
    if args.lattice and mode == "full":
        mode = "kappa_primary" if args.dataset != "scifact" else args.index_mode

    print(f"Building GlassBoxRetriever ({args.dataset}, lattice={args.lattice})...", flush=True)
    if args.lattice:
        retriever = GlassBoxRetriever.from_corpus(
            corpus, queries, train_q, glossary=gloss,
            min_pairs=min_pairs,
            scifact_lattice=True,
            index_mode=mode,
            build_kappa_index=(mode == "kappa_primary"),
        )
    else:
        retriever = GlassBoxRetriever.from_corpus(
            corpus, queries, train_q, glossary=gloss,
            min_pairs=min_pairs,
            index_mode=mode,
        )

    # pass 1: collect polluter scores
    polluter_scores: Counter[str] = Counter()
    rows_pass1 = []
    for qid in test_ids:
        fusion = glass_box_fusion_details(
            retriever.idx, retriever.bridges, queries[qid],
            glossary=retriever.glossary,
            config=retriever.config,
            scorer=retriever.lexical_scorer,
            pool_cap=10,
            corpus=corpus,
            kappa_index=retriever.kappa_index,
            registry=retriever.registry,
        )
        top1 = fusion["ranked"][0] if fusion["ranked"] else None
        golds = {d for d, s in test_q[qid].items() if s > 0}
        if top1 and top1 not in golds:
            polluter_scores[top1] += 3

    rows = []
    for qid in test_ids:
        rows.append(audit_query(
            retriever, corpus, qid, queries[qid], test_q[qid],
            dict(polluter_scores),
        ))

    summary = summarize(rows)
    polluters = find_polluters(rows)
    summary["polluter_docs"] = polluters

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "queries": rows}, indent=2), encoding="utf-8")

    rules = Path(args.rules_out)
    write_rules_md(rules, summary, polluters, rows)

    print(f"\n  Top-1 gold: {summary['top1_gold_pct']}%  miss-top-1: {summary['miss_top1_count']}")
    print("  Tests firing on miss-top-1 (top 8):")
    for t, pct in sorted(
        summary.get("test_fire_pct_on_miss", {}).items(),
        key=lambda x: -x[1],
    )[:8]:
        print(f"    {t}: {pct}%")
    print(f"\n  JSON: {out}")
    print(f"  Rules: {rules}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
