#!/usr/bin/env python3
"""
SciFact top-100 noise audit — which docs pollute answers and why.

For each held-out query:
  - Rank top-100 via scale_search path
  - Profile gold vs false docs (especially false ABOVE gold)
  - Find global polluter docs (appear in too many wrong top-10s)
  - Derive glass-box demotion rules from what separates gold from noise

Output: logs/scifact_top100_noise_audit.json
        logs/scifact_top100_rules.md

Run:  python scripts/audit_scifact_top100_noise.py
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
from aethos_bridges import bridge_expansion
from aethos_glass_box_metrics import GlassBoxMemory, _bridge_paths, _prefix_hits, _rarest_terms
from aethos_multi_corpus import IdfCache, MultiCorpusBrain, score_candidates
from pipeline.bit_04_candidate_router import query_words_for_routing
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def _doc_words(text: str) -> set[str]:
    return set(words(text))


def profile_doc(
    brain,
    branch,
    query: str,
    doc_gid: str,
    *,
    idf,
    scores: dict[str, float],
    keys,
    rarest: list[str],
) -> dict:
    local = doc_gid.split("/", 1)[1] if "/" in doc_gid else doc_gid
    text = branch.texts.get(local, "")
    toks = _doc_words(text)
    qtoks = set(words(query))
    rare = [w for w in rarest if idf(w) >= 3.0]

    h1, w1 = _prefix_hits(toks, rarest, 1)
    h2, w2 = _prefix_hits(toks, rarest, 2)
    bp = _bridge_paths(branch.pair_bridges, query, toks)

    hub_hits = [w for w in qtoks if idf(w) < 2.0 and w in toks]
    rare_hits = [w for w in qtoks if idf(w) >= 3.0 and w in toks]

    kappa_ov = 0.0
    if branch.kappa_index and keys:
        kappa_ov = branch.kappa_index.score_doc_overlap(keys, doc_gid)

    dl = branch.idx.doc_len.get(doc_gid, 0.0)
    qmatch = len(qtoks & toks)
    density = qmatch / max(len(qtoks), 1)
    distinct = len(toks) / max(len(words(text)), 1)

    return {
        "doc_id": local,
        "lex_score": round(scores.get(doc_gid, 0.0), 4),
        "kappa_overlap": round(kappa_ov, 4),
        "n_bridge_paths": len(bp),
        "bridge_paths": bp[:4],
        "has_rarest_1": h1,
        "has_rarest_2": h2,
        "hit_rarest_1": w1,
        "n_rare_query_hits": len(rare_hits),
        "n_hub_query_hits": len(hub_hits),
        "hub_hits": hub_hits[:6],
        "doc_len": round(dl, 1),
        "query_match": qmatch,
        "query_density": round(density, 3),
        "vocab_distinct": round(distinct, 3),
        "rare_pair_in_doc": sum(
            1 for a, b in __import__("itertools").combinations(rare[:4], 2)
            if a in toks and b in toks
        ),
    }


def audit_query_top100(
    brain,
    branch,
    qid: str,
    query: str,
    gold_local: dict[str, int],
    *,
    idf,
) -> dict:
    qws = query_words_for_routing(words(query))
    pool: set[str] = set()
    keys = frozenset()

    if branch.kappa_index and qws:
        kdocs, keys = brain._attractor_route(branch, qws)
        pool.update(kdocs[:600])

    idx = branch.idx
    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p and 0 < idx.df.get(p, 0) <= 256:
            pl = idx.postings.get(p)
            if pl:
                pool.update(d for d in pl if d in idx.alive)

    if branch.pair_bridges:
        pool.update(
            d for d in bridge_expansion(
                idx, branch.pair_bridges, query,
                idf=idf,
                hub_idf_gate=brain.HUB_IDF_GATE if brain.HUB_IDF_GATE < 50 else 0.0,
                hub_blocklist=brain._learned_hub_blocklist(),
            )
            if d in idx.alive
        )

    scores = score_candidates(idx, query, pool)
    if keys and branch.kappa_index and scores:
        lmax = max(scores.values()) or 1.0
        klam = brain.KAPPA_LAM * brain.glass_box.kappa_lam_scale(query, idf)
        for d in pool:
            ov = branch.kappa_index.score_doc_overlap(keys, d)
            if ov > 0:
                scores[d] = scores.get(d, 0.0) + klam * ov * lmax

    ranked = sorted(scores, key=scores.get, reverse=True)[:100]
    ranked_local = [d.split("/", 1)[1] for d in ranked if "/" in d]

    gold_gids = {
        branch.global_id(d)
        for d, s in gold_local.items()
        if s > 0 and d in branch.texts
    }
    gold_locals = {d for d, s in gold_local.items() if s > 0}

    rarest = _rarest_terms(words(query), idf)

    gold_ranks = {}
    for gid in gold_gids:
        if gid in ranked:
            gold_ranks[gid.split("/", 1)[1]] = ranked.index(gid) + 1

    best_gold_rank = min(gold_ranks.values()) if gold_ranks else None
    top1_local = ranked[0].split("/", 1)[1] if ranked else None
    top1_is_gold = ranked[0] in gold_gids if ranked else False

    false_above_gold = []
    if best_gold_rank and best_gold_rank > 1:
        for gid in ranked[: best_gold_rank - 1]:
            if gid not in gold_gids:
                false_above_gold.append(gid)

    gold_profiles = [
        profile_doc(brain, branch, query, gid, idf=idf, scores=scores, keys=keys, rarest=rarest)
        for gid in gold_gids
        if gid in scores
    ]
    false_profiles = [
        profile_doc(brain, branch, query, gid, idf=idf, scores=scores, keys=keys, rarest=rarest)
        for gid in false_above_gold
    ]

    return {
        "query_id": qid,
        "query": query[:100],
        "pool_size": len(pool),
        "ndcg10": round(
            ndcg10(ranked_local, {d: s for d, s in gold_local.items() if s > 0}), 4
        ),
        "recall10": round(recall10(ranked_local, gold_local), 4),
        "top1_is_gold": top1_is_gold,
        "top1_doc": top1_local,
        "best_gold_rank": best_gold_rank,
        "gold_ranks": gold_ranks,
        "n_false_above_gold": len(false_above_gold),
        "gold_profiles": gold_profiles,
        "false_above_gold": false_profiles,
        "ranked_top10": ranked_local[:10],
    }


def _median_feature(profiles: list[dict], key: str, default=0.0) -> float:
    vals = [p[key] for p in profiles if key in p]
    return statistics.median(vals) if vals else default


def derive_rules(gold_feats: list[dict], false_feats: list[dict]) -> list[dict]:
    """Compare gold vs false-above-gold medians → actionable demotion rules."""
    keys = [
        "lex_score", "kappa_overlap", "n_bridge_paths", "n_rare_query_hits",
        "n_hub_query_hits", "query_density", "vocab_distinct", "doc_len",
        "rare_pair_in_doc", "has_rarest_1", "has_rarest_2",
    ]
    rules = []
    for k in keys:
        gm = _median_feature(gold_feats, k)
        fm = _median_feature(false_feats, k)
        if k.startswith("has_"):
            rules.append({
                "feature": k,
                "gold_median": gm,
                "false_median": fm,
                "rule": f"prefer docs where {k} is true" if gm > fm else None,
            })
            continue
        diff = gm - fm
        if abs(diff) < 0.05 and k not in ("lex_score", "kappa_overlap"):
            continue
        direction = "boost" if diff > 0 else "penalize"
        rules.append({
            "feature": k,
            "gold_median": round(gm, 3),
            "false_median": round(fm, 3),
            "delta": round(diff, 3),
            "rule": f"{direction} when {k} {'high' if diff > 0 else 'low'}",
        })
    return sorted(rules, key=lambda r: -abs(r.get("delta", 0)))


def find_polluter_docs(rows: list[dict], *, top_n: int = 25) -> list[dict]:
    """Docs that appear too often as wrong top-1 or false-above-gold."""
    top1_false = Counter()
    top10_false = Counter()
    above_gold = Counter()

    for row in rows:
        if not row["top1_is_gold"] and row["top1_doc"]:
            top1_false[row["top1_doc"]] += 1
        for lid in row["ranked_top10"]:
            if lid not in row["gold_ranks"]:
                top10_false[lid] += 1
        for fp in row["false_above_gold"]:
            above_gold[fp["doc_id"]] += 1

    merged: Counter = Counter()
    for doc, c in top1_false.items():
        merged[doc] += c * 3
    for doc, c in top10_false.items():
        merged[doc] += c
    for doc, c in above_gold.items():
        merged[doc] += c * 2

    out = []
    for doc, score in merged.most_common(top_n):
        out.append({
            "doc_id": doc,
            "pollution_score": score,
            "wrong_top1": top1_false[doc],
            "wrong_top10": top10_false[doc],
            "above_gold": above_gold[doc],
        })
    return out


def write_rules_md(path: Path, summary: dict, rules: list[dict], polluters: list[dict]) -> None:
    lines = [
        "# SciFact top-100 noise rules (glass-box)\n",
        f"Queries audited: {summary['queries']}\n",
        f"Top-1 gold rate: {summary['top1_gold_pct']}%\n",
        f"Gold in top-10: {summary['gold_top10_pct']}%\n",
        f"Mean best gold rank (when in pool): {summary['mean_best_gold_rank']}\n",
        "\n## What separates gold from false (above gold)\n",
        "| Feature | Gold median | False median | Rule |\n",
        "|---------|-------------|--------------|------|\n",
    ]
    for r in rules[:12]:
        if r.get("rule"):
            lines.append(
                f"| {r['feature']} | {r.get('gold_median', '-')} | "
                f"{r.get('false_median', '-')} | {r['rule']} |\n"
            )
    lines.append("\n## Top polluter docs (wrong top-10 frequency)\n")
    for p in polluters[:15]:
        lines.append(
            f"- `{p['doc_id']}` score={p['pollution_score']} "
            f"(wrong#1={p['wrong_top1']} top10={p['wrong_top10']} above_gold={p['above_gold']})\n"
        )
    lines.append("\n## Recommended demotion rules\n")
    lines.append("1. **Hub-only match** — penalize docs matching only idf<2 query terms without rarest word.\n")
    lines.append("2. **High hub / low rare** — when `n_hub_query_hits` > `n_rare_query_hits`, down-weight.\n")
    lines.append("3. **Global polluter list** — demote docs in polluter table above threshold.\n")
    lines.append("4. **Rare pair boost** — boost when `rare_pair_in_doc` >= 1 (gold median higher).\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--out", default="logs/scifact_top100_noise_audit.json")
    p.add_argument("--rules-out", default="logs/scifact_top100_rules.md")
    args = p.parse_args()

    corpus, queries, train_q, test_q = load(args.dataset)
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    print(f"Building brain + auditing top-100 noise ({args.dataset})...", flush=True)
    brain = MultiCorpusBrain()
    branch = brain.stack_corpus(
        args.dataset, corpus, queries=queries, train_qrels=train_q,
    )

    rows = []
    for qid in test_ids:
        idf = IdfCache(branch.idx, branch.n_docs)
        rows.append(audit_query_top100(
            brain, branch, qid, queries[qid], test_q[qid], idf=idf,
        ))

    all_gold = [p for r in rows for p in r["gold_profiles"]]
    all_false = [p for r in rows for p in r["false_above_gold"]]
    rules = derive_rules(all_gold, all_false)
    polluters = find_polluter_docs(rows)

    n = len(rows)
    top1_gold = sum(1 for r in rows if r["top1_is_gold"])
    gold_top10 = sum(1 for r in rows if r["best_gold_rank"] and r["best_gold_rank"] <= 10)
    ranks = [r["best_gold_rank"] for r in rows if r["best_gold_rank"]]
    mean_rank = round(statistics.mean(ranks), 2) if ranks else None

    summary = {
        "dataset": args.dataset,
        "queries": n,
        "top1_gold_pct": round(100 * top1_gold / n, 1),
        "gold_top10_pct": round(100 * gold_top10 / n, 1),
        "mean_best_gold_rank": mean_rank,
        "queries_gold_not_top1": n - top1_gold,
        "rules": rules,
        "polluter_docs": polluters,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "queries": rows}, indent=2), encoding="utf-8")
    write_rules_md(Path(args.rules_out), summary, rules, polluters)

    print(f"\n{'='*60}")
    print(f"  TOP-100 NOISE AUDIT — {args.dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Top-1 is gold:     {summary['top1_gold_pct']}% ({top1_gold}/{n})")
    print(f"  Gold in top-10:    {summary['gold_top10_pct']}%")
    print(f"  Mean gold rank:    {mean_rank}")
    print(f"\n  Top separators (gold vs false above gold):")
    for r in rules[:6]:
        if r.get("rule"):
            print(f"    {r['feature']:<22} gold={r.get('gold_median')} false={r.get('false_median')} -> {r['rule']}")
    print(f"\n  Worst polluter docs:")
    for p in polluters[:8]:
        print(f"    {p['doc_id']:<12} pollution={p['pollution_score']} wrong#1={p['wrong_top1']}")
    print(f"\n  JSON: {out}")
    print(f"  Rules: {args.rules_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
