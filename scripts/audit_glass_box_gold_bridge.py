#!/usr/bin/env python3
"""
Glass-box audit — what builds bridges to gold docs vs dilutes the pool.

For each held-out query on SciFact (MultiCorpusBrain, kappa-primary default):

  RAREST WORD COVERAGE (gold doc text)
    - has rarest query word / top-2 / top-3 (by corpus idf)

  POOL & RANK
    - gold in kappa pool? in top-10? why (kappa keys, rare posting, bridge)

  BRIDGE / TEACH CORRELATION
    - supervised bridge qt->dt paths that land on gold vs top false doc
    - teach-store edges query->gold partner vs false

  SUBWORDS & COMPOUNDS
    - morph subword pieces of rare query terms in gold
    - rare query pairs/triples co-occurring in gold (compound signal)

  HUB DILUTION
    - high-df (dull) query words over-represented in top-10 false docs

  COMPOUND QUERY SIMULATION
    - if we AND the 2-3 rarest query words for pool intersection, how many
      gold docs would survive vs current pool?

Run:
  python scripts/audit_glass_box_gold_bridge.py
  python scripts/audit_glass_box_gold_bridge.py nfcorpus --max-queries 100
  python scripts/audit_glass_box_gold_bridge.py scifact --out logs/glass_box_gold_audit.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aethos_append_index import words
from aethos_bridges import bridge_expansion
from aethos_glass_box_metrics import GlassBoxMemory, run_corpus_audit, summarize_audit_rows
from aethos_multi_corpus import IdfCache, MultiCorpusBrain
from aethos_promotion import _chunk_subwords
from pipeline.bit_04_candidate_router import (
    candidates_from_attractors,
    query_words_for_routing,
)
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def _doc_words(text: str) -> set[str]:
    return set(words(text))


def build_scale_pool(
    brain: MultiCorpusBrain,
    branch,
    query: str,
    *,
    max_candidates: int = 600,
    rare_df_cap: int = 256,
) -> tuple[set[str], frozenset, int, dict[str, str]]:
    """Mirror scale_search candidate pool + trace why each doc entered."""
    idx = branch.idx
    qws = query_words_for_routing(words(query))
    pool: set[str] = set()
    reasons: dict[str, set[str]] = defaultdict(set)

    if branch.kappa_index is not None and qws:
        kdocs, keys = brain._attractor_route(branch, qws)
        for d in kdocs[:max_candidates]:
            pool.add(d)
            reasons[d].add("kappa")

    for w in set(words(query)):
        p = idx.token_prime.get(("w", w))
        if p is None:
            continue
        dfp = idx.df.get(p, 0)
        if 0 < dfp <= rare_df_cap:
            pl = idx.postings.get(p)
            if pl:
                for d in pl:
                    if d in idx.alive:
                        pool.add(d)
                        reasons[d].add(f"rare:{w}")

    if branch.pair_bridges is not None:
        for d in bridge_expansion(idx, branch.pair_bridges, query):
            if d in idx.alive:
                pool.add(d)
                reasons[d].add("bridge")

    return pool, keys, len(keys), {d: sorted(v) for d, v in reasons.items()}


def _rarest_terms(qterms: list[str], idf) -> list[str]:
    uniq = list(dict.fromkeys(qterms))
    return sorted(uniq, key=lambda w: (idf(w), w), reverse=True)


def _prefix_hits(doc_toks: set[str], rarest: list[str], n: int) -> tuple[bool, list[str]]:
    want = rarest[:n]
    hit = [w for w in want if w in doc_toks]
    return len(hit) >= n, hit


def _subword_hits(rare_terms: list[str], doc_toks: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for w in rare_terms[:5]:
        pieces = [p for p in _chunk_subwords(w) if len(p) >= 3]
        hits = [p for p in pieces if p in doc_toks and p != w]
        if hits:
            out[w] = hits
    return out


def _pair_triple_hits(rare_sorted: list[str], doc_toks: set[str]) -> dict[str, object]:
    top = rare_sorted[:6]
    pairs_ok = []
    triples_ok = []
    for a, b in itertools.combinations(top, 2):
        if a in doc_toks and b in doc_toks:
            pairs_ok.append((a, b))
    for a, b, c in itertools.combinations(top, 3):
        if a in doc_toks and b in doc_toks and c in doc_toks:
            triples_ok.append((a, b, c))
    return {
        "n_rare_pairs_in_doc": len(pairs_ok),
        "n_rare_triples_in_doc": len(triples_ok),
        "sample_pairs": pairs_ok[:4],
        "sample_triples": triples_ok[:2],
    }


def _bridge_paths(br, query: str, doc_toks: set[str]) -> list[dict]:
    if br is None:
        return []
    paths = []
    for qt in set(words(query)):
        for dt, wt in br.bridge.get(qt, ()):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "weight": round(wt, 4)})
    paths.sort(key=lambda x: -x["weight"])
    return paths[:12]


def _teach_paths(teach, query: str, doc_toks: set[str]) -> list[dict]:
    if teach is None:
        return []
    paths = []
    for qt in set(words(query)):
        partners = teach.edges.get(qt)
        if not partners:
            continue
        for dt, c in partners.most_common(12):
            if dt in doc_toks:
                paths.append({"qt": qt, "dt": dt, "count": c})
    paths.sort(key=lambda x: -x["count"])
    return paths[:12]


def _kappa_overlap(branch, keys: frozenset, gid: str) -> int:
    if branch.kappa_index is None or not keys:
        return 0
    return branch.kappa_index.score_doc_overlap(keys, gid)


def _compound_pool_sim(
    brain: MultiCorpusBrain,
    branch,
    query: str,
    rarest: list[str],
    gold_ids: set[str],
) -> dict[str, object]:
    """Intersect kappa buckets of top-2 and top-3 rarest words — gold retention."""
    if branch.kappa_index is None or len(rarest) < 2:
        return {"n_intersect_2": 0, "gold_in_intersect_2": 0, "n_intersect_3": 0, "gold_in_intersect_3": 0}
    from pipeline.bit_01_word_cell import word_to_spacetime_cell
    from pipeline.bit_02_attractor_key import kappa_from_cell

    def docs_for_word(w: str) -> set[str]:
        cell = word_to_spacetime_cell(brain._registry, w)
        if cell is None:
            return set()
        k = kappa_from_cell(cell)
        return set(branch.kappa_index.by_key.get(k, ()))

    d1 = docs_for_word(rarest[0])
    d2 = docs_for_word(rarest[1]) if len(rarest) > 1 else set()
    inter2 = d1 & d2
    inter3 = inter2
    if len(rarest) > 2:
        d3 = docs_for_word(rarest[2])
        inter3 = inter2 & d3
    return {
        "rarest_for_compound": rarest[:3],
        "n_bucket_1": len(d1),
        "n_intersect_2": len(inter2),
        "gold_in_intersect_2": len(inter2 & gold_ids),
        "n_intersect_3": len(inter3),
        "gold_in_intersect_3": len(inter3 & gold_ids),
    }


def audit_one(
    brain: MultiCorpusBrain,
    branch,
    qid: str,
    query: str,
    gold_local: dict[str, int],
    *,
    idf,
    rare_df_cap: int,
) -> dict[str, object]:
    gold_gids = {
        branch.global_id(d)
        for d, s in gold_local.items()
        if s > 0 and d in branch.texts
    }
    qterms = words(query)
    rarest = _rarest_terms(qterms, idf)
    rare_gate = 3.0
    rare_terms = [w for w in rarest if idf(w) >= rare_gate]

    pool, keys, n_keys, pool_reasons = build_scale_pool(
        brain, branch, query, rare_df_cap=rare_df_cap,
    )
    res = brain.search(query, corpus=branch.name, k=10)
    ranked = res.ranked
    top10 = set(ranked[:10])
    false_top = [d for d in ranked[:10] if d not in gold_gids]

    search_ndcg = ndcg10(
        res.local_ids,
        {d: s for d, s in gold_local.items() if s > 0},
    )

    gold_profiles = []
    for local_id, score in gold_local.items():
        if score <= 0 or local_id not in branch.texts:
            continue
        gid = branch.global_id(local_id)
        text = branch.texts[local_id]
        toks = _doc_words(text)
        h1, w1 = _prefix_hits(toks, rarest, 1)
        h2, w2 = _prefix_hits(toks, rarest, 2)
        h3, w3 = _prefix_hits(toks, rarest, 3)
        bp = _bridge_paths(branch.pair_bridges, query, toks)
        tp = _teach_paths(branch.teach, query, toks)
        compound = _pair_triple_hits(rare_terms or rarest[:6], toks)
        rank = ranked.index(gid) + 1 if gid in ranked else None

        gold_profiles.append({
            "local_id": local_id,
            "has_rarest_1": h1,
            "has_rarest_2": h2,
            "has_rarest_3": h3,
            "hit_words_1": w1,
            "hit_words_2": w2,
            "hit_words_3": w3,
            "in_pool": gid in pool,
            "pool_reasons": pool_reasons.get(gid, []),
            "in_top10": gid in top10,
            "rank": rank,
            "kappa_overlap": _kappa_overlap(branch, keys, gid),
            "bridge_paths": bp,
            "teach_paths": tp,
            "n_bridge_paths": len(bp),
            "n_teach_paths": len(tp),
            "subword_hits": _subword_hits(rare_terms or rarest[:5], toks),
            **compound,
        })

    false_profiles = []
    for gid in false_top[:5]:
        local = gid.split("/", 1)[1]
        text = branch.texts.get(local, "")
        toks = _doc_words(text)
        h1, w1 = _prefix_hits(toks, rarest, 1)
        h2, w2 = _prefix_hits(toks, rarest, 2)
        false_profiles.append({
            "local_id": local,
            "rank": ranked.index(gid) + 1,
            "has_rarest_1": h1,
            "has_rarest_2": h2,
            "hit_rarest": w1,
            "bridge_paths": _bridge_paths(branch.pair_bridges, query, toks),
            "pool_reasons": pool_reasons.get(gid, []),
        })

    hub_words = sorted(qterms, key=lambda w: idf(w))
    hub_diluters = []
    for w in hub_words[:8]:
        if idf(w) >= 2.5:
            continue
        in_false = sum(1 for fp in false_profiles if w in _doc_words(branch.texts.get(fp["local_id"], "")))
        in_gold = sum(
            1 for gp in gold_profiles
            if w in _doc_words(branch.texts.get(gp["local_id"], ""))
        )
        hub_diluters.append({
            "word": w,
            "idf": round(idf(w), 2),
            "df": branch.idx.df.get(branch.idx.token_prime.get(("w", w), 0), 0),
            "in_false_top5": in_false,
            "in_gold": in_gold,
        })

    any_gold_in_pool = any(g["in_pool"] for g in gold_profiles)
    any_gold_top10 = any(g["in_top10"] for g in gold_profiles)

    if not any_gold_top10:
        bucket = "missed_top10"
    elif not any_gold_in_pool:
        bucket = "rank_miss_not_in_pool"
    elif any(g["has_rarest_1"] and not g["has_rarest_2"] for g in gold_profiles if g["in_pool"]):
        bucket = "rarest_only"
    elif any(g["n_rare_pairs_in_doc"] >= 1 for g in gold_profiles if g["in_pool"]):
        bucket = "compound_pair"
    else:
        bucket = "kappa_only"

    return {
        "query_id": qid,
        "query": query[:120],
        "rarest_query_words": rarest[:8],
        "rare_terms_idf3": rare_terms[:6],
        "pool_size": len(pool),
        "n_kappa_keys": n_keys,
        "gold_in_pool": any_gold_in_pool,
        "gold_in_top10": any_gold_top10,
        "bucket": bucket,
        "ndcg10": round(search_ndcg, 4),
        "gold": gold_profiles,
        "false_top": false_profiles,
        "hub_diluters": hub_diluters,
        "compound_sim": _compound_pool_sim(brain, branch, query, rarest, gold_gids),
    }


def summarize(rows: list[dict]) -> dict[str, object]:
    n_q = len(rows)
    n_gold_inst = sum(len(r["gold"]) for r in rows)
    agg = Counter()
    bucket = Counter()
    pool_hit_q = top10_hit_q = 0
    bridge_gold = bridge_false = 0
    teach_gold = teach_false = 0
    compound_pair_gold = 0
    subword_gold = 0
    hub_counter: Counter = Counter()
    compound2_gold = compound3_gold = 0
    compound2_q = compound3_q = 0

    for row in rows:
        bucket[row["bucket"]] += 1
        if row["gold_in_pool"]:
            pool_hit_q += 1
        if row["gold_in_top10"]:
            top10_hit_q += 1

        cs = row["compound_sim"]
        if cs.get("n_intersect_2", 0) > 0:
            compound2_q += 1
            compound2_gold += cs.get("gold_in_intersect_2", 0)
        if cs.get("n_intersect_3", 0) > 0:
            compound3_q += 1
            compound3_gold += cs.get("gold_in_intersect_3", 0)

        for g in row["gold"]:
            if g["has_rarest_1"]:
                agg["gold_rarest_1"] += 1
            if g["has_rarest_2"]:
                agg["gold_rarest_2"] += 1
            if g["has_rarest_3"]:
                agg["gold_rarest_3"] += 1
            if g["in_pool"]:
                agg["gold_in_pool"] += 1
            if g["in_top10"]:
                agg["gold_in_top10"] += 1
            if g["n_bridge_paths"]:
                bridge_gold += 1
            if g["n_teach_paths"]:
                teach_gold += 1
            if g["n_rare_pairs_in_doc"]:
                compound_pair_gold += 1
            if g["subword_hits"]:
                subword_gold += 1

        for f in row["false_top"]:
            if f["has_rarest_1"]:
                agg["false_rarest_1"] += 1
            if f.get("bridge_paths"):
                bridge_false += 1

        for h in row["hub_diluters"]:
            if h["in_false_top5"] > h["in_gold"]:
                hub_counter[h["word"]] += 1

    n_false = sum(len(r["false_top"]) for r in rows)

    def pct(a, b):
        return round(100.0 * a / max(b, 1), 1)

    return {
        "queries": n_q,
        "gold_doc_instances": n_gold_inst,
        "false_doc_instances": n_false,
        "queries_gold_in_pool_pct": pct(pool_hit_q, n_q),
        "queries_gold_in_top10_pct": pct(top10_hit_q, n_q),
        "gold_has_rarest_1_pct": pct(agg["gold_rarest_1"], n_gold_inst),
        "gold_has_rarest_2_pct": pct(agg["gold_rarest_2"], n_gold_inst),
        "gold_has_rarest_3_pct": pct(agg["gold_rarest_3"], n_gold_inst),
        "gold_in_pool_pct": pct(agg["gold_in_pool"], n_gold_inst),
        "gold_in_top10_pct": pct(agg["gold_in_top10"], n_gold_inst),
        "gold_with_bridge_path_pct": pct(bridge_gold, n_gold_inst),
        "gold_with_teach_path_pct": pct(teach_gold, n_gold_inst),
        "false_with_bridge_path_pct": pct(bridge_false, n_false),
        "gold_rare_pair_cooccur_pct": pct(compound_pair_gold, n_gold_inst),
        "gold_subword_hit_pct": pct(subword_gold, n_gold_inst),
        "false_has_rarest_1_pct": pct(agg["false_rarest_1"], n_false),
        "bucket_counts": dict(bucket),
        "top_hub_diluters": hub_counter.most_common(15),
        "compound_intersect_2": {
            "queries_with_intersection": compound2_q,
            "gold_docs_in_intersect": compound2_gold,
        },
        "compound_intersect_3": {
            "queries_with_intersection": compound3_q,
            "gold_docs_in_intersect": compound3_gold,
        },
    }


def print_report(summary: dict, name: str) -> None:
    s = summary
    gi = s["gold_doc_instances"]
    print(f"\n{'='*72}")
    print(f"  GLASS-BOX GOLD BRIDGE AUDIT — {name.upper()}")
    print(f"{'='*72}")
    print(f"  Queries: {s['queries']}   Gold doc instances: {gi}")
    print()
    print("  RAREST QUERY WORD IN GOLD DOC TEXT (by corpus idf, rarest first)")
    print(f"    has rarest 1 word:  {s['gold_has_rarest_1_pct']}% of gold instances")
    print(f"    has rarest 1+2:     {s['gold_has_rarest_2_pct']}%")
    print(f"    has rarest 1+2+3:   {s['gold_has_rarest_3_pct']}%")
    print()
    print("  POOL & RANK (kappa-primary scale_search path)")
    print(f"    query has gold in pool:   {s['queries_gold_in_pool_pct']}% of queries")
    print(f"    query has gold in top-10: {s['queries_gold_in_top10_pct']}%")
    print(f"    gold instance in pool:    {s['gold_in_pool_pct']}%")
    print(f"    gold instance in top-10:  {s['gold_in_top10_pct']}%")
    print()
    print("  SUPERVISED BRIDGE + TEACH CORRELATION -> GOLD TEXT")
    print(f"    gold with bridge qt->dt path: {s['gold_with_bridge_path_pct']}%")
    print(f"    gold with teach edge path:    {s['gold_with_teach_path_pct']}%")
    print(f"    false top with bridge path:   {s['false_with_bridge_path_pct']}%")
    print()
    print("  COMPOUND / SUBWORD SIGNAL IN GOLD")
    print(f"    gold with rare pair co-occur: {s['gold_rare_pair_cooccur_pct']}%")
    print(f"    gold with subword piece hit:  {s['gold_subword_hit_pct']}%")
    print()
    print("  FALSE TOP-5 HAS RAREST WORD (dilution check)")
    print(f"    false docs with rarest query word: {s['false_has_rarest_1_pct']}%")
    print()
    print("  QUERY BUCKETS (recovery strategy)")
    for k, v in sorted(s["bucket_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:<28} {v}")
    print()
    print("  TOP HUB WORDS (low idf, more in false than gold)")
    for w, c in s["top_hub_diluters"][:10]:
        print(f"    {w:<20} queries diluted: {c}")
    print()
    c2 = s["compound_intersect_2"]
    c3 = s["compound_intersect_3"]
    print("  COMPOUND KAPPA SIM (intersect buckets of 2/3 rarest words)")
    print(f"    2-word intersect: {c2['queries_with_intersection']} queries, "
          f"{c2['gold_docs_in_intersect']} gold docs would survive")
    print(f"    3-word intersect: {c3['queries_with_intersection']} queries, "
          f"{c3['gold_docs_in_intersect']} gold docs would survive")
    print("    (narrowing pool — use only if gold retention stays high)")


def main() -> int:
    p = argparse.ArgumentParser(description="Glass-box audit: bridges, rarest words, pool")
    p.add_argument("dataset", nargs="?", default="scifact")
    p.add_argument("--max-queries", type=int, default=0, help="0 = all test queries")
    p.add_argument("--out", default="logs/glass_box_gold_audit.json")
    p.add_argument("--index-mode", default="kappa_primary", choices=("kappa_primary", "full"))
    p.add_argument("--rare-df-cap", type=int, default=256)
    args = p.parse_args()

    corpus, queries, train_q, test_q = load(args.dataset)
    test_ids = [q for q in test_q if q in queries]
    if args.max_queries:
        test_ids = test_ids[: args.max_queries]

    print(f"Building brain ({args.dataset}, index_mode={args.index_mode}) ...", flush=True)
    brain = MultiCorpusBrain()
    branch = brain.stack_corpus(
        args.dataset, corpus,
        queries=queries, train_qrels=train_q,
        index_mode=args.index_mode,
    )

    rows, summary = run_corpus_audit(
        brain, args.dataset, queries, test_q, qids=test_ids,
        rare_df_cap=args.rare_df_cap,
    )
    brain.learn_glass_box_metrics(
        args.dataset, queries, test_q, qids=test_ids,
    )
    nd = rc = 0.0
    for qid in test_ids:
        res = brain.search(queries[qid], corpus=args.dataset, k=10)
        nd += ndcg10(res.local_ids, test_q[qid])
        rc += recall10(res.local_ids, test_q[qid])
    summary["held_out_ndcg10"] = round(nd / len(test_ids), 4)
    summary["held_out_recall10"] = round(rc / len(test_ids), 4)
    summary["index_mode"] = args.index_mode
    summary["corpus"] = args.dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "queries": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print_report(summary, args.dataset)
    print(f"\n  Held-out nDCG@10: {summary['held_out_ndcg10']}  R@10: {summary['held_out_recall10']}")
    print(f"  JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
