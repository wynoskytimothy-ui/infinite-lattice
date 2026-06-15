#!/usr/bin/env python3
"""Count docs sharing 2-way / 3-way crossovers with queries (SciFact test)."""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

from eval_beir import load_paths, load_qrels, load_queries, resolve_beir_root
from eval_beir_symbol import load_brain_and_plane, query_words
from aethos_rare_rank import (
    _DocFreqCache,
    _rare_word_cached,
    degree_map_from_plane,
)
from pipeline.bit_12_symbol_plane_index import (
    _query_pair_meets,
    query_symbol_plane_keys,
    route_symbol_plane_candidates,
)

TOKEN = re.compile(r"[a-z]+")


def lex_overlap(qset: set[str], gold_ids, corpus) -> int:
    best = 0
    for gid in gold_ids:
        gt = {t for t in TOKEN.findall(corpus.get(gid, "").lower()) if len(t) >= 3}
        best = max(best, len(qset & gt))
    return best


def report(label: str, vals: list[int], gold_flags: list[bool]) -> None:
    n = len(vals)
    print(f"--- {label} ---")
    for thr in (1, 10, 50, 100, 200, 500):
        c = sum(1 for v in vals if v >= thr)
        print(f"  queries with >={thr} docs: {c}/{n} ({100 * c / n:.1f}%)")
    print(f"  mean docs/query: {sum(vals) / n:.1f}  median: {sorted(vals)[n // 2]}")
    g = sum(gold_flags)
    print(f"  gold hits crossover (in pool): {g}/{n} ({100 * g / n:.1f}%)")
    print()


def main() -> None:
    root = Path(resolve_beir_root())
    paths = load_paths(root, "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    knowledge, plane = load_brain_and_plane("scifact")
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)

    stats: list[dict] = []
    for qid in qrels:
        if qid not in queries:
            continue
        try:
            words = query_words(queries[qid])
            qset = set(words)
            keys = query_symbol_plane_keys(knowledge, plane, words)
            routed = route_symbol_plane_candidates(
                knowledge, plane, words, max_candidates=1200,
            )
            pool = list(routed.doc_ids)
            canon = list(dict.fromkeys(
                knowledge.morph_canonical_surface(w) for w in words
            ))
            pair_meets = _query_pair_meets(knowledge, plane, canon)
            gold_ids = set(qrels[qid])

            k2 = k3 = 0
            gold_k2 = gold_k3 = False
            meet2 = meet3pairs = 0
            gold_meet2 = gold_meet3pairs = False

            for did in pool:
                dk = plane.doc_keys.get(did, set())
                inter = len(keys & dk)
                if inter >= 2:
                    k2 += 1
                    if did in gold_ids:
                        gold_k2 = True
                if inter >= 3:
                    k3 += 1
                    if did in gold_ids:
                        gold_k3 = True

                pairs_hit = sum(1 for meet, _ in pair_meets if meet & dk)
                if pairs_hit >= 1:
                    meet2 += 1
                    if did in gold_ids:
                        gold_meet2 = True
                if pairs_hit >= 2:
                    meet3pairs += 1
                    if did in gold_ids:
                        gold_meet3pairs = True

            rare_cache: dict[str, bool] = {}
            rq = [
                w.lower() for w in words
                if _rare_word_cached(
                    knowledge, w, df_cache=cache,
                    rare_cache=rare_cache, degrees=degrees,
                )
            ]
            triple_docs = 0
            gold_triple = False
            if len(rq) >= 2:
                rqset = set(rq)
                for did in pool:
                    text = knowledge.corpus.get(did, "")
                    doc_toks = {
                        t for t in TOKEN.findall(text.lower())
                        if len(t) >= 3 and _rare_word_cached(
                            knowledge, t, df_cache=cache,
                            rare_cache=rare_cache, degrees=degrees,
                        )
                    }
                    hit = False
                    for a, b in combinations(sorted(rqset), 2):
                        if knowledge.correlates(a, b) is None:
                            continue
                        for c in doc_toks:
                            if c in rqset:
                                continue
                            if knowledge.correlates(a, c) and knowledge.correlates(b, c):
                                hit = True
                                break
                        if hit:
                            break
                    if hit:
                        triple_docs += 1
                        if did in gold_ids:
                            gold_triple = True

            stats.append({
                "lex": lex_overlap(qset, qrels[qid], knowledge.corpus),
                "k2": k2,
                "k3": k3,
                "meet2": meet2,
                "meet3pairs": meet3pairs,
                "triple": triple_docs,
                "gold_k2": gold_k2,
                "gold_k3": gold_k3,
                "gold_meet2": gold_meet2,
                "gold_meet3pairs": gold_meet3pairs,
                "gold_triple": gold_triple,
                "n_pair_meets": len(pair_meets),
            })
        except Exception:
            pass

    n = len(stats)
    print(f"queries analyzed: {n}\n")
    report(
        "2-way kappa crossover (doc shares >=2 query kappa keys)",
        [s["k2"] for s in stats],
        [s["gold_k2"] for s in stats],
    )
    report(
        "3-way kappa crossover (doc shares >=3 query kappa keys)",
        [s["k3"] for s in stats],
        [s["gold_k3"] for s in stats],
    )
    report(
        "2-way pair-meet crossover (doc hits >=1 query pair meet)",
        [s["meet2"] for s in stats],
        [s["gold_meet2"] for s in stats],
    )
    report(
        "3-way pair-meet crossover (doc hits >=2 query pair meets)",
        [s["meet3pairs"] for s in stats],
        [s["gold_meet3pairs"] for s in stats],
    )
    report(
        "3-way rare triple (query rare pair + shared rare doc token)",
        [s["triple"] for s in stats],
        [s["gold_triple"] for s in stats],
    )

    print("--- By lexical query-gold word overlap ---")
    buckets = [
        (0, 0, "0 words"),
        (1, 1, "1 word"),
        (2, 2, "2 words"),
        (3, 3, "3 words"),
        (4, 999, ">=4 words"),
    ]
    for lo, hi, label in buckets:
        sub = [s for s in stats if lo <= s["lex"] <= hi]
        if not sub:
            continue
        ns = len(sub)
        mk2 = sum(s["k2"] for s in sub) / ns
        mk3 = sum(s["k3"] for s in sub) / ns
        mm2 = sum(s["meet2"] for s in sub) / ns
        gk2 = sum(1 for s in sub if s["gold_k2"])
        gk3 = sum(1 for s in sub if s["gold_k3"])
        gm2 = sum(1 for s in sub if s["gold_meet2"])
        print(
            f"  {label} ({ns}q): avg docs k2={mk2:.0f} k3={mk3:.0f} "
            f"meet2={mm2:.0f} | gold k2={gk2} k3={gk3} meet2={gm2}"
        )


if __name__ == "__main__":
    main()
