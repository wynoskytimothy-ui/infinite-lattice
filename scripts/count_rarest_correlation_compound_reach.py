#!/usr/bin/env python3
"""
From the query's rarest trigger word, follow correlation neighbors and shared
compound (pair-meet) corridors. Count docs reaching gold vs other docs.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from eval_beir import load_paths, load_qrels, load_queries, resolve_beir_root
from eval_beir_symbol import load_brain_and_plane, query_words
from aethos_rare_rank import (
    _DocFreqCache,
    _pair_strength_from_adj,
    degree_map_from_plane,
    rare_neighbors,
)
from pipeline.bit_12_symbol_plane_index import (
    canonical_pair_key,
    get_pair_meet_keys,
    resolve_pair_link,
)


def docs_for_word(plane, word: str) -> set[str]:
    out: set[str] = set()
    for k in plane.keys_for_word(word):
        out.update(plane.by_key.get(k, ()))
    return out


def docs_for_meet(plane, meet: frozenset) -> set[str]:
    out: set[str] = set()
    for k in meet:
        out.update(plane.by_key.get(k, ()))
    return out


def main() -> None:
    root = Path(resolve_beir_root())
    paths = load_paths(root, "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    knowledge, plane = load_brain_and_plane("scifact")
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)
    adj = plane.word_adjacency
    nbr_kw = dict(df_cache=cache, adjacency=adj, rare_cache={}, degrees=degrees)

    rows: list[dict] = []
    for qid in qrels:
        if qid not in queries:
            continue
        try:
            words = query_words(queries[qid])
        except Exception:
            continue
        if not words:
            continue

        word_dfs = [(w, cache.get(w)) for w in words]
        rarest_w = min(word_dfs, key=lambda x: (x[1], len(x[0]), x[0]))[0]
        ca = knowledge.morph_canonical_surface(rarest_w)
        gold = set(qrels[qid])

        rarest_solo = docs_for_word(plane, ca)
        correlation_meet: set[str] = set()
        neighbor_solo: set[str] = set()
        n_corr_links = 0
        n_meet_links = 0
        corr_examples: list[tuple[str, float, int]] = []

        # brain neighbors (all strengths) from rarest — semantic trigger words
        brain_nbrs: list[tuple[str, float, str]] = []
        for other, strength, kind in adj.get(ca, ()):
            brain_nbrs.append((other, strength, kind))
        brain_nbrs.sort(key=lambda x: (-x[1], x[0]))
        brain_nbrs = brain_nbrs[:12]

        # rare-only correlation neighbors
        rare_nbrs = rare_neighbors(knowledge, ca, limit=12, **nbr_kw)

        seen_pairs: set[tuple[str, str]] = set()
        for other, strength in rare_nbrs:
            pair = tuple(sorted((ca, other)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            n_corr_links += 1
            cb = knowledge.morph_canonical_surface(other)
            cpk = canonical_pair_key(knowledge, ca, cb)
            meet = plane.pair_keys.get(cpk)
            lk = knowledge.correlates(ca, cb) or resolve_pair_link(knowledge, ca, cb)
            if meet is None and lk:
                meet = get_pair_meet_keys(plane, knowledge, ca, cb, link=lk)
            if meet:
                n_meet_links += 1
                hit = docs_for_meet(plane, meet)
                correlation_meet |= hit
                corr_examples.append((other, strength, len(hit)))

        # neighbor solo: neighbor kappa docs not in compound meet path
        for other, strength, _kind in brain_nbrs:
            nd = docs_for_word(plane, other)
            for did in nd:
                if did not in correlation_meet:
                    neighbor_solo.add(did)

        # rarest solo: rarest docs not reached via compound meet
        rarest_only = rarest_solo - correlation_meet
        union_all = rarest_solo | correlation_meet | neighbor_solo

        def split(docs: set[str]) -> tuple[int, int]:
            g = len(docs & gold)
            return g, len(docs) - g

        g_meet, o_meet = split(correlation_meet)
        g_rare, o_rare = split(rarest_only)
        g_nbr, o_nbr = split(neighbor_solo - rarest_solo)
        g_all, o_all = split(union_all)

        rows.append({
            "qid": qid,
            "rarest": ca,
            "n_corr_neighbors": n_corr_links,
            "n_with_meet": n_meet_links,
            "n_compound_docs": len(correlation_meet),
            "n_rarest_only": len(rarest_only),
            "n_neighbor_solo": len(neighbor_solo - rarest_solo),
            "n_union": len(union_all),
            "gold_compound": g_meet,
            "other_compound": o_meet,
            "gold_rarest_only": g_rare,
            "gold_union": g_all,
            "gold_in_compound": g_meet > 0,
            "gold_in_union": g_all > 0,
            "top_corr": corr_examples[:3],
        })

    n = len(rows)
    print(f"queries: {n}\n")

    def block(label: str, data: list[dict]) -> None:
        if not data:
            return
        nd = len(data)
        print(f"--- {label} ({nd} queries) ---")
        for key in ("n_compound_docs", "n_rarest_only", "n_neighbor_solo", "n_union"):
            vals = [r[key] for r in data]
            print(f"  {key}: mean={statistics.mean(vals):.1f} median={statistics.median(vals):.0f}")

        g_comp = sum(r["gold_compound"] for r in data)
        o_comp = sum(r["other_compound"] for r in data)
        g_rare = sum(r["gold_rarest_only"] for r in data)
        q_g_comp = sum(1 for r in data if r["gold_in_compound"])
        q_g_union = sum(1 for r in data if r["gold_in_union"])
        print(f"  compound corridor docs: {g_comp} gold / {o_comp} other (total doc-slots)")
        print(f"  rarest-only docs: {sum(r['gold_rarest_only'] for r in data)} gold doc-slots")
        print(f"  queries with gold in compound path: {q_g_comp}/{nd} ({100*q_g_comp/nd:.1f}%)")
        print(f"  queries with gold anywhere (union): {q_g_union}/{nd} ({100*q_g_union/nd:.1f}%)")
        with_meet = [r for r in data if r["n_with_meet"] > 0]
        if with_meet:
            nm = len(with_meet)
            print(f"  queries with >=1 correlation+meet link: {nm}/{nd}")
            qg = sum(1 for r in with_meet if r["gold_in_compound"])
            print(f"    gold via compound on those: {qg}/{nm} ({100*qg/nm:.1f}%)")
        print()

    block("ALL", rows)
    block("compound docs > 0", [r for r in rows if r["n_compound_docs"] > 0])
    block("no compound meet (0 docs)", [r for r in rows if r["n_compound_docs"] == 0])

    print("--- Examples: compound path hits gold ---")
    for r in sorted([x for x in rows if x["gold_in_compound"]], key=lambda x: -x["gold_compound"])[:5]:
        print(f"  Q{r['qid']} rarest={r['rarest']!r} compound_docs={r['n_compound_docs']} "
              f"gold={r['gold_compound']} corr={r['top_corr']}")


if __name__ == "__main__":
    main()
