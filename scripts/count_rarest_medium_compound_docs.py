#!/usr/bin/env python3
"""
Among docs touched by the query's rarest word, how many also connect via
pair-meet / brain compound (rarest + medium-common query word)?

Not solo medium-word kappa — compound corridor intersection only.
"""
from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

from eval_beir import load_paths, load_qrels, load_queries, resolve_beir_root
from eval_beir_symbol import load_brain_and_plane, query_words
from aethos_rare_rank import _DocFreqCache, is_hub_word, is_rare_word, degree_map_from_plane
from pipeline.bit_12_symbol_plane_index import (
    canonical_pair_key,
    get_pair_meet_keys,
    resolve_pair_link,
)

TOKEN = re.compile(r"[a-z]+")


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


def word_tier(
    knowledge, w: str, *, cache, degrees, n_docs: int,
) -> str:
    if is_hub_word(knowledge, w, df_cache=cache, degrees=degrees):
        return "hub"
    if is_rare_word(knowledge, w, df_cache=cache, degrees=degrees):
        return "rare"
    df = cache.get(w)
    if df <= 500:
        return "medium"
    return "common"


def doc_has_both_tokens(text: str, a: str, b: str) -> bool:
    toks = set(TOKEN.findall(text.lower()))
    return a in toks and b in toks


def main() -> None:
    root = Path(resolve_beir_root())
    paths = load_paths(root, "scifact")
    queries = load_queries(paths.queries)
    qrels = load_qrels(paths.qrels_test)
    knowledge, plane = load_brain_and_plane("scifact")
    cache = _DocFreqCache(knowledge)
    cache.warm_corpus()
    degrees = degree_map_from_plane(plane)
    n_docs = len(knowledge.corpus)

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

        stats = []
        for w in words:
            stats.append({
                "word": w,
                "df": cache.get(w),
                "tier": word_tier(knowledge, w, cache=cache, degrees=degrees, n_docs=n_docs),
            })
        rarest = min(stats, key=lambda x: (x["df"], len(x["word"]), x["word"]))
        rare_words = [s["word"] for s in stats if s["tier"] == "rare"]
        rw = rarest["word"]

        # medium-common partners: other query words that are not hubs (not just tier=medium)
        partners = [
            s["word"] for s in stats
            if s["word"] != rw
            and not is_hub_word(knowledge, s["word"], df_cache=cache, degrees=degrees)
        ]
        medium_words = [
            s["word"] for s in stats
            if s["word"] != rw
            and s["tier"] == "medium"
        ]
        n_rare = len([s for s in stats if is_rare_word(
            knowledge, s["word"], df_cache=cache, degrees=degrees,
        )])
        rare_docs = docs_for_word(plane, rw)
        if not rare_docs:
            continue

        # pair-meet compound corridors: rarest x each medium word
        compound_meet_docs: set[str] = set()
        compound_pairs: list[tuple[str, str]] = []
        for mw in partners:
            ca = knowledge.morph_canonical_surface(rw)
            cb = knowledge.morph_canonical_surface(mw)
            cpk = canonical_pair_key(knowledge, ca, cb)
            meet = plane.pair_keys.get(cpk)
            lk = knowledge.correlates(ca, cb) or resolve_pair_link(knowledge, ca, cb)
            if meet is None and lk:
                meet = get_pair_meet_keys(plane, knowledge, ca, cb, link=lk)
            if not meet:
                continue
            hit = docs_for_meet(plane, meet) & rare_docs
            if hit:
                compound_meet_docs |= hit
                compound_pairs.append((rw, mw))

        # also: brain direct/bridge link rarest-medium AND doc has both tokens
        cooccur_compound: set[str] = set()
        for mw in partners:
            ca, cb = knowledge.morph_canonical_surface(rw), knowledge.morph_canonical_surface(mw)
            if knowledge.correlates(ca, cb) is None:
                continue
            for did in rare_docs:
                text = knowledge.corpus.get(did, "")
                if doc_has_both_tokens(text, ca, cb) or doc_has_both_tokens(text, rw, mw):
                    cooccur_compound.add(did)

        # medium solo: doc in rare_docs, hits medium word kappa, NOT in compound_meet
        medium_solo: set[str] = set()
        for mw in partners:
            md = docs_for_word(plane, mw)
            for did in rare_docs & md:
                if did not in compound_meet_docs:
                    medium_solo.add(did)

        union_compound = compound_meet_docs | cooccur_compound
        gold = set(qrels[qid])
        rows.append({
            "qid": qid,
            "n_rare_words": n_rare,
            "lt2_rare": n_rare < 2,
            "rarest": rw,
            "rarest_df": rarest["df"],
            "n_partners": len(partners),
            "n_medium": len(medium_words),
            "medium_words": medium_words[:6],
            "n_rare_docs": len(rare_docs),
            "n_compound_meet": len(compound_meet_docs),
            "n_cooccur_link": len(cooccur_compound),
            "n_union_compound": len(union_compound),
            "n_medium_solo_only": len(medium_solo - union_compound),
            "n_rarest_only": len(rare_docs - union_compound - medium_solo),
            "n_compound_pairs": len(compound_pairs),
            "gold_in_rare": any(g in rare_docs for g in gold),
            "gold_in_compound": any(g in union_compound for g in gold),
            "gold_in_meet": any(g in compound_meet_docs for g in gold),
        })

    n = len(rows)
    sub = [r for r in rows if r["lt2_rare"]]
    ns = len(sub)

    def summarize(label: str, data: list[dict]) -> None:
        if not data:
            return
        nd = len(data)
        print(f"--- {label} ({nd} queries) ---")
        for key in (
            "n_rare_docs", "n_compound_meet", "n_union_compound",
            "n_medium_solo_only", "n_rarest_only",
        ):
            vals = [r[key] for r in data]
            print(f"  {key}: mean={statistics.mean(vals):.1f} median={statistics.median(vals):.0f}")
        pct_compound = [
            100 * r["n_union_compound"] / r["n_rare_docs"] if r["n_rare_docs"] else 0
            for r in data
        ]
        print(f"  compound / rare_docs: mean={statistics.mean(pct_compound):.1f}%")
        g_rare = sum(1 for r in data if r["gold_in_rare"])
        g_comp = sum(1 for r in data if r["gold_in_compound"])
        g_meet = sum(1 for r in data if r["gold_in_meet"])
        print(f"  gold in rare_docs: {g_rare}/{nd} ({100*g_rare/nd:.1f}%)")
        print(f"  gold in compound (meet+cooccur): {g_comp}/{nd} ({100*g_comp/nd:.1f}%)")
        print(f"  gold in pair-meet compound: {g_meet}/{nd} ({100*g_meet/nd:.1f}%)")
        has_partners = [r for r in data if r["n_partners"] > 0]
        if has_partners:
            nm = len(has_partners)
            frac = [
                100 * r["n_union_compound"] / r["n_rare_docs"]
                for r in has_partners if r["n_rare_docs"]
            ]
            print(f"  queries with non-hub partners: {nm}/{nd}")
            print(f"  compound/rare_docs (those q): mean={statistics.mean(frac):.1f}%")
            meet_frac = [
                100 * r["n_compound_meet"] / r["n_rare_docs"]
                for r in has_partners if r["n_rare_docs"]
            ]
            print(f"  pair-meet compound/rare_docs: mean={statistics.mean(meet_frac):.1f}%")
        print()

    print(f"queries: {n}  corpus: {n_docs}\n")
    summarize("ALL queries", rows)
    summarize("<2 rare words in query", sub)
    summarize(">=2 rare words in query", [r for r in rows if not r["lt2_rare"]])

    # bucket: rare_docs pool size ~430
    band = [r for r in sub if 100 <= r["n_rare_docs"] <= 800]
    summarize("lt2 rare AND rare_docs 100-800 (~typical pool)", band)


if __name__ == "__main__":
    main()
