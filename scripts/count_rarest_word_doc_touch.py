#!/usr/bin/env python3
"""How many docs does each query's rarest word touch via kappa keys?"""
from __future__ import annotations

import math
import statistics
from pathlib import Path

from eval_beir import load_paths, load_qrels, load_queries, resolve_beir_root
from eval_beir_symbol import load_brain_and_plane, query_words
from aethos_rare_rank import _DocFreqCache, is_rare_word, degree_map_from_plane


def docs_for_word(plane, word: str) -> set[str]:
    out: set[str] = set()
    for k in plane.keys_for_word(word):
        out.update(plane.by_key.get(k, ()))
    return out


def idf(df: int, n: int) -> float:
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


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

        word_stats = []
        for w in words:
            df = cache.get(w)
            word_stats.append({
                "word": w,
                "df": df,
                "idf": idf(df, n_docs),
                "rare": is_rare_word(knowledge, w, df_cache=cache, degrees=degrees),
                "n_docs_kappa": len(docs_for_word(plane, w)),
            })

        # rarest = lowest DF (highest IDF); tie-break shortest word
        rarest = min(word_stats, key=lambda x: (x["df"], len(x["word"]), x["word"]))
        rare_docs = docs_for_word(plane, rarest["word"])
        gold_ids = set(qrels[qid])
        gold_hit = any(g in rare_docs for g in gold_ids)

        rows.append({
            "qid": qid,
            "rarest": rarest["word"],
            "rarest_df": rarest["df"],
            "rarest_idf": round(rarest["idf"], 3),
            "rarest_rare_flag": rarest["rare"],
            "docs_kappa": rarest["n_docs_kappa"],
            "frac_corpus": round(rarest["n_docs_kappa"] / n_docs, 4),
            "gold_in_rare_docs": gold_hit,
            "n_query_words": len(words),
        })

    n = len(rows)
    print(f"queries: {n}  corpus: {n_docs}\n")

    touches = [r["docs_kappa"] for r in rows]
    fracs = [r["frac_corpus"] for r in rows]
    print("--- Rarest query word -> docs touched (kappa inverted index) ---")
    print(f"  mean docs touched: {statistics.mean(touches):.0f}")
    print(f"  median: {statistics.median(touches):.0f}")
    print(f"  min / max: {min(touches)} / {max(touches)}")
    print(f"  mean fraction of corpus: {100 * statistics.mean(fracs):.1f}%")
    print()

    for thr, label in [
        (n_docs, "all docs (100%)"),
        (n_docs * 0.9, ">=90% corpus"),
        (n_docs * 0.5, ">=50% corpus"),
        (1000, ">=1000 docs"),
        (500, ">=500 docs"),
        (100, ">=100 docs"),
        (50, ">=50 docs"),
        (10, ">=10 docs"),
    ]:
        c = sum(1 for t in touches if t >= thr)
        print(f"  rarest word touches {label}: {c}/{n} queries ({100 * c / n:.1f}%)")

    print()
    rarest_actually_rare = sum(1 for r in rows if r["rarest_rare_flag"])
    print(f"  rarest word passes is_rare_word(): {rarest_actually_rare}/{n} ({100 * rarest_actually_rare / n:.1f}%)")
    print(f"  gold doc in rarest-word kappa docs: {sum(1 for r in rows if r['gold_in_rare_docs'])}/{n} "
          f"({100 * sum(1 for r in rows if r['gold_in_rare_docs']) / n:.1f}%)")
    print()

    print("--- Rarest word DF distribution ---")
    for lo, hi, lab in [(0, 10, "DF 0-10"), (11, 50, "DF 11-50"), (51, 200, "DF 51-200"),
                        (201, 1000, "DF 201-1000"), (1001, 99999, "DF 1000+")]:
        sub = [r for r in rows if lo <= r["rarest_df"] <= hi]
        if not sub:
            continue
        avg_touch = statistics.mean(r["docs_kappa"] for r in sub)
        print(f"  {lab}: {len(sub)} queries  avg docs touched={avg_touch:.0f}")

    print()
    print("--- Examples: rarest word touches almost ALL docs ---")
    heavy = sorted(rows, key=lambda r: -r["docs_kappa"])[:5]
    for r in heavy:
        print(f"  Q{r['qid']}: rarest={r['rarest']!r} df={r['rarest_df']} "
              f"touches={r['docs_kappa']}/{n_docs} ({100*r['frac_corpus']:.0f}%)")

    print()
    print("--- Examples: rarest word is truly selective ---")
    light = sorted(rows, key=lambda r: r["docs_kappa"])[:5]
    for r in light:
        print(f"  Q{r['qid']}: rarest={r['rarest']!r} df={r['rarest_df']} "
              f"touches={r['docs_kappa']}/{n_docs} gold_in={r['gold_in_rare_docs']}")


if __name__ == "__main__":
    main()
