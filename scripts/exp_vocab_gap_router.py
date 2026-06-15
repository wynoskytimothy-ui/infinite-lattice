#!/usr/bin/env python3
"""
Vocabulary-gap router experiment — auto-gate PRF + teach where they help.

Methods (held-out test, full corpus):
  LEX           multi-view BM25 floor
  ZS-PRF        corridor PRF always on (drifts on clean corpora)
  TEACH         glossary correlations always on
  UNGATED       PRF + teach always fused
  ROUTED        PRF + teach only when measure_vocab_gap() fires
  SUP           supervised bridge ceiling (choose_bridge)

Run:  python scripts/exp_vocab_gap_router.py [scifact|nfcorpus|...]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from aethos_bridges import choose_bridge
from aethos_vocab_gap_router import (
    choose_expansion_mode,
    fuse_lex_expansion,
    measure_vocab_gap,
    prf_expansion,
    routed_search,
)
from scripts.bench_supervised_bridges import load, ndcg10, recall10
from aethos_teach_store import TeachStore
from scripts.exp_teach_no_forget import load_glossary, taught_rank
from scripts.exp_zeroshot_triangulation import IdfCache, build_index, recall_at


def teach_all(idx, N, glossary):
    teach = TeachStore(idx, N)
    for term, definition in glossary.items():
        teach.teach(f"{term} {definition}")
    return teach.finalize(top_k=16)


def ungated_search(idx, corpus, idf, query, lex, teach, rare_doc_cache, k=100):
    exp = defaultdict(float)
    for d, s in prf_expansion(idx, corpus, idf, query, lex, rare_doc_cache=rare_doc_cache).items():
        exp[d] += s
    for d, s in teach.expand_scores(query).items():
        exp[d] += s
    return fuse_lex_expansion(lex, dict(exp), k=k)


def eval_by_qid(label, fn, test_ids, queries, test_q):
    nd = r10 = r100 = 0.0
    t0 = time.time()
    for qid in test_ids:
        ranked = fn(qid)
        rels = test_q[qid]
        nd += ndcg10(ranked, rels)
        r10 += recall10(ranked, rels)
        r100 += recall_at(ranked, rels, 100)
    n = len(test_ids)
    return {"name": label, "ndcg10": nd / n, "recall10": r10 / n,
            "recall100": r100 / n, "ms": (time.time() - t0) / n * 1000}


def run_dataset(name):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    print(f"\n{'='*72}\n{name}: {len(corpus)} docs | test {len(test_ids)} q", flush=True)

    idx, N = build_index(corpus)
    idf = IdfCache(idx, N)
    glossary = load_glossary(name)
    teach = teach_all(idx, N, glossary) if glossary else None

    lex_cache = {qid: idx._score(queries[qid]) for qid in test_ids}
    rare_doc_cache = {}

    def f_lex(qid):
        lex = lex_cache[qid]
        return sorted(lex, key=lex.get, reverse=True)[:100]

    def f_prf(qid):
        lex = lex_cache[qid]
        exp = prf_expansion(idx, corpus, idf, queries[qid], lex, rare_doc_cache=rare_doc_cache)
        return fuse_lex_expansion(lex, exp, k=100)

    def f_teach(qid):
        from scripts.exp_teach_no_forget import taught_rank
        return taught_rank(idx, teach, queries[qid], lex_cache[qid], k=100)

    def f_ungated(qid):
        return ungated_search(idx, corpus, idf, queries[qid], lex_cache[qid],
                              teach, rare_doc_cache, k=100)

    exp_mode, exp_info = choose_expansion_mode(queries, train_q, corpus)
    print(f"  expansion mode: {exp_mode} {exp_info}", flush=True)

    routed_count = prf_routed = teach_routed = 0
    route_examples = []

    def f_routed(qid):
        nonlocal routed_count, prf_routed, teach_routed
        ranked, sig = routed_search(
            idx, corpus, idf, queries[qid], lex_cache[qid], teach,
            rare_doc_cache=rare_doc_cache, k=100, mode=exp_mode,
        )
        if sig.route_prf or sig.route_teach:
            routed_count += 1
            if sig.route_prf:
                prf_routed += 1
            if sig.route_teach:
                teach_routed += 1
            if len(route_examples) < 3:
                route_examples.append((qid, sig, queries[qid][:50]))
        return ranked

    res = []
    res.append(eval_by_qid("LEX (floor)", f_lex, test_ids, queries, test_q))
    res.append(eval_by_qid("ZS-PRF (always)", f_prf, test_ids, queries, test_q))
    if teach:
        res.append(eval_by_qid("TEACH (always)", f_teach, test_ids, queries, test_q))
        res.append(eval_by_qid("UNGATED (PRF+teach)", f_ungated, test_ids, queries, test_q))
    res.append(eval_by_qid("ROUTED (gated)", f_routed, test_ids, queries, test_q))

    t0 = time.time()
    bridge_obj, search_fn, bname, binfo = choose_bridge(idx, N, queries, train_q, corpus)
    sup_learn_s = time.time() - t0

    def f_sup(qid):
        return search_fn(idx, bridge_obj, queries[qid], lam=0.25, n_expand=30, k=100)

    res.append(eval_by_qid(f"SUP [{bname}]", f_sup, test_ids, queries, test_q))

    print(f"  router fired on {routed_count}/{len(test_ids)} queries "
          f"({100*routed_count/len(test_ids):.1f}%)  "
          f"PRF={prf_routed} teach={teach_routed}", flush=True)
    for qid, sig, qt in route_examples:
        print(f"    q{qid}: prf={sig.route_prf} teach={sig.route_teach} "
              f"gap={sig.gap_score:.2f} overlap={sig.overlap_ratio:.2f} "
              f"key_miss={sig.key_missing[:3]}  {qt!r}", flush=True)
    print(f"  (SUP learn {sup_learn_s:.1f}s | route={bname} {binfo})", flush=True)
    print(f"  {'method':<24}{'nDCG@10':>9}{'R@10':>8}{'R@100':>8}{'ms/q':>8}", flush=True)
    base = res[0]
    for r in res:
        dnd = r["ndcg10"] - base["ndcg10"]
        dr100 = r["recall100"] - base["recall100"]
        tag = "" if r is base else f"  (nDCG {dnd:+.4f}, R@100 {dr100:+.4f})"
        print(f"  {r['name']:<24}{r['ndcg10']:>9.4f}{r['recall10']:>8.4f}"
              f"{r['recall100']:>8.4f}{r['ms']:>8.1f}{tag}", flush=True)

    routed = next(r for r in res if r["name"] == "ROUTED (gated)")
    sup = next(r for r in res if r["name"].startswith("SUP"))
    prf = next(r for r in res if "ZS-PRF" in r["name"])
    gap_to_sup = sup["ndcg10"] - routed["ndcg10"]
    print(f"  ROUTED closes {100*(1 - gap_to_sup/(sup['ndcg10']-base['ndcg10']+1e-9)):.0f}% "
          f"of LEX-to-SUP nDCG gap  (ROUTED {routed['ndcg10']:.4f} vs SUP {sup['ndcg10']:.4f})",
          flush=True)

    return {
        "dataset": name,
        "routed_pct": routed_count / len(test_ids),
        "results": res,
        "bridge_route": bname,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    datasets = args or ["scifact", "nfcorpus"]
    print("Vocabulary-gap router: gate PRF + teach vs always-on vs SUP ceiling", flush=True)
    summary = []
    for ds in datasets:
        summary.append(run_dataset(ds))

    print(f"\n{'='*72}\nVERDICT", flush=True)
    for s in summary:
        r = {x["name"]: x for x in s["results"]}
        lex = r["LEX (floor)"]
        routed = r["ROUTED (gated)"]
        prf = r["ZS-PRF (always)"]
        print(f"  {s['dataset']}: route {100*s['routed_pct']:.0f}% | "
              f"PRF nDCG {prf['ndcg10']-lex['ndcg10']:+.4f} | "
              f"ROUTED nDCG {routed['ndcg10']-lex['ndcg10']:+.4f} | "
              f"R@100 {routed['recall100']-lex['recall100']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
