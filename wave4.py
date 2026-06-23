#!/usr/bin/env python3
"""Wave 4 of the glass-box tuning campaign: WIDEN to 6 BEIR corpora (sequential build = memory-safe) and
test the MULTI-SIGNAL router (query length + corpus alignment) that adds fiqa's word-only fragment-cap for
everyday/aligned corpora. Reports nDCG@10 AND MRR@10 (the fragment-cap was an MRR win) vs the wave-3
query-length router, the best single rule, and the oracle ceiling.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.bench_supervised_bridges import load, ndcg10
from tune import build, corridors, search, toks

NAMES = ["scifact", "nfcorpus", "fiqa", "arguana", "scidocs", "trec-covid"]
CONFIGS = {
    "baseline": {},
    "div0.25 (single)": {"div": 0.25},
    "router_qlen (w3)": {"router": True},
    "router_multi (w4)": {"router2": True},
}


def evals(eng, corr, queries, test_q, ids, cfg):
    nd = mr = 0.0
    for q in ids:
        r = search(eng, corr, queries[q], cfg, 10)
        nd += ndcg10(r, test_q[q])
        for i, d in enumerate(r):
            if test_q[q].get(d, 0) > 0:
                mr += 1.0 / (i + 1); break
    n = len(ids); return nd / n, mr / n


def main():
    nd = {c: {} for c in CONFIGS}; mr = {c: {} for c in CONFIGS}; awls = {}
    for nm in NAMES:
        corpus, queries, train_q, test_q = load(nm)
        eng = build(corpus); corr = corridors(eng, corpus, queries, train_q) if train_q else {}
        ids = [q for q in test_q if q in queries]
        awls[nm] = eng["awl"]
        route = "len+div" if False else ("frag-cap" if eng["awl"] < 6.3 else "div")
        print(f"  {nm}: {eng['M']:,} docs, {len(ids)} q, awl {eng['awl']:.2f} -> short-q routes to '{route}'", flush=True)
        for cn, cfg in CONFIGS.items():
            nd[cn][nm], mr[cn][nm] = evals(eng, corr, queries, test_q, ids, cfg)
        del eng, corr, corpus

    n = len(NAMES); base_nd = sum(nd["baseline"].values()) / n
    print(f"\n  WAVE 4 -- {n} BEIR corpora (sequential). avg over corpora; baseline nDCG {base_nd:.4f}\n")
    print(f"  {'config':<20}{'nDCG':>7}{'dNDCG':>8}{'MRR':>8}  " + "".join(f"{x[:7]:>9}" for x in NAMES))
    for cn in CONFIGS:
        a = sum(nd[cn].values()) / n; m = sum(mr[cn].values()) / n
        print(f"  {cn:<20}{a:>7.4f}{a-base_nd:>+8.4f}{m:>8.4f}  " + "".join(f"{nd[cn][x]:>9.4f}" for x in NAMES))
    orc = {x: max(nd[c][x] for c in CONFIGS) for x in NAMES}; oa = sum(orc.values()) / n
    print(f"  {'ORACLE (nDCG)':<20}{oa:>7.4f}{oa-base_nd:>+8.4f}{'':>8}  " + "".join(f"{orc[x]:>9.4f}" for x in NAMES))


if __name__ == "__main__":
    main()
