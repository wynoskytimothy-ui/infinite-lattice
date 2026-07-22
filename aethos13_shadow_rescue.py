#!/usr/bin/env python3
"""EdgeRAG "SHADOW RESCUE" -- the Shadow Query as a strict, append-only recall fallback behind BM25.

Phase 5 verdict: the Shadow Query crosses the vocabulary gap (reaches zero-literal-overlap gold that BM25
scores 0) but DEMOTES ranking if used everywhere. So it is wired here as a SAFETY NET, never a reranker:

    1. BM25 runs first and OWNS the ranking. Its top-100 order is never touched.
    2. A query is a "failure" only if BM25's top-100 misses some gold. The Shadow stays DORMANT on all others.
    3. On failures, the Shadow fires: it expands the query to its baked O(1) twins, retrieves docs that BM25
       could not, and APPENDS them BELOW rank 100. Append-only => P@1 of every query is unchanged BY
       CONSTRUCTION (verified numerically, not asserted).

Two honesty rails:
  * TWO TRIGGERS. The ORACLE trigger (fire where gold is truly missing) measures the recall CEILING -- the
    most the shadow could ever recover. It is NOT deployable (you don't know at query time which failed).
    The DEPLOYABLE trigger fires on a gold-agnostic weakness signal (thin pool / low BM25 confidence);
    the gap between the two is the honest cost of not having the labels.
  * MISSING GOLD IS PARTITIONED. zero-overlap gold (shares NO content term with the query -> BM25 score 0,
    unreachable at ANY depth) vs drowned gold (has overlap, just ranked > 100). The Shadow's unique claim is
    ONLY the zero-overlap slice; a fair BM25 baseline given the SAME extra budget can recover drowned gold on
    its own, so we report BM25@(100+S) too.

usage:  python aethos13_shadow_rescue.py [corpus] [shadow_budget]
        python aethos13_shadow_rescue.py scifact 100
"""
from __future__ import annotations
import os, sys, math, json, collections
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from aethos13_phase5 import LatticeIndex, content
from _freq_cascade_lib import load_eval
from eval_beir import recall_at_k
from marco_baseline import BM25, tok


class ScoredBM25(BM25):
    """marco_baseline.BM25 + a scored search, so we can read BM25's own top-score as a confidence signal."""
    def search_scored(self, query, k=100) -> List[Tuple[str, float]]:
        sc = collections.defaultdict(float)
        for w in set(tok(query)):
            if w not in self.post:
                continue
            idf = self.idf[w]
            for di, c in self.post[w]:
                dl = self.doclen[di]
                sc[di] += idf * c * (self.k1 + 1) / (c + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        top = sorted(sc, key=sc.get, reverse=True)[:k]
        return [(self.docids[di], sc[di]) for di in top]


def shadow_retrieve(idx: LatticeIndex, qterms: List[str], exclude: Set[str], budget: int) -> List[str]:
    """The Shadow Query: expand each query term to its baked twins (O(1) lookup), gather docs that contain a
    twin but are NOT already in the BM25 pool, score by summed twin-idf, return the top `budget`."""
    shadow_terms = []
    for t in dict.fromkeys(qterms):
        shadow_terms += idx.twins(t)
    shadow_terms = [w for w in dict.fromkeys(shadow_terms) if w not in set(qterms)]
    if not shadow_terms:
        return []
    grav: Dict[str, float] = {}
    for w in shadow_terms:
        iw = idx.idf(w)
        for d in idx.post.get(w, ()):        # postings of the twin term
            if d in exclude:
                continue
            grav[d] = grav.get(d, 0.0) + iw
    return [d for d, _ in sorted(grav.items(), key=lambda z: -z[1])[:budget]]


def _line(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def run(corpus_name: str = "scifact", shadow_budget: int = 100, pool: int = 100):
    ev = load_eval(corpus_name)
    corpus = {d: (v.get("title", "") + " " + v.get("text", "")).strip() for d, v in ev["corpus"].items()}
    idx = LatticeIndex(corpus, mindf=3)
    bm = ScoredBM25(k1=1.5, b=0.75)
    bm.index([(d, corpus[d]) for d in corpus])
    qids = [q for q in ev["qids"] if ev["qrels"].get(q)]

    # gold-agnostic deployable signal: BM25 top score. Calibrate a low-confidence threshold on the
    # distribution itself (bottom quartile of top-scores) -- no labels used.
    tops = []
    base_cache = {}
    for q in qids:
        b = bm.search_scored(ev["queries"][q], k=pool)
        base_cache[q] = b
        tops.append(b[0][1] if b else 0.0)
    tops_sorted = sorted(tops)
    conf_thresh = tops_sorted[max(0, len(tops_sorted) // 4 - 1)]     # ~25th percentile of top-scores

    # accumulators
    n = 0
    base_p1 = 0.0
    base_r = collections.defaultdict(float)            # recall at pool and at pool+budget (BM25 own budget)
    # per-trigger rescue stats
    triggers = {"ORACLE (gold-aware, ceiling)": "oracle", "DEPLOYABLE (thin pool | low BM25 conf)": "deploy"}
    resc = {t: {"fired": 0, "p1": 0.0, "recall": 0.0, "zo_missing": 0, "zo_recovered": 0,
                "drown_missing": 0, "drown_recovered": 0, "wasted_fires": 0} for t in triggers}

    for q in qids:
        rel = set(ev["qrels"][q])
        G = rel
        if not G:
            continue
        n += 1
        base = base_cache[q]
        base_ids = [d for d, _ in base]
        base_set = set(base_ids)
        qterms = content(ev["queries"][q])
        qset = set(qterms)

        # baseline
        base_p1 += 1.0 if base_ids and base_ids[0] in G else 0.0
        base_r["@pool"] += recall_at_k(base_ids, {g: 1 for g in G}, pool)

        # BM25 given the SAME extra budget (fair: could deeper BM25 recover it?)
        big = bm.search_scored(ev["queries"][q], k=pool + shadow_budget)
        big_ids = [d for d, _ in big]
        base_r["@pool+budget(bm25)"] += recall_at_k(big_ids, {g: 1 for g in G}, pool + shadow_budget)

        # partition the MISSING gold (not in BM25 top-pool)
        missing = [g for g in G if g not in base_set]
        zo_missing = [g for g in missing if not (idx.SET.get(g, set()) & qset)]     # zero literal overlap
        drown_missing = [g for g in missing if g not in zo_missing]

        # the shadow pool (computed once; triggers decide whether to USE it)
        shadow_ids = shadow_retrieve(idx, qterms, base_set, shadow_budget)
        rescued_pool = base_ids + shadow_ids                       # APPEND-ONLY: head is BM25, untouched
        rescued_set = set(rescued_pool)
        zo_rec = [g for g in zo_missing if g in rescued_set]
        drown_rec = [g for g in drown_missing if g in rescued_set]

        # trigger decisions
        oracle_fire = len(missing) > 0
        deploy_fire = (len(base) < pool) or (base and base[0][1] < conf_thresh)

        for tname, kind in triggers.items():
            fire = oracle_fire if kind == "oracle" else deploy_fire
            if fire:
                resc[tname]["fired"] += 1
                pool_ids = rescued_pool
                if not oracle_fire:      # this query was actually a success -> shadow fired needlessly
                    resc[tname]["wasted_fires"] += 1
                resc[tname]["zo_recovered"] += len(zo_rec)
                resc[tname]["drown_recovered"] += len(drown_rec)
            else:
                pool_ids = base_ids
            # metrics on whatever pool this trigger produced
            resc[tname]["p1"] += 1.0 if pool_ids and pool_ids[0] in G else 0.0
            resc[tname]["recall"] += recall_at_k(pool_ids, {g: 1 for g in G}, pool + shadow_budget)
            resc[tname]["zo_missing"] += len(zo_missing)
            resc[tname]["drown_missing"] += len(drown_missing)

    # ---------- report ----------
    _line(f"EDGERAG SHADOW RESCUE  |  {corpus_name}  |  {n} queries, pool={pool}, shadow_budget={shadow_budget}")
    b_p1 = base_p1 / n
    b_r_pool = base_r["@pool"] / n
    b_r_big = base_r["@pool+budget(bm25)"] / n
    print(f"  BASELINE  BM25 (k1=1.5, b=0.75):")
    print(f"    P@1              = {b_p1:.4f}")
    print(f"    Recall@{pool:<4}      = {b_r_pool:.4f}")
    print(f"    Recall@{pool+shadow_budget:<4} (BM25 given the SAME extra depth) = {b_r_big:.4f}")
    print(f"    deployable low-confidence threshold (25th pct of BM25 top-score) = {conf_thresh:.3f}")

    for tname in triggers:
        r = resc[tname]
        p1 = r["p1"] / n
        rec = r["recall"] / n
        print(f"\n  {tname}")
        print(f"    fired on            {r['fired']}/{n} queries ({100*r['fired']/n:.1f}%)"
              + (f"   [{r['wasted_fires']} needless fires on already-successful queries]" if r["wasted_fires"] else ""))
        print(f"    P@1                 = {p1:.4f}   (delta vs BM25 = {p1 - b_p1:+.4f}  <-- MUST be 0.0000: append-only)")
        print(f"    Recall (returned pool) = {rec:.4f}")
        print(f"       vs BM25@{pool} baseline           = {rec - b_r_pool:+.4f}   <-- the 'strictly increases recall' claim (append-only guarantees >= 0)")
        print(f"       vs deeper BM25@{pool+shadow_budget} (same budget) = {rec - b_r_big:+.4f}   <-- honesty check: did shadow beat just going deeper?")
        zo_m, zo_r = r["zo_missing"], r["zo_recovered"]
        dr_m, dr_r = r["drown_missing"], r["drown_recovered"]
        print(f"    zero-overlap gold   recovered {zo_r}/{zo_m}   (gold BM25 scores 0 -> UNREACHABLE at ANY BM25 depth: the shadow's unique win)")
        print(f"    drowned gold        recovered {dr_r}/{dr_m}   (had overlap, ranked > {pool}; deeper BM25 can also get these)")

    _line("VERDICT")
    print("  P@1 is provably untouched (delta 0.0000) -- the rescue is APPEND-ONLY, it never reorders BM25's head.")
    print(f"  Against the real BM25@{pool} baseline the rescue STRICTLY INCREASES recall (delta >= 0 by construction).")
    print("  HONEST DECOMPOSITION of that gain: most of it is DROWNED gold (overlap, rank > pool) that a deeper")
    print(f"  BM25@{pool+shadow_budget} would also recover -- so vs deeper-BM25 the net can be ~0. The shadow's UNIQUE,")
    print("  can't-do-with-BM25 contribution is the ZERO-OVERLAP slice (BM25 score 0). That slice is small on")
    print("  well-aligned corpora but it is the only recall no BM25 depth can reach. DEPLOYMENT NOTE: keep BM25's")
    print(f"  FULL depth AND append shadow (don't cap BM25 at {pool}) so the deployable trigger never trails BM25@{pool+shadow_budget}.")
    print("  ORACLE = ceiling (labels pick failures); DEPLOYABLE = a real gold-agnostic trigger; the gap = the")
    print("  cost of not knowing which queries failed.")

    here = os.path.dirname(os.path.abspath(__file__))
    out = {"corpus": corpus_name, "n": n, "pool": pool, "shadow_budget": shadow_budget,
           "baseline": {"p@1": round(b_p1, 4), f"recall@{pool}": round(b_r_pool, 4),
                        f"recall@{pool+shadow_budget}_bm25": round(b_r_big, 4)},
           "triggers": {t: {"fired": resc[t]["fired"], "wasted_fires": resc[t]["wasted_fires"],
                            "p@1": round(resc[t]["p1"] / n, 4),
                            f"recall@{pool+shadow_budget}": round(resc[t]["recall"] / n, 4),
                            "zero_overlap_recovered": resc[t]["zo_recovered"],
                            "zero_overlap_missing": resc[t]["zo_missing"],
                            "drowned_recovered": resc[t]["drown_recovered"],
                            "drowned_missing": resc[t]["drown_missing"]} for t in triggers}}
    json.dump(out, open(os.path.join(here, f"_shadow_rescue_{corpus_name}.json"), "w"), indent=2)
    print(f"\n  wrote _shadow_rescue_{corpus_name}.json")
    return out


if __name__ == "__main__":
    cname = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    run(cname, shadow_budget=budget)
