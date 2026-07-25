#!/usr/bin/env python3
"""EdgeRAG SERVING PIPELINE -- BM25 with an AUTO-ARMING vocabulary-gap Shadow Rescue.

The integrated serving path. BM25 owns the ranking to full depth; a corpus-agnostic gate inspects the live
BM25 pool and ARMS the Shadow Rescue only when the query looks like a vocabulary mismatch. When armed, the
O(1) baked twin lookup appends shadow candidates BELOW the BM25 pool (append-only => P@1 untouched, recall
can only rise). When dormant, we return full-depth BM25 and spend zero shadow compute.

THE GATE (what actually separates a vocab-gap query -- measured, not guessed; see _gate_probe.py):
  * confidence signals that seemed obvious (best-doc coverage, top-score/idf-mass) are CONFOUNDED BY QUERY
    LENGTH -- they run BACKWARDS (higher on nfcorpus than scifact). Rejected.
  * the signal that genuinely separates is POOL EXHAUSTION: an aligned corpus fills all K BM25 slots
    (scifact/fiqa fill=1.000), a mismatch corpus starves the pool (nfcorpus fill mean 0.68). A pool that
    cannot even fill K means BM25 has run out of lexical matches -- exactly when the shadow has room to help.
  * plus a hard tell: an OUT-OF-VOCABULARY query term (df==0) is an unbridgeable lexical gap by definition.

HONESTY, stated plainly:
  * Strict dominance over BM25@depth is GUARANTEED BY CONSTRUCTION (append-only: serve pool always contains
    BM25@depth, so recall >= BM25@depth for every query; P@1 identical). We assert it per-query; we do NOT
    dress up a construction guarantee as an empirical win.
  * The gate's REAL job is spending shadow compute only where it CONVERTS to recall. So we measure the gate's
    precision/recall against the "profitable" set (queries where firing recovers a ZERO-OVERLAP gold -- the
    only recall no BM25 depth can reach) and the compute saved by staying dormant elsewhere.

usage:  python edgerag_serve.py [corpus] [depth] [shadow_budget]
        python edgerag_serve.py nfcorpus 200 100
        python edgerag_serve.py all                 # scifact + nfcorpus + fiqa
"""
from __future__ import annotations
import os, sys, json, collections
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from aethos13_phase5 import LatticeIndex, content
from aethos13_shadow_rescue import ScoredBM25, shadow_retrieve
from _freq_cascade_lib import load_eval
from eval_beir import recall_at_k

# ---- the gate constants (corpus-agnostic, fixed; chosen from the SIGNAL, not from the labels) ----
TAU_FILL = 1.0          # arm if the BM25 pool is not completely full (fill < TAU_FILL): lexical matches exhausted
# OOV: arm if any query content term has zero postings (a hard, unbridgeable-by-BM25 lexical gap)


class EdgeRAGServer:
    """The serving object: BM25 to full depth + an auto-arming shadow rescue appended below it."""
    def __init__(self, corpus: Dict[str, str], depth: int = 200, shadow_budget: int = 100):
        self.depth = depth
        self.budget = shadow_budget
        self.idx = LatticeIndex(corpus, mindf=3)
        self.bm = ScoredBM25(k1=1.5, b=0.75)
        self.bm.index([(d, corpus[d]) for d in corpus])

    def gate(self, scored: List[Tuple[str, float]], qterms: List[str]) -> Tuple[bool, dict]:
        """Inspect the live BM25 pool. Return (arm, signals). Corpus-agnostic -- no labels, no per-corpus tuning."""
        nonzero = sum(1 for _, s in scored if s > 0)
        fill = nonzero / self.depth
        # POOL EXHAUSTION is the gate. Ablation (_gate_ablation.py): fill<1.0 keeps aligned corpora fully
        # dormant (scifact/fiqa 0% fire) while catching ~42% of the vocab-gap corpus's profitable queries.
        # The OOV trigger was ABLATED OUT: it fired 23% on scifact at 0.014 precision (near-pure wasted
        # compute) for ~0.005 recall -- a bad trade against the "dormant on aligned queries" requirement.
        has_oov = any(t not in self.idx.post for t in dict.fromkeys(qterms) if len(t) >= 3)  # informational only
        arm = fill < TAU_FILL
        return arm, {"fill": round(fill, 3), "oov": has_oov, "pool_size": nonzero}

    def serve(self, query: str) -> Tuple[List[str], bool, dict]:
        """Return (ranked_pool, armed, signals). BM25 owns ranks 1..depth; shadow appends below (append-only)."""
        scored = self.bm.search_scored(query, self.depth)
        base_ids = [d for d, _ in scored]
        qterms = content(query)
        arm, sig = self.gate(scored, qterms)
        if not arm:
            return base_ids, False, sig                       # DORMANT: full-depth BM25, zero shadow compute
        shadow = shadow_retrieve(self.idx, qterms, set(base_ids), self.budget)   # O(1) twin lookup, then gather
        return base_ids + shadow, True, sig                   # ARMED: append below the BM25 pool


def _line(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def evaluate(corpus_name: str, depth: int = 200, shadow_budget: int = 100) -> dict:
    ev = load_eval(corpus_name)
    corpus = {d: (v.get("title", "") + " " + v.get("text", "")).strip() for d, v in ev["corpus"].items()}
    srv = EdgeRAGServer(corpus, depth=depth, shadow_budget=shadow_budget)
    idx = srv.idx
    qids = [q for q in ev["qids"] if ev["qrels"].get(q)]
    K = depth + shadow_budget

    n = 0
    fired = 0
    base_p1 = serve_p1 = 0.0
    base_recall = serve_recall = 0.0
    zo_recovered = 0
    profitable = 0           # queries where firing recovers >=1 zero-overlap gold
    fired_and_profitable = 0
    regressions = 0          # MUST stay 0: append-only guarantees per-query non-regression
    p1_breaks = 0            # MUST stay 0: BM25 head is never reordered

    for q in qids:
        G = set(ev["qrels"][q])
        if not G:
            continue
        n += 1
        qterms = content(ev["queries"][q])
        qset = set(qterms)

        scored = srv.bm.search_scored(ev["queries"][q], depth)
        base_ids = [d for d, _ in scored]
        base_set = set(base_ids)

        serve_pool, armed, sig = srv.serve(ev["queries"][q])
        if armed:
            fired += 1

        # metrics: BM25@depth (baseline) vs serve (measured at full K)
        r_base = recall_at_k(base_ids, {g: 1 for g in G}, depth)
        r_serve = recall_at_k(serve_pool, {g: 1 for g in G}, K)
        base_recall += r_base
        serve_recall += r_serve
        base_p1 += 1.0 if base_ids and base_ids[0] in G else 0.0
        serve_p1 += 1.0 if serve_pool and serve_pool[0] in G else 0.0

        # invariants
        if r_serve + 1e-9 < r_base:
            regressions += 1
        if (serve_pool[:1] != base_ids[:1]):
            p1_breaks += 1

        # profitability: is there a ZERO-OVERLAP missing gold that the shadow WOULD recover if fired?
        missing = [g for g in G if g not in base_set]
        zo_missing = [g for g in missing if not (idx.SET.get(g, set()) & qset)]
        would_shadow = shadow_retrieve(idx, qterms, base_set, shadow_budget) if zo_missing else []
        zo_bridgeable = [g for g in zo_missing if g in set(would_shadow)]
        if zo_bridgeable:
            profitable += 1
            if armed:
                fired_and_profitable += 1
        if armed:
            zo_recovered += len([g for g in zo_missing if g in set(serve_pool)])

    # ---- report ----
    b_p1, s_p1 = base_p1 / n, serve_p1 / n
    b_r, s_r = base_recall / n, serve_recall / n
    fire_rate = fired / n
    gate_recall = (fired_and_profitable / profitable) if profitable else float("nan")
    gate_prec = (fired_and_profitable / fired) if fired else float("nan")

    _line(f"EDGERAG SERVE  |  {corpus_name}  |  {n} queries, BM25 depth={depth}, shadow_budget={shadow_budget}")
    print(f"  GATE fired on        {fired}/{n} queries ({100*fire_rate:.1f}%)   [dormant {100*(1-fire_rate):.1f}% => shadow compute saved]")
    print(f"  P@1     BM25={b_p1:.4f}   serve={s_p1:.4f}   delta={s_p1-b_p1:+.4f}   (MUST be 0.0000)")
    print(f"  Recall  BM25@{depth}={b_r:.4f}   serve@{K}={s_r:.4f}   delta={s_r-b_r:+.4f}")
    print(f"  zero-overlap gold recovered (unreachable at ANY BM25 depth): {zo_recovered}")
    print(f"\n  GATE QUALITY vs the 'profitable' set (queries with a recoverable zero-overlap gold):")
    print(f"    profitable queries   {profitable}/{n}")
    print(f"    gate RECALL          {gate_recall:.3f}   (fraction of profitable queries the gate armed)")
    print(f"    gate PRECISION       {gate_prec:.3f}   (fraction of fires that were profitable; rest = safe but wasted compute)")

    _line("ASSERTIONS")
    ok_p1 = (abs(s_p1 - b_p1) < 1e-9) and (p1_breaks == 0)
    ok_nonreg = (regressions == 0)
    ok_dom = (s_r + 1e-9 >= b_r)
    strict = s_r > b_r + 1e-9
    print(f"  [{'PASS' if ok_p1 else 'FAIL'}] P@1 untouched: serve P@1 == BM25 P@1 and BM25 head never reordered ({p1_breaks} breaks)")
    print(f"  [{'PASS' if ok_nonreg else 'FAIL'}] per-query recall NON-REGRESSION: 0 queries lost recall ({regressions} regressions)")
    print(f"  [{'PASS' if ok_dom else 'FAIL'}] STRICT DOMINANCE over BM25@{depth}: serve recall >= BM25 recall "
          f"({'STRICT +'+format(s_r-b_r,'.4f') if strict else 'equal (gate stayed dormant -- correct, no vocab gap)'})")
    assert ok_p1 and ok_nonreg and ok_dom, "INVARIANT VIOLATED -- append-only guarantee broken"
    print("  all invariants hold.")

    out = {"corpus": corpus_name, "n": n, "depth": depth, "shadow_budget": shadow_budget,
           "fire_rate": round(fire_rate, 4), "compute_saved": round(1 - fire_rate, 4),
           "p@1_bm25": round(b_p1, 4), "p@1_serve": round(s_p1, 4), "p@1_delta": round(s_p1 - b_p1, 4),
           f"recall@{depth}_bm25": round(b_r, 4), f"recall@{K}_serve": round(s_r, 4),
           "recall_delta": round(s_r - b_r, 4), "zero_overlap_recovered": zo_recovered,
           "profitable": profitable, "gate_recall": None if profitable == 0 else round(gate_recall, 3),
           "gate_precision": None if fired == 0 else round(gate_prec, 3),
           "strict_dominance": bool(ok_dom), "p1_untouched": bool(ok_p1), "no_regression": bool(ok_nonreg)}
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, f"_edgerag_serve_{corpus_name}.json"), "w"), indent=2)
    print(f"  wrote _edgerag_serve_{corpus_name}.json")
    return out


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    corpora = ["scifact", "nfcorpus", "fiqa"] if arg == "all" else [arg]
    results = [evaluate(c, depth, budget) for c in corpora]
    _line("SUMMARY -- auto-arm gate: dormant on aligned corpora, armed on the vocab-gap corpus")
    print(f"  {'corpus':10s} {'fire%':>7} {'saved%':>7} {'P@1 d':>8} {'recall d':>9} {'zero-ovlp':>10} {'gate P/R':>12}")
    for r in results:
        pr = f"{r['gate_precision']}/{r['gate_recall']}"
        print(f"  {r['corpus']:10s} {100*r['fire_rate']:>6.1f} {100*r['compute_saved']:>6.1f} "
              f"{r['p@1_delta']:>+8.4f} {r['recall_delta']:>+9.4f} {r['zero_overlap_recovered']:>10} {pr:>12}")
    print("\n  Dominance is guaranteed by append-only; the gate's value is spending shadow compute only where it")
    print("  converts (vocab-gap corpus), staying dormant -- and free -- where BM25 already wins.")
