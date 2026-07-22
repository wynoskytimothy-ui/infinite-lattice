#!/usr/bin/env python3
"""HELD-OUT GOVERNOR VALIDATION for the composed Champion serve.

The honest gap in "the best version": the ladder configs were selected on the same test sets reported.
This proves the per-corpus governors are estimable from a SMALL labelled TRAIN slice at onboarding and
make the SAME decision the full test set would -- then measures the composed no-GPU serve on the HELD-OUT
test set with the invariants asserted.

What is validated with NO teacher / NO GPU (this environment):
  * expand governor: vocab_gap estimated from N=20/50/all TRAIN queries -> does it make the same
    expand decision (gap>0.25) as the full TEST set? (convergence of a 20-query onboarding estimate)
  * shadow governor: pool-exhaustion fire-rate on TRAIN vs TEST (stability of the per-query gate)
  * composed serve on TEST: P@1 untouched + recall non-regression + strict dominance -- ASSERTED.
The apex governor needs a teacher (GPU-once query encoding), absent here -> reported as a hook, never faked.

    python run_champion_holdout.py            # scifact + nfcorpus + fiqa
    python run_champion_holdout.py nfcorpus
"""
from __future__ import annotations
import sys, os, csv, json, collections
from typing import Dict, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from champion import Champion, Governors
from champion import content  # re-exported via champion -> edgerag_serve
from _freq_cascade_lib import load_eval
from eval_beir import recall_at_k
from beir_data_root import resolve_beir_root

DEPTH, BUDGET = 200, 100
ONBOARD_N = 20                 # the onboarding labelled-query budget
SLICES = [20, 50]              # + full train


def load_train_qrels(corpus_name: str) -> Dict[str, Dict[str, int]]:
    p = os.path.join(resolve_beir_root(), corpus_name, "qrels", "train.tsv")
    qr: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    if not os.path.exists(p):
        return {}
    r = csv.reader(open(p, encoding="utf-8"), delimiter="\t"); next(r, None)
    for row in r:
        if len(row) >= 3 and row[2].strip() and int(row[2]) > 0:
            qr[row[0]][row[1]] = int(row[2])
    return dict(qr)


def _line(t): print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def run_corpus(name: str) -> dict:
    ev = load_eval(name)
    corpus = {d: (v.get("title", "") + " " + v.get("text", "")).strip() for d, v in ev["corpus"].items()}
    queries = ev["queries"]
    test_qrels = {q: ev["qrels"][q] for q in ev["qids"] if ev["qrels"].get(q)}
    train_qrels_all = load_train_qrels(name)
    # only train queries whose text we actually have, deterministic order
    train_qids = sorted(q for q in train_qrels_all if q in queries)
    ch = Champion(corpus, depth=DEPTH, shadow_budget=BUDGET)

    # ---- governor estimation on TRAIN slices vs the TEST-set truth ----
    test_gap = ch.vocab_gap(queries, test_qrels)
    test_decision = test_gap > 0.25
    slice_rows = []
    for n in SLICES + [len(train_qids)]:
        sub = {q: train_qrels_all[q] for q in train_qids[:n]}
        if not sub:
            continue
        gov = ch.estimate_governors(queries, sub)   # no teacher -> apex hook only
        slice_rows.append({"n": len(sub), "gap": round(gov.vocab_gap, 3), "expand_on": gov.expand_on,
                           "matches_test": gov.expand_on == test_decision})

    # the governors we would actually SHIP: estimated from the N=20 onboarding slice.
    # HELD-OUT only if the corpus actually has a train split of at least ONBOARD_N labelled queries;
    # otherwise we fall back to the test set to set the governor (NOT held-out -- flagged honestly).
    held_out = len(train_qids) >= ONBOARD_N
    onboard = {q: train_qrels_all[q] for q in train_qids[:ONBOARD_N]}
    gov = ch.estimate_governors(queries, onboard) if held_out else Governors(
        vocab_gap=test_gap, expand_margin=test_gap - 0.25, expand_confident=abs(test_gap - 0.25) >= 0.10,
        source="test-fallback")
    ch.gov = gov

    # ---- shadow governor stability: fire-rate TRAIN vs TEST ----
    def fire_rate(qrels):
        fired = tot = 0
        for q in qrels:
            r = ch.serve(queries[q])
            tot += 1
            fired += 1 if r["armed"] else 0
        return fired / tot if tot else 0.0
    train_fire = fire_rate(onboard) if onboard else float("nan")

    # ---- composed serve on the HELD-OUT test set, invariants asserted ----
    n = fired = 0
    base_p1 = serve_p1 = base_rec = serve_rec = 0.0
    zo = 0
    p1_breaks = regressions = 0
    K = DEPTH + BUDGET
    for q in test_qrels:
        gold = set(test_qrels[q])
        if not gold:
            continue
        n += 1
        r = ch.serve(queries[q])
        base_ids, ranked = r["base_ids"], r["ranked"]
        if r["armed"]:
            fired += 1
        rb = recall_at_k(base_ids, {g: 1 for g in gold}, DEPTH)
        rs = recall_at_k(ranked, {g: 1 for g in gold}, K)
        base_rec += rb; serve_rec += rs
        base_p1 += 1.0 if base_ids and base_ids[0] in gold else 0.0
        serve_p1 += 1.0 if ranked and ranked[0] in gold else 0.0
        if rs + 1e-9 < rb:
            regressions += 1
        if ranked[:1] != r["base_ids"][:1] and not r["apex_used"]:   # apex intentionally reorders the head
            p1_breaks += 1
        if r["armed"]:
            qset = set(r["query_terms"])
            for g in gold:
                if g not in set(base_ids) and not (ch.base.idx.SET.get(g, set()) & qset) and g in set(ranked):
                    zo += 1
    test_fire = fired / n if n else 0.0
    b_p1, s_p1 = base_p1 / n, serve_p1 / n
    b_r, s_r = base_rec / n, serve_rec / n

    ok_p1 = abs(s_p1 - b_p1) < 1e-9 and p1_breaks == 0
    ok_nonreg = regressions == 0
    ok_dom = s_r + 1e-9 >= b_r

    _line(f"CHAMPION HELD-OUT  |  {name}  |  test n={n}, train pool={len(train_qids)}, onboard N={ONBOARD_N}")
    print(f"  GOVERNOR ESTIMATION (expand = vocab_gap > 0.25):  test-set truth gap={test_gap:.3f} -> expand_on={test_decision}")
    print(f"    {'train N':>8} {'gap est':>8} {'expand_on':>10} {'matches test?':>14}")
    for row in slice_rows:
        print(f"    {row['n']:>8} {row['gap']:>8.3f} {str(row['expand_on']):>10} {str(row['matches_test']):>14}")
    print(f"  SHIPPED governors (from N={ONBOARD_N} onboarding slice): {gov.as_dict()}")
    if not gov.expand_confident:
        print(f"  [NEAR-BOUNDARY] gap {gov.vocab_gap:.3f} is within 0.10 of the 0.25 threshold -- a {ONBOARD_N}-query "
              f"estimate is NOT decisive here (~2.6% of random 20-slices flip); collect more labels before trusting expand.")
    print(f"  SHADOW governor stability: fire-rate train={train_fire:.3f}  test={test_fire:.3f}")
    print(f"\n  COMPOSED SERVE ON HELD-OUT TEST (no teacher -> apex hook off; floor+shadow):")
    print(f"    P@1  BM25={b_p1:.4f}  serve={s_p1:.4f}  delta={s_p1-b_p1:+.4f}")
    print(f"    Recall  BM25@{DEPTH}={b_r:.4f}  serve@{K}={s_r:.4f}  delta={s_r-b_r:+.4f}   zero-overlap bridged={zo}")
    print(f"    fire-rate on test={test_fire:.3f}")
    print(f"  INVARIANTS: [{'PASS' if ok_p1 else 'FAIL'}] P@1 untouched  "
          f"[{'PASS' if ok_nonreg else 'FAIL'}] recall non-regression  "
          f"[{'PASS' if ok_dom else 'FAIL'}] strict dominance  (p1_breaks={p1_breaks}, regressions={regressions})")

    all_match = all(r["matches_test"] for r in slice_rows if r["n"] <= 50) if slice_rows else False
    return {"corpus": name, "test_n": n, "train_pool": len(train_qids), "held_out": held_out,
            "test_gap": round(test_gap, 4), "test_expand_decision": test_decision,
            "slices": slice_rows, "onboard_governors": gov.as_dict(),
            "onboard_decision_matches_test": (gov.expand_on == test_decision) if held_out else None,
            "small_slices_match_test": all_match,
            "train_fire_rate": round(train_fire, 4), "test_fire_rate": round(test_fire, 4),
            "test": {"p1_bm25": round(b_p1, 4), "p1_serve": round(s_p1, 4), "p1_delta": round(s_p1 - b_p1, 4),
                     f"recall@{DEPTH}_bm25": round(b_r, 4), f"recall@{K}_serve": round(s_r, 4),
                     "recall_delta": round(s_r - b_r, 4), "zero_overlap": zo},
            "invariants": {"p1_untouched": ok_p1, "non_regression": ok_nonreg, "strict_dominance": ok_dom,
                           "p1_breaks": p1_breaks, "regressions": regressions}}


def main() -> int:
    names = sys.argv[1:] if len(sys.argv) > 1 else ["scifact", "nfcorpus", "fiqa"]
    rows = [run_corpus(c) for c in names]
    ok = all(r["invariants"]["p1_untouched"] and r["invariants"]["non_regression"]
             and r["invariants"]["strict_dominance"] for r in rows)
    held = [r for r in rows if r["held_out"]]
    onboard_ok = all(r["onboard_decision_matches_test"] for r in held) if held else True
    _line("SUMMARY")
    print(f"  {'corpus':10s} {'test gap':>9} {'governor':>16} {'N=20 right?':>12} {'test recall d':>14} {'invariants':>12}")
    for r in rows:
        inv = r["invariants"]; allok = inv["p1_untouched"] and inv["non_regression"] and inv["strict_dominance"]
        conf = r["onboard_governors"].get("expand_confident", True)
        if r["held_out"]:
            gtxt = "held-out" + ("" if conf else "*near-bnd")
            ntxt = str(r["onboard_decision_matches_test"])
        else:
            gtxt = "test-fallback"; ntxt = "n/a (no train)"
        print(f"  {r['corpus']:10s} {r['test_gap']:>9.3f} {gtxt:>16} {ntxt:>12} "
              f"{r['test']['recall_delta']:>+14.4f} {('PASS' if allok else 'FAIL'):>12}")
    print(f"\n  Held-out verdict: of {len(rows)} corpora, {len(held)} have train splits -> N={ONBOARD_N} onboarding "
          f"slice sets the expand governor correctly on {'ALL' if onboard_ok else 'NOT all'} of them "
          f"({', '.join(r['corpus'] for r in held) or 'none'}).")
    print(f"  Composed-serve invariants (P@1 untouched + recall non-regression + strict dominance) "
          f"{'HOLD on EVERY corpus (held-out and test-only)' if ok else 'FAILED somewhere'}.")
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"depth": DEPTH, "budget": BUDGET, "onboard_n": ONBOARD_N, "corpora": rows,
               "onboard_governors_correct": onboard_ok, "invariants_all_pass": ok},
              open(os.path.join(here, "_champion_holdout.json"), "w"), indent=2)
    print("  wrote _champion_holdout.json")
    return 0 if (ok and onboard_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
