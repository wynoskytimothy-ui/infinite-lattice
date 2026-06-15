#!/usr/bin/env python3
"""
Pretrain brain memory test — gold doc before SciFact, lazy compound learn.

  python scripts/test_pretrain_brain_memory.py
  python scripts/test_pretrain_brain_memory.py --full

Doc: derivations/lazy_correlation_brain.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import (
    PRETRAIN_QUANTUM_GOLD,
    SymbolKnowledgeIndex,
    knowledge_path,
)
from pipeline.bit_12_symbol_plane_index import (
    build_symbol_plane_index,
    route_symbol_plane_candidates,
)

GOLD_ID = "gold_quantum_biometrics"
PROBE_PAIRS = [
    ("quantum", "dimension"),
    ("quantum", "zero"),
    ("zero", "dimension"),
    ("inductive", "biometrics"),
    ("quantum", "inductive"),
]
QUERY = ["quantum", "zero", "dimension", "inductive", "biometrics"]


def run_test(*, full: bool) -> dict[str, object]:
    report: dict[str, object] = {"full_scifact": full, "steps": [], "passed": False}

    def step(name: str, ok: bool, detail: object = None) -> None:
        report["steps"].append({"name": name, "ok": ok, "detail": detail})

    # --- Step 1: SciFact baseline (no pretrain) ---
    scifact_path = knowledge_path("scifact")
    if not scifact_path.is_file():
        print("Building SciFact knowledge (subset) ...")
        brain = SymbolKnowledgeIndex.build_from_beir(
            "scifact", max_docs=None if full else 500, download=True,
        )
        brain.save()
    else:
        print(f"Loading {scifact_path} ...")
        t0 = time.perf_counter()
        brain = SymbolKnowledgeIndex.load("scifact")
        print(f"  loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    baseline = {f"{a}+{b}": brain.remembers(a, b) for a, b in PROBE_PAIRS}
    step("scifact_baseline_loaded", True, baseline)

    missing_before = [p for p, ok in baseline.items() if not ok]
    step("scifact_missing_pretrain_pairs", len(missing_before) > 0, missing_before)

    links_before = brain.summary()["total_cross_links"]

    # --- Step 2: Compound learn gold doc (lazy deepen) ---
    t0 = time.perf_counter()
    learn_report = brain.compound_learn(PRETRAIN_QUANTUM_GOLD)
    learn_ms = (time.perf_counter() - t0) * 1000.0
    learn_report["learn_ms"] = round(learn_ms, 1)
    step("compound_learn", learn_report["links_added"] > 0, learn_report)

    after = {f"{a}+{b}": brain.remembers(a, b) for a, b in PROBE_PAIRS}
    step("pretrain_pairs_remembered", all(after.values()), after)

    gold_check = brain.query_gold_links(QUERY, GOLD_ID)
    step("query_gold_links", gold_check["all_pairs_linked"], gold_check)

    links_after = brain.summary()["total_cross_links"]
    step(
        "links_grew_not_shrank",
        links_after >= links_before,
        {"before": links_before, "after": links_after},
    )

    # --- Step 3: Save / reload memory ---
    out = knowledge_path("scifact_compound")
    brain.save(out)
    reloaded = SymbolKnowledgeIndex.load("scifact_compound", path=out)
    reload_ok = all(reloaded.remembers(a, b) for a, b in PROBE_PAIRS)
    step("save_reload_memory", reload_ok, {"path": str(out)})

    # --- Step 4: Plane router finds gold doc ---
    t0 = time.perf_counter()
    plane = build_symbol_plane_index(reloaded, pair_key_limit=50_000)
    route = route_symbol_plane_candidates(reloaded, plane, QUERY)
    route_ms = (time.perf_counter() - t0) * 1000.0
    gold_in_candidates = GOLD_ID in route.doc_ids
    step(
        "plane_router_finds_gold",
        gold_in_candidates,
        {
            "route_ms": round(route_ms, 1),
            "n_candidates": len(route.doc_ids),
            "gold_rank": route.doc_ids.index(GOLD_ID) if gold_in_candidates else -1,
        },
    )

    oks = [s["ok"] for s in report["steps"]]
    report["passed"] = all(oks)
    report["missing_before_pretrain"] = missing_before
    report["remembered_after"] = after
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="use full scifact.pkl")
    p.add_argument("--out", default=None, help="JSON report path")
    args = p.parse_args()

    print("=" * 70)
    print("PRETRAIN BRAIN MEMORY TEST")
    print("=" * 70)

    report = run_test(full=args.full)

    for s in report["steps"]:
        tag = "PASS" if s["ok"] else "FAIL"
        print(f"  [{tag}] {s['name']}")
        if s["detail"] and s["name"] in (
            "compound_learn", "query_gold_links", "plane_router_finds_gold",
        ):
            print(f"         {s['detail']}")

    print()
    overall = "PASS" if report["passed"] else "FAIL"
    print(f"  OVERALL: {overall}")
    print("=" * 70)

    out = Path(args.out or _ROOT / "logs" / "pretrain_brain_memory.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  report: {out}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
