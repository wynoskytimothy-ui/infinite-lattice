#!/usr/bin/env python3
"""
Concrete Plane gate — side-audit harness for phased symbol retrieval.

  python scripts/concrete_plane_gate.py --phase 0 --dataset scifact
  python scripts/concrete_plane_gate.py --phase 0 --strict --skip-eval

Exit 0 when all checks for the requested phase pass (and --strict metrics hold).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_complex_plane import imaginary_start
from aethos_symbol_entangle import find_morph_pieces
from aethos_symbol_morph import meet_2way_on_line
from eval_beir_symbol import load_brain_and_plane, plane_index_path
from pipeline.bit_02_attractor_key import kappa_branch_fan
from pipeline.bit_12_symbol_plane_index import (
    correlation_meet_keys,
    symbol_word_chain,
    verify_bit12_gate,
)

BASELINE_PATH = _ROOT / "logs" / "gate_baseline.json"
def _probe_pairs(knowledge, plane, n: int = 3) -> list[tuple[str, str]]:
    """Cross-linked vocabulary pairs whose meet keys are on the plane."""
    from pipeline.bit_12_symbol_plane_index import canonical_pair_key

    if plane.pair_keys:
        out: list[tuple[str, str]] = []
        for a, b in knowledge.cross_links:
            if a == b:
                continue
            if knowledge.correlates(a, b) is None:
                continue
            if canonical_pair_key(knowledge, a, b) not in plane.pair_keys:
                continue
            out.append((a, b))
            if len(out) >= n:
                break
        if out:
            return out
    return [("diminished", "lower"), ("hypothesis", "test")]


PHASE_METRICS: dict[int, dict[str, float]] = {
    0: {"ndcg_at_10": 0.48, "route_recall": 0.90, "mean_query_ms": 25.0},
    2: {"route_recall": 0.88},
    3: {"route_recall": 0.88},
    4: {"route_recall": 0.88},
    5: {"route_recall": 0.88},
    6: {"ndcg_at_10": 0.46, "mean_query_ms": 50.0},
    10: {"ndcg_at_10": 0.49, "mean_query_ms": 20.0},
    12: {"ndcg_at_10": 0.50, "recall_at_10": 0.60, "route_recall": 0.90, "mean_query_ms": 35.0},
}


def _load_baseline() -> dict:
    if BASELINE_PATH.is_file():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {}


def _check_g0() -> tuple[bool, str]:
    for n in (1, 3, 7, 100):
        if abs(imaginary_start(n).modulus_squared - 2 * n * n) > 1e-6:
            return False, f"layer-0 fail n={n}"
    return True, "layer-0 |z0|^2=2n^2"


def _check_g1(knowledge, sample: int = 200, *, phase: int = 0) -> tuple[bool, str]:
    vocab = sorted(knowledge.vocab)
    if len(vocab) < 10:
        return False, "vocab too small"
    rng = random.Random(42)
    pick = rng.sample(vocab, min(sample, len(vocab)))
    morph_hits = 0
    for w in pick:
        chain = symbol_word_chain(knowledge, w)
        if not chain:
            continue
        from pipeline.bit_12_symbol_plane_index import symbol_word_chain_query
        qchain = symbol_word_chain_query(knowledge, w)
        if w in knowledge.morph.composites or w in knowledge.morph.subwords:
            morph_hits += 1
        elif len(qchain) <= 2 or qchain != chain:
            morph_hits += 1
    ratio = morph_hits / len(pick)
    floor = 0.50 if phase >= 1 else 0.10
    ok = ratio >= floor
    return ok, f"morph-aware chain ratio {ratio:.2f} (phase {phase} floor {floor})"


def _check_g2() -> tuple[bool, str]:
    if meet_2way_on_line(3, 5) != 8:
        return False, "meet_2way_on_line(3,5)!=8"
    return True, "imag meet 3+5=8"


def _check_g3() -> tuple[bool, str]:
    keys = kappa_branch_fan((3, 5, 7), 10.0)
    if len(keys) != 4:
        return False, f"kappa_branch_fan len={len(keys)}"
    return True, "4-branch kappa fan"


def _check_g4(knowledge, plane) -> tuple[bool, str]:
    ok, failures = verify_bit12_gate(knowledge, plane, _probe_pairs(knowledge, plane))
    if not ok:
        return False, "; ".join(failures[:3])
    return True, "verify_bit12_gate probes"


def _check_g5(knowledge, plane) -> tuple[bool, str]:
    for a, b in _probe_pairs(knowledge, plane):
        lk = knowledge.correlates(a, b)
        if lk is None:
            continue
        meet = correlation_meet_keys(knowledge, a, b, link=lk)
        if not meet:
            return False, f"no meet keys {a}+{b}"
        from pipeline.bit_12_symbol_plane_index import canonical_pair_key

        stored = plane.pair_keys.get(canonical_pair_key(knowledge, a, b))
        if stored is None and lk.strength >= 2.0:
            return False, f"pair_keys missing {a}+{b}"
    return True, "pair_keys indexed for probes"


def _check_g6(knowledge) -> tuple[bool, str]:
    from aethos_symbol_cellular import CellularRole

    leaks = 0
    for lk in knowledge.cross_links.values():
        if knowledge.cellular.role_of(lk.left) == CellularRole.MEMBRANE:
            if knowledge.cellular.role_of(lk.right) == CellularRole.SIGNAL:
                leaks += 1
        if knowledge.cellular.role_of(lk.right) == CellularRole.MEMBRANE:
            if knowledge.cellular.role_of(lk.left) == CellularRole.SIGNAL:
                leaks += 1
    ok = leaks == 0
    return ok, f"membrane-rare leaks={leaks}"


def _check_g7(knowledge) -> tuple[bool, str]:
    from aethos_symbol_morph_pieces import morph_pieces
    from aethos_rare_rank import morph_trigger_pieces
    from aethos_query_oov import morph_subword_pieces

    tokens = ["cellular", "cells", "diminished", "expression", "protein"]
    mismatches = 0
    for tok in tokens:
        unified = set(morph_pieces(knowledge, tok, mode="query"))
        if unified != set(morph_trigger_pieces(knowledge, tok)):
            mismatches += 1
        if unified != set(morph_subword_pieces(knowledge, tok)):
            mismatches += 1
    ok = mismatches == 0
    return ok, f"unified morph_pieces mismatches={mismatches}"


def _run_eval(dataset: str, mode: str, max_queries: int) -> dict:
    out = _ROOT / "logs" / f"gate_eval_p{max_queries}_{mode}.json"
    cmd = [
        sys.executable,
        str(_ROOT / "eval_beir_symbol.py"),
        "--dataset",
        dataset,
        "--split",
        "test",
        "--max-queries",
        str(max_queries),
        "--mode",
        mode,
        "--out",
        str(out),
    ]
    subprocess.run(cmd, cwd=_ROOT, check=True, capture_output=True, text=True)
    return json.loads(out.read_text(encoding="utf-8"))


def _metrics_ok(phase: int, metrics: dict, strict: bool) -> tuple[bool, list[str]]:
    if not strict:
        return True, []
    gates = PHASE_METRICS.get(phase, PHASE_METRICS[0])
    fails: list[str] = []
    for key, floor in gates.items():
        val = metrics.get(key)
        if val is None:
            fails.append(f"missing metric {key}")
            continue
        if key == "mean_query_ms":
            if val > floor:
                fails.append(f"{key}={val:.1f} > {floor}")
        else:
            if val < floor:
                fails.append(f"{key}={val:.3f} < {floor}")
    return len(fails) == 0, fails


def main() -> int:
    ap = argparse.ArgumentParser(description="Concrete Plane phase gate")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--max-queries", type=int, default=30)
    ap.add_argument("--mode", default="kappa", choices=("kappa", "witness", "cascade", "rare"))
    args = ap.parse_args()

    knowledge, plane = load_brain_and_plane(args.dataset)
    checks: dict[str, bool] = {}
    detail: dict[str, str] = {}

    for gid, fn in (
        ("G0", lambda: _check_g0()),
        ("G1", lambda: _check_g1(knowledge, phase=args.phase)),
        ("G2", lambda: _check_g2()),
        ("G3", lambda: _check_g3()),
        ("G4", lambda: _check_g4(knowledge, plane)),
        ("G5", lambda: _check_g5(knowledge, plane)),
        ("G6", lambda: _check_g6(knowledge)),
        ("G7", lambda: _check_g7(knowledge)),
    ):
        ok, msg = fn()
        checks[gid] = ok
        detail[gid] = msg

    metrics: dict = {}
    if not args.skip_eval and args.phase in (0, 6, 10, 12):
        try:
            metrics = _run_eval(args.dataset, args.mode, args.max_queries)
        except subprocess.CalledProcessError as e:
            checks["G10"] = False
            detail["G10"] = f"eval failed: {e.stderr[:200] if e.stderr else e}"
        else:
            checks["G10"] = True
            detail["G10"] = (
                f"nDCG={metrics.get('ndcg_at_10', 0):.3f} "
                f"route={metrics.get('route_recall', 0):.3f} "
                f"ms={metrics.get('mean_query_ms', 0):.1f}"
            )
            m_ok, m_fails = _metrics_ok(args.phase, metrics, args.strict)
            if args.strict and not m_ok:
                checks["G10"] = False
                detail["G10"] += " FAIL: " + "; ".join(m_fails)

    plane_path = plane_index_path(args.dataset)
    report = {
        "phase": args.phase,
        "dataset": args.dataset,
        "passed": all(checks.values()),
        "checks": checks,
        "detail": detail,
        "metrics": metrics,
        "artifacts": {
            "plane_mb": round(plane_path.stat().st_size / 1_048_576, 2) if plane_path.is_file() else None,
            "kappa_buckets": len(plane.by_key),
            "pair_keys": len(plane.pair_keys),
        },
        "formula_refs": ["ONTOLOGY B.2", "BIT1 cell", "BIT2 kappa", "BIT12 meet imag sum"],
        "baseline": _load_baseline(),
    }

    out_path = _ROOT / "logs" / f"concrete_plane_gate_p{args.phase}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Concrete Plane gate — phase {args.phase}")
    for gid, ok in checks.items():
        mark = "OK" if ok else "FAIL"
        print(f"  {gid} [{mark}] {detail.get(gid, '')}")
    if metrics:
        print(f"  metrics: nDCG@10={metrics.get('ndcg_at_10')} route={metrics.get('route_recall')}")
    print(f"  passed={report['passed']}  -> {out_path}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
