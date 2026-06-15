#!/usr/bin/env python3
"""
Audit symbol knowledge correlations — sanity checks + sample report.

  python scripts/audit_symbol_knowledge.py
  python scripts/audit_symbol_knowledge.py --dataset scifact
  python scripts/audit_symbol_knowledge.py --dataset scifact --out logs/scifact_correlation_audit.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aethos_symbol_cellular import _DEFAULT_MEMBRANE
from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path

# Known SciFact / biomedical pairs we expect to correlate in full corpus
SCIFACT_PROBES: list[tuple[str, str, str]] = [
    ("covid", "virus", "pandemic topic"),
    ("rna", "virus", "molecular biology"),
    ("vaccine", "immune", "immunology"),
    ("cancer", "cell", "oncology"),
    ("protein", "gene", "genetics"),
    ("antibody", "infection", "immunity"),
    ("diabetes", "insulin", "metabolism"),
    ("neuron", "brain", "neuroscience"),
    ("dna", "mutation", "genetics"),
    ("clinical", "trial", "medicine"),
    ("heart", "cardiac", "cardiology"),
    ("bacteria", "antibiotic", "microbiology"),
]

# Pairs that should NOT be linked (membrane or nonsense)
NEGATIVE_PROBES: list[tuple[str, str, str]] = [
    ("the", "covid", "membrane filler"),
    ("and", "protein", "membrane filler"),
    ("was", "cancer", "membrane filler"),
    ("ed", "gene", "suffix membrane"),
]


def _degree_map(idx: SymbolKnowledgeIndex) -> dict[str, int]:
    deg: dict[str, int] = defaultdict(int)
    for lk in idx.cross_links.values():
        deg[lk.left] += 1
        deg[lk.right] += 1
    return dict(deg)


def audit_membrane_leaks(idx: SymbolKnowledgeIndex) -> dict[str, object]:
    """Membrane tokens must never appear in cross_links."""
    leaks: list[dict[str, str]] = []
    for key, lk in idx.cross_links.items():
        for w in key:
            if w in _DEFAULT_MEMBRANE:
                leaks.append({"pair": list(key), "kind": lk.kind})
    return {
        "passed": len(leaks) == 0,
        "leak_count": len(leaks),
        "samples": leaks[:20],
    }


def audit_bridge_validity(idx: SymbolKnowledgeIndex) -> dict[str, object]:
    """Every bridge link must have a via word and a supporting direct edge."""
    invalid: list[dict[str, object]] = []
    direct_keys = {k for k, lk in idx.cross_links.items() if lk.kind == "direct"}
    for key, lk in idx.cross_links.items():
        if lk.kind != "bridge":
            continue
        if not lk.via:
            invalid.append({"pair": list(key), "reason": "missing via"})
            continue
        via = lk.via.lower()
        a, b = key
        # via should bridge: (via,b) or (a,via) direct, and sibling relationship
        supports = (
            tuple(sorted((via, b))) in direct_keys
            or tuple(sorted((a, via))) in direct_keys
        )
        if not supports:
            invalid.append({"pair": list(key), "via": via, "reason": "no direct support"})
    return {
        "passed": len(invalid) == 0,
        "bridge_count": sum(1 for lk in idx.cross_links.values() if lk.kind == "bridge"),
        "invalid_count": len(invalid),
        "samples": invalid[:15],
    }


def audit_kind_counts(idx: SymbolKnowledgeIndex) -> dict[str, int]:
    c: Counter[str] = Counter()
    for lk in idx.cross_links.values():
        c[lk.kind] += 1
    return dict(c)


def audit_strength_distribution(idx: SymbolKnowledgeIndex) -> dict[str, object]:
    by_kind: dict[str, list[float]] = defaultdict(list)
    for lk in idx.cross_links.values():
        by_kind[lk.kind].append(lk.strength)
    out: dict[str, object] = {}
    for kind, vals in by_kind.items():
        vals.sort()
        n = len(vals)
        out[kind] = {
            "count": n,
            "min": vals[0] if n else 0,
            "median": vals[n // 2] if n else 0,
            "max": vals[-1] if n else 0,
            "p95": vals[int(n * 0.95)] if n else 0,
        }
    return out


def audit_top_pairs(
    idx: SymbolKnowledgeIndex,
    *,
    kind: str | None = None,
    limit: int = 25,
) -> list[dict[str, object]]:
    rows: list[tuple[float, CrossLinkLike]] = []
    for lk in idx.cross_links.values():
        if kind and lk.kind != kind:
            continue
        rows.append((lk.strength, lk))
    rows.sort(key=lambda x: -x[0])
    out: list[dict[str, object]] = []
    for strength, lk in rows[:limit]:
        out.append({
            "left": lk.left,
            "right": lk.right,
            "kind": lk.kind,
            "strength": strength,
            "via": lk.via,
            "opposite": lk.opposite,
        })
    return out


# type alias for cross link in audit
CrossLinkLike = object


def audit_probes(idx: SymbolKnowledgeIndex) -> dict[str, object]:
    positive: list[dict[str, object]] = []
    hits = 0
    for a, b, topic in SCIFACT_PROBES:
        lk = idx.correlates(a, b)
        row = {"a": a, "b": b, "topic": topic, "linked": lk is not None}
        if lk:
            hits += 1
            row["kind"] = lk.kind
            row["strength"] = lk.strength
            if lk.via:
                row["via"] = lk.via
        else:
            na = idx.neighbors(a, kinds={"direct"})[:4]
            row["a_neighbors"] = [
                n.right if n.left == a else n.left for n in na
            ]
        positive.append(row)

    negative: list[dict[str, object]] = []
    neg_ok = 0
    for a, b, reason in NEGATIVE_PROBES:
        lk = idx.correlates(a, b)
        ok = lk is None
        if ok:
            neg_ok += 1
        negative.append({
            "a": a, "b": b, "reason": reason,
            "correctly_blocked": ok,
            "leaked": None if ok else {"kind": lk.kind, "strength": lk.strength},
        })

    return {
        "positive_hits": hits,
        "positive_total": len(SCIFACT_PROBES),
        "positive_rate": round(hits / max(len(SCIFACT_PROBES), 1), 3),
        "positive": positive,
        "negative_blocked": neg_ok,
        "negative_total": len(NEGATIVE_PROBES),
        "negative": negative,
    }


def audit_hub_words(idx: SymbolKnowledgeIndex, limit: int = 20) -> list[dict[str, object]]:
    deg = _degree_map(idx)
    top = sorted(deg.items(), key=lambda x: -x[1])[:limit]
    return [{"word": w, "degree": d} for w, d in top]


def audit_gaps(idx: SymbolKnowledgeIndex, limit: int = 30) -> dict[str, object]:
    gaps = idx.gap_words()
    return {
        "count": len(gaps),
        "samples": gaps[:limit],
    }


def audit_cooccur_vs_stored(idx: SymbolKnowledgeIndex, sample: int = 50) -> dict[str, object]:
    """Verify random direct links match raw doc co-occurrence counts."""
    direct = [(k, lk) for k, lk in idx.cross_links.items() if lk.kind == "direct"]
    if not direct:
        return {"passed": True, "checked": 0, "mismatches": []}
    random.seed(42)
    picks = random.sample(direct, min(sample, len(direct)))
    mismatches: list[dict[str, object]] = []
    cooccur = idx._cooccur_pairs
    for key, lk in picks:
        expected = cooccur.get(key)
        if expected is None:
            mismatches.append({
                "pair": list(key),
                "stored": lk.strength,
                "cooccur": expected,
                "reason": "missing from cooccur index",
            })
        elif abs(float(expected) - lk.strength) > 0.01:
            mismatches.append({
                "pair": list(key),
                "stored": lk.strength,
                "cooccur": expected,
                "reason": "strength mismatch",
            })
    return {
        "passed": len(mismatches) == 0,
        "checked": len(picks),
        "mismatches": mismatches[:10],
    }


def audit_morph_samples(idx: SymbolKnowledgeIndex, limit: int = 15) -> list[dict[str, object]]:
    morph = [(k, lk) for k, lk in idx.cross_links.items() if lk.kind == "morph"]
    morph.sort(key=lambda x: -x[1].strength)
    out: list[dict[str, object]] = []
    for key, lk in morph[:limit]:
        out.append({
            "pair": list(key),
            "strength": lk.strength,
            "shared_root": _shared_root(idx, key[0], key[1]),
        })
    return out


def _shared_root(idx: SymbolKnowledgeIndex, a: str, b: str) -> str | None:
    ca = idx.morph.composites.get(a)
    cb = idx.morph.composites.get(b)
    if ca and cb and ca.correlation.root_prime == cb.correlation.root_prime:
        return ca.correlation.root_text
    return None


def run_audit(idx: SymbolKnowledgeIndex) -> dict[str, object]:
    t0 = time.perf_counter()
    report: dict[str, object] = {
        "dataset": idx.dataset,
        "summary": idx.summary(),
        "kind_counts": audit_kind_counts(idx),
        "strength_distribution": audit_strength_distribution(idx),
        "membrane_leaks": audit_membrane_leaks(idx),
        "bridge_validity": audit_bridge_validity(idx),
        "cooccur_consistency": audit_cooccur_vs_stored(idx),
        "semantic_probes": audit_probes(idx),
        "hub_words": audit_hub_words(idx),
        "gaps": audit_gaps(idx),
        "top_direct_pairs": audit_top_pairs(idx, kind="direct", limit=30),
        "top_bridge_pairs": audit_top_pairs(idx, kind="bridge", limit=20),
        "morph_samples": audit_morph_samples(idx),
    }
    checks = [
        report["membrane_leaks"]["passed"],
        report["bridge_validity"]["passed"],
        report["cooccur_consistency"]["passed"],
        report["semantic_probes"]["negative_blocked"] == report["semantic_probes"]["negative_total"],
        report["semantic_probes"]["positive_hits"] >= report["semantic_probes"]["positive_total"] * 0.5,
    ]
    report["overall_passed"] = all(checks)
    report["checks"] = {
        "no_membrane_leaks": report["membrane_leaks"]["passed"],
        "bridges_valid": report["bridge_validity"]["passed"],
        "cooccur_consistent": report["cooccur_consistency"]["passed"],
        "negatives_blocked": report["semantic_probes"]["negative_blocked"] == report["semantic_probes"]["negative_total"],
        "semantic_hit_rate_ok": report["semantic_probes"]["positive_hits"] >= report["semantic_probes"]["positive_total"] * 0.5,
    }
    report["audit_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report


def print_report(report: dict[str, object]) -> None:
    s = report["summary"]
    print("=" * 70)
    print(f"CORRELATION AUDIT — {report['dataset']}")
    print("=" * 70)
    print(f"  docs={s['n_docs']}  vocab={s['vocab']}  total_links={s['total_cross_links']}")
    print(f"  direct={s['direct_pairs']}  morph={s['morph_links']}  bridge={s['bridge_links']}")
    print(f"  gap_signal_words={s['gap_signal_words']}  build_ms={s['build_ms']}")
    print()

    checks = report["checks"]
    for name, ok in checks.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}")

    print("\n  strength distribution:")
    for kind, dist in report["strength_distribution"].items():
        print(f"    {kind}: n={dist['count']}  median={dist['median']}  max={dist['max']}  p95={dist['p95']}")

    probes = report["semantic_probes"]
    print(f"\n  semantic probes: {probes['positive_hits']}/{probes['positive_total']} linked")
    for row in probes["positive"]:
        if row["linked"]:
            print(f"    OK  {row['a']}+{row['b']}  {row['kind']}={row['strength']:.0f}  ({row['topic']})")
        else:
            nbrs = row.get("a_neighbors", [])
            print(f"    --  {row['a']}+{row['b']}  missing  {row['a']} neighbors={nbrs}")

    print("\n  top hub words (most correlations):")
    for row in report["hub_words"][:12]:
        print(f"    {row['word']!r:18} degree={row['degree']}")

    print("\n  top direct pairs (by co-doc count):")
    for row in report["top_direct_pairs"][:15]:
        print(f"    {row['left']!r}+{row['right']!r}  strength={row['strength']:.0f}")

    if report["morph_samples"]:
        print("\n  morph family samples:")
        for row in report["morph_samples"][:8]:
            print(f"    {row['pair']}  root={row['shared_root']}")

    if report["top_bridge_pairs"]:
        print("\n  bridge samples (non-touching):")
        for row in report["top_bridge_pairs"][:8]:
            print(f"    {row['left']!r}+{row['right']!r}  via={row['via']!r}")

    gaps = report["gaps"]
    print(f"\n  unlinked signal words: {gaps['count']}")
    if gaps["samples"]:
        print(f"    sample: {gaps['samples'][:15]}")

    print()
    overall = "PASS" if report["overall_passed"] else "FAIL"
    print(f"  OVERALL AUDIT: {overall}  ({report['audit_ms']} ms)")
    print("=" * 70)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--path", default=None, help="override .pkl path")
    p.add_argument("--out", default=None, help="write JSON report path")
    p.add_argument("--rebuild", action="store_true", help="rebuild from BEIR before audit")
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    if args.rebuild:
        print(f"Building {args.dataset} ...", flush=True)
        idx = SymbolKnowledgeIndex.build_from_beir(
            args.dataset, max_docs=args.max_docs, download=True,
        )
        path = idx.save()
        print(f"Saved: {path}", flush=True)
    else:
        src = Path(args.path) if args.path else knowledge_path(args.dataset)
        if not src.is_file():
            print(f"Missing {src} — run with --rebuild", flush=True)
            return 1
        print(f"Loading {src} ...", flush=True)
        idx = SymbolKnowledgeIndex.load(args.dataset, path=src)

    report = run_audit(idx)
    print_report(report)

    out = args.out or (_ROOT / "logs" / f"{args.dataset}_correlation_audit.json")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  JSON report: {out}")
    return 0 if report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
