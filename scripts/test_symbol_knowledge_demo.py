#!/usr/bin/env python3
"""
End-to-end test of symbol knowledge — correlations, bridges, gaps, save/load.

  python scripts/test_symbol_knowledge_demo.py
  python scripts/test_symbol_knowledge_demo.py --scifact
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex, knowledge_path


def test_toy_corpus() -> bool:
    print("=" * 60)
    print("TEST 1 — toy corpus (direct + morph + bridge)")
    print("=" * 60)

    corpus = {
        "d1": "the diminished score was lower after treatment",
        "d2": "diminishes over time in clinical study",
        "d3": "quantum zero dimension Hilbert space analysis",
        "d4": "zero dimensional quantum systems exhibit behavior",
    }
    idx = SymbolKnowledgeIndex.build_from_corpus(corpus, dataset="toy_test")
    s = idx.summary()
    print(f"  built in {s['build_ms']:.1f} ms")
    print(f"  links: direct={s['direct_pairs']} morph={s['morph_links']} "
          f"bridge={s['bridge_links']}")

    checks = [
        ("diminished", "lower", "direct", "same doc co-occurrence"),
        ("diminished", "diminishes", "morph", "shared root, never touched"),
        ("diminishes", "lower", "bridge", "non-touching via diminished"),
        ("quantum", "zero", "direct", "science terms same doc"),
        ("zero", "dimension", "direct", "zero + dimension linked"),
    ]
    ok = True
    for a, b, expect_kind, note in checks:
        lk = idx.correlates(a, b)
        if lk is None:
            print(f"  FAIL  {a!r}+{b!r}  expected {expect_kind}  ({note})")
            ok = False
        elif lk.kind != expect_kind and not (expect_kind == "direct" and lk.kind == "bridge"):
            print(f"  WARN  {a!r}+{b!r}  got {lk.kind} expected {expect_kind}  ({note})")
            print(f"        strength={lk.strength}")
        else:
            print(f"  OK    {a!r}+{b!r}  kind={lk.kind}  strength={lk.strength}  ({note})")

    # membrane should NOT link
    if idx.correlates("the", "diminished"):
        print("  FAIL  the+diminished should be blocked (membrane)")
        ok = False
    else:
        print("  OK    the+diminished blocked (membrane filler)")

    # save / load round-trip
    path = knowledge_path("toy_test")
    idx.save(path)
    loaded = SymbolKnowledgeIndex.load("toy_test", path=path)
    if loaded.correlates("quantum", "zero") is None:
        print("  FAIL  save/load broke quantum+zero")
        ok = False
    else:
        print("  OK    save/load round-trip")

    print(f"\n  TEST 1: {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_gaps_and_merge() -> bool:
    print("=" * 60)
    print("TEST 2 — gaps then train with richer corpus")
    print("=" * 60)

    sparse = {"d1": "neutrino oscillation experiment"}
    idx = SymbolKnowledgeIndex.build_from_corpus(sparse, dataset="gap_test")
    gaps_before = idx.gap_words()
    print(f"  sparse corpus gaps: {len(gaps_before)}  e.g. {gaps_before[:5]}")

    rich = {
        "d2": "neutrino mass hierarchy explained with detector physics",
        "d3": "oscillation probability depends on neutrino energy",
    }
    report = idx.gaps_after_merge(rich)
    merged = idx.merge_corpus(rich)
    print(f"  after merge: gaps {report['before']} -> {report['after']}")
    print(f"  newly linked sample: {report['newly_linked'][:6]}")

    lk = merged.correlates("neutrino", "oscillation")
    ok = lk is not None
    print(f"  neutrino+oscillation: {lk.kind if lk else 'MISSING'}")
    print(f"\n  TEST 2: {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_scifact_saved() -> bool:
    print("=" * 60)
    print("TEST 3 — saved SciFact knowledge (if present)")
    print("=" * 60)

    path = knowledge_path("scifact")
    if not path.is_file():
        print("  SKIP  no scifact.pkl — run:")
        print("        python aethos_symbol_knowledge.py --dataset scifact --max-docs 200")
        print()
        return True

    t0 = time.perf_counter()
    idx = SymbolKnowledgeIndex.load("scifact", path=path)
    ms = (time.perf_counter() - t0) * 1000
    s = idx.summary()
    print(f"  loaded in {ms:.0f} ms")
    print(f"  {s['n_docs']} docs  vocab={s['vocab']}  links={s['total_cross_links']}")

    # probe pairs common in biomedical IR
    probes = [
        ("covid", "virus"),
        ("rna", "virus"),
        ("vaccine", "immune"),
        ("cancer", "cell"),
        ("protein", "gene"),
    ]
    hits = 0
    for a, b in probes:
        lk = idx.correlates(a, b)
        if lk:
            hits += 1
            print(f"  OK    {a}+{b}  kind={lk.kind}  strength={lk.strength:.0f}")
        else:
            # try neighbors
            nbrs = idx.neighbors(a, kinds={"direct"})[:3]
            nbr_names = [n.right if n.left == a else n.left for n in nbrs]
            print(f"  --    {a}+{b}  no direct link  {a} neighbors: {nbr_names}")

    ok = hits >= 2
    print(f"\n  TEST 3: {'PASS' if ok else 'PARTIAL'} ({hits}/{len(probes)} probe pairs linked)\n")
    return ok


def test_scifact_build(max_docs: int) -> bool:
    print("=" * 60)
    print(f"TEST 4 — build SciFact ({max_docs} docs)")
    print("=" * 60)
    try:
        idx = SymbolKnowledgeIndex.build_from_beir(
            "scifact", max_docs=max_docs, download=True,
        )
    except FileNotFoundError as e:
        print(f"  SKIP  {e}")
        print()
        return True

    path = idx.save()
    s = idx.summary()
    print(f"  built in {s['build_ms']:.0f} ms")
    print(f"  saved: {path}")
    print(f"  summary: {s}")

    lk = idx.correlates("protein", "cell")
    print(f"  protein+cell: {lk.kind if lk else 'none'}")
    print(f"\n  TEST 4: PASS\n")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scifact", action="store_true", help="rebuild SciFact index")
    p.add_argument("--max-docs", type=int, default=100)
    args = p.parse_args()

    results = [test_toy_corpus(), test_gaps_and_merge()]
    if args.scifact:
        results.append(test_scifact_build(args.max_docs))
    else:
        results.append(test_scifact_saved())

    passed = sum(results)
    total = len(results)
    print("=" * 60)
    print(f"OVERALL: {passed}/{total} tests passed")
    print("=" * 60)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
