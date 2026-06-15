#!/usr/bin/env python3
"""Browse saved symbol knowledge correlations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from aethos_symbol_knowledge import SymbolKnowledgeIndex


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--word", default=None, help="show top neighbors for one word")
    p.add_argument("--pair", nargs=2, metavar=("A", "B"), help="check one pair")
    args = p.parse_args()

    idx = SymbolKnowledgeIndex.load(args.dataset)
    s = idx.summary()
    print(f"Loaded {args.dataset}: {s['n_docs']} docs, {s['total_cross_links']} links\n")

    if args.pair:
        a, b = args.pair
        lk = idx.correlates(a, b)
        if lk:
            print(f"{a} + {b}: kind={lk.kind} strength={lk.strength} via={lk.via}")
        else:
            print(f"{a} + {b}: no link")
        return

    if args.word:
        w = args.word.lower()
        print(f"Neighbors of {w!r}:")
        for lk in idx.neighbors(w)[:20]:
            other = lk.right if lk.left == w else lk.left
            print(f"  {other:18}  {lk.kind:7}  strength={lk.strength:.1f}")
        return

    probes = [
        ("cancer", "cell"), ("protein", "gene"), ("breast", "cancer"),
        ("rna", "virus"), ("clinical", "trial"), ("heart", "cardiac"),
        ("quantum", "energy"), ("hilbert", "space"), ("zero", "dimension"),
        ("tumor", "immunotherapy"), ("mrna", "vaccine"),
    ]
    print("Sample correlation probes:")
    for a, b in probes:
        lk = idx.correlates(a, b)
        tag = f"{lk.kind}={lk.strength:.0f}" if lk else "---"
        print(f"  {a:14} + {b:14}  {tag}")

    cov = sorted(t for t in idx.vocab if "cov" in t or "sars" in t or "corona" in t)
    print(f"\nCov-related tokens ({len(cov)}): {cov[:12]}")


if __name__ == "__main__":
    main()
