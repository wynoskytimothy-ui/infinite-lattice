#!/usr/bin/env python3
"""
Honest per-corpus auto-tuning of BM25 (k1, b) on a held-out DEV split.

    python scripts/auto_tune.py <dataset>

Picks (k1, b) by nDCG on a DEV split (dev.tsv if present, else a slice of train
held out from bridge training), then reports the gain on the untouched TEST
split. NEVER tunes on test. Corpora without a dev split (trec-covid, touché)
can't be honestly tuned this way - we only note the sensitivity.

This is the principled "tune it better" loop: the engine sets its own length-
normalisation per corpus, on held-out data, no test peeking.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex
from scripts.bench_supervised_bridges import find_ds, load, ndcg10

GRID = [(k1, b) for k1 in (1.2, 1.8) for b in (0.3, 0.4, 0.5, 0.6, 0.75, 0.9)]


def load_dev(name):
    root = find_ds(name)
    p = root / "qrels" / "dev.tsv"
    if not p.exists():
        return None
    rel = {}
    r = csv.reader(open(p, encoding="utf-8"), delimiter="\t")
    next(r)
    for qid, cid, sc in r:
        rel.setdefault(qid, {})[cid] = int(sc)
    return rel


def run(name):
    corpus, queries, train_q, test_q = load(name)
    dev_q = load_dev(name)
    src = "dev.tsv"
    if dev_q is None:
        # carve a dev set from train (held out; not used for anything else here)
        tr = sorted(train_q)
        if not tr:
            print(f"{name}: no dev and no train split -> cannot honestly tune (note "
                  f"sensitivity only)")
            return
        cut = max(1, len(tr) // 5)
        dev_q = {q: train_q[q] for q in tr[:cut]}
        src = f"train-slice ({cut} q)"
    dev_ids = [q for q in dev_q if q in queries]
    test_ids = [q for q in test_q if q in queries]

    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    print(f"\n{name}: {len(corpus):,} docs | dev {len(dev_ids)} ({src}) | test {len(test_ids)}")

    def ev(ids, qrels):
        return sum(ndcg10(idx.search(queries[q], 10), qrels[q]) for q in ids) / len(ids)

    idx.k1, idx.b = 1.2, 0.75
    idx.finalize()
    test_def = ev(test_ids, test_q)

    best = (0.0, None)
    for k1, b in GRID:
        idx.k1, idx.b = k1, b
        idx.finalize()
        nd = ev(dev_ids, dev_q)
        if nd > best[0]:
            best = (nd, (k1, b))
    idx.k1, idx.b = best[1]
    idx.finalize()
    test_tuned = ev(test_ids, test_q)

    print(f"  default  k1=1.2 b=0.75:           TEST nDCG {test_def:.4f}")
    print(f"  dev-tuned k1={best[1][0]} b={best[1][1]} (dev nDCG {best[0]:.4f}): "
          f"TEST nDCG {test_tuned:.4f}  ({test_tuned-test_def:+.4f} held-out)")


def main():
    run(sys.argv[1] if len(sys.argv) > 1 else "scifact")


if __name__ == "__main__":
    main()
