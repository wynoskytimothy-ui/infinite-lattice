#!/usr/bin/env python3
"""Verify the supervised-bridge semantic-correlation lift on held-out SciFact,
using the CURRENT aethos_bridges API (bridge_search). Bridges learned from
TRAIN qrels only; every number on held-out TEST queries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex
from aethos_bridges import RelevanceBridges, bridge_search
from scripts.bench_supervised_bridges import load, ndcg10, recall10

ds = sys.argv[1] if len(sys.argv) > 1 else "scifact"
corpus, queries, train_q, test_q = load(ds)
test_ids = [q for q in test_q if q in queries]

idx = AppendOnlyLatticeIndex()
for d, t in corpus.items():
    idx.add(d, t)
N = len(idx.alive)
br = RelevanceBridges(idx, N, min_pairs=2).learn(queries, train_q, corpus)


def lex_search(q, k=10):
    s = idx._score(q)
    return sorted(s, key=s.get, reverse=True)[:k]


def ev(fn):
    nd = rc = 0.0
    for qid in test_ids:
        r = fn(queries[qid])
        nd += ndcg10(r, test_q[qid])
        rc += recall10(r, test_q[qid])
    n = len(test_ids)
    return nd / n, rc / n


nd0, rc0 = ev(lex_search)
nd1, rc1 = ev(lambda q: bridge_search(idx, br, q))
nt, nb = br.stats()
print(f"\n{ds}: {len(corpus)} docs | train {len(train_q)} q | test {len(test_ids)} q")
print(f"  learned {nb} bridges over {nt} query-terms (min_pairs=2)")
print(f"  baseline lexical:   nDCG@10 {nd0:.4f}   Recall@10 {rc0:.4f}")
print(f"  + supervised bridges: nDCG@10 {nd1:.4f}   Recall@10 {rc1:.4f}   "
      f"({nd1-nd0:+.4f} nDCG, {rc1-rc0:+.4f} recall)")
print(f"\n  the semantic-correlation lift = the bridge delta above "
      f"(learned query->doc term links from qrels, no neural net).")
