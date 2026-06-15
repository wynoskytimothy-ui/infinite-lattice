#!/usr/bin/env python3
"""
Where we stand - footprint / speed / accuracy on scifact + nfcorpus.

Current best stack: append-only multi-view lattice index (word + char-trigram +
prefix, positional) + supervised relevance bridges with pool expansion
(min_pairs=1 for scifact's paraphrase structure, =2 for nfcorpus; lam=0.25,
n_expand=20). All held-out: bridges from train qrels, measured on test queries.

Reports, per corpus:
  FOOTPRINT - vocab, total postings, compact serialized bytes (4B docid + 1B tf
              per posting + bridge table), bytes/doc; in-memory RSS delta if
              psutil is present.
  SPEED     - ingest docs/sec, bridge-learn time, per-query latency (mean / p50 /
              p99) for lexical and for lexical+bridges.
  ACCURACY  - nDCG@10 / Recall@10, lexical baseline vs +bridges (held-out).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges
from scripts.bench_active_learning import best_search

try:
    import psutil
    _PROC = psutil.Process()
except Exception:
    _PROC = None


def rss_mb():
    return _PROC.memory_info().rss / 1e6 if _PROC else None


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, int(p / 100 * len(xs)))
    return xs[i]


def measure(name, min_pairs):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]

    r0 = rss_mb()
    t0 = time.perf_counter()
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    build_s = time.perf_counter() - t0
    N = len(idx.alive)

    # bridges
    tb = time.perf_counter()
    br = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    learn_s = time.perf_counter() - tb
    n_bridge_terms, n_bridge_edges = br.stats()
    r1 = rss_mb()

    # ---- footprint ----
    vocab = len(idx.token_prime)
    total_postings = sum(len(p) for p in idx.postings.values())
    # compact serialized: per posting 4B docid + 1B tf; per bridge edge 4B + 1B
    compact_b = total_postings * 5 + n_bridge_edges * 5
    bytes_doc = compact_b / N

    # ---- query latency ----
    def time_queries(fn):
        lat = []
        for qid in test_ids:
            t = time.perf_counter()
            fn(queries[qid])
            lat.append((time.perf_counter() - t) * 1000)
        return lat

    lat_lex = time_queries(lambda q: idx.search(q, 10))
    lat_br = time_queries(lambda q: best_search(idx, br, q))

    # ---- accuracy (held-out) ----
    def acc(fn):
        nd = rc = 0.0
        for qid in test_ids:
            ranked = fn(queries[qid])
            nd += ndcg10(ranked, test_q[qid])
            rc += recall10(ranked, test_q[qid])
        return nd / len(test_ids), rc / len(test_ids)

    nd0, rc0 = acc(lambda q: idx.search(q, 10))
    nd1, rc1 = acc(lambda q: best_search(idx, br, q))

    return {
        "name": name, "docs": N, "test_q": len(test_ids), "vocab": vocab,
        "postings": total_postings, "compact_b": compact_b, "bytes_doc": bytes_doc,
        "bridge_terms": n_bridge_terms, "bridge_edges": n_bridge_edges,
        "min_pairs": min_pairs,
        "build_s": build_s, "ingest_dps": N / build_s, "learn_s": learn_s,
        "lat_lex": lat_lex, "lat_br": lat_br,
        "rss_delta": (r1 - r0) if (r0 and r1) else None,
        "nd0": nd0, "rc0": rc0, "nd1": nd1, "rc1": rc1,
    }


def report(m):
    print(f"\n{'='*60}\n{m['name'].upper()}  ({m['docs']:,} docs, "
          f"{m['test_q']} test queries, min_pairs={m['min_pairs']})")
    print(f"\n  FOOTPRINT")
    print(f"    vocab (multi-view primes) : {m['vocab']:,}")
    print(f"    total postings            : {m['postings']:,}")
    print(f"    bridge edges              : {m['bridge_edges']:,} "
          f"over {m['bridge_terms']:,} terms")
    print(f"    compact index             : {m['compact_b']/1e6:.1f} MB "
          f"({m['bytes_doc']:.0f} B/doc)")
    if m["rss_delta"]:
        print(f"    process RSS delta         : {m['rss_delta']:.0f} MB "
              f"(in-memory, Python overhead incl.)")
    print(f"\n  SPEED")
    print(f"    ingest                    : {m['build_s']:.1f}s "
          f"({m['ingest_dps']:.0f} docs/sec)")
    print(f"    bridge learn              : {m['learn_s']:.1f}s")
    ll, lb = m["lat_lex"], m["lat_br"]
    print(f"    query lexical             : {sum(ll)/len(ll):.1f} ms/q  "
          f"(p50 {pct(ll,50):.1f}, p99 {pct(ll,99):.1f})")
    print(f"    query +bridges            : {sum(lb)/len(lb):.1f} ms/q  "
          f"(p50 {pct(lb,50):.1f}, p99 {pct(lb,99):.1f})")
    print(f"\n  ACCURACY (held-out test)")
    print(f"    lexical baseline          : nDCG {m['nd0']:.4f}  Recall {m['rc0']:.4f}")
    print(f"    + supervised bridges      : nDCG {m['nd1']:.4f}  Recall {m['rc1']:.4f}  "
          f"({m['nd1']-m['nd0']:+.4f} / {m['rc1']-m['rc0']:+.4f})")


def main():
    print("FOOTPRINT / SPEED / ACCURACY - current best stack, held-out")
    ms = [measure("scifact", 1), measure("nfcorpus", 2)]
    for m in ms:
        report(m)
    print(f"\n{'='*60}\nSUMMARY")
    print(f"  {'corpus':<9} {'docs':>7} {'B/doc':>7} {'ingest':>9} "
          f"{'q ms':>7} {'nDCG':>7} {'Recall':>7}")
    for m in ms:
        print(f"  {m['name']:<9} {m['docs']:>7,} {m['bytes_doc']:>7.0f} "
              f"{m['ingest_dps']:>6.0f}/s {sum(m['lat_br'])/len(m['lat_br']):>6.1f} "
              f"{m['nd1']:>7.4f} {m['rc1']:>7.4f}")


if __name__ == "__main__":
    main()
