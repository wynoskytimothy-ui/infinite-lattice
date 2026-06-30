#!/usr/bin/env python3
"""GLASS-BOX recall analysis (Q20, the master key): of the gold STILL not reachable by the best expansion union,
classify WHY — does it connect to the query at hop-1/2/3 of the co-occurrence graph, or is it DISCONNECTED
(co-occurrence fundamentally can't reach it)? This tells us which lever (deeper hops vs neural) actually raises
recall, instead of guessing. Shows the missing-bridge term for examples."""
import os, sys
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_edge_rag import EdgeRAG
from scripts.bench_supervised_bridges import load
from _fast_tok import words
from _o1_drift import build_drift, drift_bag, score_bag


def reachset(seeds, drift, hops):
    """terms reachable from seeds within `hops` of the co-occurrence graph."""
    frontier = set(seeds); seen = set(seeds)
    for _ in range(hops):
        nxt = set()
        for w in frontier:
            for cw, _ in drift.get(w, ()): nxt.add(cw)
        frontier = nxt - seen; seen |= nxt
    return seen


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "nfcorpus"
    corpus, queries, train_q, test_q = load(name)
    doc_ids = list(corpus.keys()); d2i = {d: i for i, d in enumerate(doc_ids)}
    ids = [q for q in test_q if q in queries and any(test_q[q].get(d, 0) > 0 and d in d2i for d in test_q[q])]
    eng = EdgeRAG().build(corpus)
    allw = {w for d in corpus.values() for w in words(d)}
    drift, _, _ = build_drift(corpus, allw)               # graph for reachability (all terms)
    docw = {d: set(words(corpus[d])) for d in corpus}

    # best expansion pool the engine reaches now (lexical + 1-hop drift), top-100
    cat = Counter(); examples = []
    n_unreach = 0
    for qid in ids:
        q = queries[qid]; qw = set(words(q))
        gold = [d for d in test_q[qid] if test_q[qid].get(d, 0) > 0 and d in corpus]
        sc = score_bag(eng, drift_bag(q, drift, alpha=0.4))      # current best query-side expansion
        top = set(np.argsort(sc)[::-1][:100].tolist())
        h1 = reachset(qw, drift, 1); h2 = reachset(qw, drift, 2); h3 = reachset(qw, drift, 3)
        for g in gold:
            if d2i[g] in top: continue                          # already reached
            n_unreach += 1
            gt = docw[g]
            if qw & gt: cat["literal (drowned)"] += 1           # shares a literal word but ranked >100
            elif h1 & gt: cat["hop-1 reachable"] += 1
            elif h2 & gt: cat["hop-2 reachable"] += 1
            elif h3 & gt: cat["hop-3 reachable"] += 1
            else:
                cat["DISCONNECTED (no co-occ path)"] += 1
                if len(examples) < 4:                           # glass-box: show the gap
                    examples.append((q[:46], sorted(gt & allw - qw, key=lambda w: -len(w))[:5]))

    print("=" * 84); print(f"GLASS-BOX RECALL — {name}: {len(ids)} q | {n_unreach} unreached gold (after 1-hop drift)"); print("=" * 84)
    tot = sum(cat.values())
    print("  WHY each unreached gold is missed (the lever each implies):")
    levers = {"literal (drowned)": "rerank/pool-depth", "hop-1 reachable": "stronger 1-hop weight",
              "hop-2 reachable": "2-hop propagation (the prop lever)", "hop-3 reachable": "3-hop / iterative expand",
              "DISCONNECTED (no co-occ path)": "NEURAL only (co-occurrence cannot reach)"}
    for k in ["literal (drowned)", "hop-1 reachable", "hop-2 reachable", "hop-3 reachable", "DISCONNECTED (no co-occ path)"]:
        c = cat.get(k, 0)
        print(f"    {k:<32}{c:>6} ({100*c/max(1,tot):>4.0f}%)  -> {levers[k]}")
    print(f"\n  glass-box examples (DISCONNECTED gold — its distinctive terms vs the query):")
    for q, gt in examples: print(f"    q='{q}...'  gold terms not in query: {gt}")
    print(f"\n  READ: the % at hop-2/hop-3 = recall recoverable by DEEPER co-occurrence expansion (free); the")
    print(f"  DISCONNECTED % = the hard ceiling for co-occurrence -> only the neural model reaches it.")


if __name__ == "__main__":
    main()
