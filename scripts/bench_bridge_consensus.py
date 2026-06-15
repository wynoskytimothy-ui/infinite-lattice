#!/usr/bin/env python3
"""
Bench - BRIDGE CONSENSUS (a meet): trust a bridge-reached doc by agreement.

The diagnosis (diagnose_min_pairs.py): min_pairs=1 forms the rich rare-concept
bridges (biomaterials->nanotechnologies) and wins on scifact (paraphrase
structure) but adds noise on nfcorpus. The fix is not a global gate but a MEET:
a document pulled into the pool purely by bridges (no lexical anchor) is trusted
only when >= min_agree DISTINCT query terms' bridges converge on it. Precision
through agreement - a single noisy bridge can't inject a doc; multiple
independent query signals intersecting on the same doc can.

This should (a) recover the "0-dimensional biomaterials" query (all 4 query terms
bridge to its gold doc) and (b) avoid the nfcorpus noise penalty (random single
bridges rarely converge) - lifting BOTH corpora instead of trading them off.

Configs (held-out test, bridges from train only):
  A  min2, no consensus      (current best: best_search)
  B  min1, no consensus      (recall, noisy)
  C  min1, consensus gate    (>=2 query terms agree, no weighting)
  D  min1, consensus gate + agreement weighting (bonus * frac of q terms agreeing)
  E  min2, consensus gate + weighting (consensus on top of the safe gate)
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10, RelevanceBridges
from scripts.bench_active_learning import best_search

LAM, N_EXPAND = 0.25, 20


def consensus_rank(idx, br, q, lam=LAM, n_expand=N_EXPAND, min_agree=2, weight=True):
    """Full ranked pool with a consensus gate on unanchored expansion docs."""
    lex = idx._score(q)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    cset = set(cand)
    contrib = defaultdict(float)        # doc -> summed bridge weight
    agree = defaultdict(set)            # doc -> {query terms whose bridge hit it}
    qbridged = [qt for qt in set(words(q)) if qt in br.bridge]
    for qt in qbridged:
        for dt, w in br.bridge[qt]:
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            post = idx.postings.get(p)
            if not post:
                continue
            for d, tf in post.items():
                contrib[d] += w * tf / (tf + 1.0)
                agree[d].add(qt)
    nqb = len(qbridged) or 1
    # unanchored docs must clear the consensus gate; anchored (lexical) docs stay
    exp_docs = [d for d in contrib if d not in cset and len(agree[d]) >= min_agree]
    exp_docs.sort(key=lambda d: contrib[d], reverse=True)
    pool = list(cand) + exp_docs[:n_expand]
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(contrib.values()) if contrib else 1.0
    final = {}
    for d in pool:
        s = lex.get(d, 0.0) / lmax
        c = contrib.get(d, 0.0)
        if c > 0:
            f = (len(agree[d]) / nqb) if weight else 1.0
            s += lam * (c / emax) * f
        final[d] = s
    return sorted(final, key=lambda d: final[d], reverse=True)


def evaluate(rank_fn, queries, test_q, test_ids, watch=None, watch_gold=None):
    nd = rc = 0.0
    watch_rank = None
    for qid in test_ids:
        full = rank_fn(queries[qid])
        ranked = full[:10]
        nd += ndcg10(ranked, test_q[qid])
        rc += recall10(ranked, test_q[qid])
        if watch and qid == watch and watch_gold:
            watch_rank = (full.index(watch_gold) if watch_gold in full else None)
    return nd / len(test_ids), rc / len(test_ids), watch_rank


def run(name, watch=None, watch_gold=None):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    br1 = RelevanceBridges(idx, N, min_pairs=1).learn(queries, train_q, corpus)
    br2 = RelevanceBridges(idx, N, min_pairs=2).learn(queries, train_q, corpus)

    print(f"\n{'='*70}\n{name}: {len(corpus)} docs | test {len(test_ids)} q")
    hdr = "  {:<32} {:>7} {:>7}".format("config", "nDCG", "Recall")
    if watch:
        hdr += "  {:>13}".format(f"q{watch} goldrank")
    print(hdr)

    def show(tag, fn):
        nd, rc, wr = evaluate(fn, queries, test_q, test_ids, watch, watch_gold)
        line = f"  {tag:<32} {nd:>7.4f} {rc:>7.4f}"
        if watch:
            line += f"  {str(wr):>13}"
        print(line)
        return nd, rc

    a = show("A min2  no-consensus (best)", lambda q: full_best(idx, br2, q))
    b = show("B min1  no-consensus", lambda q: full_best(idx, br1, q))
    c = show("C min1  consensus gate", lambda q: consensus_rank(idx, br1, q, min_agree=2, weight=False))
    d = show("D min1  consensus + weight", lambda q: consensus_rank(idx, br1, q, min_agree=2, weight=True))
    e = show("E min2  consensus + weight", lambda q: consensus_rank(idx, br2, q, min_agree=2, weight=True))
    base = a[0]
    best = max([("A", a), ("B", b), ("C", c), ("D", d), ("E", e)], key=lambda x: x[1][0])
    print(f"  => best: {best[0]} nDCG {best[1][0]:.4f} ({best[1][0]-base:+.4f} vs A), "
          f"recall {best[1][1]:.4f}")
    return base, best


def full_best(idx, br, q):
    """best_search but returning the FULL ranked pool (to locate watched gold)."""
    from collections import defaultdict as dd
    lex = idx._score(q)
    cand = sorted(lex, key=lambda d: lex[d], reverse=True)[:100]
    exp = dd(float)
    for qt in set(words(q)):
        for dt, w in br.bridge.get(qt, ()):
            p = idx.token_prime.get(("w", dt))
            if p is None:
                continue
            for d, tf in idx.postings.get(p, {}).items():
                exp[d] += w * tf / (tf + 1.0)
    cset = set(cand)
    extra = [d for d in sorted(exp, key=lambda d: exp[d], reverse=True)
             if d not in cset][:N_EXPAND]
    pool = cand + extra
    if not pool:
        return []
    lmax = max((lex.get(d, 0.0) for d in pool), default=1.0) or 1.0
    emax = max(exp.values()) if exp else 1.0
    final = {d: lex.get(d, 0.0) / lmax + LAM * exp.get(d, 0.0) / emax for d in pool}
    return sorted(final, key=lambda d: final[d], reverse=True)


def main():
    print("BRIDGE CONSENSUS (meet): trust bridge-reached docs by agreement")
    print("held-out test; bridges from train only; LAM=0.25 n_expand=20")
    s_base, s_best = run("scifact", watch="1", watch_gold="31715818")
    n_base, n_best = run("nfcorpus")
    print(f"\n{'='*70}\nVERDICT (held-out)")
    print(f"  scifact : A {s_base:.4f} -> {s_best[0]} {s_best[1][0]:.4f} "
          f"({s_best[1][0]-s_base:+.4f})")
    print(f"  nfcorpus: A {n_base:.4f} -> {n_best[0]} {n_best[1][0]:.4f} "
          f"({n_best[1][0]-n_base:+.4f})")
    print()
    print("  Consensus = an FTA meet: a doc is trusted when multiple independent")
    print("  query-term bridges intersect on it. Goal: keep min1's rare-concept")
    print("  recall while filtering single-bridge noise - lift BOTH corpora.")


if __name__ == "__main__":
    main()
