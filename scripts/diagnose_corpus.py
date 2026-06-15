#!/usr/bin/env python3
"""
Glass-box failure analysis - WHY does each query miss, per corpus?

    python scripts/diagnose_corpus.py <dataset> [min_pairs]

For every test query it finds the rank of each gold doc in the full lex+bridge
ranking and buckets the failures into "goblins":

  HIT        best gold in top-10 (no goblin)
  NEAR       best gold rank 11-30  (tuning can recover)
  RANKING    best gold rank 31-500 (findable but outranked)
  SEMANTIC   gold scored but query<->gold word overlap = 0 (vocabulary gap)
  POOL-MISS  gold not scored at all (unreachable; needs a doc / bridge)

Then reports the recoverable headroom (nDCG if NEAR misses were fixed) and the
top uncovered query terms - the ranked "what to fix next" list. Glass box: every
miss is explained, not guessed.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from aethos_bridges import RelevanceBridges
from scripts.bench_supervised_bridges import load, ndcg10, recall10

LAM = 0.25


def full_ranking(idx, br, query, lam=LAM):
    lex = idx._score(query)
    exp = defaultdict(float)
    if br is not None:
        for qt in set(words(query)):
            for dt, w in br.bridge.get(qt, ()):
                p = idx.token_prime.get(("w", dt))
                if p is None:
                    continue
                for d, tf in idx.postings.get(p, {}).items():
                    exp[d] += w * tf / (tf + 1.0)
    pool = set(lex) | set(exp)
    lmax = max(lex.values()) if lex else 1.0
    emax = max(exp.values()) if exp else 1.0
    scored = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
    ranked = sorted(scored, key=lambda d: scored[d], reverse=True)
    return ranked, scored


def run(name, min_pairs=2):
    corpus, queries, train_q, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    br = (RelevanceBridges(idx, len(idx.alive), min_pairs=min_pairs)
          .learn(queries, train_q, corpus)) if train_q else None
    print(f"\n{'='*64}\n{name}: {len(corpus):,} docs | {len(test_ids)} test q"
          f" | bridges={'yes' if br else 'no'}")

    N = len(idx.alive)

    def idf(w):
        p = idx.token_prime.get(("w", w))
        return idx._idf(p, N) if p else 0.0

    goblins = Counter()
    tunable_nd = cur_nd = 0.0
    lex_worst = []
    for qid in test_ids:
        ranked, scored = full_ranking(idx, br, queries[qid])
        rels = test_q[qid]
        golds = {d for d, s in rels.items() if s > 0}
        cur_nd += ndcg10(ranked[:10], rels)
        rank = {d: (ranked.index(d) + 1 if d in scored else None) for d in golds}
        best = min((r for r in rank.values() if r), default=None)
        ideal = ranked[:10]
        if best and best > 10:
            bestdoc = min((d for d in golds if rank[d]), key=lambda d: rank[d])
            qwords = set(words(queries[qid]))
            gwords = set(words(corpus[bestdoc]))
            ov = qwords & gwords
            top_w = max(qwords, key=idf, default="")           # query's key (rare) term
            if not ov:
                goblins["SEMANTIC (0 overlap)"] += 1            # needs bridges/author-doc
            elif top_w and top_w not in gwords:
                goblins["DISCRIMINATIVE-GAP (key term absent)"] += 1   # needs semantics
            else:
                goblins[f"LEXICAL (has key term, rank {'11-30' if best<=30 else '>30'})"] += 1
                # LEXICALLY tunable: the gold has the key term but ranks low
                ideal = [bestdoc] + [d for d in ranked[:10] if d != bestdoc][:9]
                lex_worst.append((best, qid, top_w, round(idf(top_w), 1),
                                  sorted(ov & {w for w in qwords if idf(w) >= 2})[:4]))
        elif best is None:
            goblins["POOL-MISS (unreachable)"] += 1
        else:
            goblins["HIT (top-10)"] += 1
        tunable_nd += ndcg10(ideal, rels)

    n = len(test_ids)
    print(f"  current nDCG@10 {cur_nd/n:.4f}   if-LEXICAL-misses-fixed {tunable_nd/n:.4f} "
          f"(+{(tunable_nd-cur_nd)/n:.4f} = the lexically-tunable headroom)")
    print("  goblins (best gold-doc rank):")
    for g, c in goblins.most_common():
        print(f"     {c:4d}  {g}")
    lex_worst.sort()
    print(f"  LEXICAL misses (tunable: gold HAS the key term but ranks low):")
    for r, qid, tw, tidf, ov in lex_worst[:6]:
        print(f"     rank {r:>3}  q{qid}: key '{tw}'(idf {tidf}) present in gold; "
              f"other matched rare terms {ov or '-'}  '{queries[qid][:38]}'")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run(name, mp)


if __name__ == "__main__":
    main()
