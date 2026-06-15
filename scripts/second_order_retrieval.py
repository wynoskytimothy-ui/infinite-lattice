#!/usr/bin/env python3
"""
Second-order rare-anchored semantics as an UNSUPERVISED retrieval bridge.

Builds term synonymy from the corpus alone (no qrels): two terms are linked if
they share rare context (second-order). Then expands the query with each term's
high-confidence synonyms and fuses conservatively with the lexical ranking.

The honest test: does corpus-derived SYNONYMY (not association) help retrieval -
especially where there is a vocabulary gap (nfcorpus) and where there is NO
training data (trec-covid/touche), the one place supervised bridges can't go?
Measured vs the lexical baseline; conservative (rerank lexical candidates +
bounded pool expansion) so a bad link can't inject a drift doc.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def build_second_order(corpus, top_anchors=8, min_df=4, max_inv=400,
                       top_per=6, min_sim=0.5):
    """term -> [(neighbour, sim)] from shared rare context.

    Anchors = the top-K RAREST words in each document (the user's "rarest words
    mean the most" - and this bounds the work). A term too rare to have a
    distribution (df<min_df) is skipped; an anchor shared by >max_inv terms is
    not discriminative and is skipped at query time. Second-order similarity =
    idf-weighted cosine of the rare-anchor context vectors."""
    df = Counter()
    docterms = []
    for txt in corpus.values():
        ts = set(words(txt))
        docterms.append(ts)
        for w in ts:
            df[w] += 1
    N = len(docterms)

    def idf(w):
        return math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))

    co = defaultdict(Counter)
    for ts in docterms:
        rare = heapq.nlargest(top_anchors, ts, key=idf)     # K rarest words = anchors
        for t in ts:
            if df[t] < min_df:
                continue
            ct = co[t]
            for a in rare:
                if a != t:
                    ct[a] += 1
    inv = defaultdict(list)
    norm = {}
    for t, ctx in co.items():
        ss = 0.0
        for a, c in ctx.items():
            w = c * idf(a)
            ss += w * w
            inv[a].append(t)
        norm[t] = math.sqrt(ss) or 1.0

    nbr_cache = {}

    def neighbours(term):
        if term in nbr_cache:
            return nbr_cache[term]
        out = []
        if term in co and df[term] >= min_df:
            cand = Counter()
            for a, c in co[term].items():
                lst = inv[a]
                if len(lst) > max_inv:                      # common anchor: not discriminative
                    continue
                wa2 = c * idf(a) * idf(a)
                ca = co
                for t in lst:
                    cand[t] += wa2 * ca[t][a]
            na = norm[term]
            scored = [(dot / (na * norm[t]), t) for t, dot in cand.items() if t != term]
            scored.sort(reverse=True)
            out = [(t, s) for s, t in scored[:top_per] if s >= min_sim]
        nbr_cache[term] = out
        return out

    return neighbours, idf


def run(name, lam=0.10):
    corpus, queries, _, test_q = load(name)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    neighbours, sidf = build_second_order(corpus)
    print(f"\n{name}: {len(corpus):,} docs | {len(test_ids)} test q")

    def search_lex(q):
        return idx.search(q, 10)

    def search_sem(q):
        lex = idx._score(q)
        cand = sorted(lex, key=lex.get, reverse=True)[:100]
        if not cand:
            return []
        lmax = max(lex[d] for d in cand) or 1.0
        exp = defaultdict(float)
        for qt in set(words(q)):
            pq = idx.token_prime.get(("w", qt))
            if pq is None or idx._idf(pq, N) < 2.5:        # only expand RARE query terms
                continue
            for nb, s in neighbours(qt):
                p = idx.token_prime.get(("w", nb))
                if p is None:
                    continue
                idf_nb = idx._idf(p, N)
                for d, tf in idx.postings.get(p, {}).items():
                    exp[d] += s * idf_nb * tf / (tf + 1.0)
        cset = set(cand)
        extra = [d for d in sorted(exp, key=exp.get, reverse=True) if d not in cset][:20]
        pool = cand + extra
        emax = max(exp.values()) if exp else 1.0
        final = {d: lex.get(d, 0.0) / lmax + lam * exp.get(d, 0.0) / emax for d in pool}
        return heapq.nlargest(10, final, key=final.get)

    def ev(fn):
        nd = rc = 0.0
        for qid in test_ids:
            r = fn(queries[qid])
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    nd0, rc0 = ev(search_lex)
    nd1, rc1 = ev(search_sem)
    print(f"  lexical:            nDCG {nd0:.4f}  Recall {rc0:.4f}")
    print(f"  + 2nd-order semant: nDCG {nd1:.4f}  Recall {rc1:.4f}   "
          f"({nd1-nd0:+.4f} nDCG, {rc1-rc0:+.4f} recall)")
    # verifiability: a few learned synonym links
    shown = 0
    for qid in test_ids[:40]:
        for qt in set(words(queries[qid])):
            nb = neighbours(qt)
            if nb and shown < 4:
                print(f"     '{qt}' ~ " + ", ".join(f"{t}({s:.2f})" for t, s in nb[:3]))
                shown += 1
                break
    return nd1 - nd0


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    run(name)


if __name__ == "__main__":
    main()
