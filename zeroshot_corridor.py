#!/usr/bin/env python3
"""Zero-shot retrieval lift from UNSUPERVISED corridor semantics (no qrels). For
each query, the branch-MEET of its rare words (terms shared across >=2 query-word
corridors, weighted by rarity) = the query's topic. Retrieve by that, RRF-FUSE with
the plain lexical ranking. Measured on an aligned corpus (scifact) and a mismatch
one (nfcorpus) -- prior: signals help mismatch vocab, neutral on aligned."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, ndcg10, recall10


def run(ds):
    corpus, queries, train_q, test_q = load(ds)
    test_ids = [q for q in test_q if q in queries]
    idx = AppendOnlyLatticeIndex()
    for d, t in corpus.items():
        idx.add(d, t)
    N = len(idx.alive)
    _idf = {}

    def idf(w):
        v = _idf.get(w)
        if v is None:
            p = idx.token_prime.get(("w", w))
            v = idx._idf(p, N) if p else 0.0
            _idf[w] = v
        return v

    RARE = 3.0
    cooc = defaultdict(Counter)                         # unsupervised corridor (corpus only)
    for d, t in corpus.items():
        ws = [w for w in set(words(t)) if idf(w) >= RARE]
        for a in ws:
            for b in ws:
                if a != b:
                    cooc[a][b] += 1

    def meet_terms(q, k=10):
        present, strength = Counter(), Counter()
        qrare = [w for w in set(words(q)) if idf(w) >= RARE]
        for w in qrare:
            for c, n in cooc.get(w, {}).items():
                if c not in qrare:
                    present[c] += 1
                    strength[c] += n * idf(c)
        need = 2 if len(qrare) >= 2 else 1
        cand = [c for c in present if present[c] >= need]
        return sorted(cand, key=lambda c: -strength[c])[:k]

    def lex(text, k=100):
        s = idx._score(text)
        return sorted(s, key=s.get, reverse=True)[:k]

    def rrf(weighted, k=60):
        sc = defaultdict(float)
        for r, w in weighted:
            for i, d in enumerate(r):
                sc[d] += w / (k + i + 1)
        return sorted(sc, key=sc.get, reverse=True)

    def baseline(q):
        return lex(q, 10)

    def corridor(q, w=0.15):
        mt = meet_terms(q)
        if not mt:
            return lex(q, 10)
        return rrf([(lex(q, 100), 1.0), (lex(" ".join(mt), 100), w)])[:10]

    def ev(fn):
        nd = rc = 0.0
        for qid in test_ids:
            r = fn(queries[qid])
            nd += ndcg10(r, test_q[qid])
            rc += recall10(r, test_q[qid])
        n = len(test_ids)
        return nd / n, rc / n

    b = ev(baseline)
    c = ev(corridor)
    return b, c, len(corpus), len(test_ids)


def main():
    print("ZERO-SHOT corridor-meet expansion (UNSUPERVISED, no qrels), RRF-fused\n")
    print(f"  {'corpus':<10}{'baseline nDCG':>14}{'+ corridor':>12}{'delta':>9}{'recall delta':>14}")
    for ds in ("scifact", "nfcorpus"):
        try:
            b, c, nd, nq = run(ds)
        except Exception as e:
            print(f"  {ds:<10} skipped ({type(e).__name__})")
            continue
        print(f"  {ds:<10}{b[0]:>14.4f}{c[0]:>12.4f}{c[0]-b[0]:>+9.4f}{c[1]-b[1]:>+14.4f}")
    print("\n  zero-shot = NO qrels used (vs the supervised bridge which needs them).")
    print("  honest: corridor expansion bridges divergent vocabulary -> expect it to help")
    print("  the mismatch corpus (nfcorpus) more than the aligned one (scifact).")


if __name__ == "__main__":
    main()
