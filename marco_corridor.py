#!/usr/bin/env python3
"""MS MARCO -- the FORMULA-NATIVE engine: the prime-lattice CORRIDOR as the retrieval
substrate (not BM25+patch). Embodies "the rarest company narrows": each query term
activates its rare-term corridor, docs score by corridor overlap, and the score is
multiplied super-linearly by how many distinct query-term corridors MEET on the doc
(intersection narrowing -- topical single-corridor hits suppressed, multi-corridor
convergence amplified). Measured vs BM25 floor on the same pool, with hit/miss split.
"""
import sys, time
from collections import defaultdict, Counter
from marco_baseline import load_dev, build_pool, load_texts, BM25, tok
from marco_bridges import search_scored


def build_corridor(bm, texts, idf_gate=4.0, top_per=10, min_co=3):
    cooc = defaultdict(Counter)
    freq = Counter()
    t0 = time.perf_counter()
    for text in texts.values():
        rare = [w for w in set(tok(text)) if bm.idf.get(w, 0.0) >= idf_gate]
        for a in rare:
            freq[a] += 1
        for i, a in enumerate(rare):
            ca = cooc[a]
            for b in rare:
                if a != b:
                    ca[b] += 1
    corridor = {}
    for a, cnt in cooc.items():
        fa = freq[a]
        scored = [(b, (c / fa) * bm.idf.get(b, 0.0)) for b, c in cnt.items() if c >= min_co]
        scored.sort(key=lambda x: -x[1])
        if scored:
            corridor[a] = scored[:top_per]
    print(f"  corridor: {len(corridor)} rare anchor-terms ({time.perf_counter()-t0:.0f}s)", flush=True)
    return corridor


def corridor_search(bm, query, corridor, beta=0.3, alpha=1.5, k=100):
    qterms = [w for w in set(tok(query)) if w in bm.idf]
    # activated term -> {qidx: weight}  (self full idf-weight 1.0; corridor neighbor beta*strength)
    term_src = defaultdict(dict)
    for qi, qt in enumerate(qterms):
        term_src[qt][qi] = max(term_src[qt].get(qi, 0.0), 1.0)
        for ct, strength in corridor.get(qt, []):
            term_src[ct][qi] = max(term_src[ct].get(qi, 0.0), beta * strength)
    score = defaultdict(float)
    hits = defaultdict(set)
    for term, srcmap in term_src.items():
        if term not in bm.post:
            continue
        idft = bm.idf[term]
        mass = sum(srcmap.values())
        qis = set(srcmap.keys())
        for di, _c in bm.post[term]:
            score[di] += mass * idft
            hits[di] |= qis
    final = {di: score[di] * (len(hits[di]) ** alpha) for di in score}
    top = sorted(final, key=final.get, reverse=True)[:k]
    return [bm.docids[di] for di in top], {bm.docids[di]: final[di] for di in top}


def rr10(ranked, rel):
    for i, pid in enumerate(ranked[:10], 1):
        if pid in rel:
            return 1.0 / i
    return 0.0


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    nd = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    qrels, queries = load_dev()
    qids, pool, rel = build_pool(qrels, queries, nq, nd)
    texts = load_texts(pool)
    qids = [q for q in qids if all(p in texts for p in qrels[q])]
    bm = BM25(); bm.index(list(texts.items()))
    corridor = build_corridor(bm, texts)

    K = 60
    methods = {"bm25": {}, "corridor": {}, "rrf(bm25+corr)": {}}
    for m in methods:
        methods[m] = {"all": 0.0, "hit": 0.0, "miss": 0.0}
    nhit = nmiss = 0
    t0 = time.perf_counter()
    for q in qids:
        r = qrels[q]
        bm_ranked = [bm.docids[di] for di, _ in search_scored(bm, queries[q], 100)]
        co_ranked, _ = corridor_search(bm, queries[q], corridor, k=100)
        # rrf
        rrf = defaultdict(float)
        for rank, pid in enumerate(bm_ranked):
            rrf[pid] += 1.0 / (K + rank + 1)
        for rank, pid in enumerate(co_ranked):
            rrf[pid] += 1.0 / (K + rank + 1)
        rrf_ranked = sorted(rrf, key=rrf.get, reverse=True)
        rb = rr10(bm_ranked, r)
        bucket = "hit" if rb > 0 else "miss"
        nhit += bucket == "hit"; nmiss += bucket == "miss"
        for m, ranked in (("bm25", bm_ranked), ("corridor", co_ranked), ("rrf(bm25+corr)", rrf_ranked)):
            v = rr10(ranked, r)
            methods[m]["all"] += v
            methods[m][bucket] += v
    n = len(qids)
    print(f"\n  eval {n} queries in {time.perf_counter()-t0:.0f}s  ({nhit} BM25-hit, {nmiss} BM25-miss)")
    print(f"\n   {'method':>16}{'MRR@10':>9}{'hit-set':>10}{'miss-set':>10}")
    for m in ("bm25", "corridor", "rrf(bm25+corr)"):
        print(f"   {m:>16}{methods[m]['all']/n:>9.4f}{methods[m]['hit']/max(1,nhit):>10.4f}"
              f"{methods[m]['miss']/max(1,nmiss):>10.4f}")
    print(f"\n   corridor as the ENGINE (formula-native). miss-set = where BM25 failed:")
    print(f"   corridor/rrf > 0 there = the meet/narrowing rescued vocabulary-mismatch queries.")


if __name__ == "__main__":
    main()
