#!/usr/bin/env python3
"""Score-attribution glass box: WHY do pollutant docs out-rank the gold? For every RANKING
failure (gold is in the BM25 top-100 but NOT top-10), decompose the score gap (rank-1 pollutant
minus gold) term by term, and classify the polluting terms by frequency. Answers the user's
questions: what triggered what, what is polluting the answers, and whether HIGH-FREQUENCY words
are causing the mismatch (winning by volume) vs the gold genuinely lacking a rare word.

Full 8.8M, dev queries. For each failure (wrong = BM25 rank-1, gold = the in-pool gold doc):
  - per query term: its BM25 contribution to wrong vs gold; gap = wrong - gold.
  - attribute the total gap to COMMON (idf<2) / MEDIUM (2-4) / RARE (>=4) terms.
  - top polluter: is it ABSENT in gold (gold lacks it) or present-but-lower-tf (volume/stuffing)?
"""
import sys, random
from collections import Counter, defaultdict
import numpy as np
from marco_full_eval import FullIndex, stoks, RARE, QGATE, K1, B


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    idx = FullIndex()

    qrels = defaultdict(set)
    with open(idx.cf.name.replace("collection.tsv", "qrels.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 4 and int(p[3]) > 0:
                qrels[p[0]].add(p[2])
    queries = {}
    with open(idx.cf.name.replace("collection.tsv", "queries.dev.tsv"), encoding="utf-8") as f:
        for line in f:
            a = line.rstrip("\n").split("\t", 1)
            if len(a) == 2:
                queries[a[0]] = a[1]
    qids = [q for q in qrels if q in queries]
    random.Random(42).shuffle(qids); qids = qids[:nq]

    def contrib(tf, dl, idf):
        return 0.0 if tf == 0 else idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * dl / idx.avgdl))

    gap_by_class = defaultdict(float); n_gap_by_class = defaultdict(int)
    total_gap = 0.0
    pol_idf = []                       # idf of the single biggest polluting term per failure
    pol_absent = pol_volume = pol_other = 0
    gold_missing = []; wrong_missing = []
    examples = []
    nfail = nrankfail = 0

    for q in qids:
        rel = qrels[q]
        qs = [w for w in set(stoks(queries[q])) if idx.idf_of(w) >= QGATE]
        if not qs:
            continue
        order, _ = idx.bm25_top(qs, k=100)
        ranked = [int(d) for d in order]
        rstr = [str(d) for d in ranked]
        if any(d in rel for d in rstr[:10]):
            continue                                    # gold already in top-10 (win)
        gpos = next((i for i, d in enumerate(rstr) if d in rel), None)
        if gpos is None:
            continue                                    # gold not in top-100 = recall failure, skip
        nrankfail += 1
        wrong_pid = ranked[0]
        gold_pid = ranked[gpos]
        wt = Counter(stoks(idx.text(wrong_pid))); wdl = max(1, sum(wt.values()))
        gt = Counter(stoks(idx.text(gold_pid))); gdl = max(1, sum(gt.values()))
        gaps = {}
        for w in qs:
            iw = idx.idf_of(w)
            gaps[w] = contrib(wt.get(w, 0), wdl, iw) - contrib(gt.get(w, 0), gdl, iw)
        for w, g in gaps.items():
            if g > 0:
                iw = idx.idf_of(w)
                cls = "rare" if iw >= RARE else ("medium" if iw >= 2.0 else "common")
                gap_by_class[cls] += g; n_gap_by_class[cls] += 1; total_gap += g
        topw = max(gaps, key=gaps.get)
        tiw = idx.idf_of(topw)
        pol_idf.append(tiw)
        if gt.get(topw, 0) == 0:
            pol_absent += 1                              # gold simply lacks this word
        elif wt.get(topw, 0) > gt.get(topw, 0):
            pol_volume += 1                              # both have it, wrong has MORE (volume/stuffing)
        else:
            pol_other += 1
        gold_missing.append(sum(1 for w in qs if gt.get(w, 0) == 0))
        wrong_missing.append(sum(1 for w in qs if wt.get(w, 0) == 0))
        if len(examples) < 12 and tiw < 2.0:            # capture common-word pollution examples
            examples.append((queries[q][:42], topw, round(tiw, 1), wt.get(topw, 0), gt.get(topw, 0)))

    N = nrankfail
    print(f"\nPOLLUTION ATTRIBUTION -- full 8.8M, {N} RANKING failures (gold in top-100, not top-10)\n")
    print(f"   score-gap (pollutant rank-1 minus gold) attributed by term frequency class:")
    for cls in ("common", "medium", "rare"):
        share = gap_by_class[cls] / total_gap if total_gap else 0
        print(f"     {cls:<8} (idf {'<2' if cls=='common' else '2-4' if cls=='medium' else '>=4'}): "
              f"{share:>6.1%} of the gap   ({n_gap_by_class[cls]} polluting terms)")
    print(f"\n   the single biggest polluting term per failure -- WHY it wins:")
    print(f"     gold simply LACKS it (absent in gold): {pol_absent/N:>6.1%}")
    print(f"     VOLUME (both have it, pollutant has more tf): {pol_volume/N:>6.1%}")
    print(f"     other: {pol_other/N:>6.1%}")
    print(f"     median idf of the top polluter: {np.median(pol_idf):.2f}  (high=rare word gold lacks; low=common-word volume)")
    print(f"\n   gold query-terms MISSING (median): {int(np.median(gold_missing))}  vs  pollutant missing: {int(np.median(wrong_missing))}")
    print(f"   (gold missing more terms -> it's a weaker lexical match -> semantic gap, not fixable pollution)")
    print(f"\n   common-word pollution examples (query | top polluter | idf | tf_wrong | tf_gold):")
    for ex in examples[:10]:
        print(f"     {ex[0]:<44} {ex[1]:<14} idf {ex[2]:<4} tf {ex[3]} vs {ex[4]}")


if __name__ == "__main__":
    main()
