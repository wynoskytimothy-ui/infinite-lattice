#!/usr/bin/env python3
"""The discrimination-cascade diagnostic on the FULL 8.8M collection (same battery we ran on
the 298k pool: marco_overlap / marco_rarest / marco_crossover / marco_recover).

The full index makes this exact: a term's posting list IS the set of docs containing it, and
the lists are sorted by doc-id (streaming build), so reach = len(posting) and gold-coverage =
membership test, no text scanning. For the same 3000 dev queries (seed 42) we measure, per
rare query word (idf>=4):
  - rarest word alone:   how many docs it reaches, is the gold among them (gold vs non-gold)
  - 2nd-rarest alone:    same
  - 2-way meet (r1 & r2): how many docs share BOTH, gold among them
  - 3-way meet:          how many share all three, gold among them
plus the rare-word availability distribution and the meet's narrowing power.

  python marco_full_diagnose.py [n_queries]   (default 3000)
"""
import sys, random
from collections import defaultdict, Counter
import numpy as np
from marco_full_eval import FullIndex, stoks, RARE


def posting(idx, term):
    i = idx.tid.get(term)
    if i is None:
        return None
    s, e = int(idx.ptr[i]), int(idx.ptr[i + 1])
    return idx.di[s:e]                      # sorted ascending (streaming build)


def has(post, dis):
    """is any di in dis present in the sorted posting array?"""
    if post is None or len(post) == 0:
        return False
    for di in dis:
        j = np.searchsorted(post, di)
        if j < len(post) and post[j] == di:
            return True
    return False


def med(xs):
    return int(np.median(xs)) if xs else 0


def pct(num, den):
    return 100.0 * num / den if den else 0.0


def main():
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
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
    random.Random(42).shuffle(qids)
    qids = qids[:nq]

    n_rare_hist = Counter()          # # rare words in query
    n_rare_in_gold_hist = Counter()  # # of those rare words that appear in the gold doc
    # per-level: reaches (doc counts) and gold-covered flags
    reach = defaultdict(list); goldcov = defaultdict(int); have = defaultdict(int)
    narrow_keep = 0; narrow_n = 0; narrow_r1 = []; narrow_r2 = []

    for q in qids:
        gold = set(int(p) for p in qrels[q])
        terms = [(w, idx.idf_of(w)) for w in set(stoks(queries[q]))]
        terms = [(w, v) for w, v in terms if v > 0]
        rare = sorted([(w, v) for w, v in terms if v >= RARE], key=lambda x: -x[1])
        n_rare_hist[min(len(rare), 5)] += 1
        # how many rare words appear in the gold doc
        ng = sum(1 for w, _ in rare if has(posting(idx, w), gold))
        n_rare_in_gold_hist[min(ng, 5)] += 1
        if not rare:
            continue
        p1 = posting(idx, rare[0][0])
        reach["r1"].append(len(p1) if p1 is not None else 0); have["r1"] += 1
        if has(p1, gold):
            goldcov["r1"] += 1
        if len(rare) >= 2:
            p2 = posting(idx, rare[1][0])
            reach["r2"].append(len(p2) if p2 is not None else 0); have["r2"] += 1
            if has(p2, gold):
                goldcov["r2"] += 1
            m2 = np.intersect1d(p1, p2, assume_unique=True)
            reach["m2"].append(len(m2)); have["m2"] += 1
            if has(m2, gold):
                goldcov["m2"] += 1
            # narrowing power: among queries where the rarest already covers gold,
            # does the 2-way keep gold, and how much does reach shrink?
            if has(p1, gold):
                narrow_n += 1; narrow_r1.append(len(p1)); narrow_r2.append(len(m2))
                if has(m2, gold):
                    narrow_keep += 1
        if len(rare) >= 3:
            p3 = posting(idx, rare[2][0])
            m3 = np.intersect1d(np.intersect1d(p1, posting(idx, rare[1][0]), assume_unique=True),
                                p3, assume_unique=True)
            reach["m3"].append(len(m3)); have["m3"] += 1
            if has(m3, gold):
                goldcov["m3"] += 1

    N = len(qids)
    print(f"\nFULL 8.8M DISCRIMINATION AUDIT -- {N} dev queries (same sample as the eval)\n")
    print("  rare words available per query (idf>=4):")
    for k in range(6):
        lbl = f"{k}+" if k == 5 else str(k)
        print(f"     {lbl} rare: {n_rare_hist.get(k,0):>5}  ({pct(n_rare_hist.get(k,0),N):4.1f}%)")
    print("\n  of those rare words, how many appear in the GOLD doc:")
    for k in range(6):
        lbl = f"{k}+" if k == 5 else str(k)
        print(f"     {lbl} in gold: {n_rare_in_gold_hist.get(k,0):>5}  ({pct(n_rare_in_gold_hist.get(k,0),N):4.1f}%)")

    print("\n  THE CASCADE -- docs reached (median) and gold-coverage, full collection:")
    print(f"     {'level':<22}{'queries':>9}{'med docs':>10}{'gold in set':>13}")
    rows = [("rarest word alone", "r1"), ("2nd-rarest alone", "r2"),
            ("rarest & 2nd (2-way)", "m2"), ("rarest&2nd&3rd (3-way)", "m3")]
    for lbl, key in rows:
        h = have[key]
        print(f"     {lbl:<22}{h:>9}{med(reach[key]):>10}{pct(goldcov[key],h):>12.1f}%")
    print("\n  pool reference (298k):  rarest ~76 docs/89% -> 2-way ~12/69% -> 3-way ~1/29%")

    print(f"\n  MEET NARROWING POWER (queries where the rarest already covers gold, n={narrow_n}):")
    if narrow_n:
        print(f"     adding the 2nd rare word: median reach {med(narrow_r1)} -> {med(narrow_r2)} docs "
              f"({med(narrow_r1)/max(1,med(narrow_r2)):.0f}x narrower), gold KEPT "
              f"{pct(narrow_keep,narrow_n):.1f}% of the time")
        print("     = the crossover/meet cuts the non-gold competition while keeping the gold = the +5.7% rerank lever.")


if __name__ == "__main__":
    main()
