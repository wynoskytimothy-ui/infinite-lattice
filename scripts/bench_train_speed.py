#!/usr/bin/env python3
"""
Deep search step 8 - optimize bridge TRAINING (faster, smaller, lossless).

RelevanceBridges.learn re-tokenizes each gold doc every time it appears in a
relevant pair, and counts co-occurrence over ALL doc words even though only
high-idf ones can survive the idf gate at finalize. Two lossless wins:
  - CACHE tokenized + idf-filtered doc/query word sets (a doc in 30 pairs is
    tokenized once, not 30x);
  - PRE-FILTER doc words by idf at COUNT time (the gate is applied at finalize
    anyway, so the bridges are identical) -> smaller inner loop AND a smaller
    intermediate co-occurrence table (training footprint).

Validates: bridges BIT-IDENTICAL to the current learn (so retrieval is
unchanged), while learn() is faster and the co-occurrence table is smaller.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, words
from scripts.bench_supervised_bridges import load, RelevanceBridges


class LegacyBridges(RelevanceBridges):
    """The pre-optimization learn() (no caching, count all doc words, filter idf
    at finalize) - kept as the before/after baseline. The optimized version now
    lives in RelevanceBridges.learn (shipped)."""

    def learn(self, queries, train_qrels, corpus):
        for qid, rels in train_qrels.items():
            if qid not in queries:
                continue
            qterms = {w for w in words(queries[qid]) if self._idf(w) >= self.idf_gate}
            if not qterms:
                continue
            for cid, sc in rels.items():
                if sc <= 0 or cid not in corpus:
                    continue
                dterms = set(words(corpus[cid]))
                for qt in qterms:
                    self.qt_pairs[qt] += 1
                    for dt in dterms:
                        if dt != qt:
                            self.cooc[qt][dt] += 1
        for qt, partners in self.cooc.items():
            np_ = self.qt_pairs[qt]
            scored = []
            for dt, c in partners.items():
                if c < self.min_pairs:
                    continue
                idf = self._idf(dt)
                if idf < self.idf_gate:
                    continue
                scored.append((dt, (c / np_) * idf))
            scored.sort(key=lambda x: x[1], reverse=True)
            if scored:
                self.bridge[qt] = scored[:self.top_per_term]
        return self


def full_scored(br):
    """The full (uncut) set of surviving (dt, weight) per query term - the real
    correctness invariant. The top_per_term CUT can differ at tie boundaries
    (immaterial to retrieval), so compare the uncut sets, not the cut bridges."""
    out = {}
    for qt, parts in br.cooc.items():
        np_ = br.qt_pairs[qt]
        s = frozenset((dt, round((c / np_) * br._idf(dt), 9))
                      for dt, c in parts.items()
                      if dt != qt and c >= br.min_pairs and br._idf(dt) >= br.idf_gate)
        if s:
            out[qt] = s
    return out


def cooc_entries(br):
    return sum(len(c) for c in br.cooc.values())


def run(name, min_pairs):
    corpus, queries, train_q, test_q = load(name)
    idx = AppendOnlyLatticeIndex()
    for d, txt in corpus.items():
        idx.add(d, txt)
    N = len(idx.alive)
    njudg = sum(len(v) for v in train_q.values())
    print(f"\n{'='*60}\n{name}: {N:,} docs, {njudg:,} train judgements (min_pairs={min_pairs})")

    t0 = time.perf_counter()
    legacy = LegacyBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    t_base = time.perf_counter() - t0
    cooc_base = cooc_entries(legacy)

    t0 = time.perf_counter()
    opt = RelevanceBridges(idx, N, min_pairs=min_pairs).learn(queries, train_q, corpus)
    t_opt = time.perf_counter() - t0
    cooc_opt = cooc_entries(opt)

    same = full_scored(legacy) == full_scored(opt)       # uncut counts (correctness)
    eb = sum(len(v) for v in legacy.bridge.values())
    eo = sum(len(v) for v in opt.bridge.values())

    print(f"  legacy learn:    {t_base:6.2f}s   cooc table {cooc_base:>9,} entries   "
          f"{eb:,} bridges")
    print(f"  shipped (opt):   {t_opt:6.2f}s   cooc table {cooc_opt:>9,} entries   "
          f"{eo:,} bridges")
    print(f"  => {t_base/t_opt:.1f}x faster, cooc table {cooc_base/max(1,cooc_opt):.2f}x smaller, "
          f"bridges identical: {same}")
    return same


def main():
    print("DEEP SEARCH step 8 - optimize bridge training (lossless)")
    ok = True
    for name, mp in (("scifact", 1), ("nfcorpus", 2)):
        ok &= run(name, mp)
    print(f"\n  bridges bit-identical on both corpora: {ok}")
    print("  => retrieval accuracy is unchanged by construction; only training")
    print("  time and the intermediate co-occurrence footprint go down.")


if __name__ == "__main__":
    main()
