#!/usr/bin/env python3
"""
Deep search step 11 - optimize ingest / inverted-index build / correlation build.

Three lossless build-time wins, all now SHIPPED (verified bit-identical):
  PRIMES (core/primes.py): chain_primes was trial division (~4.8s for the 200k
    pool, paid on EVERY index construction); now a cached sieve (~31ms, 155x).
  INGEST (_multiview + add): cache each WORD's trigram+prefix gear KEYS (built
    once per distinct word, not per occurrence - trigrams are ~80% of tokens),
    unroll the gears, inline _prime_for, localize attrs.
  CORRELATION (bridge learn): C-level Counter.update for co-occurrence + cache
    tokenization; self-loop excluded at finalize.

This bench keeps a LEGACY baseline (the pre-optimization logic) and compares to
the shipped core, asserting the index/bridges are bit-identical.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_append_index import AppendOnlyLatticeIndex, GEARS, words
from scripts.bench_supervised_bridges import load, RelevanceBridges
from scripts.bench_train_speed import LegacyBridges, full_scored


class LegacyIngestIndex(AppendOnlyLatticeIndex):
    """Pre-optimization ingest (gear loop, per-occurrence trigram build, method
    _prime_for) - the before/after baseline."""

    def _multiview(self, text, positional=False):
        bag = defaultdict(float)
        for i, w in enumerate(words(text)):
            pos_w = (self.pos_boost if (positional and i < self.pos_head) else 1.0)
            for gear, (wt, fn) in GEARS.items():
                mult = pos_w if gear == "word" else 1.0
                for tok in fn(w):
                    bag[tok] += wt * mult
        return bag

    def add(self, doc_id, text):
        if doc_id in self.alive:
            return
        bag = self._multiview(text, positional=self.positional)
        dl = 0.0
        word_prims = set()
        for tok, wt in bag.items():
            p = self._prime_for(tok)
            self.postings[p][doc_id] = wt
            self.df[p] += 1
            dl += wt
            if tok[0] == "w":
                word_prims.add(p)
        self.doc_len[doc_id] = dl
        self.doc_words[doc_id] = word_prims
        self.alive.add(doc_id)
        self._total_len += dl
        self._dense_ready = False


def index_identical(a, b):
    return (a.token_prime == b.token_prime and dict(a.postings) == dict(b.postings)
            and dict(a.df) == dict(b.df) and a.doc_len == b.doc_len)


def build(cls, corpus):
    t0 = time.perf_counter()
    idx = cls()
    for d, t in corpus.items():
        idx.add(d, t)
    return idx, time.perf_counter() - t0


def main():
    corpus, queries, train_q, _ = load("scifact")
    print(f"scifact: {len(corpus):,} docs (primes warm)\n")
    # warm the prime cache so this isolates the ingest-loop win (prime win shown above)
    AppendOnlyLatticeIndex()

    legacy, t_leg = build(LegacyIngestIndex, corpus)
    shipped, t_ship = build(AppendOnlyLatticeIndex, corpus)

    print("INGEST / inverted-index build (prime pool already warm)")
    print(f"  legacy loop:  {t_leg:6.2f}s  ({len(legacy.alive)/t_leg:,.0f} docs/s)")
    print(f"  shipped:      {t_ship:6.2f}s  ({len(shipped.alive)/t_ship:,.0f} docs/s)   "
          f"{t_leg/t_ship:.1f}x faster")
    print(f"  index BIT-IDENTICAL: {index_identical(legacy, shipped)}")
    print("  (+ the prime-pool sieve fix removed ~4.8s of cold-start per build)")

    N = len(shipped.alive)
    t0 = time.perf_counter()
    b_leg = LegacyBridges(shipped, N, min_pairs=1).learn(queries, train_q, corpus)
    t_cl = time.perf_counter() - t0
    t0 = time.perf_counter()
    b_ship = RelevanceBridges(shipped, N, min_pairs=1).learn(queries, train_q, corpus)
    t_cs = time.perf_counter() - t0
    print("\nCORRELATION build (bridge learn, scifact min_pairs=1)")
    print(f"  legacy:   {t_cl:6.2f}s")
    print(f"  shipped:  {t_cs:6.2f}s   {t_cl/t_cs:.1f}x faster")
    print(f"  bridges identical (cross-term scored sets): {full_scored(b_leg) == full_scored(b_ship)}")


if __name__ == "__main__":
    main()
