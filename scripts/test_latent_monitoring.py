#!/usr/bin/env python3
"""
Test 37 - Latent monitoring: capabilities we built but never turned toward watching.

The architecture has monitoring powers beyond Test 25's supervision and
Test 28's prediction. Four of them, each falling out of a primitive we
already proved:

  (A) SURPRISE MONITOR - the compressor IS an anomaly detector. Surprise =
      -log2(p) = bits-per-symbol (Tests 15-23). Whatever the model did not
      expect compresses badly, so a bits/byte spike is an alarm. No separate
      detector needed; the codec already computes it.

  (B) FTA LOCALIZER - factoring pinpoints the broken element. A state is a
      prime composite (Test 3); divide expected by actual and the leftover
      primes NAME exactly which element changed - detection AND location,
      not just a "something is wrong" checksum.

  (C) ROOT-CAUSE PROMOTER - correlated failures share a hidden factor
      (Test 6). Promote it and the prime IS the common cause - automatic
      root-cause analysis.

  (D) INVARIANT WATCHDOG - the five Zeno roles (Test 32) are continuous
      assertions. Positivity, bounded budget, address exactness - violate
      any and the watchdog trips at the exact step.

Each is verified: clean separation of anomalies by surprise, 100% corruption
localization, correct root cause, and the watchdog catching an injected fault
on the precise tick.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.primes import chain_primes


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


PRIMES = chain_primes(4096)


# ======================================================================
# (A) SURPRISE MONITOR - the codec as an anomaly detector
# ======================================================================

class SurpriseMonitor:
    """Adaptive order-1 byte model; emits per-byte surprise = -log2(p).
    Same machinery as the lattice codec (Tests 15-23), read as an alarm."""

    def __init__(self, alphabet=256):
        self.A = alphabet
        self.ctx_counts: dict[int, list] = {}
        self.prev = 0

    def observe(self, sym: int) -> float:
        row = self.ctx_counts.get(self.prev)
        if row is None:
            row = [1] * self.A            # Laplace prior
            self.ctx_counts[self.prev] = row
        total = sum(row)
        p = row[sym] / total
        surprise = -math.log2(p)
        row[sym] += 8                      # adapt
        self.prev = sym
        return surprise


def main():
    header("Latent monitoring - four watchers hiding in the primitives")
    rng = random.Random(0x37A1)

    # ------------------------------------------------------------------
    print("\n(A) SURPRISE MONITOR - compressor surprise IS the anomaly alarm")
    print("-" * 72)
    # normal source: an order-1 Markov chain over a small alphabet (very
    # predictable once learned). Anomaly: a burst of uniform-random bytes.
    ALPHA = 16
    trans = {a: rng.sample(range(ALPHA), ALPHA) for a in range(ALPHA)}

    def markov_byte(prev):
        # 90% follow a fixed successor, 10% wander - learnable structure
        if rng.random() < 0.9:
            return trans[prev][0]
        return rng.randrange(ALPHA)

    mon = SurpriseMonitor(ALPHA)
    stream, labels = [], []                # label 1 = anomaly
    prev = 0
    # warmup + normal
    for _ in range(4000):
        prev = markov_byte(prev)
        stream.append(prev)
        labels.append(0)
    # anomalous burst (uniform random - no structure)
    for _ in range(200):
        b = rng.randrange(ALPHA)
        stream.append(b)
        labels.append(1)
    # back to normal
    for _ in range(2000):
        prev = markov_byte(prev)
        stream.append(prev)
        labels.append(0)

    surprises = [mon.observe(s) for s in stream]
    # evaluate on the second half (model is warmed up)
    warm = 4000
    normal_s = [surprises[i] for i in range(warm, len(stream)) if labels[i] == 0]
    anom_s = [surprises[i] for i in range(warm, len(stream)) if labels[i] == 1]
    mean_n = sum(normal_s) / len(normal_s)
    mean_a = sum(anom_s) / len(anom_s)
    print(f"  mean surprise  normal = {mean_n:.2f} bits   anomaly = {mean_a:.2f} bits")
    # threshold at midpoint; measure precision/recall
    thr = (mean_n + mean_a) / 2
    tp = sum(1 for i in range(warm, len(stream)) if labels[i] == 1 and surprises[i] > thr)
    fp = sum(1 for i in range(warm, len(stream)) if labels[i] == 0 and surprises[i] > thr)
    fn = sum(1 for i in range(warm, len(stream)) if labels[i] == 1 and surprises[i] <= thr)
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    print(f"  threshold {thr:.2f}: recall {recall*100:.0f}%, precision {precision*100:.0f}%")
    assertion(mean_a > mean_n * 2,
              "anomalous bytes are >2x more surprising - the codec's bits/byte "
              "is a ready-made anomaly score (no separate detector)")
    assertion(recall > 0.8,
              "surprise threshold flags >80% of injected anomalies")

    # ------------------------------------------------------------------
    print("\n(B) FTA LOCALIZER - factoring names the broken element (Test 3)")
    print("-" * 72)
    # a 'system snapshot' = a composite over N component readings. Each
    # component i with value v contributes prime PRIMES[i*K + v]. The
    # composite is the integrity fingerprint; factoring expected/actual
    # names the exact component that changed.
    N_COMP, K = 64, 16

    def snapshot(values):
        comp = 1
        for i, v in enumerate(values):
            comp *= PRIMES[i * K + v]
        return comp

    def localize(expected_vals, actual_comp):
        """Return the components whose prime is missing from actual_comp."""
        broken = []
        for i, v in enumerate(expected_vals):
            if actual_comp % PRIMES[i * K + v] != 0:
                broken.append(i)
        return broken

    detected = located = 0
    trials = 1000
    for _ in range(trials):
        vals = [rng.randrange(K) for _ in range(N_COMP)]
        good = snapshot(vals)
        # corrupt exactly one component
        idx = rng.randrange(N_COMP)
        bad_vals = list(vals)
        bad_vals[idx] = (vals[idx] + 1 + rng.randrange(K - 1)) % K
        bad = snapshot(bad_vals)
        if bad != good:
            detected += 1
        found = localize(vals, bad)
        if found == [idx]:
            located += 1
    print(f"  {trials} single-component corruptions:")
    print(f"    detected:  {detected}/{trials}")
    print(f"    localized: {located}/{trials} (named the exact broken component)")
    assertion(detected == trials and located == trials,
              "every corruption detected AND pinpointed by factoring - a "
              "checksum says 'broken'; FTA says WHICH (Test 3 as a monitor)")

    # ------------------------------------------------------------------
    print("\n(C) ROOT-CAUSE PROMOTER - correlated faults share a factor (Test 6)")
    print("-" * 72)
    # 300 snapshots; a hidden faulty sensor (component 42) intermittently
    # corrupts. The monitor collects the localized components and promotes
    # the most frequent - that promoted prime IS the root cause.
    faulty = 42
    blame = Counter()
    n_faults = 0
    for _ in range(300):
        vals = [rng.randrange(K) for _ in range(N_COMP)]
        good = snapshot(vals)
        bad_vals = list(vals)
        corrupted = []
        # the faulty sensor fails 60% of the time
        if rng.random() < 0.6:
            bad_vals[faulty] = (vals[faulty] + 1) % K
            corrupted.append(faulty)
            n_faults += 1
        # plus rare random noise elsewhere
        if rng.random() < 0.1:
            j = rng.randrange(N_COMP)
            bad_vals[j] = (vals[j] + 1) % K
            corrupted.append(j)
        bad = snapshot(bad_vals)
        for i in localize(vals, bad):
            blame[i] += 1
    root, hits = blame.most_common(1)[0]
    print(f"  blame histogram top-3: {blame.most_common(3)}")
    print(f"  promoted root cause: component {root} ({hits} incidents)")
    assertion(root == faulty,
              "the monitor fingers the faulty sensor as the common factor - "
              "automatic root-cause, the same promotion that resolved 525/547 "
              "checker anomalies (Test 6)")

    # ------------------------------------------------------------------
    print("\n(D) INVARIANT WATCHDOG - the five Zeno roles as live assertions (32)")
    print("-" * 72)
    # run a frame descent; the watchdog asserts positivity (security) and a
    # budget bound (janitor) every step. Inject a fault at a known step and
    # confirm it trips exactly there.
    def descend_with_watchdog(fault_at=None):
        width = Fraction(1)
        budget = Fraction(0)
        primes = (2,) + tuple(chain_primes(40))
        for step in range(60):
            p = primes[step % len(primes)]
            width = width / p
            if fault_at is not None and step == fault_at:
                width = -width                 # inject a corruption
            budget += width
            # WATCHDOG: continuous invariants
            if width <= 0:                     # SECURITY: positivity
                return ("TRIP-positivity", step)
            if budget >= 1:                    # JANITOR: bounded budget
                return ("TRIP-budget", step)
        return ("clean", 60)

    clean = descend_with_watchdog(fault_at=None)
    tripped = descend_with_watchdog(fault_at=12)
    print(f"  healthy descent:  {clean[0]} (ran all {clean[1]} steps)")
    print(f"  fault injected at step 12: watchdog {tripped[0]} at step {tripped[1]}")
    assertion(clean[0] == "clean",
              "healthy run passes every invariant (no false alarm)")
    assertion(tripped[0] == "TRIP-positivity" and tripped[1] == 12,
              "injected fault caught on the EXACT step by the positivity "
              "invariant (the Zeno security role as an always-on watchdog)")

    # ------------------------------------------------------------------
    header("RESULT - monitoring was latent the whole time")
    print(f"  (A) SURPRISE   codec bits/byte = anomaly score "
          f"(normal {mean_n:.1f} vs anomaly {mean_a:.1f} bits, {recall*100:.0f}% recall)")
    print(f"  (B) LOCALIZE   FTA factoring pinpoints the broken element "
          f"({located}/{trials})")
    print(f"  (C) ROOT CAUSE promotion fingers the faulty sensor (component {root})")
    print(f"  (D) WATCHDOG   Zeno invariants trip on the exact faulty step")
    print()
    print("  None of these needed new machinery - they are existing primitives")
    print("  pointed at WATCHING instead of doing:")
    print("    compression surprise  -> change / intrusion / novelty detection")
    print("    FTA factorization     -> exact fault localization (not just CRC)")
    print("    promotion clustering  -> root-cause analysis with no training")
    print("    Zeno invariants       -> runtime verification / always-on asserts")
    print()
    print("  A system that compresses, addresses, promotes, and descends is")
    print("  ALSO monitoring itself for free: every byte carries a surprise")
    print("  score, every state carries an integrity fingerprint that names")
    print("  its own faults, and every descent checks its own invariants.")


if __name__ == "__main__":
    main()
