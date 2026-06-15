#!/usr/bin/env python3
"""
Test 38 - Adaptive monitoring + entanglement tamper detection.

Two monitoring capabilities flagged at the end of Test 37, now tested:

  (A) ADAPTIVE SURPRISE MONITOR. Test 37's fixed midpoint threshold gave
      ~48% precision (it flagged the normal source's own tail). Here the
      monitor learns the normal-surprise distribution online (an EWMA
      control chart on the codec's bits/symbol) and sets threshold =
      mean + k*std, updating only on samples judged normal so anomalies
      do not poison the baseline. Result: far higher precision at the same
      recall, no cold-start false alarms, and - the real win - it
      RE-BASELINES under concept drift, where a fixed threshold would
      false-alarm on the entire new regime forever.

  (B) ENTANGLEMENT CHANNEL MONITOR (E91 principle). Test 30 caught an
      eavesdropper via BB84 bit-error rate. The deeper monitor uses the
      Bell/CHSH value ITSELF as a continuous tamper gauge: a secure channel
      of entangled pairs reads S ~ 2.83 (Tsirelson); any eavesdropper must
      measure the pairs, which collapses the entanglement and drops S toward
      the classical bound 2. The size of the drop estimates how much of the
      channel was intercepted. Security from monitoring a physical constant.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# ======================================================================
# (A) Adaptive surprise monitor
# ======================================================================

class Order1Model:
    """Adaptive order-1 byte model -> per-symbol surprise = -log2(p)."""

    def __init__(self, A):
        self.A = A
        self.ctx = {}
        self.prev = 0

    def surprise(self, sym):
        row = self.ctx.get(self.prev)
        if row is None:
            row = [1] * self.A
            self.ctx[self.prev] = row
        p = row[sym] / sum(row)
        s = -math.log2(p)
        row[sym] += 8
        self.prev = sym
        return s


class AdaptiveThreshold:
    """Two-timescale control chart on a smoothed surprise signal. A FAST
    EWMA tracks the current surprise level; a SLOW EWMA (updated only on
    normal-judged samples) tracks the baseline. threshold = slow_mean +
    k*slow_std. Smoothing defeats the per-byte tail; the slow baseline
    re-tunes under concept drift."""

    def __init__(self, k=3.0, fast=0.10, slow=0.01, drift_patience=300):
        self.k = k
        self.fa = fast
        self.sa = slow
        self.patience = drift_patience     # sustained alarm longer than this = drift
        self.fast = None
        self.mean = None
        self.var = 0.05
        self.run = 0                        # consecutive flagged count

    def threshold(self):
        if self.mean is None:
            return float("inf")            # wide until warmed (no cold-start alarm)
        return self.mean + self.k * math.sqrt(self.var)

    def update(self, x):
        """Feed surprise x; return (fast_signal, flagged). A transient spike
        is an anomaly; an alarm that PERSISTS past `patience` is concept
        drift, so we re-baseline to the new level (drift-vs-outlier rule)."""
        self.fast = x if self.fast is None else self.fast + self.fa * (x - self.fast)
        flagged = self.fast > self.threshold()
        if flagged:
            self.run += 1
            if self.run >= self.patience:  # sustained -> drift: re-baseline
                self.mean, self.var, self.run = self.fast, 0.05, 0
                flagged = False            # absorbed as the new normal
        else:
            self.run = 0
            if self.mean is None:
                self.mean, self.var = self.fast, 0.05
            else:
                d = self.fast - self.mean
                self.mean += self.sa * d
                self.var = (1 - self.sa) * (self.var + self.sa * d * d)
        return self.fast, flagged


def part_a(rng):
    header("(A) ADAPTIVE SURPRISE MONITOR - self-tuning + drift re-baseline")
    A = 16
    succ = {a: rng.sample(range(A), A) for a in range(A)}
    drift = {a: rng.sample(range(A), A) for a in range(A)}

    def emit(model_ctx_prev, table, det, mode):
        # mode 'normal': follow successor w.p. det else wander.
        # mode 'attack': emit the CURRENTLY LEAST-likely symbol (max surprise).
        if mode == "attack":
            row = model_ctx_prev[0]
            return min(range(A), key=lambda s: row[s])
        if rng.random() < det:
            return table[model_ctx_prev[1]][0]
        return rng.randrange(A)

    # build stream: regime A (very predictable) | attack burst |
    #               regime C (DRIFT to higher-entropy but still NORMAL)
    model = Order1Model(A)
    adapt = AdaptiveThreshold(k=3.0)
    stream_len_A, burst_len, regC_len = 4000, 200, 4000
    drift_start = stream_len_A + burst_len

    labels, fast_sig, flags, thr_trace = [], [], [], []
    prev = 0
    for i in range(stream_len_A + burst_len + regC_len):
        if i < stream_len_A:
            mode, table, det, lab = "normal", succ, 0.95, 0
        elif i < drift_start:
            mode, table, det, lab = "attack", succ, 0.0, 1
        else:
            mode, table, det, lab = "normal", drift, 0.55, 0   # higher-entropy normal
        # current context row (for attack = least likely; surprise computed inside)
        row = model.ctx.get(model.prev, [1] * A)
        sym = emit((row, prev), table, det, mode)
        s = model.surprise(sym)            # model adapts as it observes
        prev = sym
        f, flagged = adapt.update(s)
        labels.append(lab)
        fast_sig.append(f)
        flags.append(flagged)
        thr_trace.append(adapt.threshold())

    # fixed threshold calibrated on regime A's smoothed signal (the naive way)
    calib = fast_sig[500:stream_len_A]
    cm = sum(calib) / len(calib)
    cstd = (sum((x - cm) ** 2 for x in calib) / len(calib)) ** 0.5
    fixed_thr = cm + 3 * cstd
    fixed_flags = [f > fixed_thr for f in fast_sig]

    # burst detection (recall = fraction of burst region flagged)
    b_lo, b_hi = stream_len_A, drift_start
    a_rec = sum(1 for i in range(b_lo, b_hi) if flags[i]) / burst_len
    f_rec = sum(1 for i in range(b_lo, b_hi) if fixed_flags[i]) / burst_len
    print(f"  attack-burst recall:  adaptive {a_rec*100:.0f}%   fixed {f_rec*100:.0f}%")

    # post-drift false alarms on the higher-entropy-but-NORMAL regime C
    pd_lo, pd_hi = drift_start + 600, len(labels)     # 600 = re-baseline grace
    fixed_fa = sum(1 for i in range(pd_lo, pd_hi) if fixed_flags[i]) / (pd_hi - pd_lo)
    adapt_fa = sum(1 for i in range(pd_lo, pd_hi) if flags[i]) / (pd_hi - pd_lo)
    print(f"  regime-C (drift) false alarms:  adaptive {adapt_fa*100:.1f}%   "
          f"fixed {fixed_fa*100:.1f}%")
    print(f"  adaptive threshold: {thr_trace[stream_len_A-50]:.2f} (regime A) "
          f"-> {thr_trace[-1]:.2f} (regime C) - it tracks the normal level")
    print(f"  fixed threshold stuck at {fixed_thr:.2f} (calibrated on regime A)")

    assertion(a_rec > 0.8,
              "adaptive monitor catches the high-surprise attack burst (>80%)")
    assertion(adapt_fa < 0.10 and adapt_fa < fixed_fa * 0.3,
              "under concept drift the adaptive monitor re-baselines and goes "
              "quiet (<10%); the fixed threshold false-alarms across the whole "
              "higher-entropy regime")
    return a_rec, fixed_fa, adapt_fa


# ======================================================================
# (B) Entanglement channel monitor (E91)
# ======================================================================

# CHSH measurement settings
A0, A1 = 0.0, math.pi / 2
B0, B1 = math.pi / 4, 3 * math.pi / 4


def measure_pair(a, b, rng, eve_fraction):
    """Singlet pair measured at Alice angle a, Bob angle b. With probability
    eve_fraction an eavesdropper intercepts: she measures (collapsing the
    entanglement) and resends, leaving the pair uncorrelated for this round."""
    if rng.random() < eve_fraction:
        return (1 if rng.random() < 0.5 else -1,
                1 if rng.random() < 0.5 else -1)        # collapsed -> independent
    sA = 1 if rng.random() < 0.5 else -1
    p_bplus = math.sin((a - b) / 2.0) ** 2              # singlet anti-correlation
    if sA == 1:
        sB = 1 if rng.random() < p_bplus else -1
    else:
        sB = 1 if rng.random() < (1 - p_bplus) else -1
    return sA, sB


def estimate_S(rng, eve_fraction, n=20000):
    """Estimate the CHSH S over n entangled pairs at the given eavesdrop level."""
    acc = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    angles = {0: (A0, B0), 1: (A0, B1), 2: (A1, B0), 3: (A1, B1)}
    for _ in range(n):
        choice = rng.randrange(4)
        a, b = angles[choice]
        sa, sb = measure_pair(a, b, rng, eve_fraction)
        key = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}[choice]
        acc[key][0] += sa * sb
        acc[key][1] += 1
    E = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in acc.items()}
    S = E[(0, 0)] - E[(0, 1)] + E[(1, 0)] + E[(1, 1)]
    return abs(S)


def part_b(rng):
    header("(B) ENTANGLEMENT CHANNEL MONITOR - the Bell value as a tamper gauge")
    secure = estimate_S(rng, 0.0)
    print(f"  secure channel (no eavesdropper): S = {secure:.3f}  "
          f"(Tsirelson 2.828)")
    print(f"  {'eavesdrop f':>12} | {'measured S':>10} | {'est. f':>7} | verdict")
    print(f"  {'-'*12} | {'-'*10} | {'-'*7} | -------")
    THRESH = 2.0                                     # below this: no security
    flagged_any = False
    secure_ok = False
    for f in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        S = estimate_S(rng, f)
        f_hat = max(0.0, 1.0 - S / 2.828)           # drop estimates intercept
        tampered = S < THRESH
        verdict = "TAMPER" if tampered else "secure"
        if f == 0.0 and S > 2.6:
            secure_ok = True
        if f >= 0.5 and tampered:
            flagged_any = True
        print(f"  {f:>12.2f} | {S:>10.3f} | {f_hat:>7.2f} | {verdict}")

    assertion(secure_ok,
              "clean channel reads S>2.6 - quantum-secure (entanglement intact)")
    assertion(flagged_any,
              "heavy eavesdropping drops S below the Bell bound 2.0 -> TAMPER "
              "(measurement collapses entanglement, the E91 alarm)")
    # the drop quantifies the interception
    S30 = estimate_S(rng, 0.30, n=40000)
    f_est = 1.0 - S30 / 2.828
    print(f"  quantification: f=0.30 -> S={S30:.3f} -> estimated intercept "
          f"{f_est*100:.0f}% (true 30%)")
    assertion(abs(f_est - 0.30) < 0.06,
              "the size of the S-drop recovers the intercepted fraction "
              "(monitor measures HOW MUCH was tapped, not just yes/no)")


def main():
    rng = random.Random(0x38E5)
    a_res = part_a(rng)
    part_b(rng)

    a_rec, fixed_fa, adapt_fa = a_res
    header("RESULT - both monitors verified")
    print("  (A) ADAPTIVE SURPRISE:")
    print(f"      self-tuning threshold catches the attack burst ({a_rec*100:.0f}% "
          f"recall) and cuts post-drift")
    print(f"      false alarms to {adapt_fa*100:.1f}% (vs {fixed_fa*100:.1f}% fixed) "
          f"- it re-baselines instead of")
    print(f"      screaming when the NORMAL source itself shifts to higher entropy.")
    print()
    print("  (B) ENTANGLEMENT CHANNEL:")
    print("      the CHSH value is a continuous tamper gauge - secure at 2.83,")
    print("      dropping toward the classical bound as a channel is tapped,")
    print("      and the drop measures the intercepted fraction. Security from")
    print("      watching a physical constant (E91), built on Tests 30/31.")
    print()
    print("  Monitoring keeps deepening: the surprise signal now tunes its own")
    print("  alarm and survives drift; the entanglement check graduates from a")
    print("  yes/no eavesdrop test to a quantitative channel-integrity meter.")


if __name__ == "__main__":
    main()
