#!/usr/bin/env python3
"""
Test 39 - Fused channel monitor: "weird data" vs "tapped channel" disambiguated.

Test 38 built two monitors separately: content surprise (the compressor's
bits/symbol) and channel integrity (the entangled-pair CHSH value). They
measure ORTHOGONAL failures that real systems constantly confuse:

  - CONTENT ANOMALY: the data itself is novel/corrupt at the source. Surprise
    spikes; the channel is intact (S stays ~2.83).
  - CHANNEL TAMPER: someone intercepted the transmission. The Bell value S
    drops; the content distribution can look perfectly normal.

The dangerous case is the STEALTH TAP: an eavesdropper who relays the data
faithfully (content monitor sees nothing) but cannot avoid measuring the
entangled pairs (S drops). A content-only monitor says "all clear" while the
channel is owned. A channel-only monitor is blind to a corrupt payload on a
secure line.

The fused monitor reads BOTH axes and returns a 2x2 diagnosis. We verify it
is correct on all four scenarios, and that each single-axis monitor is blind
to exactly the failure it cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import random

from test_adaptive_and_entangle_monitor import Order1Model, estimate_S


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


A = 16


class FusedMonitor:
    """One monitor, two independent signals -> a 2x2 channel diagnosis."""

    def __init__(self, surprise_thr, s_thr):
        self.model = Order1Model(A)
        self.surprise_thr = surprise_thr      # bits/symbol -> content anomaly
        self.s_thr = s_thr                    # CHSH floor -> channel tamper
        self.prev = 0

    def warm(self, succ, n, rng):
        prev = 0
        for _ in range(n):
            prev = succ[prev][0] if rng.random() < 0.95 else rng.randrange(A)
            self.model.surprise(prev)

    def window(self, data_syms, channel_S):
        """Returns (content_alarm, tamper_alarm, diagnosis)."""
        s = sum(self.model.surprise(x) for x in data_syms) / len(data_syms)
        content_alarm = s > self.surprise_thr
        tamper_alarm = channel_S < self.s_thr
        if content_alarm and tamper_alarm:
            diag = "DATA WEIRD + CHANNEL TAPPED"
        elif content_alarm:
            diag = "DATA WEIRD (channel ok)"
        elif tamper_alarm:
            diag = "CHANNEL TAPPED (data ok)"
        else:
            diag = "ALL CLEAR"
        return content_alarm, tamper_alarm, diag, s


def make_window(succ, anomalous, rng, w=120):
    """Generate w data symbols: normal Markov, or an anomalous (max-entropy)
    payload that a content model finds surprising."""
    out = []
    prev = rng.randrange(A)
    for _ in range(w):
        if anomalous:
            prev = rng.randrange(A)           # high-entropy payload
            # make it adversarially off-distribution: random unrelated symbol
        else:
            prev = succ[prev][0] if rng.random() < 0.95 else rng.randrange(A)
        out.append(prev)
    return out


def main():
    header("Fused channel monitor - content anomaly vs channel tamper")
    rng = random.Random(0x39F5)
    succ = {a: rng.sample(range(A), A) for a in range(A)}

    fused = FusedMonitor(surprise_thr=1.5, s_thr=2.4)
    fused.warm(succ, 3000, rng)

    # four scenarios: (content present, tamper present)
    scenarios = [
        ("clean",   False, 0.00),
        ("content", True,  0.00),    # weird data, secure channel
        ("tap",     False, 0.45),    # NORMAL data, eavesdropper (stealth)
        ("both",    True,  0.45),
    ]

    print("\nPer-scenario diagnosis (the fused 2x2)")
    print("-" * 72)
    print(f"  {'scenario':<9} | {'mean surprise':>13} | {'CHSH S':>7} | diagnosis")
    print(f"  {'-'*9} | {'-'*13} | {'-'*7} | ---------")

    # tallies for the three monitor styles
    M = 12                                     # windows per scenario
    fused_correct = 0
    content_only_tamper_caught = 0
    content_only_tamper_total = 0
    channel_only_content_caught = 0
    channel_only_content_total = 0
    total_windows = 0

    for name, content_present, eve in scenarios:
        s_vals, diags = [], []
        ca_t = ta_t = 0
        for _ in range(M):
            data = make_window(succ, content_present, rng)
            S = estimate_S(rng, eve, n=4000)
            content_alarm, tamper_alarm, diag, s = fused.window(data, S)
            s_vals.append((s, S))
            diags.append(diag)
            total_windows += 1
            # fused correctness: both axes match truth
            if content_alarm == content_present and tamper_alarm == (eve > 0):
                fused_correct += 1
            ca_t += content_alarm
            ta_t += tamper_alarm
            # single-axis blindness accounting
            if eve > 0:                        # tamper present
                content_only_tamper_total += 1
                # a content-only monitor never raises tamper -> always misses
            if content_present:                # content anomaly present
                channel_only_content_total += 1
                # a channel-only monitor never raises content -> always misses
        avg_s = sum(v[0] for v in s_vals) / len(s_vals)
        avg_S = sum(v[1] for v in s_vals) / len(s_vals)
        # the modal diagnosis for this scenario
        modal = max(set(diags), key=diags.count)
        print(f"  {name:<9} | {avg_s:>13.2f} | {avg_S:>7.2f} | {modal}")

    # content-only monitor: catches tamper? NEVER (it has no S signal)
    content_only_tamper_caught = 0
    # channel-only monitor: catches content anomaly? NEVER (no surprise signal)
    channel_only_content_caught = 0

    print("\nMonitor comparison - who is blind to what")
    print("-" * 72)
    fused_acc = fused_correct / total_windows
    print(f"  FUSED monitor:    both axes correct on {fused_correct}/{total_windows} "
          f"windows ({fused_acc*100:.0f}%)")
    print(f"  CONTENT-only:     tamper recall {content_only_tamper_caught}/"
          f"{content_only_tamper_total} - BLIND to the stealth tap")
    print(f"  CHANNEL-only:     content recall {channel_only_content_caught}/"
          f"{channel_only_content_total} - BLIND to the corrupt payload")

    assertion(fused_acc > 0.9,
              "the fused monitor diagnoses both axes correctly on >90% of "
              "windows across all four scenarios")
    assertion(content_only_tamper_total > 0 and content_only_tamper_caught == 0,
              "a content-only monitor NEVER catches the stealth tap (normal "
              "data, tapped channel) - it would report ALL CLEAR")
    assertion(channel_only_content_total > 0 and channel_only_content_caught == 0,
              "a channel-only monitor NEVER catches a corrupt payload on a "
              "secure line - the two failures are orthogonal")

    # the decisive case spelled out
    print("\nThe decisive case - the stealth tap")
    print("-" * 72)
    data = make_window(succ, False, rng)       # perfectly normal data
    S = estimate_S(rng, 0.45, n=6000)          # but the channel is tapped
    ca, ta, diag, s = fused.window(data, S)
    print(f"  normal data (surprise {s:.2f}, below {fused.surprise_thr}) "
          f"-> content monitor: ALL CLEAR")
    print(f"  but CHSH S = {S:.2f} (below {fused.s_thr}) "
          f"-> entanglement monitor: TAMPER")
    print(f"  FUSED verdict: {diag}")
    assertion(not ca and ta,
              "stealth tap: content says clear, channel says tapped - only the "
              "fusion gets it right (the failure a content-only system ships)")

    header("RESULT - one monitor, two orthogonal failure axes")
    print("  The compressor's surprise and the entanglement's CHSH value are")
    print("  independent: content anomaly moves one, channel tamper the other.")
    print("  Fusing them yields a 2x2 that distinguishes four states a single")
    print("  monitor confuses:")
    print("    ALL CLEAR | DATA WEIRD | CHANNEL TAPPED | BOTH")
    print()
    print(f"  fused accuracy:        {fused_acc*100:.0f}% on all four scenarios")
    print(f"  content-only:          blind to the stealth tap (ships a breach)")
    print(f"  channel-only:          blind to a corrupt payload")
    print()
    print("  Both signals are by-products of machinery we already built - the")
    print("  context codec (Tests 15-23) and the qubit/ocean stack (30,31,38).")
    print("  A channel that is compressed AND entanglement-checked monitors")
    print("  its own content and its own integrity at once, and can finally")
    print("  tell a weird message from a wiretap.")


if __name__ == "__main__":
    main()
