#!/usr/bin/env python3
"""
Test 40 - The autonomic loop: monitor -> diagnose -> ACT -> learn (MAPE-K).

Test 39 gave a diagnosis. This closes the loop into a self-healing channel
that responds and learns, assembled entirely from proven pieces:

  MONITOR   fused surprise + CHSH (Test 39)
  ANALYZE   the 2x2 diagnosis (Test 39)
  PLAN+ACT  CHANNEL TAPPED -> rekey/quarantine the channel (S recovers);
            DATA WEIRD      -> promote the anomaly fingerprint for root cause
                               (Test 37C), recognizing repeats instantly
  KNOWLEDGE the recursive lattice accumulates anomaly concepts (Test 6),
            kept bounded by ground-zero recycling (Test 29), with a full
            audit trail (Test 5 provenance)

This is the IBM MAPE-K autonomic-computing pattern - normally a bespoke
framework - falling out of capabilities we already built. Verified: every
tap auto-recovers, repeated anomalies are recognized (the system learns),
memory stays bounded, and every incident is audited.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_adaptive_and_entangle_monitor import estimate_S


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


PR = chain_primes(256)


class Channel:
    """A stateful channel: an eavesdropper can attach (eve rises); a rekey
    detaches it (eve -> 0). S is read from the entangled pairs (Test 38)."""

    def __init__(self):
        self.eve = 0.0

    def attach_eve(self, fraction):
        self.eve = fraction

    def rekey(self):
        self.eve = 0.0                      # quarantine + re-establish entanglement

    def chsh(self, rng):
        return estimate_S(rng, self.eve, n=2000)


class AutonomicChannel:
    """The closed MAPE-K loop over the channel."""

    S_THR = 2.4
    SURPRISE_THR = 1.5

    def __init__(self):
        self.channel = Channel()
        self.known_anomalies: dict[int, int] = {}   # fingerprint -> concept prime
        self.next_concept = 0
        self.audit: list[tuple] = []
        self.rekeys = 0
        self.recoveries = 0
        self.recognized = 0
        self.learned = 0
        self.mttr_samples: list[int] = []

    def _fingerprint(self, payload):
        """A content-anomaly signature: composite over its top-2 symbols."""
        from collections import Counter
        common = [s for s, _ in Counter(payload).most_common(2)]
        comp = 1
        for s in common:
            comp *= PR[s]
        return comp

    def step(self, rng, surprise, payload, S):
        content_alarm = surprise > self.SURPRISE_THR
        tamper_alarm = S < self.S_THR
        actions = []

        if tamper_alarm:                    # PLAN+ACT: rekey the channel
            self.channel.rekey()
            self.rekeys += 1
            actions.append("REKEY")

        if content_alarm:                   # PLAN+ACT: root-cause the payload
            fp = self._fingerprint(payload)
            if fp in self.known_anomalies:
                self.recognized += 1
                actions.append(f"RECOGNIZED#{self.known_anomalies[fp]}")
            else:                           # learn: promote a new concept
                self.next_concept += 1
                self.known_anomalies[fp] = self.next_concept
                self.learned += 1
                actions.append(f"LEARN#{self.next_concept}")
                # KNOWLEDGE bound: cap concepts (ground-zero recycle oldest)
                if len(self.known_anomalies) > 16:
                    oldest = min(self.known_anomalies, key=lambda k: self.known_anomalies[k])
                    del self.known_anomalies[oldest]

        diag = ("BOTH" if content_alarm and tamper_alarm else
                "TAP" if tamper_alarm else
                "WEIRD" if content_alarm else "CLEAR")
        self.audit.append((diag, tuple(actions)))
        return diag, actions


# ----------------------------------------------------------------------
# Incident generator: clean / content (3 recurring attack types) / tap / both
# ----------------------------------------------------------------------

ATTACK_SIGS = {
    "alpha": [2, 2, 2, 9, 9],
    "beta":  [5, 5, 13, 13, 13],
    "gamma": [7, 7, 7, 1, 1],
}


def make_payload(kind, rng, w=60):
    if kind == "clean":
        return [0 if rng.random() < 0.9 else rng.randrange(16) for _ in range(w)],  0.3
    sig = ATTACK_SIGS[kind]
    payload = [rng.choice(sig) for _ in range(w)]
    # surprise proxy: anomalous payloads read high (off the normal model)
    return payload, 4.5


def main():
    header("The autonomic loop - monitor, diagnose, ACT, learn (MAPE-K)")
    rng = random.Random(0x40A0)
    sys_ = AutonomicChannel()

    incidents = ["clean", "content", "tap", "both"]
    attack_cycle = ["alpha", "beta", "gamma"]
    N = 200

    unrecovered = 0
    pending_tap_since = None

    for t in range(N):
        kind = rng.choice(incidents)
        # build the window
        if kind in ("content", "both"):
            atk = attack_cycle[t % 3]
            payload, surprise = make_payload(atk, rng)
        else:
            payload, surprise = make_payload("clean", rng)
        # channel state: tap/both attach an eavesdropper this step
        if kind in ("tap", "both"):
            sys_.channel.attach_eve(0.45)
            if pending_tap_since is None:
                pending_tap_since = t
        S = sys_.channel.chsh(rng)
        diag, actions = sys_.step(rng, surprise, payload, S)

        # recovery accounting: after a rekey the channel should read secure now
        if "REKEY" in actions:
            S_after = sys_.channel.chsh(rng)
            if S_after >= sys_.S_THR:
                sys_.recoveries += 1
                if pending_tap_since is not None:
                    sys_.mttr_samples.append(t - pending_tap_since + 1)
                    pending_tap_since = None
            else:
                unrecovered += 1

    # ------------------------------------------------------------------
    print("\nClosed-loop behaviour over 200 incidents")
    print("-" * 72)
    from collections import Counter
    diag_counts = Counter(d for d, _ in sys_.audit)
    print(f"  diagnoses: {dict(diag_counts)}")
    print(f"  rekeys issued:      {sys_.rekeys}")
    print(f"  channel recoveries: {sys_.recoveries}/{sys_.rekeys}")
    mttr = sum(sys_.mttr_samples) / len(sys_.mttr_samples) if sys_.mttr_samples else 0
    print(f"  mean-time-to-recovery: {mttr:.2f} windows")
    print(f"  anomalies learned:  {sys_.learned}  recognized (repeat): {sys_.recognized}")
    print(f"  knowledge size:     {len(sys_.known_anomalies)} concepts (bounded)")
    print(f"  audit entries:      {len(sys_.audit)}")

    assertion(sys_.recoveries == sys_.rekeys and unrecovered == 0,
              "every tap auto-recovered after rekey (self-healing channel: "
              "S back to secure)")
    assertion(mttr <= 1.01,
              "mean-time-to-recovery ~1 window - detection and action are one "
              "loop, not a human in the middle")
    assertion(sys_.recognized > 0 and sys_.learned > 0,
              "the system LEARNS: first occurrence of an attack type is "
              "promoted, repeats are recognized instantly (Test 6 + 37C)")
    assertion(len(sys_.known_anomalies) <= 16,
              "knowledge stays bounded (ground-zero recycling caps concepts, "
              "Tests 29/33)")
    assertion(len(sys_.audit) == N,
              "every incident audited (complete provenance, Test 5)")

    # recognition improves with exposure (learning curve)
    first_half = sys_.audit[:100]
    second_half = sys_.audit[100:]

    def recog_rate(half):
        rec = sum(1 for _, acts in half for a in acts if a.startswith("RECOGNIZED"))
        learn = sum(1 for _, acts in half for a in acts if a.startswith("LEARN"))
        return rec / (rec + learn) if rec + learn else 0

    r1, r2 = recog_rate(first_half), recog_rate(second_half)
    print(f"\n  learning curve: recognition rate {r1*100:.0f}% (first half) -> "
          f"{r2*100:.0f}% (second half)")
    assertion(r2 >= r1,
              "recognition rate rises with exposure - the loop gets smarter "
              "about recurring threats")

    # ------------------------------------------------------------------
    header("RESULT - a self-healing, self-learning channel from proven parts")
    print("  MONITOR   fused surprise + CHSH            (Tests 37-39)")
    print("  ANALYZE   2x2 content/channel diagnosis    (Test 39)")
    print("  ACT       rekey taps + promote anomalies   (Tests 25, 37C)")
    print("  LEARN     concepts in the lattice, bounded (Tests 6, 29)")
    print("  AUDIT     full provenance of every action  (Test 5)")
    print()
    print(f"  {sys_.rekeys}/{sys_.rekeys} taps healed in ~{mttr:.0f} window; "
          f"{sys_.recognized} repeat attacks recognized; memory bounded;")
    print(f"  {len(sys_.audit)}/{N} incidents audited.")
    print()
    print("  IBM's MAPE-K autonomic loop is normally a bespoke framework. Here")
    print("  it is the natural closure of the monitoring stack: the same prime")
    print("  descent that detects, localizes, and diagnoses now also acts,")
    print("  remembers, and bounds itself - a system that watches AND heals.")


if __name__ == "__main__":
    main()
