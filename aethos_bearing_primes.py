#!/usr/bin/env python3
"""
Prime-address anomaly classifier: which sensor events are REAL degradation and
which are noise/jitter -- decided by recurrence in prime space, not amplitude.

Every bearing sits, each snapshot, in one of 4 electron quadrants = the signs of
(own-level z, coupling-residual z):

    (+,+) rising & ABOVE prediction   <- the SOURCE signature
    (-,-) below  & BELOW prediction   <- a DRAGGED victim (propagation)
    (+,-) rising & below              (-,+) below & above   (mixed)

Each (bearing, quadrant) that crosses a SENSITIVE gate gets a prime address. A
snapshot's signature is the PRODUCT of the active primes (FTA: it factors back to
exactly who is in what state -- glass-box). Then:

  * noise / jitter -> hits a prime ONCE and never returns: an "empty" address,
    scattered through unused prime space. Discarded.
  * real degradation -> keeps hitting the SAME prime: mass accumulates and GROWS
    there. The fault lives at an address.

Because the discriminator is recurrence, not magnitude, we can run at a sensitive
3-sigma gate -- where a raw threshold monitor drowns in false alarms -- and still
keep only the real degrade. Append-only: a never-seen state is just the next
prime; the vast unused address space is the confirmed-healthy manifold.

    python aethos_bearing_primes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from core.primes import chain_primes
except Exception:                                   # self-contained fallback
    def chain_primes(n):
        ps, c = [], 2
        while len(ps) < n:
            if all(c % p for p in ps if p * p <= c):
                ps.append(c)
            c += 1
        return ps

GATE = 3.0          # sensitive: let noise in, then reject it by recurrence
MIN_MASS = 4        # fewer fires than this = one-off jitter
RUN_MIN = 4         # a real degrade holds for >= this many consecutive snapshots
QUAD = {(True, True): "rising & above pred", (False, True): "quiet & above pred",
        (True, False): "rising & below pred", (False, False): "below & below pred"}


class PrimeAddressBook:
    """append-only token -> prime, with per-prime recurrence mass over time."""

    def __init__(self):
        self._pool = chain_primes(64)
        self.prime = {}                 # token -> prime
        self.fires = {}                 # prime -> [snapshot indices]
        self.mag = {}                   # prime -> [|z| at each fire]

    def hit(self, token, t, z):
        p = self.prime.get(token)
        if p is None:
            p = self._pool[len(self.prime)]
            self.prime[token] = p
            self.fires[p], self.mag[p] = [], []
        self.fires[p].append(t)
        self.mag[p].append(abs(z))
        return p

    def longest_run(self, p):
        idx = self.fires[p]
        best = run = 1
        for a, b in zip(idx, idx[1:]):
            run = run + 1 if b == a + 1 else 1
            best = max(best, run)
        return best if idx else 0

    def run_start(self, p):
        """index where the LONGEST sustained run begins (true degrade onset)."""
        idx = self.fires[p]
        best = run = 1
        start = bs = idx[0]
        for a, b in zip(idx, idx[1:]):
            if b == a + 1:
                run += 1
            else:
                run, start = 1, b
            if run > best:
                best, bs = run, start
        return bs

    def role(self, p, T, quad):
        """jitter | source | propagation | transient -- quadrant gives the ROLE,
        recurrence gives reality. A dragged victim is NOT a degrade source."""
        if len(self.fires[p]) < MIN_MASS or self.longest_run(p) < RUN_MIN:
            return "jitter"                      # one-off / flickering -> empty prime
        qm, qc = quad                            # m>0 (rising), c>0 (above prediction)
        to_end = self.fires[p][-1] >= 0.85 * T
        if qm and qc and self.growth(p) > 1.3 and to_end:
            return "source"                      # rising & above & growing = the fault
        if (not qm) and (not qc) and to_end:
            return "propagation"                 # below & below = dragged by the source
        return "transient"

    def growth(self, p):
        m = self.mag[p]
        if len(m) < MIN_MASS:
            return float("nan")
        h = len(m) // 2
        a, b = np.mean(m[:h]), np.mean(m[h:])
        return b / max(a, 1e-6)


def main():
    src = Path(__file__).resolve().parent / "bearing_run_2nd_test.json"
    if not src.exists():
        print("run aethos_bearing_monitor.py first to produce bearing_run_2nd_test.json")
        return
    d = json.loads(src.read_text())
    hours = np.array(d["hours"])
    mz = np.array(d["marginal_z"])        # (T, B) own-level z
    cz = np.array(d["coupling_z"])        # (T, B) coupling-residual z
    truth = d.get("truth", 0)
    T, B = mz.shape

    book = PrimeAddressBook()
    raw_alarm_snaps = 0                   # what a naive 3-sigma monitor would flag
    for t in range(T):
        fired_any = False
        for b in range(B):
            m, c = mz[t, b], cz[t, b]
            if max(abs(m), abs(c)) <= GATE:
                continue
            fired_any = True
            token = (b, m > 0, c > 0)     # bearing + electron quadrant
            book.hit(token, t, c if abs(c) >= abs(m) else m)
        raw_alarm_snaps += fired_any

    # tally each allocated prime
    rows = []
    order = {"source": 0, "propagation": 1, "transient": 2, "jitter": 3}
    for token, p in book.prime.items():
        b, qm, qc = token
        rows.append({
            "bearing": b + 1, "state": QUAD[(qm, qc)], "prime": p,
            "fires": len(book.fires[p]), "run": book.longest_run(p),
            "onset_h": float(hours[book.run_start(p)]),
            "last_h": float(hours[book.fires[p][-1]]),
            "growth": book.growth(p), "kind": book.role(p, T, (qm, qc)),
        })
    rows.sort(key=lambda r: (order[r["kind"]], -r["fires"]))

    print(f"IMS 2nd_test: {T} snapshots, gate {GATE:.0f}-sigma (sensitive)\n")
    print(f"a raw {GATE:.0f}-sigma threshold monitor fires on "
          f"{raw_alarm_snaps}/{T} snapshots = {100*raw_alarm_snaps/T:.0f}% of the run "
          f"(an unusable alarm storm).")
    print(f"the lattice collapses that storm onto {len(book.prime)} prime addresses "
          f"(of {B*4} possible; {B*4-len(book.prime)} stay empty = states never entered).\n")

    print(f"  {'addr':>4} {'bearing':<8} {'state':<20} {'fires':>5} {'run':>4} "
          f"{'onset':>7} {'grow':>6}  verdict")
    TAG = {"source": "REAL DEGRADE (source)", "propagation": "propagation (dragged)",
           "transient": "transient", "jitter": "noise / jitter"}
    for r in rows:
        g = f"{r['growth']:>4.0f}x" if r["growth"] == r["growth"] else "   -"
        print(f"  p={r['prime']:>2} Bearing {r['bearing']:<1} {r['state']:<20} "
              f"{r['fires']:>5} {r['run']:>4} {r['onset_h']:>6.0f}h "
              f"{g:>6}  {TAG[r['kind']]}")

    src = sorted([r for r in rows if r["kind"] == "source"],
                 key=lambda r: (r["onset_h"], -r["fires"]))
    victims = sorted({r["bearing"] for r in rows if r["kind"] == "propagation"})
    jitter = [r for r in rows if r["kind"] == "jitter"]
    print(f"\n  noise/jitter: {len(jitter)} one-off addresses "
          f"({sum(r['fires'] for r in jitter)} fires) that never recurred -- discarded.")
    if src:
        s = src[0]
        hit = "HIT" if s["bearing"] - 1 == truth else "MISS"
        print(f"  REAL degrade: Bearing {s['bearing']} lives at prime {s['prime']} "
              f"(rising & above prediction), recurring from hour {s['onset_h']:.0f}, "
              f"grew {s['growth']:.0f}x.")
        print(f"     => source = Bearing {s['bearing']} [{hit} vs truth Bearing {truth+1}]"
              + (f"; it dragged Bearings {victims} (propagation)" if victims else ""))
        if len(src) > 1:
            extra = ", ".join(f"Bearing {r['bearing']} (from {r['onset_h']:.0f}h)"
                              for r in src[1:])
            print(f"     secondary rising signatures (weaker, not ground-truth-documented): {extra}")

    # dump for visualization: per-prime fire timelines + kinds
    out = Path(__file__).resolve().parent / "bearing_primes_2nd_test.json"
    out.write_text(json.dumps({
        "T": T, "hours": d["hours"], "gate": GATE,
        "raw_alarm_frac": raw_alarm_snaps / T,
        "primes": [{**r, "fire_h": [float(hours[i]) for i in book.fires[r["prime"]]]}
                   for r in rows],
    }))
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    main()
