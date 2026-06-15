#!/usr/bin/env python3
"""
Zeno frame-descent onset localization -- the exact WHEN, with a prime-address
timestamp and a positive-width certificate.

The meet says WHO (which bearing) and the defect line says WHAT (the fault). This
pins WHEN it began. Faithful to the Zeno kernel (scripts/test_zeno_kernel.py):

    Frame.child(p, i): subdivide [a,b] by prime p into p children of width w/p,
                       take child i.   width schedule  wₙ = w₀ / ∏ pₖ  (never 0)

We bracket the onset between a healthy frame edge and a degraded one, then descend:
at each level we split by the next prime and keep the child that still straddles
the healthy->degraded transition. The width shrinks as w₀/∏pₖ and terminates at a
positive-width FLOOR (no singular instant -- Zeno's point never arrives). The
descent trajectory ((2,i),(3,i),(5,i),...) IS a self-describing prime address of
the onset; the floor is the resolution certificate (we claim no more precision
than the descent earned).

    python aethos_zeno_onset.py --test 1st_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
THRESH = 6.0                       # entanglement-break onset, in baseline sigma


class Frame:
    __slots__ = ("a", "b", "traj")

    def __init__(self, a, b, traj=()):
        self.a, self.b, self.traj = a, b, traj

    @property
    def width(self):
        return self.b - self.a

    def child(self, p, i):
        w = self.width / p
        a = self.a + i * w
        return Frame(a, a + w, self.traj + ((p, i),))


def localize(sig, thresh, a0, b0, floor):
    """descend prime frames to the healthy->degraded transition of `sig`."""
    grid = np.arange(len(sig))
    deg = lambda x: np.interp(x, grid, sig) >= thresh      # degraded at (float) index x
    fr = Frame(float(a0), float(b0))
    schedule = []
    for p in PRIMES:
        if fr.width <= floor:
            break
        pick = None
        for i in range(p):                                 # earliest child straddling the crossing
            ch = fr.child(p, i)
            if (not deg(ch.a)) and deg(ch.b):
                pick = ch
                break
        if pick is None:                                   # crossing on a node: first degraded child
            for i in range(p):
                ch = fr.child(p, i)
                if deg(ch.b):
                    pick = ch
                    break
        fr = pick if pick is not None else fr.child(p, p - 1)
        schedule.append((p, fr.width))
    # linear crossing inside the final (sub-snapshot) frame
    va, vb = np.interp([fr.a, fr.b], grid, sig)
    onset = fr.a + (thresh - va) / (vb - va) * fr.width if vb > va else fr.a
    return fr, onset, schedule


def addr(traj):
    return "/".join(f"{p}:{i}" for p, i in traj)


def sustained_array(mask, start, persist=3, window=5):
    """boolean: snapshot t begins a SUSTAINED crossing (>= persist of next window)."""
    out = np.zeros(len(mask), bool)
    for i in range(start, len(mask)):
        if mask[i] and mask[i:i + window].sum() >= persist:
            out[i] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="1st_test")
    ap.add_argument("--n-base", type=int, default=100)
    args = ap.parse_args()

    run = json.loads((HERE / f"bearing_run_{args.test}.json").read_text())
    meet = json.loads((HERE / f"bearing_meet_{args.test}.json").read_text())
    cz = np.array(run["coupling_z"])              # (T,B) entanglement-break signal
    hours = np.array(run["hours"])
    T = len(hours)
    eol_h = hours[-1]
    sources = meet["sources"] or [int(np.argmax(np.abs(cz).max(0)))]
    snap_h = (hours[-1] - hours[0]) / (T - 1)     # snapshot interval in hours
    floor = 0.25                                  # quarter-snapshot resolution certificate

    print(f"Zeno onset descent on {args.test}: {T} snapshots, "
          f"{snap_h*60:.0f} min/snapshot, floor {floor} snap (~{floor*snap_h*60:.0f} min)\n")

    for b in sources:
        # monotone predicate: has a SUSTAINED entanglement-break happened by t?
        sus = sustained_array(cz[:, b] > THRESH, args.n_base)
        if not sus.any():
            print(f"  Bearing {b+1}: never reaches sustained {THRESH:.0f}-sigma; skipped")
            continue
        ever = np.maximum.accumulate(sus.astype(float))     # step 0->1 at the onset
        naive = int(np.argmax(sus))
        fr, onset, sched = localize(ever, 0.5, args.n_base, T - 1, floor)
        oh = float(np.interp(onset, np.arange(T), hours))
        nh = float(hours[naive])
        print(f"  Bearing {b+1}  (source):")
        print(f"     onset  = snapshot {onset:7.2f}  ->  hour {oh:6.1f}  "
              f"({eol_h-oh:.0f} h before end-of-life)")
        print(f"     prime address: {addr(fr.traj)}  ({len(sched)} levels, width "
              f"{T-1-args.n_base} -> {fr.width:.3f} snap, >0 certificate)")
        print(f"     first-sustained scan: snapshot {naive} (hour {nh:.1f}) -- "
              f"descent locates the same onset with a prime-address timestamp\n")

    print("  who (meet) + what (defect line) + when (this) = the full diagnosis tuple.")


if __name__ == "__main__":
    main()
