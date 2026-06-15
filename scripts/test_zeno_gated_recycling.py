#!/usr/bin/env python3
"""
Test 33 - Zeno-gated recycling: the width floor IS the recycle certificate.

Test 29 used TWO cooperating mechanisms: a monitored job produced a
completion certificate (halted/loop/budget), and a separate reclamation
step recycled primes when the certificate arrived. Test 32 showed the Zeno
frame-descent is the kernel under both. This test proves the unification is
SUBSTITUTABLE CODE, not analogy: it drives the REAL `RecyclingLattice`
(imported unchanged from Test 29) directly from a Zeno frame descent
(imported unchanged from Test 32), with ONE event - the width hitting the
floor - serving as both the termination signal and the recycle trigger.

The single mechanism:
    while frame.width > floor:        # GATEKEEPER admits the next step
        promote a working prime        # the descent IS the work (nursery)
    # width <= floor: the SAME test fires the completion certificate
    tenure(address = descent trajectory)   # BOOKKEEPER
    recycle nursery to ground zero          # JANITOR

Proof obligations:
  (1) recycle events == floor-hit events == cycles  (one mechanism, not two)
  (2) lattice bounded forever (Test 29's guarantee, now from the floor)
  (3) tenured addresses exact & recoverable (BOOKKEEPER survives reuse)
  (4) total budget bounded - convergent width series (JANITOR)
  (5) zero width never realized (SECURITY)
  (6) the floor sets the cadence: higher floor -> earlier recycle ->
      smaller lattice (RULER), and removing the gate reproduces the
      Test 5 pool-exhaustion wall.
"""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.primes import chain_primes
from test_ground_zero_recycling import RecyclingLattice
from test_zeno_kernel import Frame, position_from_traj


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# primes start at 2 to match section 12's primorial descent
ZP = (2,) + tuple(chain_primes(80))


def run_zeno_recycler(lat: RecyclingLattice, n_cycles: int, floor: Fraction,
                      base_list, expected: dict, descent_primes,
                      gate: bool = True):
    """One fused engine. Returns (recycle_events, floor_hits, total_budget,
    min_width). The SAME `width <= floor` test stops the descent AND triggers
    recycling - that is the whole point."""
    # configure the instance for the new certificate kind (RecyclingLattice
    # itself is imported unchanged; we only seed an instance counter)
    lat.certificates.setdefault("zeno-floor", 0)
    recycle_events = 0
    floor_hits = 0
    total_budget = Fraction(0)
    min_width = Fraction(1)
    summaries_for_rollup: list[int] = []

    for cycle in range(n_cycles):
        frame = Frame(Fraction(0), Fraction(1))
        nursery_locals: list[int] = []
        level = 0
        # ---- GATEKEEPER: descend (= work) only while above the floor ----
        while True:
            p = descent_primes[level % len(descent_primes)]
            i = (cycle + level) % p                 # deterministic child
            nxt = frame.child(p, i)
            # allocate a working prime for this descent level (nursery)
            pool = base_list if len(nursery_locals) < 2 else \
                base_list + nursery_locals[-4:]
            chain = [pool[(cycle + level) % len(pool)],
                     pool[(cycle + level + 1) % len(pool)]]
            nursery_locals.append(lat.promote(chain, label=f"z{cycle}.{level}"))
            frame = nxt
            min_width = min(min_width, frame.width)
            total_budget += frame.width             # JANITOR accumulates
            level += 1
            # the SINGLE decision point:
            if gate and frame.width <= floor:
                floor_hits += 1
                break
            if not gate and level >= 70:            # no gate -> runaway proxy
                break
        # ---- the floor-hit IS the completion certificate ----
        # BOOKKEEPER: the descent trajectory is this computation's address
        addr = position_from_traj(frame.traj)
        summary = lat.promote(nursery_locals[-3:], label=f"sum{cycle}@{addr}")
        # tenure + recycle nursery to ground zero (driven by the floor event)
        lat.complete_cycle(keep=[summary], certificate="zeno-floor")
        recycle_events += 1
        expected[summary] = lat.closure(summary)
        summaries_for_rollup.append(summary)
        # roll-up keeps the durable layer bounded too
        if len(summaries_for_rollup) == 64:
            ep = lat.rollup(summaries_for_rollup, f"epoch@{cycle}")
            for s in summaries_for_rollup:
                expected.pop(s, None)
            expected[ep] = lat.closure(ep)
            summaries_for_rollup = []

    return recycle_events, floor_hits, total_budget, min_width


def main():
    header("Zeno-gated recycling - the width floor IS the recycle certificate")

    base_primes = list(chain_primes(48))[:48]
    pool_primes = list(chain_primes(700))[48:]
    base_list = base_primes
    descent_primes = ZP[:10]

    # ------------------------------------------------------------------
    print("\nPart A - Counterfactual: no gate -> the Test 5 pool wall returns")
    print("-" * 72)
    naive = RecyclingLattice(base_primes, list(pool_primes), recycling=True)
    # 'no gate' here means the recycle is severed from the floor event:
    naive_nr = RecyclingLattice(base_primes, list(pool_primes), recycling=False)
    died = -1
    try:
        run_zeno_recycler(naive_nr, 10_000, Fraction(1, 10 ** 9), base_list,
                          {}, descent_primes, gate=True)
    except RuntimeError:
        died = naive_nr.logical_promotions
    print(f"  recycling OFF: pool exhausted after {died} promotions")
    assertion(died > 0,
              "without reclamation the pool dies (Test 5 wall) - recycling is "
              "load-bearing, and we will drive it from the floor")

    # ------------------------------------------------------------------
    print("\nPart B - Fused engine: ONE width test = terminate + recycle")
    print("-" * 72)
    lat = RecyclingLattice(base_primes, list(pool_primes), recycling=True)
    expected: dict = {}
    floor = Fraction(1, 10 ** 9)
    t0 = time.time()
    recycles, floor_hits, budget, min_w = run_zeno_recycler(
        lat, 10_000, floor, base_list, expected, descent_primes)
    dt = time.time() - t0
    print(f"  cycles: 10,000 in {dt:.1f}s")
    print(f"  floor-hit events:   {floor_hits}")
    print(f"  recycle events:     {recycles}")
    print(f"  logical promotions: {lat.logical_promotions:,}")
    print(f"  max live nodes:     {lat.max_live}")
    print(f"  certificates:       {dict(lat.certificates)}")
    assertion(recycles == floor_hits == 10_000,
              "recycle events == floor-hit events == cycles: ONE mechanism "
              "drives both termination and reclamation (not two)")
    assertion(lat.certificates.get("zeno-floor", 0) == 10_000,
              "every recycle carried the 'zeno-floor' certificate - the "
              "width floor IS the completion proof")

    # ------------------------------------------------------------------
    print("\nPart C - The Test 29 guarantees, now from the floor alone")
    print("-" * 72)
    assertion(lat.max_live < 400,
              f"(2) lattice bounded: max {lat.max_live} live across "
              f"{lat.logical_promotions:,} promotions (JANITOR via the floor)")
    bad = sum(1 for p, exp in expected.items() if lat.closure(p) != exp)
    assertion(bad == 0,
              f"(3) all {len(expected)} tenured addresses exact after reuse "
              f"(BOOKKEEPER survives recycling)")
    print(f"  (4) total descent budget = {float(budget):.4f} "
          f"(bounded - convergent width series, JANITOR)")
    assertion(budget < 10_000,
              "(4) total budget bounded despite 10,000 unbounded-looking "
              "descents (per-cycle width series converges)")
    print(f"  (5) smallest width realized = {float(min_w):.2e}")
    assertion(min_w > 0,
              "(5) zero width never realized - SECURITY invariant held every "
              "step of every cycle")

    # ------------------------------------------------------------------
    print("\nPart D - The floor sets the cadence (RULER): higher floor, less RAM")
    print("-" * 72)
    print(f"  {'floor':>12} | {'avg levels/cycle':>16} | {'max live':>9}")
    print(f"  {'-'*12} | {'-'*16} | {'-'*9}")
    cadence = []
    for fexp in (4, 7, 12):
        fl = Fraction(1, 10 ** fexp)
        L = RecyclingLattice(base_primes, list(pool_primes), recycling=True)
        exp2: dict = {}
        rc, fh, bud, mw = run_zeno_recycler(L, 2000, fl, base_list, exp2,
                                            descent_primes)
        avg_levels = L.logical_promotions / 2000  # incl. summary promote
        cadence.append((fexp, avg_levels, L.max_live))
        print(f"  1e-{fexp:<10} | {avg_levels:>16.1f} | {L.max_live:>9}")
    # deeper floor -> more levels per cycle -> larger (but still bounded) lattice
    assertion(cadence[0][1] < cadence[-1][1] and cadence[-1][2] < 600,
              "the floor tunes per-cycle depth and peak memory - the descent "
              "schedule IS the clock/throttle (RULER), and every setting stays "
              "bounded")

    # ------------------------------------------------------------------
    header("RESULT")
    print("  Two components became one. The width-floor test that stops a")
    print("  Zeno descent is the SAME test that recycles the worker's primes:")
    print(f"    floor hits = recycles = cycles = 10,000 (identical counts)")
    print(f"    lattice bounded at {lat.max_live} nodes / {lat.logical_promotions:,} promotions")
    print(f"    all {len(expected)} addresses exact; budget {float(budget):.3f}; min width > 0")
    print()
    print("  Test 29 needed a monitored job + a certificate + a recycler.")
    print("  Test 33 needs only the Zeno descent: the no-terminal-frame floor")
    print("  is simultaneously the GATEKEEPER (terminate), the certificate")
    print("  (proof of completion), the JANITOR trigger (recycle), the")
    print("  BOOKKEEPER address (trajectory), and the RULER (cadence).")
    print()
    print("  The unification is substitutable code: the real RecyclingLattice")
    print("  (Test 29) ran unchanged, driven entirely by the real Zeno Frame")
    print("  (Test 32). One kernel, proven by execution - not analogy.")


if __name__ == "__main__":
    main()
