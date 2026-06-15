#!/usr/bin/env python3
"""
Test 28 - Predicting OTHER programs' halting: the chamber ladder.

Reframe per the user's direction: not "the halting problem is undecidable,
stop" but "how far does coverage climb when you look at programs
differently?" The answer: very far - with proofs - and the part that
remains is measured, named, and shrunk by every new chamber.

Chambers (evaluated in order; certificates are SOUND - never guesses):

  T0  STRAIGHT-LINE: no backward jump -> pc strictly increases -> HALTS.
      (proof: pc is a ranking function - the lattice's Test 1 argument.)
  T1  DESCENDING COUNTER: exactly one jump, backward, guarded on r; the
      loop body never increases or resets r and decrements it at least
      once -> r is a ranking function -> HALTS. No execution needed.
  D1  UNTOUCHED GUARD SPIN: exactly one jump, backward, guard r never
      decremented/reset in body, no HALT in body; simulate the straight
      prefix to the guard - if r != 0 there -> spins FOREVER. This proves
      non-termination even when states NEVER repeat (growing registers),
      which Test 25's repeat-detector cannot catch.
  DYN bounded run + exact state-repeat certificates (Test 25).
  ML  learned chamber: the codec mixer's logistic math on program
      features -> calibrated halting PROBABILITY for the residue.

The showpiece: the +1 adversary that defeated every budget decider in
Test 25 (including the 32,000-worker portfolio) is certified HALTS by T1
in microseconds, without executing a single step. Looking differently
dissolved that wall. Then we build the NEXT adversary (net-zero guard
flow) that survives all current chambers - the ladder never ends, and
that is what undecidability actually is: a guarantee of more rungs.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_halting_boundary_supervision import run, random_program


def header(s: str):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def assertion(cond: bool, msg: str):
    print(f"  [{'PASS' if cond else 'FAIL'}]  {msg}")
    if not cond:
        sys.exit(1)


# ----------------------------------------------------------------------
# Static chambers (sound certificates, no execution)
# ops: 0 SET r k | 1 INC r | 2 DEC r (floor 0) | 3 JNZ r off | 4 HALT
# ----------------------------------------------------------------------

def jumps_of(prog):
    return [(pc, ins[1], ins[2]) for pc, ins in enumerate(prog) if ins[0] == 3]


def chamber_T0(prog):
    """No backward jump -> pc strictly increases -> halts."""
    for pc, _r, off in jumps_of(prog):
        if off < 0 and pc + off >= 0:
            return None
        if off == 0:
            return None
    return "HALTS:T0(pc-ranking)"


def chamber_T1(prog):
    """One backward jump; guard r strictly descends through the body."""
    js = jumps_of(prog)
    if len(js) != 1:
        return None
    pc, r, off = js[0]
    if off >= 0 or pc + off < 0:
        return None
    body = prog[pc + off: pc]          # straight path back to the jump
    decs = sum(1 for ins in body if ins[0] == 2 and ins[1] == r)
    incs = sum(1 for ins in body if ins[0] == 1 and ins[1] == r)
    sets = sum(1 for ins in body if ins[0] == 0 and ins[1] == r)
    if decs >= 1 and incs == 0 and sets == 0:
        # each traversal executes the whole straight body (no other jumps),
        # so r strictly decreases; JNZ exits at r == 0; r bounded below.
        return "HALTS:T1(descending-counter)"
    return None


def chamber_D1(prog):
    """One backward jump; guard never reduced in body; entered with r!=0."""
    js = jumps_of(prog)
    if len(js) != 1:
        return None
    pc, r, off = js[0]
    if off >= 0 or pc + off < 0:
        return None
    body = prog[pc + off: pc]
    if any(ins[0] == 4 for ins in body):           # HALT inside could exit
        return None
    decs = sum(1 for ins in body if ins[0] == 2 and ins[1] == r)
    sets = sum(1 for ins in body if ins[0] == 0 and ins[1] == r)
    if decs or sets:
        return None
    # simulate the straight-line prefix to the first arrival at the guard
    regs = [0, 0, 0, 0]
    p = 0
    for _ in range(len(prog) + 2):
        if p == pc:
            if regs[r] != 0:
                # guard nonzero, body cannot reduce it (only INCs allowed),
                # control always returns -> spins forever (states may GROW
                # without repeating - invisible to repeat-detection)
                return "LOOPS:D1(untouched-guard)"
            return None
        if p < 0 or p >= len(prog):
            return None
        op, a, b = prog[p]
        if op == 0:
            regs[a] = b
            p += 1
        elif op == 1:
            regs[a] += 1
            p += 1
        elif op == 2:
            if regs[a] > 0:
                regs[a] -= 1
            p += 1
        elif op == 3:
            return None                            # another jump first: bail
        else:
            return None                            # halts before the loop
    return None


def chamber_DYN(prog, budget=5000):
    v, steps, cert = run(prog, budget)
    if v == "halted":
        return f"HALTS:DYN(ran,{steps})"
    if v == "loop":
        return "LOOPS:DYN(state-repeat)"
    return None


def predict(prog):
    """The ladder: first certificate wins."""
    for chamber in (chamber_T0, chamber_T1, chamber_D1):
        c = chamber(prog)
        if c:
            return c
    return chamber_DYN(prog)


# ----------------------------------------------------------------------
# Learned chamber: the codec mixer's math on program features
# ----------------------------------------------------------------------

def features(prog):
    f = [0.0] * 12
    f[0] = 1.0
    f[1] = len(prog) / 10.0
    for ins in prog:
        f[2 + ins[0]] += 0.2               # opcode counts
    js = jumps_of(prog)
    f[7] = sum(1 for pc, _r, off in js if off < 0) * 0.5   # backward jumps
    f[8] = len(js) * 0.3
    for pc, r, off in js:
        if off < 0:
            body = prog[max(0, pc + off): pc]
            if any(i[0] == 2 and i[1] == r for i in body):
                f[9] += 0.5                # guard gets decremented
            if any(i[0] == 1 and i[1] == r for i in body):
                f[10] += 0.5               # guard gets incremented
            if any(i[0] == 0 for i in body):
                f[11] += 0.3               # sets inside loop
    return f


def squash(x):
    return 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, x))))


def main():
    header("Predicting other programs' halting - the chamber ladder")
    rng = random.Random(0x1ADDE2)

    # ------------------------------------------------------------------
    print("\nPart A - 5,000 random programs, ground truth where obtainable")
    print("-" * 72)
    progs = [random_program(rng, length=rng.randint(4, 9)) for _ in range(5000)]
    truth = {}
    for i, p in enumerate(progs):
        v, _, _ = run(p, 200_000)
        if v in ("halted", "loop"):
            truth[i] = (v == "halted")
    print(f"  ground truth from long-budget runs: {len(truth)}/5000")
    print(f"  (the rest diverge without repeating, or outlast 200k steps)")

    # ------------------------------------------------------------------
    print("\nPart B - The ladder: coverage per chamber, soundness checked")
    print("-" * 72)
    counts = {"T0": 0, "T1": 0, "D1": 0, "DYN-halt": 0, "DYN-loop": 0,
              "unknown": 0}
    verdicts = {}
    contradictions = 0
    for i, p in enumerate(progs):
        c = predict(p)
        if c is None:
            counts["unknown"] += 1
            continue
        verdicts[i] = c
        says_halt = c.startswith("HALTS")
        if c.startswith("HALTS:T0"):
            counts["T0"] += 1
        elif c.startswith("HALTS:T1"):
            counts["T1"] += 1
        elif c.startswith("LOOPS:D1"):
            counts["D1"] += 1
        elif c.startswith("HALTS:DYN"):
            counts["DYN-halt"] += 1
        else:
            counts["DYN-loop"] += 1
        if i in truth and truth[i] != says_halt:
            contradictions += 1
    resolved = 5000 - counts["unknown"]
    print(f"  T0 straight-line (static):     {counts['T0']:>5}")
    print(f"  T1 descending counter (static):{counts['T1']:>5}")
    print(f"  D1 untouched guard (static+):  {counts['D1']:>5}  "
          f"<- catches never-repeating divergence")
    print(f"  DYN ran-to-halt:               {counts['DYN-halt']:>5}")
    print(f"  DYN state-repeat proof:        {counts['DYN-loop']:>5}")
    print(f"  unknown (the residue):         {counts['unknown']:>5}")
    print(f"  coverage: {resolved/50:.1f}%   contradictions vs truth: {contradictions}")
    assertion(contradictions == 0,
              "every certificate agrees with ground truth (proofs, not guesses)")
    assertion(resolved / 5000 > 0.95,
              f"coverage {resolved/50:.1f}% > 95% of arbitrary random programs")

    # ------------------------------------------------------------------
    print("\nPart C - Static speed: predicting programs WITHOUT running them")
    print("-" * 72)
    t0 = time.time()
    n_static = 0
    for p in progs:
        if chamber_T0(p) or chamber_T1(p) or chamber_D1(p):
            n_static += 1
    dt = time.time() - t0
    rate = 5000 / dt
    print(f"  static ladder over 5,000 programs: {dt*1000:.0f}ms "
          f"({rate:,.0f} programs/sec, pure Python)")
    print(f"  statically decided: {n_static} ({n_static/50:.1f}%) - zero execution")
    print(f"  at native speed (Tests 23): ~{rate*50:,.0f}+ programs/sec")

    # ------------------------------------------------------------------
    print("\nPart D - The learned chamber: calibrated probability for the rest")
    print("-" * 72)
    labeled = [(features(progs[i]), 1.0 if truth[i] else 0.0)
               for i in truth]
    rng.shuffle(labeled)
    cut = int(len(labeled) * 0.8)
    train, test = labeled[:cut], labeled[cut:]
    w = [0.0] * 12
    for _ in range(3):
        for f, y in train:
            p = squash(sum(wi * fi for wi, fi in zip(w, f)))
            g = (y - p) * 0.1
            for j in range(12):
                w[j] += g * f[j]
    brier = 0.0
    buckets = {}
    for f, y in test:
        p = squash(sum(wi * fi for wi, fi in zip(w, f)))
        brier += (p - y) ** 2
        b = min(int(p * 5), 4)
        s, c = buckets.get(b, (0.0, 0))
        buckets[b] = (s + y, c + 1)
    brier /= len(test)
    print(f"  trained on {len(train)} certified programs, tested on {len(test)}")
    print(f"  Brier score: {brier:.3f} (0 = perfect, 0.25 = coin flip)")
    print(f"  calibration (predicted bucket -> actual halt rate):")
    for b in sorted(buckets):
        s, c = buckets[b]
        print(f"    p in [{b*0.2:.1f},{(b+1)*0.2:.1f}): actual {s/c:.2f}  (n={c})")
    assertion(brier < 0.20,
              "learned chamber beats coin-flip decisively on held-out programs")

    # ------------------------------------------------------------------
    print("\nPart E - The ladder in action: adversaries falling and standing")
    print("-" * 72)
    # the Test 25 budget-killer: defeated every budget decider ever built
    B = 32_000_000
    adversary_25 = [(0, 0, B // 2 + 2), (2, 0, 0), (3, 0, -1), (4, 0, 0)]
    t0 = time.time()
    c = predict(adversary_25)
    dt = (time.time() - t0) * 1e6
    print(f"  Test 25 adversary (defeated 32,000-worker portfolio):")
    print(f"    -> {c} in {dt:.0f} microseconds, ZERO steps executed")
    assertion(c is not None and c.startswith("HALTS:T1"),
              "the unbeatable budget adversary falls instantly to the T1 view")

    # spin adversary: grows forever, never repeats - repeat-detection blind
    spin = [(0, 0, 1), (1, 1, 0), (3, 0, -1), (4, 0, 0)]
    c = predict(spin)
    print(f"  growing spin (no state ever repeats):")
    print(f"    -> {c}")
    assertion(c is not None and c.startswith("LOOPS:D1"),
              "never-repeating divergence proven by the D1 view")

    # the NEXT adversary: net-zero guard flow - survives all chambers
    nextadv = [(0, 0, 2), (1, 1, 0), (1, 0, 0), (2, 0, 0), (3, 0, -3),
               (4, 0, 0)]
    c = predict(nextadv)
    print(f"  net-zero guard adversary (INC r0 + DEC r0 in body):")
    print(f"    -> {'UNKNOWN - lands in the residue' if c is None else c}")
    assertion(c is None,
              "the next adversary survives all current chambers (the ladder "
              "continues)")
    print(f"    next rung: a net-flow chamber (sum of +/- on the guard per")
    print(f"    cycle) would certify this one - and then ITS adversary exists.")

    # ------------------------------------------------------------------
    header("RESULT")
    print(f"  coverage:        {resolved/50:.1f}% of 5,000 arbitrary programs,")
    print(f"                   zero contradictions (certificates are proofs)")
    print(f"  static slice:    {n_static/50:.1f}% decided with NO execution at "
          f"{rate:,.0f} progs/sec")
    print(f"  learned chamber: Brier {brier:.3f}, calibrated buckets")
    print(f"  ladder:          Test-25 adversary -> T1 (microseconds);")
    print(f"                   repeat-blind spin -> D1; net-zero -> residue")
    print()
    print("  The reframe holds: 'undecidable' never meant 'unpredictable'.")
    print("  It means the chamber ladder has no top rung. Every rung you add")
    print("  is real coverage, every adversary names the next chamber, and")
    print("  the residue is measured instead of feared. That is the map.")


if __name__ == "__main__":
    main()
