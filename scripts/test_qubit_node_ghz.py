#!/usr/bin/env python3
"""
Test 31 - Node-as-qubit, exact: Werner threshold + GHZ all-or-nothing.

Built ON the existing AETHOS quantum stack (aethos_quantum exact statevector,
aethos_physics ocean-fill contract) via the new aethos_qubit_node module.
Where Test 30 used Monte-Carlo electron mechanics, this uses EXACT
statevectors - results are machine-precise.

Three things, each grounded in the user's physics files:

  (A) EXACT singlet: the real 2-qubit register gives E(a,b) = -cos(a-b) and
      CHSH S = -2sqrt(2) to machine precision (not sampling).
  (B) OCEAN FILL as the classical<->quantum dial: CHSH = 2sqrt2 * phi, so the
      Bell wall is crossed exactly at phi* = 1/sqrt(2) (the Werner visibility
      threshold - a real quantum-information result the ocean reproduces).
  (C) GHZ ALL-OR-NOTHING: three entangled nodes where local realism fails in
      a SINGLE measurement round (Mermin), strictly stronger than Bell's
      statistical violation. This is "measure one, the whole set is
      determined" taken to its sharpest form.

  (D) Lattice bridge: nodes addressed by primes; every pair entanglable at
      any moment with a collision-free composite address (Test 3 / Test 8).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aethos_qubit_node import (
    GHZ3, QubitNode, chsh_at_fill, entangle_pair, pair_address,
    werner_threshold,
)
from aethos_quantum import (
    TwoQubitRegister, bell_correlation_register, chsh_s_register,
)
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


def main():
    header("Node-as-qubit (exact): Werner threshold + GHZ all-or-nothing")

    # ------------------------------------------------------------------
    print("\nPart A - Exact singlet: E(a,b) = -cos(a-b), CHSH = -2sqrt(2)")
    print("-" * 72)
    reg = TwoQubitRegister.singlet()
    worst = 0.0
    for d in (0, 30, 45, 60, 90, 135, 180):
        a, b = 0.0, math.radians(d)
        e = bell_correlation_register(reg, a, b)
        qm = -math.cos(a - b)
        worst = max(worst, abs(e - qm))
        print(f"  E(0,{d:>3}) = {e:+.6f}   -cos(a-b) = {qm:+.6f}")
    assertion(worst < 1e-9,
              f"exact statevector matches -cos(a-b) to machine precision "
              f"(worst {worst:.1e})")
    S = chsh_s_register(reg)
    print(f"  CHSH S = {S:.6f}   (Tsirelson -2sqrt2 = {-2*math.sqrt(2):.6f})")
    assertion(abs(abs(S) - 2 * math.sqrt(2)) < 1e-9,
              "CHSH hits Tsirelson 2sqrt(2) exactly (no sampling error)")

    # ------------------------------------------------------------------
    print("\nPart B - Ocean fill as the classical<->quantum dial")
    print("-" * 72)
    star = werner_threshold()
    print(f"  Werner/Bell threshold phi* = 1/sqrt(2) = {star:.6f}")
    print(f"  {'phi':>6} | {'CHSH':>8} | regime")
    print(f"  {'-'*6} | {'-'*8} | ------")
    crossed_below = crossed_above = False
    for phi in (0.0, 0.25, 0.5, star, 0.75, 0.9, 1.0):
        S = chsh_at_fill(phi)
        regime = "QUANTUM (Bell-violating)" if S > 2 + 1e-9 else "classical"
        if S <= 2 + 1e-9 and phi <= star:
            crossed_below = True
        if S > 2 + 1e-9 and phi > star:
            crossed_above = True
        print(f"  {phi:>6.3f} | {S:>8.4f} | {regime}")
    assertion(abs(chsh_at_fill(star) - 2.0) < 1e-9,
              "ocean fill exactly 1/sqrt(2) sits ON the Bell wall (S = 2.0)")
    assertion(crossed_below and crossed_above,
              "fill below threshold = classical, above = quantum (the dial "
              "the user described - axis choice over a tunable medium)")
    # cross-check the analytic dial against the real register's dephase
    reg_full = entangle_pair(1.0, 1.0)
    reg_half = entangle_pair(0.5, 1.0)
    S_full = abs(chsh_s_register(reg_full))
    S_half = abs(chsh_s_register(reg_half))
    print(f"  register check: phi=1.0 -> S={S_full:.4f}, "
          f"phi=0.5 -> S={S_half:.4f}")
    assertion(S_full > 2.0 > S_half,
              "the real dephased register agrees: full fill quantum, "
              "half fill classical")

    # ------------------------------------------------------------------
    print("\nPart C - GHZ all-or-nothing: local realism fails in ONE shot")
    print("-" * 72)
    ghz = GHZ3()
    ops = ghz.mermin_operators()
    for name, val in ops.items():
        print(f"  <{name}> = {val:+.6f}")
    qm_xxx, forced_xxx = ghz.local_realism_contradiction()
    print(f"\n  local realism forces  XXX = (XYY)(YXY)(YYX) = {forced_xxx:+.1f}")
    print(f"  quantum mechanics says XXX =                   {qm_xxx:+.1f}")
    assertion(abs(qm_xxx - 1.0) < 1e-9
              and all(abs(ops[k] + 1.0) < 1e-9 for k in ("XYY", "YXY", "YYX")),
              "GHZ gives XXX=+1, XYY=YXY=YYX=-1 (exact)")
    assertion(abs(forced_xxx + 1.0) < 1e-9 and abs(qm_xxx - 1.0) < 1e-9,
              "local realism demands -1, quantum gives +1 - contradiction in a "
              "SINGLE round (Mermin), strictly stronger than Bell")
    print("  -> three entangled nodes; the four observables cannot all come")
    print("     from pre-set local values. Measuring fixes the whole set, and")
    print("     no 'hidden answer sheet' can exist - proven without statistics.")

    # ------------------------------------------------------------------
    print("\nPart D - Lattice bridge: prime-addressed nodes, any-pair entangling")
    print("-" * 72)
    primes = chain_primes(16)
    nodes = [QubitNode(p, axis=math.radians(37 * i)) for i, p in enumerate(primes)]
    # 4 sub-quadrant coverage
    quads = set(n.sub_quadrant() for n in nodes)
    print(f"  sub-quadrant definitions present: {sorted(quads)}")
    assertion(quads == {0, 1, 2, 3}, "nodes cover all 4 sub-quadrant definitions")
    # every pair gets a distinct entanglement address
    addrs = set()
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            addrs.add(pair_address(nodes[i], nodes[j]))
    n_pairs = len(nodes) * (len(nodes) - 1) // 2
    print(f"  distinct pair addresses: {len(addrs)} / {n_pairs} possible pairs")
    assertion(len(addrs) == n_pairs,
              "every node pair has a collision-free composite entanglement "
              "address (any 2 nodes, any moment - no locality constraint)")
    # the address factors back to the two nodes (provenance, Test 8)
    a, b = nodes[3], nodes[9]
    addr = pair_address(a, b)
    rec = sorted(p for p in primes if addr % p == 0)
    assertion(rec == sorted([a.prime, b.prime]),
              f"entanglement address {addr} factors back to nodes "
              f"{a.prime},{b.prime}")

    # ------------------------------------------------------------------
    header("RESULT")
    print(f"  exact singlet:     E = -cos(a-b), CHSH = -2sqrt2 (machine precise)")
    print(f"  ocean-fill dial:   classical below phi*=0.707, quantum above")
    print(f"  GHZ all-or-nothing: XXX = +1 vs local-realism -1 (single shot)")
    print(f"  lattice bridge:    16 nodes, 120 collision-free pair addresses")
    print()
    print("  Built on the repo's own quantum stack (aethos_quantum exact")
    print("  statevector + aethos_physics ocean fill), now exposed as the")
    print("  reusable aethos_qubit_node module. The electron sections, the")
    print("  ocean, and the lattice addressing are one system: any node is a")
    print("  qubit, any pair entangles with a prime-composite address, the")
    print("  ocean fill is the classical<->quantum dial, and GHZ shows the")
    print("  collapse is real - no pre-set answer sheet survives a single")
    print("  three-node measurement.")


if __name__ == "__main__":
    main()
