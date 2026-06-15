#!/usr/bin/env python3
"""Explain WHY E-check calibration values land (hidden pattern decomposition)."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aethos_physics as ap  # noqa: E402
from aethos_active import (  # noqa: E402
    BRANCHES_PER_VECTOR,
    VECTORS_PER_NODE,
    WINGS_PER_ROOM,
    ActiveNetwork100,
)
from aethos_origins import OriginTree  # noqa: E402
from aethos_sequences import SequenceKind  # noqa: E402

R0 = ap.r_pe_spring_only()
TARGET_M = ap.R_PE / R0


def section(title: str) -> None:
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    section("A: Two-factor mass ratio")
    print(f"R_pe^(0) = pi^2/8           = {R0:.6f}")
    print(f"E gap lattice_mass_multiplier = {ap.lattice_mass_multiplier():.2f}")
    print(f"L_p/L_0 = 8/pi^2            = {8/math.pi**2:.6f}  (reciprocal of R_pe^(0))")
    print(f"Target M_lat                  = {TARGET_M:.2f}")

    section("B: M_lat linear in node count (depth=3 fixed)")
    origins = len(list(OriginTree.bootstrap(max_depth=3).walk()))
    denom = origins + BRANCHES_PER_VECTOR + VECTORS_PER_NODE
    print(f"N_origins (1+3+9+27)        = {origins}")
    print(f"Denominator origins+12      = {denom}")
    print(f"Scale factor 32/denom         = {WINGS_PER_ROOM/denom:.4f}")
    print(f"Target sum_mu               = {TARGET_M * denom / WINGS_PER_ROOM:.1f}")
    print()
    print(f"{'n':>4} {'sum_mu':>8} {'M_lat':>8} {'R_pe':>8} {'err%':>7} balanced")
    for count in range(70, 101, 5):
        net = ActiveNetwork100.bootstrap(count=count, origin_max_depth=3)
        weights = [ap.chain_cascade_weight(n.chain) for n in net.nodes]
        total = sum(weights)
        mlat = total * WINGS_PER_ROOM / denom
        rpe = R0 * mlat
        err = 100 * abs(rpe - ap.R_PE) / ap.R_PE
        roles = Counter(n.role.name for n in net.nodes)
        balanced = len(set(roles.values())) == 1
        mark = " <-- E-check" if count == ap.REFERENCE_NETWORK_COUNT else ""
        print(
            f"{count:4d} {total:8.1f} {mlat:8.1f} {rpe:8.1f} {err:7.2f} {balanced}{mark}"
        )

    section("C: 80 = 16 x 5 role ledger")
    n = ap.REFERENCE_NETWORK_COUNT
    print(f"count={n} => {n//5} complete cycles of SOLO/PAIR/TRIPLE/K_CHAIN/FOUR_WAY")
    print(f"nodes per origin room (avg)   = {n/origins:.1f}")

    section("G: 1280 wing slots -> 80 active (1/16)")
    wa = ap.wing_activation_analysis()
    print(f"origins x 32 wings            = {int(wa['n_origins'])} x 32 = {int(wa['wings_total'])}")
    print(f"active nodes                  = {int(wa['active_nodes'])}")
    print(f"global activation             = {wa['global_activation_fraction']:.4f}  (= 1/16)")
    print(f"per-origin: 2 nodes / 32 wings = {wa['wing_fraction_per_origin']:.4f}  (= 1/16)")
    print(f"role_cycles n/5               = {wa['role_cycles']:.0f}  (= 16)")
    print("Pattern: proton E-check uses one sixteenth of cosmic wing address space per depth-3 tree.")

    section("E: Material blob bootstrap (C6) at n=80")
    try:
        from aethos_blob import ElectronBlob

        print(f"{'density':>8} {'coupling':>8} {'M_lat':>8} {'R_pe':>8} {'err%':>7}")
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            for c in (0.0, 0.5, 1.0):
                blob = ElectronBlob(density=d, coupling=c)
                mlat = ap.m_lat_from_material_blob(blob)
                rpe = R0 * mlat
                err = 100 * abs(rpe - ap.R_PE) / ap.R_PE
                print(f"{d:8.2f} {c:8.2f} {mlat:8.1f} {rpe:8.1f} {err:7.2f}")
    except Exception as exc:
        print(f"(blob path skipped: {exc})")

    section("D: He ratio = (f3/f4) x (m4/m3)")
    m_ratio = ap.M_HE4 / ap.M_HE3
    f3, f4 = ap.F_COIN_HE3_DISCRIMINATOR, ap.F_COIN_HE4_DISCRIMINATOR
    net = (f3 / f4) * m_ratio
    print(f"m4/m3 (mass-only split)       = {m_ratio:.4f} ({(m_ratio-1)*100:.1f}%)")
    print(f"Required f3/f4 for 1.075      = {1.075/m_ratio:.4f}")
    print(f"E-check f3/f4                 = {f3/f4:.4f}")
    print(f"Net Lambda_3He/Lambda_4He     = {net:.4f}")
    print(f"Structure cancels mass frac   = {1 - 1.075/m_ratio:.1%}")


if __name__ == "__main__":
    main()
