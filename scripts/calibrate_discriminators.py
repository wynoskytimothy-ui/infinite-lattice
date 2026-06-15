#!/usr/bin/env python3
"""Sweep He isotope ratio and M_lat chain_species for book discriminators."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aethos_physics as ap  # noqa: E402
from aethos_sequences import SequenceKind  # noqa: E402

R_PE = ap.R_PE
R0 = ap.r_pe_spring_only()
TARGET_M_LAT = R_PE / R0  # ~1488.3


def he_ratio(f3: float, f4: float, sigma: float = 1.0) -> float:
    return ap.lambda_he3_he4_ratio(
        f_coin_he3=f3,
        f_coin_he4=f4,
        sigma_ratio=sigma,
    )


def main() -> None:
    print("=== 3He/4He Lambda ratio calibration (target 1.05–1.10) ===")
    mass_only = math.sqrt(ap.M_HE4 / ap.M_HE3)
    print(f"Mass-only factor sqrt(m4/m3) = {mass_only:.4f} ({(mass_only - 1) * 100:.1f}%)")

    candidates = [
        (0.93, 1.0, 1.0, "f3=0.93, f4=1.0"),
        (0.465, 0.5, 1.0, "f3=0.465, f4=0.5"),
        (0.81, 0.87, 1.0, "f3=0.81, f4=0.87"),
        (0.75, 0.80, 0.95, "tuned from defaults"),
    ]
    for f3, f4, sig, label in candidates:
        r = he_ratio(f3, f4, sig)
        print(f"  {label}: ratio = {r:.4f} ({(r - 1) * 100:.1f}%)")

    # ratio = (f3/f4) * (m4/m3)^(3/2) at matched T
    expo = 1.5
    mass_factor = (ap.M_HE4 / ap.M_HE3) ** expo
    print(f"Proxy mass+thermal factor (m4/m3)^(3/2) = {mass_factor:.4f}")
    f_over = 1.075 / mass_factor
    print(f"f_coin_he3/f_coin_he4 for ratio=1.075: {f_over:.4f}")
    r_check = he_ratio(0.352, 0.5)
    print(f"  calibrated f3=0.352, f4=0.5: ratio = {r_check:.4f}")

    print("\n=== M_lat / R_pe sweep (target R_pe ~ 1836) ===")
    print(f"Target M_lat = R_pe^E / (pi^2/8) = {TARGET_M_LAT:.2f}\n")

    rows: list[tuple[float, str, int, int, float, float]] = []
    species_list = list(SequenceKind)
    for species in species_list:
        if species == SequenceKind.CUSTOM:
            continue
        for count in (60, 80, 100, 120):
            for depth in (2, 3, 4):
                try:
                    mlat = ap.m_lat_from_active_network(
                        count=count,
                        origin_max_depth=depth,
                        chain_species=species,
                    )
                except Exception:
                    continue
                rpe = R0 * mlat
                err = abs(rpe - R_PE) / R_PE
                rows.append((err, species.value, count, depth, mlat, rpe))

    rows.sort(key=lambda x: x[0])
    print("Top 12 closest to CODATA R_pe:")
    print(f"{'species':<14} {'count':>5} {'depth':>5} {'M_lat':>10} {'R_pe':>10} {'err%':>8}")
    for err, sp, cnt, dep, mlat, rpe in rows[:12]:
        print(f"{sp:<14} {cnt:5d} {dep:5d} {mlat:10.2f} {rpe:10.2f} {100 * err:7.3f}")

    best = rows[0]
    print(f"\nBest: {best[1]} count={best[2]} depth={best[3]} -> M_lat={best[4]:.2f}, R_pe={best[5]:.2f}")


if __name__ == "__main__":
    main()
