"""
BIT 5 — |z| band template (compression + consensus)

Math at fixed (A, n), wing w:
  |z_b| for b ∈ {VA1..VA4}
  band = argmin_b | |z_b| − |z| |
  z_obs = Re(z_VA1 + z_VA2)   (Im cancels on Y-mirror pair)

Gate: chain (3,5,7), n=5 → 4 bands, 8 wings per band (32 total).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from aethos_lattice import BranchKind, LatticeId, lattice_id_parts
from aethos_physics import SpacetimeCell
from pipeline.bit_01_word_cell import DEFAULT_ANCHOR_N, word_to_spacetime_cell

NUM_BANDS = 4


@dataclass(frozen=True)
class ZBandProfile:
    """|z| band template for one lattice address."""

    band_id: int  # 0..3
    z_modulus: float
    branch_moduli: tuple[float, float, float, float]
    z_obs: float  # Re(z_VA1 + z_VA2)


def branch_moduli(
    chain: Sequence[int | float],
    n: float,
    *,
    wing: int = 1,
) -> tuple[float, float, float, float]:
    """|z| at each VA branch for fixed (A, n, wing)."""
    return tuple(
        abs(SpacetimeCell.at(chain, n, branch, wing).z)
        for branch in BranchKind
    )


def band_id_for_modulus(
    z_modulus: float,
    branch_moduli: Sequence[float],
) -> int:
    """argmin_b | |z_b| − |z| | over four branch reference moduli."""
    if len(branch_moduli) != NUM_BANDS:
        raise ValueError(f"expected {NUM_BANDS} branch moduli, got {len(branch_moduli)}")
    return min(
        range(NUM_BANDS),
        key=lambda i: abs(float(branch_moduli[i]) - z_modulus),
    )


def z_obs_va1_va2(
    chain: Sequence[int | float],
    n: float,
    *,
    wing: int = 1,
) -> float:
    """Real observable from VA1 + VA2 spring sum."""
    va1 = SpacetimeCell.at(chain, n, BranchKind.VA1, wing)
    va2 = SpacetimeCell.at(chain, n, BranchKind.VA2, wing)
    return float((va1.z + va2.z).real)


def band_profile_for_cell(
    cell: SpacetimeCell,
    *,
    wing: int = 1,
) -> ZBandProfile:
    """Band profile for primary cell using its chain and rail n."""
    if not cell.chain:
        z_mod = abs(cell.z)
        moduli = (z_mod, z_mod, z_mod, z_mod)
        return ZBandProfile(
            band_id=0,
            z_modulus=z_mod,
            branch_moduli=moduli,
            z_obs=float(cell.z.real),
        )
    moduli = branch_moduli(cell.chain, cell.n, wing=wing)
    z_mod = abs(cell.z)
    return ZBandProfile(
        band_id=band_id_for_modulus(z_mod, moduli),
        z_modulus=z_mod,
        branch_moduli=moduli,
        z_obs=z_obs_va1_va2(cell.chain, cell.n, wing=wing),
    )


def band_profile_for_word(
    registry,
    word: str,
    *,
    n: int = DEFAULT_ANCHOR_N,
    wing: int = 1,
) -> ZBandProfile:
    """BIT 1 cell → BIT 5 band profile."""
    cell = word_to_spacetime_cell(registry, word, n=n, wing=wing)
    return band_profile_for_cell(cell, wing=wing)


def wing_band_map(
    chain: Sequence[int | float],
    n: float,
) -> dict[int, int]:
    """
    All 32 wings → band id.

    Each wing compares |z| to the four branch moduli at the same wing index.
    """
    bands: dict[int, int] = {}
    for lid in LatticeId:
        branch, vector = lattice_id_parts(lid)
        cell = SpacetimeCell.at(chain, n, branch, vector.index)
        refs = branch_moduli(chain, n, wing=vector.index)
        bands[int(lid)] = band_id_for_modulus(abs(cell.z), refs)
    return bands


def verify_bit05_gate(
    *,
    chain: tuple[int, ...] = (3, 5, 7),
    n: float = 5,
) -> tuple[bool, list[str]]:
    """
    BIT 5 gate: (3,5,7) @ n=5 has exactly 4 bands, 8 wings each.
    """
    failures: list[str] = []
    moduli = branch_moduli(chain, n, wing=1)
    if len(set(moduli)) != NUM_BANDS:
        failures.append(f"expected 4 distinct branch moduli at w1, got {moduli}")

    z_obs = z_obs_va1_va2(chain, n, wing=1)
    va1 = SpacetimeCell.at(chain, n, BranchKind.VA1, wing=1)
    va2 = SpacetimeCell.at(chain, n, BranchKind.VA2, wing=1)
    if abs((va1.z + va2.z).imag) > 1e-9:
        failures.append(f"VA1+VA2 Im should cancel, got {(va1.z + va2.z).imag}")

    bands = wing_band_map(chain, n)
    counts = Counter(bands.values())
    if len(counts) != NUM_BANDS:
        failures.append(f"expected 4 bands across 32 wings, got {dict(counts)}")
    for bid in range(NUM_BANDS):
        if counts.get(bid, 0) != 8:
            failures.append(f"band {bid}: expected 8 wings, got {counts.get(bid, 0)}")

    cell = SpacetimeCell.at(chain, n, BranchKind.VA1, wing=1)
    prof = band_profile_for_cell(cell)
    if prof.band_id != 0:
        failures.append(f"VA1 w1 should be band 0, got {prof.band_id}")
    if abs(prof.z_obs - z_obs) > 1e-9:
        failures.append(f"z_obs mismatch {prof.z_obs} vs {z_obs}")

    return len(failures) == 0, failures
