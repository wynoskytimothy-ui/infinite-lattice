"""
BIT 2 — Quantized attractor key κ(z, ζ)

Math:
  κ(z, ζ; q) = (round(Re z / q), round(Im z / q), round(ζ / q))  ∈ ℤ³
  N(κ; r)    = { κ' : ||κ' − κ||_∞ ≤ r }

Branch fan (BIT 1 rotation): κ_k = κ(Ψ_k) for k = 0..3 around j.

Default q = 1 (integer spring coords at anchor n₀).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from pipeline.bit_01_word_cell import (
    four_branch_cells,
    spacetime_cell_at_branch,
)

AttractorKey = tuple[int, int, int]

DEFAULT_QUANTIZE = 1.0


def kappa(
    z: complex,
    zeta: float,
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> AttractorKey:
    """Quantize spring (z, ζ) to integer bucket κ."""
    q = quantize if quantize > 0 else DEFAULT_QUANTIZE
    return (
        int(round(z.real / q)),
        int(round(z.imag / q)),
        int(round(zeta / q)),
    )


def kappa_from_cell(
    cell: SpacetimeCell,
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> AttractorKey:
    return kappa(cell.z, cell.zeta, quantize=quantize)


def attractor_neighbors(
    key: AttractorKey,
    *,
    radius: int = 1,
) -> set[AttractorKey]:
    """Chebyshev neighborhood N(κ; r) in quantized spring space."""
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    rx, ry, rz = key
    out: set[AttractorKey] = set()
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                out.add((rx + dx, ry + dy, rz + dz))
    return out


def keys_from_cells(
    cells: Iterable[SpacetimeCell],
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> set[AttractorKey]:
    return {kappa_from_cell(c, quantize=quantize) for c in cells}


def kappa_branch_fan(
    chain: Sequence[int | float],
    n: float,
    *,
    wing: int = 1,
    quantize: float = DEFAULT_QUANTIZE,
) -> tuple[AttractorKey, ...]:
    """κ at each quarter-turn branch k=0..3 (rotation around j)."""
    return tuple(
        kappa_from_cell(c, quantize=quantize)
        for c in four_branch_cells(chain, n, wing=wing)
    )


def kappa_at_branch(
    chain: Sequence[int | float],
    n: float,
    branch: BranchKind,
    *,
    wing: int = 1,
    quantize: float = DEFAULT_QUANTIZE,
) -> AttractorKey:
    cell = spacetime_cell_at_branch(chain, n, branch, wing)
    return kappa_from_cell(cell, quantize=quantize)


def verify_bit02_gate(
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> tuple[bool, list[str]]:
    """
    BIT 2 gate:
      - Triple witness (3,5,7) subset meet → κ = (12, 5, 15)
      - N(κ; 0) is singleton; N(κ; 1) has (2r+1)³ = 27 keys
    """
    failures: list[str] = []
    cell = SpacetimeCell.promote_witness((3, 5, 7), (3, 5))
    key = kappa_from_cell(cell, quantize=quantize)
    if key != (12, 5, 15):
        failures.append(f"triple κ expected (12,5,15), got {key}")
    n0 = attractor_neighbors(key, radius=0)
    if n0 != {key}:
        failures.append(f"N(κ;0) should be singleton, got {len(n0)} keys")
    n1 = attractor_neighbors(key, radius=1)
    if len(n1) != 27:
        failures.append(f"N(κ;1) expected 27 keys, got {len(n1)}")
    return (len(failures) == 0, failures)


def verify_bit02_rotation_gate(
    *,
    quantize: float = DEFAULT_QUANTIZE,
) -> tuple[bool, list[str]]:
    """
    BIT 2 rotation gate:
      - VA1 κ at (3,5,7) n=5 stays (12, 5, 15)
      - four-branch fan yields 4 distinct κ keys
    """
    failures: list[str] = []
    chain = (3, 5, 7)
    n = 5.0
    va1 = kappa_at_branch(chain, n, BranchKind.VA1, quantize=quantize)
    if va1 != (12, 5, 15):
        failures.append(f"VA1 rotation κ expected (12,5,15), got {va1}")
    fan = kappa_branch_fan(chain, n, quantize=quantize)
    if len(set(fan)) != 4:
        failures.append(f"branch fan expected 4 distinct κ, got {fan}")
    return (len(failures) == 0, failures)
