"""
Hub wire format — lattice address α + critical-line pin (8 B).

Stores generative transmission (prime, rail n, branch, wing, leg_sum on j=1+i)
instead of materialized float (X,Y,Z). Regenerate Ψ via SpacetimeCell.at at query time.

Same lattice as trng; cleaner 3D complex plane write-up (ONTOLOGY.md §B.6).
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_lattice import BranchKind, LatticeId, lattice_id_parts
from aethos_physics import SpacetimeCell

# prime(4) + leg_sum(2) + flags(1) + rail_n(1)
WIRE_BYTES = 8


@dataclass(frozen=True)
class CriticalLinePin:
    """
    Critical-line anchor on spring plane: leg_sum = Re(z)+Im(z) along j=(1,1).

    branch (VA1..4) + wing (1..8) complete address α with prime chain (via word).
    """

    prime: int
    leg_sum: int
    band_side: bool  # True when Im > Re (half-wedge above Re=Im)
    rail_n: int = 7
    branch: int = 1
    wing: int = 1

    @classmethod
    def from_coord(
        cls,
        coord: tuple[float, float, float],
        prime: int,
        *,
        rail_n: int = 7,
        branch: int = 1,
        wing: int = 1,
    ) -> CriticalLinePin:
        x, y, _z = coord
        return cls(
            prime=int(prime),
            leg_sum=int(round(x + y)),
            band_side=float(y) > float(x),
            rail_n=int(rail_n),
            branch=int(branch),
            wing=int(wing),
        )

    @classmethod
    def from_cell(cls, cell: SpacetimeCell, prime: int) -> CriticalLinePin:
        branch = int(cell.branch) if cell.branch is not None else 1
        wing = int(cell.wing) if cell.wing is not None else 1
        return cls(
            prime=int(prime),
            leg_sum=int(round(cell.z.real + cell.z.imag)),
            band_side=float(cell.z.imag) > float(cell.z.real),
            rail_n=int(cell.n),
            branch=branch,
            wing=wing,
        )


def leg_sum_im_led(coord: tuple[float, float, float]) -> tuple[int, bool]:
    """Scalar along critical line j and half-wedge flag."""
    x, y = int(round(coord[0])), int(round(coord[1]))
    return x + y, y > x


def hub_coord_from_word(
    registry,
    word: str,
    pin: CriticalLinePin,
) -> tuple[float, float, float]:
    """Regenerate hub formula_coord from α pin + registry chain."""
    from pipeline.bit_01_word_cell import word_to_spacetime_cell

    cell = word_to_spacetime_cell(
        registry,
        word,
        n=pin.rail_n,
        branch=BranchKind(pin.branch),
        wing=pin.wing,
    )
    return (float(cell.z.real), float(cell.z.imag), float(cell.zeta))


def wing_coords_for_word(
    registry,
    word: str,
    *,
    rail_n: int = 7,
    consensus_wings: tuple[int, ...],
) -> dict[int, tuple[float, float, float]]:
    """On-demand consensus wing coords — no stored float tables."""
    from pipeline.bit_01_word_cell import word_to_spacetime_cell

    out: dict[int, tuple[float, float, float]] = {}
    for lid in consensus_wings:
        branch, vec = lattice_id_parts(LatticeId(lid))
        cell = word_to_spacetime_cell(
            registry,
            word,
            n=rail_n,
            branch=branch,
            wing=vec.index,
        )
        out[lid] = (float(cell.z.real), float(cell.z.imag), float(cell.zeta))
    return out


def pin_wire_bytes() -> int:
    return WIRE_BYTES


@dataclass(frozen=True)
class LegSumMeetKey:
    """
    S-partner meet bucket on critical line j: Re+Im = leg_sum at rail n.

    Collapses half-wedge pairs (z, S(z)) to one meet posting — same leg_sum,
    opposite band_side.
    """

    leg_sum: int
    rail_n: int = 7


def leg_sum_meet_key(pin: CriticalLinePin) -> LegSumMeetKey:
    return LegSumMeetKey(leg_sum=int(pin.leg_sum), rail_n=int(pin.rail_n))


def leg_sum_meet_key_from_coord(
    coord: tuple[float, float, float],
    *,
    rail_n: int = 7,
) -> LegSumMeetKey:
    leg, _ = leg_sum_im_led(coord)
    return LegSumMeetKey(leg_sum=leg, rail_n=rail_n)


def pin_for_query_word(
    registry,
    word: str,
    coord: tuple[float, float, float],
    *,
    anchor_n: int = 7,
    lattice_id: LatticeId = LatticeId.L01,
) -> CriticalLinePin:
    """Build meet pin for a query/hub word at lattice address."""
    tok = registry.resolve_token(word.lower())
    branch, vec = lattice_id_parts(lattice_id)
    return CriticalLinePin.from_coord(
        coord,
        int(tok.prime),
        rail_n=anchor_n,
        branch=int(branch),
        wing=vec.index,
    )
