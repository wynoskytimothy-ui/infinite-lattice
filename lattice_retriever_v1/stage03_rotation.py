"""
Stage 03 — Frequency sets the rotation.

Each symbol's corpus frequency (doc-frequency) contributes to a wing/quadrant
1..32. Same letters in different order → different profile → different place.
Anagrams stop colliding before promotion (stage 04).

Uses SpacetimeCell placement from pipeline bit_01 (hub chain + wing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from aethos_lattice import BranchKind
from aethos_physics import SpacetimeCell
from aethos_words import word_sorted_chain
from pipeline.bit_01_word_cell import (
    DEFAULT_ANCHOR_N,
    spacetime_cell_at_branch,
)

from lattice_retriever_v1.stage01_symbols import SymbolPrimeSequence, symbols_to_primes

NUM_QUADRANTS = 32


def wing_from_frequency_profile(dfs: tuple[int, ...]) -> int:
    """
    Map ordered per-symbol doc-frequencies → quadrant 1..32.

    Position-weighted log sum: earlier symbols with high df pull rotation
    differently than the same multiset in another order.
    """
    if not dfs:
        return 1
    score = 0.0
    for i, d in enumerate(dfs):
        score += (i + 1) * math.log1p(max(1, d))
    return int(score % NUM_QUADRANTS) + 1


def wing_and_branch_from_quadrant(quadrant: int) -> tuple[int, BranchKind]:
    """32 quadrants = 8 wings × 4 branches (VA1..VA4)."""
    q = max(1, min(NUM_QUADRANTS, quadrant))
    idx = q - 1
    wing = (idx % 8) + 1
    branch = BranchKind((idx // 8) + 1)
    return wing, branch


@dataclass(frozen=True)
class RotationPlacement:
    """Token placement after frequency-weighted rotation."""

    text: str
    wing: int
    quadrant: int
    branch: BranchKind
    cell: SpacetimeCell
    frequency_profile: tuple[int, ...]

    def explain(self) -> dict:
        c = self.cell
        return {
            "text": self.text,
            "wing": self.wing,
            "quadrant": self.quadrant,
            "branch": self.branch.name,
            "frequency_profile": list(self.frequency_profile),
            "cell": {
                "z_real": round(c.z.real, 6),
                "z_imag": round(c.z.imag, 6),
                "zeta": round(c.zeta, 6),
            },
        }


def rotate_token(
    text: str,
    df_by_char: dict[str, int],
    *,
    n: int = DEFAULT_ANCHOR_N,
) -> RotationPlacement:
    """Place token in 3D complex plane using symbol order + corpus frequencies."""
    seq: SymbolPrimeSequence = symbols_to_primes(text)
    profile = tuple(max(1, df_by_char.get(s.char, 1)) for s in seq.spans)
    quadrant = wing_from_frequency_profile(profile)
    wing, branch = wing_and_branch_from_quadrant(quadrant)
    chain = word_sorted_chain(text)
    if not chain:
        raise ValueError(f"no alphabetic symbols in {text!r}")
    cell = spacetime_cell_at_branch(chain, n, branch, wing)
    return RotationPlacement(
        text=text,
        wing=wing,
        quadrant=quadrant,
        branch=branch,
        cell=cell,
        frequency_profile=profile,
    )
