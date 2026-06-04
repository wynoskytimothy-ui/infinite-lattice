"""
Material blob → active anchor set (C6).

A "giant blob of electrons" selects which counting set labels active nodes.
Different species → different addresses, meets, ocean edges, observable pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

from aethos_sequences import SequenceKind


@dataclass(frozen=True)
class ElectronBlob:
    """
    Coarse material parameters (normalized MODEL handles).

    density:   0..1 — electron packing / Fermi scale proxy
    coupling:  0..1 — ocean / spring coupling strength
    temperature_k: thermal scale (reserved for future clock/noise rules)
    """

    density: float = 0.5
    coupling: float = 0.5
    temperature_k: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "density", _clamp01(self.density))
        object.__setattr__(self, "coupling", _clamp01(self.coupling))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# Density ladder: which SequenceKind the blob prefers (MODEL — not FIT).
_SPECIES_LADDER: tuple[tuple[float, SequenceKind], ...] = (
    (0.0, SequenceKind.PRIMES),
    (0.25, SequenceKind.EVENS),
    (0.5, SequenceKind.POWERS_OF_2),
    (0.75, SequenceKind.FIBONACCI),
)


def assign_species(
    blob: ElectronBlob,
    node_index: int,
    *,
    origin_depth: int = 0,
) -> SequenceKind:
    """
    Pick anchor set for one active node from blob parameters.

    score = density + 0.15*coupling + 0.03*origin_depth + small node hash
    """
    score = blob.density + 0.15 * blob.coupling + 0.03 * float(origin_depth)
    score += 0.02 * ((node_index * 7 + 3) % 11) / 11.0
    score = _clamp01(score)
    chosen = _SPECIES_LADDER[0][1]
    for threshold, kind in _SPECIES_LADDER:
        if score >= threshold:
            chosen = kind
    return chosen


def species_profile(blob: ElectronBlob, count: int) -> dict[str, int]:
    """Count species assignment over node indices 0..count-1 (uniform depth 0)."""
    tallies: dict[str, int] = {}
    for i in range(count):
        k = assign_species(blob, i).value
        tallies[k] = tallies.get(k, 0) + 1
    return tallies
