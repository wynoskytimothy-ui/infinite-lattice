"""Neuron room types — seed, wing agents, room state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aethos_lattice import BranchKind


class RoomStatus(str, Enum):
    DORMANT = "DORMANT"
    OPEN = "OPEN"


@dataclass(frozen=True)
class NeuronSeed:
    """Formula inputs for one neuron room — wings materialize on open."""

    seed_id: int
    primes: tuple[int, ...]
    k: int
    quadrant: int
    transgressor_n: int
    invoke_order: tuple[int, ...] | None = None


@dataclass(frozen=True)
class WingAgent:
    """One of 32 lattice wings after the room opens."""

    wing_id: int
    branch: BranchKind
    quadrant: int
    coord: tuple[int, int, int]
    lit: bool


@dataclass
class NeuronRoom:
    """Dormant until open_room materializes wing agents from the seed formula."""

    seed: NeuronSeed
    wings: tuple[WingAgent, ...] | None = None
    status: RoomStatus = RoomStatus.DORMANT
