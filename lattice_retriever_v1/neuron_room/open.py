"""Open a neuron room — lazy 32-wing materialization from lattice formula."""

from __future__ import annotations

from functools import reduce
from operator import mul

from aethos_lattice import LatticeId, lattice_id_parts

from lattice_retriever_v1.neuron_room.types import (
    NeuronRoom,
    NeuronSeed,
    RoomStatus,
    WingAgent,
)
from lattice_retriever_v1.stage02_intersections import (
    DEFAULT_TRANSGRESSOR_N,
    NUM_LATTICES,
    lattice_signature,
)


def seed_from_primes(
    *primes: int,
    quadrant: int = 1,
    n: int = DEFAULT_TRANSGRESSOR_N,
    invoke_order: tuple[int, ...] | None = None,
) -> NeuronSeed:
    """Build a seed from prime meet inputs; wings stay unmaterialized."""
    ps = tuple(primes)
    order = invoke_order if invoke_order is not None else ps
    distinct = sorted(set(ps))
    seed_id = reduce(mul, distinct, 1) if distinct else 0
    return NeuronSeed(
        seed_id=seed_id,
        primes=ps,
        k=len(ps),
        quadrant=max(1, min(NUM_LATTICES, int(quadrant))),
        transgressor_n=n,
        invoke_order=order,
    )


def _wing_agent(
    wing_id: int,
    coord: tuple[int, int, int],
    *,
    lit: bool,
) -> WingAgent:
    branch, _vector = lattice_id_parts(LatticeId(wing_id))
    return WingAgent(
        wing_id=wing_id,
        branch=branch,
        quadrant=wing_id,
        coord=coord,
        lit=lit,
    )


def open_room(seed: NeuronSeed) -> NeuronRoom:
    """Materialize all 32 wings from lattice_signature / LatticeBank32K."""
    sig = lattice_signature(seed.primes, n=seed.transgressor_n)
    wings = tuple(
        _wing_agent(i + 1, sig[i], lit=True)
        for i in range(NUM_LATTICES)
    )
    return NeuronRoom(seed=seed, wings=wings, status=RoomStatus.OPEN)


def dormant_room(seed: NeuronSeed) -> NeuronRoom:
    """Seed-only room — wings not materialized."""
    return NeuronRoom(seed=seed, wings=None, status=RoomStatus.DORMANT)
