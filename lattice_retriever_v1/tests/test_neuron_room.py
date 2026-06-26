"""Neuron room — open materializes 32 wings from lattice formula."""

from lattice_retriever_v1.neuron_room import (
    RoomStatus,
    dormant_room,
    open_room,
    seed_from_primes,
)
from lattice_retriever_v1.stage02_intersections import lattice_signature


def test_open_room_l01_matches_lattice_signature() -> None:
    seed = seed_from_primes(73, 23)
    room = open_room(seed)
    direct = lattice_signature((73, 23), n=seed.transgressor_n)
    assert room.wings is not None
    assert room.wings[0].coord == direct[0]


def test_open_room_has_32_wings() -> None:
    seed = seed_from_primes(19, 29, 47)
    room = open_room(seed)
    assert room.status == RoomStatus.OPEN
    assert room.wings is not None
    assert len(room.wings) == 32
    assert all(w.lit for w in room.wings)
    assert {w.wing_id for w in room.wings} == set(range(1, 33))


def test_unopened_wings_not_materialized() -> None:
    seed = seed_from_primes(73, 23)
    room = dormant_room(seed)
    assert room.status == RoomStatus.DORMANT
    assert room.wings is None

    opened = open_room(seed)
    assert opened.wings is not None
    assert len(opened.wings) == 32
