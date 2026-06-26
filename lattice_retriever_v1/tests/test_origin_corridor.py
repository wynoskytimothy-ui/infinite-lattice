"""Origin corridor — regeneration bit-identical at depth 0."""

from __future__ import annotations

from aethos_origins import OriginTree

from lattice_retriever_v1.brain_loop import BrainLoop
from lattice_retriever_v1.origin_corridor import (
    corridor_key_with_origin,
    origin_offset_coord,
    resolve_origin,
)
from lattice_retriever_v1.stage05_free_token import free_token_address
from lattice_retriever_v1.stage08_retrieve import FIXTURE_CORPUS


def test_origin_offset_coord_depth_zero_is_identity():
    tree = OriginTree.bootstrap(max_depth=1)
    origin = resolve_origin(tree, "O0")
    base = (12, 34, 56)
    assert origin.coord == (0, 0, 0)
    assert origin_offset_coord(base, origin) == base


def test_regeneration_bit_identical_with_without_origin_at_depth_zero():
    tree = OriginTree.bootstrap(max_depth=2)
    primes = (3, 5)
    n = 7
    explicit = corridor_key_with_origin(primes, n, origin_id="O0", tree=tree)
    default = corridor_key_with_origin(primes, n, tree=tree)

    assert explicit["regenerate_bit_identical_depth_0"] is True
    assert default["regenerate_bit_identical_depth_0"] is True
    assert explicit["corridor_key"] == default["corridor_key"]
    assert explicit["lattice_L01_base"] == explicit["lattice_L01_offset"]
    assert explicit["lattice_L01_base"] == default["lattice_L01_base"]

    addr = free_token_address(primes[0], primes[1], transgressor_n=n)
    assert list(addr.corridor_key) == explicit["corridor_key"]


def test_deeper_origin_shifts_coords_not_corridor_key():
    tree = OriginTree.bootstrap(max_depth=2)
    primes = (3, 5)
    shallow = corridor_key_with_origin(primes, 7, origin_id="O0", tree=tree)
    deep_id = next(o.id for o in tree.walk() if o.depth == 1)
    deep = corridor_key_with_origin(primes, 7, origin_id=deep_id, tree=tree)

    assert shallow["corridor_key"] == deep["corridor_key"]
    assert shallow["lattice_L01_base"] == deep["lattice_L01_base"]
    assert shallow["lattice_L01_offset"] != deep["lattice_L01_offset"]


def test_brain_loop_explain_includes_origin_corridor():
    loop = BrainLoop(enable_neuron_room=True)
    loop.index_corpus(FIXTURE_CORPUS)
    loop.retrieve("cancer mutation rare", limit=3)
    ex = loop.explain_last()
    nr = ex.get("neuron_room")
    assert nr is not None
    oc = nr.get("origin_corridor")
    assert oc is not None
    assert oc["regenerate_bit_identical_depth_0"] is True
    assert "corridor_key" in oc
